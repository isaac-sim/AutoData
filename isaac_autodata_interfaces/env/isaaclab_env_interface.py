# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data generation interface for Isaac Lab environments.

Native ports of the env-configuration and synchronous step-loop helpers that the generation
entrypoint needs, so the CLI no longer depends on ``isaaclab_mimic.datagen.generation``.

* :func:`setup_env_config` — parse and adapt an env config for recording generated demos.
* :func:`env_loop` — synchronous step loop that drives the env while async data generators feed
  actions through queues.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import torch
from typing import Any

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy


def setup_output_paths(output_file_path: str) -> tuple[str, str]:
    """Split an output dataset path into ``(directory, file_name_without_extension)``.

    Creates the output directory if it does not exist.
    """
    output_dir = os.path.dirname(output_file_path)
    output_file_name = os.path.splitext(os.path.basename(output_file_path))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir, output_file_name


def get_env_name_from_dataset(input_file_path: str) -> str:
    """Read the recorded environment name from an HDF5 source dataset.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dataset file {input_file_path} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(input_file_path)
    env_name = dataset_file_handler.get_env_name()
    assert env_name is not None, "Environment name not found in dataset"
    return env_name


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_policy_params: GenerationPolicy,
    recorder_cfg: RecorderManagerBaseCfg | None = None,
) -> tuple[Any, Any]:
    """Configure the environment for data generation.

    The generation config travels as the task descriptor's
    :class:`~isaac_autodata_interfaces.tasks.generation_policy_spec.GenerationPolicy` rather than
    being read from the env cfg. This function reads only the field it needs
    (``keep_failed``) to shape the recorder export mode.

    Args:
        env_name: Name of the environment.
        output_dir: Directory to save output.
        output_file_name: Name of output file.
        num_envs: Number of environments to run.
        device: Device to run on.
        generation_policy_params: The task descriptor's generation config (source of truth).
        recorder_cfg: Optional recorder manager config; defaults to an action/state recorder.

    Returns:
        A tuple of the environment configuration and the success termination condition.
    """
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
    env_cfg.env_name = env_name

    # Extract success checking function
    assert hasattr(env_cfg.terminations, "success"), "No success termination term was found in the environment."
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Configure for data generation
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    # Setup recorders
    if recorder_cfg is None:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    else:
        env_cfg.recorders = recorder_cfg
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if generation_policy_params.keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def env_loop(
    env: Any,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    generation_policy_params: GenerationPolicy,
    stats: dict,
    data_gen_tasks: asyncio.Future | None = None,
) -> bool:
    """Synchronous step loop for the environment.

    Steps the env in inference mode, draining actions produced by the async data generators from
    ``env_action_queue`` and servicing reset requests from ``env_reset_queue``. Outcome counters
    are read from ``stats`` (populated by the data generator tasks) to report progress and decide
    when enough demos/attempts have been collected.

    The stop condition is read from ``generation_policy_params`` (the task descriptor's generation
    config) rather than from ``env.cfg.datagen_config``: ``num_trials`` interpreted as successes
    when ``guarantee_success`` is set, otherwise as total attempts.

    Args:
        env: The environment to run the main step loop on.
        env_reset_queue: The asyncio queue carrying per-env reset requests.
        env_action_queue: The asyncio queue carrying ``(env_id, action)`` pairs to execute.
        asyncio_event_loop: The main asyncio event loop.
        generation_policy_params: The task descriptor's generation config (source of truth).
        stats: Shared dict with ``num_success``, ``num_failures``, and ``num_attempts`` counters.
        data_gen_tasks: Gathered future for all data generation tasks. When provided, the loop
            exits early if all tasks finish unexpectedly (e.g. due to an unhandled exception).

    Returns:
        Whether the generation policy's requested trial target was reached.
    """
    num_trials = generation_policy_params.num_trials
    guarantee_success = generation_policy_params.guarantee_success
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # check if any environment needs to be reset while waiting for actions
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                if data_gen_tasks is not None and data_gen_tasks.done():
                    exc = data_gen_tasks.exception()
                    if exc is not None:
                        raise exc
                    return False
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)

            # batch-fetch all per-env actions in one gather instead of sequential blocking calls
            get_tasks = [env_action_queue.get() for _ in range(env.num_envs)]
            results = asyncio_event_loop.run_until_complete(asyncio.gather(*get_tasks))
            for env_id, action in results:
                actions[env_id] = action

            # perform action on environment
            env.step(actions)

            # mark done so the data generators can continue with the step results
            for _ in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != stats["num_attempts"]:
                prev_num_attempts = stats["num_attempts"]
                num_success = stats["num_success"]
                num_attempts = stats["num_attempts"]
                generated_success_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                print("")
                print("*" * 50, "\033[K")
                print(f"{num_success}/{num_attempts} ({generated_success_rate:.1f}%) successful demos generated\033[K")
                print("*" * 50, "\033[K")

                # termination condition is on enough successes if guarantee_success else enough attempts
                check_val = num_success if guarantee_success else num_attempts
                if check_val >= num_trials:
                    print(f"Reached {num_trials} {'successes' if guarantee_success else 'attempts'}. Exiting.")
                    return True

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                return False

    # Do not close env here: async data generator tasks may still be running.
    # Caller must close env after cancelling and awaiting those tasks.
    return False
