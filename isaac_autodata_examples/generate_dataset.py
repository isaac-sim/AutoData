#!/usr/bin/env python
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data generation entrypoint.

Usage::

    python isaac_autodata_examples/generate_dataset.py \\
        --task <task_name> \\
        --alg {mimicgen|dexmimicgen|skillgen} \\
        --task_descriptor <task_descriptor.yaml> \\
        --embodiment <embodiment.yaml> \\
        --input_file <source.hdf5> \\
        --output_file <out.hdf5> \\
        --generation_num_trials <N> \\
        --num_envs <N>

The ``--alg`` choice selects the :class:`GenerationAlgorithm` plug-in driving the run:

* ``mimicgen`` — single-arm MimicGen.
* ``dexmimicgen`` — two-arm MimicGen with subtask coordination constraints.
* ``skillgen`` — single-arm SkillGen. The CLI builds one
  :class:`~isaac_autodata_interfaces.motion_planners.curobo.CuroboPlanner` per env to satisfy the
  motion-planner protocol. Requires cuRobo installed in the env (see
  ``submodules/IsaacLab-Arena/submodules/IsaacLab/docs/source/overview/imitation-learning/skillgen.rst``).

The CLI composes a :class:`Datastream` from the task descriptor YAML, the embodiment YAML, the
live env, and the HDF5 source dataset, then hands it to :class:`DataGenerator`.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Hardcoded to keep argparse importable without pulling in the heavy core package.
# Add new algorithms here when registering them in isaac_autodata_core.algorithms.
_ALG_CHOICES = ["mimicgen", "dexmimicgen", "skillgen"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--alg",
    type=str,
    choices=_ALG_CHOICES,
    required=True,
    help="Generation algorithm. skillgen needs a motion planner (auto-wired from upstream curobo).",
)
parser.add_argument(
    "--task_descriptor",
    type=str,
    required=True,
    help="Path to the task descriptor YAML (defines subtasks, constraints, generation policy).",
)
parser.add_argument(
    "--embodiment",
    type=str,
    required=True,
    help="Path to the embodiment YAML (defines the robot's pose ↔ action transforms).",
)
parser.add_argument("--generation_num_trials", type=int, default=None, help="Number of demos to generate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--input_file", type=str, required=True, help="Source dataset HDF5 file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="Destination HDF5 for generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask for interactive debugging.",
)
parser.add_argument(
    "--curobo_version",
    type=str,
    choices=["v1", "v2", "auto"],
    default="auto",
    help=(
        "Which cuRobo backend to use for SkillGen motion planning. 'auto' picks v2 when the v2 "
        "MotionPlanner API is importable, otherwise falls back to v1. Ignored unless --alg skillgen."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402
import traceback  # noqa: E402

import isaaclab_mimic.envs  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab_mimic.datagen.generation import env_loop, setup_env_config  # noqa: E402
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths  # noqa: E402

from isaac_autodata_core import DataGenerator, get_algorithm  # noqa: E402
from isaac_autodata_core.algorithms import REGISTERED_ALGORITHMS  # noqa: E402
from isaac_autodata_interfaces.datastream import Datastream  # noqa: E402
from isaac_autodata_interfaces.embodiments import embodiment_adapter_from_yaml  # noqa: E402
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor  # noqa: E402


async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term,
    pause_subtask: bool,
    stats: dict,
) -> None:
    """Repeatedly call ``data_generator.generate`` and tally outcomes into ``stats``."""
    while True:
        try:
            result = await data_generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                pause_subtask=pause_subtask,
            )
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            # The upstream ``env_loop`` blocks on ``env_action_queue.get()`` in the main thread
            # and has no signal path for async-task failures. Re-raising leaves the loop hung
            # forever, holding HDF5 locks until the process is SIGKILLed. Force-exit so the
            # traceback above is the last word.
            os._exit(1)
            raise exc  # unreachable; kept for type-checkers

        if result.success:
            stats["num_success"] += 1
        else:
            stats["num_failures"] += 1
        stats["num_attempts"] += 1
        print(
            f"[TRIAL] env={env_id}  outcome={'SUCCESS' if result.success else 'FAIL'}  "
            f"running: {stats['num_success']}/{stats['num_attempts']} "
            f"({stats['num_failures']} failed)",
            flush=True,
        )


def build_datastream(
    env: ManagerBasedRLMimicEnv,
    input_file: str,
    task_descriptor_yaml: str,
    embodiment_yaml: str,
    uses_start_signals: bool,
) -> Datastream:
    """Compose the read facade (task + embodiment + source pool) over the live env.

    Built before the motion planners so the planners can read all world state through it. The
    pool lock is created up front and shared with the async generation tasks.
    """
    task_descriptor = TaskDescriptor.from_yaml(task_descriptor_yaml)
    embodiment_adapter = embodiment_adapter_from_yaml(embodiment_yaml)
    datastream = Datastream(
        env=env,
        task_descriptor=task_descriptor,
        embodiment_adapter=embodiment_adapter,
        source_dataset_path=input_file,
        asyncio_lock=asyncio.Lock(),
        uses_start_signals=uses_start_signals,
    )
    print(f"Loaded {datastream.num_source_demos} source episodes into the datagen pool")
    return datastream


def setup_async_generation(
    datastream: Datastream,
    num_envs: int,
    success_term,
    algorithm,
    pause_subtask: bool,
) -> dict:
    """Build a single :class:`DataGenerator` over ``datastream`` and ``num_envs`` async tasks."""
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue: asyncio.Queue = asyncio.Queue()
    env_action_queue: asyncio.Queue = asyncio.Queue()
    env = datastream.get_env()

    data_generator = DataGenerator(datastream=datastream, algorithm=algorithm)
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}

    tasks = []
    for env_id in range(num_envs):
        task = asyncio_event_loop.create_task(
            run_data_generator(
                env=env,
                env_id=env_id,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                data_generator=data_generator,
                success_term=success_term,
                pause_subtask=pause_subtask,
                stats=stats,
            )
        )
        tasks.append(task)

    return {
        "tasks": tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": datastream.source_pool,
        "stats": stats,
    }


def _build_motion_planners(datastream, num_envs: int, env_name: str, curobo_version: str = "auto") -> dict:
    """Construct one motion planner per env_id satisfying the SkillGen planner interface.

    Dispatches between the v1 backend (:mod:`isaac_autodata_interfaces.motion_planners.curobo`)
    and the v2 backend (:mod:`isaac_autodata_interfaces.motion_planners.curobo_v2`) via
    :func:`isaac_autodata_interfaces.motion_planners.get_curobo_planner_classes`. Both backends
    read all world state through the shared :class:`Datastream`.
    """
    from isaac_autodata_interfaces.motion_planners import detect_curobo_version, get_curobo_planner_classes

    resolved_version = detect_curobo_version() if curobo_version == "auto" else curobo_version
    print(f"[generate_dataset] Using cuRobo {resolved_version} backend")
    planner_cls, planner_cfg_cls = get_curobo_planner_classes(resolved_version)

    planners: dict[int, planner_cls] = {}
    for env_id in range(num_envs):
        planner_config = planner_cfg_cls.from_task_name(env_name)
        # Visualization is rerun-based; limit to env_id 0 to keep simulation responsive.
        if env_id != 0:
            planner_config.visualize_spheres = False
            planner_config.visualize_plan = False
        planners[env_id] = planner_cls(
            datastream=datastream,
            config=planner_config,
            env_id=env_id,
        )
    return planners


def _close_motion_planners(planners: dict | None) -> None:
    if not planners:
        return
    for env_id, planner in planners.items():
        if getattr(planner, "plan_visualizer", None) is not None:
            print(f"Closing plan visualizer for environment {env_id}")
            planner.plan_visualizer.close()
            planner.plan_visualizer = None
    planners.clear()


def main() -> None:
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError(f"Env {env_name!r} is not a ManagerBasedRLMimicEnv")

    # The algorithm's start-signal expectation is a class attribute, so we can read it before
    # instantiating (SkillGen can only be instantiated once the planners exist, and the planners
    # need the Datastream, which needs this flag). Mirror it onto the upstream env config so
    # anything downstream that still reads ``env.cfg.datagen_config.use_skillgen`` stays consistent.
    alg_cls = REGISTERED_ALGORITHMS[args_cli.alg]
    env_cfg.datagen_config.use_skillgen = alg_cls.uses_subtask_start_signals

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    env.reset()

    # Build the read facade first; the motion planners read all world state through it.
    datastream = build_datastream(
        env=env,
        input_file=args_cli.input_file,
        task_descriptor_yaml=args_cli.task_descriptor,
        embodiment_yaml=args_cli.embodiment,
        uses_start_signals=alg_cls.uses_subtask_start_signals,
    )

    # SkillGen needs one curobo planner per env. Mimic/DexMimic take no kwargs.
    motion_planners: dict | None = None
    alg_kwargs: dict = {}
    if args_cli.alg == "skillgen":
        motion_planners = _build_motion_planners(
            datastream, args_cli.num_envs, env_name, curobo_version=args_cli.curobo_version
        )
        alg_kwargs["motion_planners"] = motion_planners
    algorithm = get_algorithm(args_cli.alg, **alg_kwargs)

    try:
        async_components = setup_async_generation(
            datastream=datastream,
            num_envs=args_cli.num_envs,
            success_term=success_term,
            algorithm=algorithm,
            pause_subtask=args_cli.pause_subtask,
        )

        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        try:
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            print("Async tasks cancelled.")
        finally:
            data_gen_tasks.cancel()
            try:
                async_components["event_loop"].run_until_complete(data_gen_tasks)
            except asyncio.CancelledError:
                print("Remaining async tasks cleaned up.")
            except Exception as exc:
                print(f"Error cleaning up async tasks: {exc}")
            _close_motion_planners(motion_planners)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
    simulation_app.close()
