#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data generation entrypoint.

Usage::

    python scripts/generate_dataset.py \\
        --env_name <env_id> \\
        --alg {mimicgen|dexmimicgen|skillgen|softmimicgen} \\
        --task_descriptor <task_descriptor.yaml> \\
        --embodiment <embodiment.yaml> \\
        --env_profile <environment_profile.yaml> \\
        --input_file <source.hdf5> \\
        --output_file <out.hdf5> \\
        --generation_num_trials <N> \\
        --num_envs <N>

The ``--alg`` choice selects the :class:`GenerationAlgorithm` plug-in driving the run:

* ``mimicgen`` — single-arm MimicGen.
* ``dexmimicgen`` — two-arm MimicGen with subtask coordination constraints.
* ``skillgen`` — single-arm SkillGen. SkillGen depends on a motion-planner interface; until the
  planner code is ported into this repo, the CLI satisfies that interface with the upstream Arena
  ``CuroboPlanner``.
* ``softmimicgen`` — one- or two-arm MimicGen with deformable-object nodal registration.

The CLI composes a :class:`Datastream` from the task descriptor YAML, the embodiment YAML, the
live env, and the HDF5 source dataset, then hands it to :class:`DataGenerator`.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Hardcoded to keep argparse importable without pulling in the heavy core package.
# Add new algorithms here when registering them in autodata_core.algorithms.
_ALG_CHOICES = ["mimicgen", "dexmimicgen", "skillgen", "softmimicgen"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--env_name",
    type=str,
    default=None,
    help="Environment name. Overrides the env name recorded in the source dataset.",
)
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
parser.add_argument(
    "--env_profile",
    type=str,
    default=None,
    help=(
        "Optional environment profile YAML overlaid on the base task before env creation "
        "(scene additions, reset randomization, motion-planner profile)."
    ),
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
    "--result_file",
    type=str,
    default=None,
    help="Optional JSON path for the final structured generation result.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask for interactive debugging.",
)
parser.add_argument(
    "--visualize_plan",
    action="store_true",
    help="Visualize SkillGen motion plans in a Rerun viewer (env 0 only; requires the rerun package).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402
import traceback  # noqa: E402
from typing import Any  # noqa: E402

from autodata_core import DataGenerator, get_algorithm  # noqa: E402
from autodata_core.algorithms import REGISTERED_ALGORITHMS  # noqa: E402
from autodata_examples.envs import register_environment_for_run  # noqa: E402
from autodata_interfaces.datastream import Datastream  # noqa: E402
from autodata_interfaces.embodiments import embodiment_adapter_from_yaml  # noqa: E402
from autodata_interfaces.env import (  # noqa: E402
    EnvironmentProfile,
    env_loop,
    get_env_name_from_dataset,
    setup_env_config,
    setup_output_paths,
)
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor  # noqa: E402
from autodata_utils.generation_result import write_generation_result  # noqa: E402


async def run_data_generator(
    env: Any,
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
            raise exc

        if result.success:
            stats["num_success"] += 1
        else:
            stats["num_failures"] += 1
        stats["num_attempts"] += 1


def build_datastream(
    env: Any,
    input_file: str,
    task_descriptor: TaskDescriptor,
    embodiment_yaml: str,
) -> Datastream:
    """Compose the read interface (task + embodiment + source pool) over the live env.

    Built before the motion planners so the planners can read all world state through it.
    """
    embodiment_adapter = embodiment_adapter_from_yaml(embodiment_yaml)
    datastream = Datastream(
        env=env,
        task_descriptor=task_descriptor,
        embodiment_adapter=embodiment_adapter,
        source_dataset_path=input_file,
        uses_start_signals=task_descriptor.get_generation_policy().use_skillgen,
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


def _build_motion_planners(
    datastream, num_envs: int, env_name: str, *, planner_profile: str | None = None, visualize_plan: bool = False
) -> dict:
    """Construct one cuRobo v1 motion planner per env_id satisfying the SkillGen interface.

    Planners read all world state (collision-geometry source, object poses, joint configuration)
    through the shared :class:`Datastream`, so they never touch the env/robot handles directly.
    The planner config comes from ``planner_profile`` (named by the environment profile) when
    given, else from task-name matching. Rerun plan visualization is opt-in via
    ``visualize_plan`` and limited to env 0.
    """
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    planners: dict[int, CuroboPlanner] = {}
    for env_id in range(num_envs):
        if planner_profile is not None:
            planner_config = CuroboPlannerCfg.from_profile(planner_profile)
        else:
            planner_config = CuroboPlannerCfg.from_task_name(env_name)
        # Visualization is rerun-based; limit to env_id 0 to keep simulation responsive.
        if env_id == 0:
            planner_config.visualize_plan = planner_config.visualize_plan or visualize_plan
        else:
            planner_config.visualize_spheres = False
            planner_config.visualize_plan = False
        planners[env_id] = CuroboPlanner(
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
    # The env name (CLI override) falls back to the name recorded in the source dataset.
    env_name = args_cli.env_name.split(":")[-1] if args_cli.env_name else get_env_name_from_dataset(args_cli.input_file)

    # The task descriptor's GenerationPolicy is the source for generation policy parameters.
    task_descriptor = TaskDescriptor.from_yaml(args_cli.task_descriptor)
    generation_policy_params = task_descriptor.get_generation_policy()

    # The CLI --generation_num_trials, when given, overrides the descriptor's num_trials.
    if args_cli.generation_num_trials is not None:
        generation_policy_params.num_trials = args_cli.generation_num_trials

    random.seed(generation_policy_params.seed)
    np.random.seed(generation_policy_params.seed)
    torch.manual_seed(generation_policy_params.seed)

    env_make_kwargs = register_environment_for_run(
        env_name=env_name,
        enable_cameras=args_cli.enable_cameras,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        seed=generation_policy_params.seed,
    )

    # The algorithm's start-signal expectation is a class attribute, so we resolve it before
    # instantiating (the Datastream needs it, and SkillGen can only be instantiated once the
    # planners exist, which in turn need the Datastream).
    alg_cls = REGISTERED_ALGORITHMS[args_cli.alg]
    generation_policy_params.use_skillgen = alg_cls.uses_subtask_start_signals

    # The environment profile, when given, overlays scene/reset changes onto the base task and
    # names the motion-planner profile tuned for the resulting scene.
    env_profile = EnvironmentProfile.from_yaml(args_cli.env_profile) if args_cli.env_profile else None

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_policy_params=generation_policy_params,
        env_profile=env_profile,
    )

    env = gym.make(env_name, cfg=env_cfg, **env_make_kwargs.get(env_name, {})).unwrapped

    env.reset()

    # Build the Datastream first; the motion planners read all world state through it.
    datastream = build_datastream(
        env=env,
        input_file=args_cli.input_file,
        task_descriptor=task_descriptor,
        embodiment_yaml=args_cli.embodiment,
    )

    # SkillGen needs one curobo planner per env. Mimic/DexMimic take no kwargs.
    motion_planners: dict | None = None
    alg_kwargs: dict = {}
    if args_cli.alg == "skillgen":
        motion_planners = _build_motion_planners(
            datastream,
            args_cli.num_envs,
            env_name,
            planner_profile=env_profile.planner if env_profile else None,
            visualize_plan=args_cli.visualize_plan,
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
        generation_completed = False
        try:
            generation_completed = env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["event_loop"],
                generation_policy_params=generation_policy_params,
                stats=async_components["stats"],
                data_gen_tasks=data_gen_tasks,
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

        if generation_completed and args_cli.result_file:
            write_generation_result(
                result_file=args_cli.result_file,
                algorithm=args_cli.alg,
                requested_trials=generation_policy_params.num_trials,
                stats=async_components["stats"],
                env_profile=(
                    {"name": env_profile.name, "path": args_cli.env_profile, "planner": env_profile.planner}
                    if env_profile
                    else None
                ),
            )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
    simulation_app.close()
