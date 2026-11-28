#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data generation entrypoint.

Usage::

    python scripts/generate_dataset.py \\
        --env_name <env_id> \\
        [--alg {mimicgen|dexmimicgen|skillgen|softmimicgen|schedulegen}] \\
        --task_descriptor <task_descriptor.yaml> \\
        --embodiment <embodiment.yaml> \\
        --env_profile <environment_profile.yaml> \\
        [--input_file <source.hdf5>] \\
        --output_file <out.hdf5> \\
        --generation_num_trials <N> \\
        --num_envs <N>

The task descriptor's ``algo`` selects the :class:`GenerationAlgorithm` plug-in driving the run
(``--alg`` is optional and must match it):

* ``mimicgen`` — single-arm MimicGen.
* ``dexmimicgen`` — two-arm MimicGen with subtask coordination constraints.
* ``skillgen`` — single-arm SkillGen using the native cuRobo planner interface.
* ``schedulegen`` — complete multi-arm path construction, collision/task scheduling, state restore,
  and synchronized replay with native arm-local cuRobo configuration.
* ``softmimicgen`` — one- or two-arm MimicGen with deformable-object nodal registration.

The CLI composes a :class:`Datastream` from the task descriptor YAML, the embodiment YAML, the
live env, and an optional HDF5 source dataset, then hands it to :class:`DataGenerator`. A source
dataset is optional only for an all-motion-planned ScheduleGen task.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import warnings

from isaaclab.app import AppLauncher

from autodata_interfaces.embodiments import embodiment_adapter_from_yaml
from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from autodata_interfaces.tasks.task_descriptor_utils import resolve_task_algorithm
from autodata_utils.cli import callable_path

PROCESS_START = time.monotonic()

# Hardcoded to keep argparse importable without pulling in the heavy core package.
# Add new algorithms here when registering them in autodata_core.algorithms.
_ALG_CHOICES = ["mimicgen", "dexmimicgen", "skillgen", "softmimicgen", "schedulegen"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
planning_args = parser.add_argument_group("Motion planning (SkillGen and ScheduleGen)")
schedule_args = parser.add_argument_group("ScheduleGen execution")
recording_args = parser.add_argument_group("Recording and reports")
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
    default=None,
    help=(
        "Generation algorithm. Defaults to the task descriptor's algo, which it must match. "
        "skillgen needs a motion planner (auto-wired from upstream curobo)."
    ),
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
parser.add_argument(
    "--generation_max_failures",
    type=int,
    default=None,
    help="Bound additional failed attempts when targeting successful demos; ignored for fixed-attempt runs.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--presets", choices=("physx", "newton"), default=None, help="Physics backend preset for Arena environments."
)
schedule_args.add_argument(
    "--generation_mode",
    choices=("per_environment", "buffered"),
    default="per_environment",
    help="Per-environment generation or buffered ScheduleGen planning/scheduling with vectorized replay.",
)
schedule_args.add_argument(
    "--schedule_workers",
    type=int,
    default=1,
    help="Concurrent CPU schedule solves in buffered mode; planning and validation remain serial.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Source dataset HDF5 file; optional for an all-motion-planned ScheduleGen task.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="Destination HDF5 for generated episodes.",
)
recording_args.add_argument(
    "--result_file",
    type=str,
    default=None,
    help=(
        "Optional JSON path for the final structured generation result. With --record_metric_inputs, an attempt "
        "log with one JSON line per attempt is written next to it (<result_file stem>.<run id>.attempts.jsonl)."
    ),
)
recording_args.add_argument(
    "--record_metric_inputs",
    action="store_true",
    help=(
        "Record what the dataset metrics read: per-step end-effector and target poses and task success, run "
        "metadata in the dataset and, with --result_file, a run block and an attempt log. Off by default."
    ),
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask for interactive debugging.",
)
planning_args.add_argument(
    "--visualize_plan",
    action="store_true",
    help="Visualize SkillGen or ScheduleGen plans in Rerun (environment zero only).",
)
planning_args.add_argument(
    "--rerun_output",
    type=str,
    default=None,
    help="Save SkillGen or ScheduleGen visualization to this .rrd file instead of opening a viewer.",
)
planning_args.add_argument("--robot_planning", type=str, default=None, help="Planning-groups YAML for ScheduleGen.")
recording_args.add_argument(
    "--record_video",
    action="store_true",
    help=(
        "Save every generated rollout from the configured simulator viewport (single environment). "
        "Also records image observations in HDF5, which can substantially increase dataset size."
    ),
)
recording_args.add_argument(
    "--video_dir",
    type=str,
    default="./datasets/visual_validation",
    help="Directory for generated rollout MP4 files.",
)
recording_args.add_argument(
    "--video_observation_path",
    type=str,
    default=None,
    help="Record this dot-separated RGB observation instead of the configured simulator viewport.",
)

parser.add_argument(
    "--enable_cameras",
    dest="record_images",
    action="store_true",
    help="Render and record the task's image observations.",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--external_callback", type=callable_path, help="Callable registering environments after startup.")
parser.add_argument(
    "--recorder_cfg", type=callable_path, help="Callable returning an Isaac Lab recorder configuration."
)
args_cli, environment_args = parser.parse_known_args()
if environment_args and args_cli.external_callback is None:
    parser.error(f"unrecognized arguments: {' '.join(environment_args)}")
args_cli.alg = resolve_task_algorithm(args_cli.alg, args_cli.task_descriptor)
task_descriptor = TaskDescriptor.from_yaml(args_cli.task_descriptor)
embodiment_adapter = embodiment_adapter_from_yaml(args_cli.embodiment)
task_descriptor.validate_embodiment(embodiment_adapter)
if args_cli.input_file is None and (args_cli.alg != "schedulegen" or task_descriptor.requires_source_demos()):
    parser.error("--input_file is required for tasks containing demonstration segments.")
if args_cli.alg == "schedulegen" and args_cli.robot_planning is None:
    parser.error("--robot_planning is required for schedulegen.")
if args_cli.num_envs < 1 or args_cli.schedule_workers < 1:
    parser.error("--num_envs and --schedule_workers must be positive.")
if args_cli.generation_mode == "buffered" and args_cli.alg != "schedulegen":
    parser.error("--generation_mode buffered requires --alg schedulegen.")
if args_cli.generation_mode != "buffered" and args_cli.schedule_workers != 1:
    parser.error("--schedule_workers applies only to buffered generation.")
if args_cli.alg != "schedulegen" and args_cli.robot_planning is not None:
    parser.error("--robot_planning applies only to ScheduleGen.")
if args_cli.alg not in ("skillgen", "schedulegen") and (args_cli.visualize_plan or args_cli.rerun_output):
    parser.error("plan visualization requires SkillGen or ScheduleGen.")

if args_cli.record_video and not args_cli.record_images:
    warnings.warn(
        "--record_video also enables rendering and recording image observations in HDF5. "
        "This can substantially increase dataset size.",
        UserWarning,
    )
args_cli.record_images = args_cli.record_images or args_cli.record_video
record_viewport = args_cli.record_video and args_cli.video_observation_path is None
viewport_headless = record_viewport and "kit" not in (args_cli.visualizer or [])
if viewport_headless:
    args_cli.visualizer = [name for name in (args_cli.visualizer or []) if name != "none"] + ["kit"]
    args_cli.visualizer_explicit = True
    args_cli.headless = True
app_launcher = AppLauncher(args_cli, enable_cameras=args_cli.record_images)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import isaaclab_mimic.envs  # noqa: E402, F401
from isaaclab.utils.string import string_to_callable  # noqa: E402

from autodata_core import DataGenerator, get_algorithm  # noqa: E402
from autodata_core.algorithms import REGISTERED_ALGORITHMS  # noqa: E402
from autodata_examples.envs import register_environment_for_run  # noqa: E402
from autodata_interfaces.datastream import Datastream  # noqa: E402
from autodata_interfaces.env import ViewportRenderWrapper  # noqa: E402
from autodata_interfaces.env import (
    EnvironmentProfile,
    ObservationRenderWrapper,
    collect_autodata_entry,
    env_loop,
    get_env_name_from_dataset,
    observation_video_path_resolve,
    setup_env_config,
    setup_output_paths,
)
from autodata_interfaces.env.datagen_info_recorder import PreStepDatagenInfoRecorder, add_pose_recorder  # noqa: E402
from autodata_interfaces.env.recorders import make_action_state_recorder_manager_cfg  # noqa: E402
from autodata_interfaces.env.registration import load_callable, register_external_environment  # noqa: E402
from autodata_utils.dataset_metadata import exported_dataset_files, merge_autodata_entry  # noqa: E402
from autodata_utils.generation_result import (  # noqa: E402
    AttemptLog,
    attempts_file_for,
    code_revision,
    input_record,
    write_generation_result,
)


def _hardware(device: str) -> dict[str, str | None]:
    """Model names of the GPU and CPU the run used; physics runs on the CPU when ``device`` is ``cpu``."""
    gpu = None
    if torch.cuda.is_available():
        index = torch.device(device).index if device.startswith("cuda") else None
        gpu = torch.cuda.get_device_name(torch.cuda.current_device() if index is None else index)
    cpu = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model_lines = (line for line in cpuinfo.read_text().splitlines() if line.startswith("model name"))
        cpu = next((line.split(":", 1)[1].strip() for line in model_lines), None)
    return {"gpu": gpu, "num_gpus": 1 if gpu else 0, "cpu": cpu or platform.processor() or None}


def build_datastream(
    env: Any,
    input_file: str | None,
    task_descriptor: TaskDescriptor,
    embodiment_adapter: EmbodimentAdapter,
) -> Datastream:
    """Compose the read interface (task + embodiment + source pool) over the live env.

    Built before the motion planners so the planners can read all world state through it.
    """
    datastream = Datastream(
        env=env,
        task_descriptor=task_descriptor,
        embodiment_adapter=embodiment_adapter,
        source_dataset_path=input_file,
        uses_start_signals=task_descriptor.get_generation_policy().use_skillgen,
    )
    if datastream.num_source_demos:
        print(f"Loaded {datastream.num_source_demos} source episodes into the datagen pool")
    else:
        print("Task requires no source demonstrations; using an empty datagen pool")
    return datastream


def setup_async_generation(
    datastream: Datastream,
    num_envs: int,
    success_term,
    algorithm,
    pause_subtask: bool,
    attempt_log: AttemptLog | None = None,
) -> dict:
    """Build a single :class:`DataGenerator` over ``datastream`` and ``num_envs`` async tasks."""
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue: asyncio.Queue = asyncio.Queue()
    env_action_queue: asyncio.Queue = asyncio.Queue()

    data_generator = DataGenerator(
        datastream=datastream,
        algorithm=algorithm,
        demo_keys=datastream.get_generation_policy().source_demo_keys,
    )
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}
    attempts: list[dict[str, Any]] = []
    pending_env_ids: set[int] = set()

    tasks = []
    for env_id in range(num_envs):
        task = asyncio_event_loop.create_task(
            data_generator.generate_loop(
                env_id=env_id,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                success_term=success_term,
                pause_subtask=pause_subtask,
                stats=stats,
                attempt_log=attempt_log,
                attempts=attempts,
                pending_env_ids=pending_env_ids,
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
        "attempts": attempts,
    }


def _build_motion_planners(
    datastream,
    num_envs: int,
    env_name: str,
    *,
    planner_profile: str | None = None,
    visualize_plan: bool = False,
    rerun_output: str | None = None,
) -> dict:
    """Construct one cuRobo v1 motion planner per env_id satisfying the SkillGen interface.

    Planners read all world state (collision-geometry source, object poses, joint configuration)
    through the shared :class:`Datastream`, so they never touch the env/robot handles directly.
    The planner config comes from ``planner_profile`` (named by the environment profile) when
    given, else from task-name matching. Rerun plan visualization is opt-in via
    ``visualize_plan`` and limited to env 0.

    Args:
        datastream: Environment, task and embodiment interfaces used by each planner.
        num_envs: Number of independent planners to construct.
        env_name: Task name used when no planner profile is provided.
        planner_profile: Optional named robot/planner configuration.
        visualize_plan: Whether environment zero logs plans to Rerun.
        rerun_output: Optional recording path instead of opening a Rerun viewer.

    Returns:
        Motion planners keyed by environment index.
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
            planner_config.rerun_output = rerun_output
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
    """Close each shared plan visualizer once and release planner references."""
    if not planners:
        return
    closed_visualizers: set[int] = set()
    for env_id, planner_or_groups in planners.items():
        if hasattr(planner_or_groups, "group_planners") and hasattr(planner_or_groups, "full_state"):
            groups = (*planner_or_groups.group_planners.values(), planner_or_groups.full_state)
        else:
            groups = planner_or_groups.values() if isinstance(planner_or_groups, dict) else (planner_or_groups,)
        for planner in groups:
            visualizer = getattr(planner, "plan_visualizer", None)
            if visualizer is not None and id(visualizer) not in closed_visualizers:
                print(f"Closing plan visualizer for environment {env_id}")
                visualizer.close()
                closed_visualizers.add(id(visualizer))
            planner.plan_visualizer = None
    planners.clear()


def main() -> None:
    """Create the selected environment and algorithm, generate episodes and save run results."""
    register_external_environment(args_cli, environment_args)
    if args_cli.rerun_output is not None:
        assert args_cli.visualize_plan, "--rerun_output requires --visualize_plan"
    if args_cli.record_video:
        assert args_cli.num_envs == 1, "video recording currently supports exactly one environment"
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    env_profile = EnvironmentProfile.from_yaml(args_cli.env_profile) if args_cli.env_profile else None
    # A profile names the base task it patches. Without a profile, preserve the ordinary source
    # dataset fallback; an explicit CLI environment remains the highest-priority override.
    if args_cli.env_name:
        env_name = args_cli.env_name.split(":")[-1]
    elif env_profile is not None:
        env_name = env_profile.base_env
    else:
        assert args_cli.input_file is not None, "--env_name or --env_profile is required without --input_file"
        env_name = get_env_name_from_dataset(args_cli.input_file)

    # The task descriptor's GenerationPolicy is the source for generation policy parameters.
    generation_policy_params = task_descriptor.get_generation_policy()

    # The CLI --generation_num_trials, when given, overrides the descriptor's num_trials.
    if args_cli.generation_num_trials is not None:
        generation_policy_params.num_trials = args_cli.generation_num_trials
    if args_cli.generation_max_failures is not None:
        assert args_cli.generation_max_failures >= 0
        generation_policy_params.max_num_failures = args_cli.generation_max_failures

    random.seed(generation_policy_params.seed)
    np.random.seed(generation_policy_params.seed)
    torch.manual_seed(generation_policy_params.seed)

    env_make_kwargs = register_environment_for_run(
        env_name=env_name,
        enable_cameras=args_cli.record_images,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        seed=generation_policy_params.seed,
        presets=args_cli.presets,
    )

    # The algorithm's start-signal expectation is a class attribute, so we resolve it before
    # instantiating (the Datastream needs it, and SkillGen can only be instantiated once the
    # planners exist, which in turn need the Datastream).
    alg_cls = REGISTERED_ALGORITHMS[args_cli.alg]
    generation_policy_params.use_skillgen = alg_cls.uses_subtask_start_signals

    # The environment profile, when given, overlays scene/reset changes onto the base task and
    # names the motion-planner profile tuned for the resulting scene.
    recorder_cfg = load_callable(args_cli.recorder_cfg)() if args_cli.recorder_cfg else None
    if args_cli.record_metric_inputs:
        recorder_cfg = recorder_cfg or make_action_state_recorder_manager_cfg(record_images=args_cli.record_images)
        add_pose_recorder(recorder_cfg)
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_policy_params=generation_policy_params,
        recorder_cfg=recorder_cfg,
        record_step_success=args_cli.record_metric_inputs,
        env_profile=env_profile,
        enable_cameras=args_cli.record_images,
    )
    env_cfg.seed = generation_policy_params.seed

    if viewport_headless:
        from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg

        from autodata_interfaces.env.kit_visualizer import KitVisualizerHeadless

        visualizers = env_cfg.sim.visualizer_cfgs or []
        if not isinstance(visualizers, list):
            visualizers = [visualizers]
        kit_cfg = next((cfg for cfg in visualizers if cfg.visualizer_type == "kit"), None)
        if kit_cfg is None:
            kit_cfg = KitVisualizerCfg(class_type=KitVisualizerHeadless)
            visualizers = [*visualizers, kit_cfg]
        elif kit_cfg.class_type is KitVisualizer:
            kit_cfg.class_type = KitVisualizerHeadless
        kit_cfg.headless = True
        env_cfg.sim.visualizer_cfgs = visualizers
    base_env = gym.make(env_name, cfg=env_cfg, **env_make_kwargs.get(env_name, {})).unwrapped
    random.seed(generation_policy_params.seed)
    np.random.seed(generation_policy_params.seed)
    torch.manual_seed(generation_policy_params.seed)
    env = base_env
    if args_cli.record_video:
        if args_cli.video_observation_path is not None:
            observation_path = observation_video_path_resolve(
                args_cli.video_observation_path,
                default_image_observation_names=tuple(getattr(env_cfg, "image_obs_list", ())),
            )
            env = ObservationRenderWrapper(env, observation_path=observation_path)
        else:
            env = ViewportRenderWrapper(env)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            episode_trigger=(lambda _: True) if args_cli.generation_mode == "per_environment" else None,
            step_trigger=(lambda step: step == 0) if args_cli.generation_mode == "buffered" else None,
            name_prefix=f"{task_descriptor.name}-{args_cli.alg}",
            disable_logger=True,
        )

    if args_cli.record_images:
        # Initialize dynamic render transforms before resetting the scene used for recording.
        base_env.sim.step(render=False)
        base_env.scene.update(base_env.cfg.sim.dt)
    base_env.reset()

    # Build the Datastream first; the motion planners read all world state through it.
    datastream = build_datastream(
        env=base_env,
        input_file=args_cli.input_file,
        task_descriptor=task_descriptor,
        embodiment_adapter=embodiment_adapter,
    )
    PreStepDatagenInfoRecorder.datastream = datastream
    run_id = uuid.uuid4().hex
    # SkillGen needs one curobo planner per env. Mimic/DexMimic take no kwargs.
    motion_planners: dict | None = None
    alg_kwargs: dict = {}
    planner_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args_cli.alg == "skillgen":
        motion_planners = _build_motion_planners(
            datastream,
            args_cli.num_envs,
            env_name,
            planner_profile=env_profile.planner if env_profile else None,
            visualize_plan=args_cli.visualize_plan,
            rerun_output=args_cli.rerun_output,
        )
        alg_kwargs["motion_planners"] = motion_planners
    elif args_cli.alg == "schedulegen":
        from autodata_interfaces.motion_planners.curobo import curobo_planning_context_create

        diagnostics_path = task_descriptor.get_attempt_diagnostics()
        attempt_diagnostics = string_to_callable(diagnostics_path) if diagnostics_path is not None else None
        motion_planners = {}
        planning_env_count = 1 if args_cli.generation_mode == "buffered" else args_cli.num_envs
        for env_id in range(planning_env_count):
            planning_context = curobo_planning_context_create(
                datastream,
                args_cli.robot_planning,
                env_id=env_id,
                visualize_plan=args_cli.visualize_plan,
                rerun_output=args_cli.rerun_output if env_id == 0 else None,
            )
            motion_planners[env_id] = planning_context
        alg_kwargs.update(
            planning_contexts=motion_planners,
            attempt_diagnostics=attempt_diagnostics,
        )
    algorithm = get_algorithm(args_cli.alg, **alg_kwargs)
    autodata_entry = None
    if args_cli.record_metric_inputs:
        noises = [spec.action_noise for specs in task_descriptor.subtasks.values() for spec in specs]
        noises += [
            getattr(planner.config, "motion_noise_scale", 0.0)
            for planner_or_context in (motion_planners or {}).values()
            for planner in (
                planner_or_context.group_planners.values()
                if hasattr(planner_or_context, "group_planners")
                else (planner_or_context,)
            )
        ]
        autodata_entry = {
            **collect_autodata_entry(base_env, embodiment_adapter, run_id),
            "action_noise": max(noises, default=0.0),
        }
    planner_init_s = time.perf_counter() - planner_started
    generation_started = time.perf_counter()
    generation_start = time.monotonic()

    attempts_file = attempts_file_for(args_cli.result_file, run_id) if autodata_entry and args_cli.result_file else None
    attempt_log = AttemptLog(attempts_file, run_id) if attempts_file else None
    try:
        if args_cli.generation_mode == "buffered":
            from autodata_core.schedulegen.batching import generate_buffered

            data_generator = DataGenerator(
                datastream=datastream, algorithm=algorithm, demo_keys=generation_policy_params.source_demo_keys
            )
            async_components = generate_buffered(
                env, data_generator, algorithm, success_term, args_cli.schedule_workers, attempt_log=attempt_log
            )
            generation_completed = async_components["completed"]
        else:
            async_components = setup_async_generation(
                datastream=datastream,
                num_envs=args_cli.num_envs,
                success_term=success_term,
                algorithm=algorithm,
                pause_subtask=args_cli.pause_subtask,
                attempt_log=attempt_log,
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
        generation_end = time.monotonic()
        if args_cli.result_file:
            write_generation_result(
                result_file=args_cli.result_file,
                algorithm=args_cli.alg,
                requested_trials=generation_policy_params.num_trials,
                stats=async_components["stats"],
                attempts=async_components["attempts"],
                execution={
                    **async_components.get("execution", {}),
                    "mode": args_cli.generation_mode,
                    "num_envs": args_cli.num_envs,
                    "planner_context_count": len(motion_planners or {}),
                    "planner_init_s": planner_init_s,
                    "generation_s": time.perf_counter() - generation_started,
                    "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    **(
                        {
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                        if torch.cuda.is_available()
                        else {}
                    ),
                },
                environment={
                    "name": env_name,
                    "physics_manager": (
                        f"{base_env.sim.physics_manager.__module__}:{base_env.sim.physics_manager.__qualname__}"
                    ),
                    "control_dt_s": float(base_env.step_dt),
                },
                env_profile=(
                    {"name": env_profile.name, "path": args_cli.env_profile, "planner": env_profile.planner}
                    if env_profile
                    else None
                ),
                run_metadata=(
                    None
                    if autodata_entry is None
                    else {
                        "run_id": run_id,
                        "seed": generation_policy_params.seed,
                        "revision": code_revision(),
                        "inputs": {
                            "source": input_record(args_cli.input_file),
                            "embodiment": input_record(args_cli.embodiment),
                            "task_descriptor": input_record(args_cli.task_descriptor),
                            "env_profile": input_record(args_cli.env_profile),
                        },
                        "device": args_cli.device,
                        **_hardware(args_cli.device),
                        "num_envs": args_cli.num_envs,
                        "setup_time_s": generation_start - PROCESS_START,
                        "generation_time_s": generation_end - generation_start,
                        "attempts_file": Path(attempts_file).name,
                    }
                ),
            )
            if not generation_completed:
                print(f"Generation target was not reached; wrote the run outcome to {args_cli.result_file}", flush=True)
    finally:
        _close_motion_planners(motion_planners)
        env.close()
        if autodata_entry is not None:
            for dataset_file in exported_dataset_files(
                output_dir, output_file_name, generation_policy_params.keep_failed
            ):
                merge_autodata_entry(dataset_file, autodata_entry)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    sys.exit(exit_code)
