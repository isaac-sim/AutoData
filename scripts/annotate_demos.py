# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:

python scripts/annotate_demos.py \
--task <ENV_ID> \
--task_descriptor <TASK_DESCRIPTOR_YAML> \
--embodiment <EMBODIMENT_YAML> \
--input_file ./datasets/source.hdf5 \
--output_file ./datasets/source_annotated.hdf5

If ``--task`` is omitted the env id is read from the source dataset (if available).

Controls:

* ``N`` -- begin / resume playback
* ``B`` -- pause playback
* ``S`` -- mark a subtask signal at the current step
* ``Q`` -- skip the current episode

Automatic mode:

Pass ``--auto`` to annotate subtask termination signals without keyboard input (``--headless``
supported). ``--signal_obs_group`` selects the observation group holding the per-subtask boolean
terms (default ``subtask_terms``). SkillGen start signals still require manual mode.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Manually annotate AutoData source demonstrations with subtask signals.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--env_name",
    type=str,
    default=None,
    help="Environment name. Overrides the env name recorded in the source dataset.",
)
parser.add_argument(
    "--task_descriptor",
    type=str,
    required=True,
    help="Path to the task descriptor YAML (defines subtasks and their term/start signal names).",
)
parser.add_argument(
    "--embodiment",
    type=str,
    required=True,
    help="Path to the embodiment YAML (defines the robot's pose <-> action transforms).",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="./datasets/dataset.hdf5",
    help="Source dataset HDF5 to annotate.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="Destination HDF5 for the annotated dataset.",
)
parser.add_argument(
    "--auto",
    action="store_true",
    default=False,
    help="Annotate automatically by sampling the env's subtask-term observation terms during replay.",
)
parser.add_argument(
    "--signal_obs_group",
    type=str,
    default="subtask_terms",
    help="Observation group holding the per-subtask boolean terms read in --auto mode.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402
import gymnasium as gym  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402
from collections.abc import Callable  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers gym envs)
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler  # noqa: E402

from isaac_autodata_core.pool import DataGenInfoPool  # noqa: E402
from isaac_autodata_interfaces.datastream import Datastream  # noqa: E402
from isaac_autodata_interfaces.embodiments import embodiment_adapter_from_yaml  # noqa: E402
from isaac_autodata_interfaces.env import get_env_name_from_dataset, setup_env_config, setup_output_paths  # noqa: E402
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor  # noqa: E402

is_paused = False
current_action_index = 0
marked_subtask_action_indices: list[int] = []
skip_episode = False

_datastream: Datastream | None = None


def play_cb() -> None:
    global is_paused
    is_paused = False


def pause_cb() -> None:
    global is_paused
    is_paused = True


def skip_episode_cb() -> None:
    global skip_episode
    skip_episode = True


def mark_subtask_cb() -> None:
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


# Recorder term that writes per-step datagen info read through the Datastream.
class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records ``obs/datagen_info`` from the Datastream on each step."""

    def record_pre_step(self):
        assert _datastream is not None, "Datastream must be initialized before recording."
        datagen_info = {
            "object_pose": _datastream.get_object_poses(),
            "eef_pose": _datastream.embodiment_adapter.get_eef_poses(env_ids=None),
            "target_eef_pose": _datastream.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`PreStepDatagenInfoRecorder`."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


@configclass
class AnnotationRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus the datagen-info term used during annotation."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()


def _build_signal_names(
    task_descriptor: TaskDescriptor,
    annotate_start_signals: bool,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Resolve the per-EEF subtask term (and start) signal names to annotate.

    Returns:
        ``(term_signal_names, start_signal_names)`` dicts, each mapping eef name to an ordered list
        of signal names.
    """
    term_signal_names: dict[str, list[str]] = {}
    start_signal_names: dict[str, list[str]] = {}
    for eef_name in task_descriptor.get_eef_names():
        eef_term_names = task_descriptor.get_term_signal_names(eef_name)

        # Start signals (one per subtask, keyed by term-signal name) are needed only for SkillGen.
        if annotate_start_signals:
            assert all(name for name in eef_term_names), (
                f"Missing 'subtask_term_signal' for one or more subtasks of eef '{eef_name}'. When "
                "annotating for SkillGen, every subtask (including the last) must specify"
                "'subtask_term_signal' (its name is reused as the start-signal key)."
            )
            start_signal_names[eef_name] = list(eef_term_names)
        else:
            start_signal_names[eef_name] = []

        # Termination signals: every subtask except the final one (which ends the trajectory).
        non_final_term_names = eef_term_names[:-1]
        assert all(name for name in non_final_term_names), (
            f"Missing 'subtask_term_signal' for a non-final subtask of eef '{eef_name}'. Every "
            "subtask except the last must specify 'subtask_term_signal'."
        )
        term_signal_names[eef_name] = non_final_term_names

    return term_signal_names, start_signal_names


def main() -> int:
    """Annotate the source dataset and export the annotated copy."""
    global _datastream, is_paused, skip_episode, marked_subtask_action_indices

    assert args_cli.auto or not args_cli.headless, (
        "annotate_demos.py performs manual keyboard annotation and cannot run headless. "
        "Re-run with a window (drop --headless / unset HEADLESS), or pass --auto."
    )

    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")

    # Resolve the env id (CLI override, else the name recorded in the dataset).
    env_name = args_cli.env_name.split(":")[-1] if args_cli.env_name else get_env_name_from_dataset(args_cli.input_file)

    task_descriptor = TaskDescriptor.from_yaml(args_cli.task_descriptor)
    generation_policy = task_descriptor.get_generation_policy()

    # Start signals are required only by SkillGen.
    annotate_start_signals = generation_policy.use_skillgen

    assert not (args_cli.auto and annotate_start_signals), (
        "--auto only annotates subtask termination signals; SkillGen additionally needs subtask "
        "start signals, which have no automatic source yet. Annotate SkillGen datasets in manual "
        "mode (drop --auto)."
    )

    # Resolve the subtask signals to annotate
    term_signal_names, start_signal_names = _build_signal_names(task_descriptor, annotate_start_signals)
    total_signals = sum(len(names) for names in term_signal_names.values()) + sum(
        len(names) for names in start_signal_names.values()
    )
    assert total_signals > 0, (
        "No subtask signals to annotate: every eef declares only a single end-of-trajectory subtask. "
        "Manual annotation requires at least one non-final subtask with a 'subtask_term_signal'."
    )

    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    # Build the env config for replay+recording
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=1,
        device=args_cli.device,
        generation_policy_params=generation_policy,
        recorder_cfg=AnnotationRecorderManagerCfg(),
    )
    # Only export episodes we explicitly mark successful (i.e. fully annotated).
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    try:
        # Create the Datastream
        embodiment_adapter = embodiment_adapter_from_yaml(args_cli.embodiment)
        empty_pool = DataGenInfoPool(
            task_descriptor=task_descriptor,
            embodiment_adapter=embodiment_adapter,
            device=env.device,
            uses_start_signals=annotate_start_signals,
        )
        _datastream = Datastream(
            env=env,
            task_descriptor=task_descriptor,
            embodiment_adapter=embodiment_adapter,
            source_pool=empty_pool,
            uses_start_signals=annotate_start_signals,
        )

        env.reset()

        if args_cli.auto:
            # Fail fast if the env does not publish the expected signal observation terms.
            _datastream.get_subtask_term_signals(obs_group=args_cli.signal_obs_group)
        else:
            # Set up the keyboard used to mark subtask boundaries during replay.
            keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
            keyboard_interface.add_callback("N", play_cb)
            keyboard_interface.add_callback("B", pause_cb)
            keyboard_interface.add_callback("S", mark_subtask_cb)
            keyboard_interface.add_callback("Q", skip_episode_cb)
            keyboard_interface.reset()

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)
        episode_names = list(dataset_file_handler.get_episode_names())
        if len(episode_names) == 0:
            print("No episodes found in the dataset.")
            return 0

        exported_episode_count = 0
        processed_episode_count = 0
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            for episode_index, episode_name in enumerate(episode_names):
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                if args_cli.auto:
                    annotated = annotate_episode_in_auto_mode(
                        env, episode, success_term, term_signal_names, task_descriptor, args_cli.signal_obs_group
                    )
                else:
                    annotated = annotate_episode_in_manual_mode(
                        env, episode, success_term, term_signal_names, start_signal_names, annotate_start_signals
                    )

                if annotated and not skip_episode:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")

        print(
            f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
            f" episode{'s' if exported_episode_count != 1 else ''} to {args_cli.output_file}."
        )
        print("Exiting the app.")
        return exported_episode_count
    finally:
        env.close()


def replay_episode(
    env,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    on_step: Callable[[int], None] | None = None,
) -> bool:
    """Replay a recorded episode, recording datagen info each step.


    Args:
        env: The environment to replay in.
        episode: The recorded episode to replay.
        success_term: Optional termination term used to verify the task succeeded.
        on_step: Optional callback invoked with the action index before each step is applied,
            mirroring where manual marks are taken (state reflects all previous actions).

    Returns:
        True if the episode replayed to completion and the success condition held on any step (or no
        success term was given).
    """
    global current_action_index, skip_episode, is_paused
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]

    # Reset the simulation before restoring state (use of env.sim.reset() for deterministic replay)
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

    # Always return True if no success term was given.
    task_succeeded = success_term is None
    first_action = True
    for action_index in range(len(actions)):
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        if on_step is not None:
            on_step(action_index)
        action = actions[action_index]
        action_tensor = action.reshape(1, action.shape[0]).to(device=env.device)
        env.step(action_tensor)
        if success_term is not None:
            task_succeeded = task_succeeded or bool(success_term.func(env, **success_term.params)[0])

    return task_succeeded


def annotate_episode_in_manual_mode(
    env,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None,
    term_signal_names: dict[str, list[str]],
    start_signal_names: dict[str, list[str]],
    annotate_start: bool,
) -> bool:
    """Interactively annotate one episode's subtask signals.

    For each EEF that needs annotation, replays the episode and prompts the user to mark its subtask
    signals. When the marked-signal count matches the expected count the marks are accepted,
    otherwise the episode is replayed again for re-marking.

    Args:
        env: The environment to replay in.
        episode: The recorded episode to annotate.
        success_term: Optional success termination term.
        term_signal_names: Per-EEF ordered subtask termination-signal names to annotate.
        start_signal_names: Per-EEF ordered subtask start-signal names to annotate (may be empty).
        annotate_start: Whether subtask start signals are annotated (SkillGen); when True the marks
            interleave start/termination per subtask, otherwise every mark is a termination.

    Returns:
        True if the episode was fully annotated, False if it was skipped or never succeeded.
    """
    global is_paused, marked_subtask_action_indices, skip_episode

    term_signal_action_indices: dict[str, int] = {}
    start_signal_action_indices: dict[str, int] = {}

    for eef_name, eef_term_names in term_signal_names.items():
        eef_start_names = start_signal_names[eef_name]
        # Nothing to annotate for this eef (e.g. a single end-of-trajectory subtask).
        if len(eef_term_names) == 0 and len(eef_start_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            if len(eef_start_names) > 0:
                print(f"\t\t- Start:\t{eef_start_names}")
            print(f"\t\t- Termination:\t{eef_term_names}")
            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')

            marked_subtask_action_indices = []
            task_succeeded = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_count = len(eef_term_names) + len(eef_start_names)
            if task_succeeded and expected_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_count} subtask signals for eef "{eef_name}" were annotated.')
                # When annotating both signal types the marks interleave start, term, start, term, ...
                # ending with the final subtask's start, otherwise every mark is a termination.
                for marked_index in range(expected_count):
                    if not annotate_start:
                        term_signal_action_indices[eef_term_names[marked_index]] = marked_subtask_action_indices[
                            marked_index
                        ]
                    elif marked_index % 2 == 0:
                        start_signal_action_indices[eef_start_names[marked_index // 2]] = marked_subtask_action_indices[
                            marked_index
                        ]
                    else:
                        term_signal_action_indices[eef_term_names[math.floor(marked_index / 2)]] = (
                            marked_subtask_action_indices[marked_index]
                        )
                break

            if not task_succeeded:
                print("\tThe final task was not completed.")
                return False

            print(
                f"\tOnly {len(marked_subtask_action_indices)} out of {expected_count} subtask signals"
                f' for eef "{eef_name}" were annotated.'
            )
            print(f'\tThe episode will be replayed again for re-marking subtask signals for eef "{eef_name}".\n')

    num_steps = len(episode.data["actions"])
    annotated_episode = env.recorder_manager.get_episode(0)

    for signal_name, action_index in term_signal_action_indices.items():
        # Termination signal: False until the subtask completes, True from that step onward.
        signal = torch.ones(num_steps, dtype=torch.bool)
        signal[:action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{signal_name}", signal)

    if annotate_start:
        for signal_name, action_index in start_signal_action_indices.items():
            signal = torch.ones(num_steps, dtype=torch.bool)
            signal[:action_index] = False
            annotated_episode.add(f"obs/datagen_info/subtask_start_signals/{signal_name}", signal)

    return True


def _first_rising_edge(values: torch.Tensor) -> int | None:
    """Return the index of the first False -> True transition, or None if there is none.

    A signal already active at step 0 has no rising edge and is treated as invalid: the
    downstream boundary parser derives subtask ends from the transition, not the level.
    """
    if bool(values[0]):
        return None
    true_indices = values.nonzero()
    if len(true_indices) == 0:
        return None
    return int(true_indices[0][0])


def _validate_boundary_spacing(
    task_descriptor: TaskDescriptor, eef_name: str, term_edges: list[int], num_steps: int
) -> str | None:
    """Check detected boundaries against the descriptor's term-offset ranges; None if valid.

    Mirrors :meth:`DataGenInfoPool._validate_subtask_offsets` (non-SkillGen branch) so episodes
    the pool would reject are skipped at annotation time with a readable reason instead.
    """
    # Pool boundary convention: a signal ramp rising at edge index e ends its subtask at e + 1;
    # the final subtask ends at the trajectory end.
    ends = [edge + 1 for edge in term_edges] + [num_steps]
    term_offsets = [st.subtask_term_offset_range for st in task_descriptor.get_subtasks(eef_name)]
    for i in range(1, len(ends)):
        if not ends[i - 1] + term_offsets[i - 1][1] < ends[i] + term_offsets[i][0]:
            return (
                f"subtasks {i - 1} and {i} are too close for the configured offsets: "
                f"end={ends[i - 1]} +max_offset={term_offsets[i - 1][1]} vs "
                f"end={ends[i]} +min_offset={term_offsets[i][0]}"
            )
    return None


def annotate_episode_in_auto_mode(
    env,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None,
    term_signal_names: dict[str, list[str]],
    task_descriptor: TaskDescriptor,
    signal_obs_group: str,
) -> bool:
    """Automatically annotate one episode's subtask termination signals.

    Replays the episode once, sampling the env's subtask-term observation terms before each
    action (the state then reflects all previous actions, matching where manual marks are
    taken). Each signal's first rising edge becomes its subtask boundary; the boundaries are
    validated against subtask order and the descriptor's term-offset spacing before writing the
    same monotonic ramps manual mode produces.

    Args:
        env: The environment to replay in.
        episode: The recorded episode to annotate.
        success_term: Optional success termination term.
        term_signal_names: Per-EEF ordered subtask termination-signal names to annotate.
        task_descriptor: Task descriptor providing subtask offset ranges for validation.
        signal_obs_group: Observation group holding the per-subtask boolean terms.

    Returns:
        True if the episode was fully annotated, False if it was rejected (with a printed reason).
    """
    assert _datastream is not None, "Datastream must be initialized before annotating."

    sampled_signals: list[dict[str, torch.Tensor]] = []

    def sample_signals(action_index: int) -> None:
        sampled_signals.append(_datastream.get_subtask_term_signals(env_ids=[0], obs_group=signal_obs_group))

    task_succeeded = replay_episode(env, episode, success_term, on_step=sample_signals)
    if not task_succeeded:
        print("\tThe final task was not completed.")
        return False

    num_steps = len(episode.data["actions"])
    term_signal_action_indices: dict[str, int] = {}

    for eef_name, eef_term_names in term_signal_names.items():
        eef_edges: list[int] = []
        for signal_name in eef_term_names:
            signal = torch.stack([step[signal_name][0] for step in sampled_signals])
            edge = _first_rising_edge(signal)
            if edge is None:
                reason = "already active at episode start" if bool(signal[0]) else "never fired"
                print(f'\tSubtask signal "{signal_name}" (eef "{eef_name}") {reason}.')
                return False
            eef_edges.append(edge)

        if any(later <= earlier for earlier, later in zip(eef_edges, eef_edges[1:])):
            print(f'\tSubtask signals for eef "{eef_name}" fired out of order: {dict(zip(eef_term_names, eef_edges))}.')
            return False

        spacing_error = _validate_boundary_spacing(task_descriptor, eef_name, eef_edges, num_steps)
        if spacing_error is not None:
            print(f'\tRejected annotations for eef "{eef_name}": {spacing_error}.')
            return False

        print(f'\tDetected subtask signals for eef "{eef_name}": {dict(zip(eef_term_names, eef_edges))}')
        for signal_name, edge in zip(eef_term_names, eef_edges):
            term_signal_action_indices[signal_name] = edge

    annotated_episode = env.recorder_manager.get_episode(0)
    for signal_name, action_index in term_signal_action_indices.items():
        # Termination signal: False until the subtask completes, True from that step onward.
        signal = torch.ones(num_steps, dtype=torch.bool)
        signal[:action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{signal_name}", signal)

    return True


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
        exit_code = 130
    finally:
        simulation_app.close()
    exit(exit_code)
