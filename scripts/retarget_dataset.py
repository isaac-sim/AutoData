#!/usr/bin/env python
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodiment dataset retargeting entrypoint (replay-based).

Usage (config-based — the task/pair parameters live in a retarget descriptor YAML)::

    python scripts/retarget_dataset.py \\
        --retarget_config <retarget_descriptor.yaml> \\
        --input_file <source.hdf5> \\
        --output_file <out.hdf5> \\
        [--keep_failed]

The retarget descriptor (see ``autodata_examples/retarget/``) is self-contained: it references the
source/target embodiments and target env, declares its own per-EEF ``subtasks`` (segmentation + object
tracking), and carries every retargeting knob (hand_policy / hand_interp_* / joint_mapping,
reference_pose, eef_reference_link, write_datagen_info, init_robot_from_ik, replay_speed, retarget_frame,
scene_translation, eef_offsets, ...) -- all descriptor-only. Without ``--retarget_config`` only the three
references (``--source_embodiment``/``--target_embodiment``/``--target_env_name``) build a config and
every other knob (including subtasks) takes its default, so a descriptor is needed to set them.

Retargeting transfers each demonstration recorded on a *source* embodiment onto a *target*
embodiment, one output demo per input demo (replay 1:1). Unlike
:mod:`scripts.generate_dataset`, no object-centric regeneration happens: the source
end-effector trajectory is replayed on the target robot and the resulting rollout is recorded.

How it works, per source episode:

* The recorded per-step target EEF pose trajectory (``obs/datagen_info/target_eef_pose``) is an
  absolute, env-relative SE(3) trajectory and is embodiment-independent. The source dataset must
  therefore already carry ``datagen_info`` (i.e. it went through ``annotate_demos.py`` or was
  produced by ``generate_dataset.py``).
* The gripper/hand "passthrough" channels are extracted from the source actions with the *source*
  embodiment adapter and remapped to the *target* embodiment's passthrough layout (see
  :func:`build_passthrough_remapper`). v1 supports embodiments whose passthrough layouts match;
  differing grippers (e.g. parallel gripper → dexterous hand) are a documented extension seam.
* The target env is reset so the *task scene* (objects) matches the source episode's initial state
  while the *target robot* starts from its own default configuration, then each step's target EEF
  pose + passthrough actions are re-encoded into the target embodiment's action via
  :meth:`EmbodimentAdapter.target_eef_pose_to_action` and stepped through the target env.

The task descriptor is shared with the source dataset (same subtasks / EEF names) and is used here
for the generation policy (export mode, seed) and to validate that both embodiments agree on the
task's EEF names.

Note: the output records target actions + states. Re-run ``annotate_demos.py`` on it before using
it as a source for ``generate_dataset.py`` (the subtask signals are not carried over).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--retarget_config",
    type=str,
    default=None,
    help=(
        "Path to a retarget descriptor YAML bundling the task/pair-specific parameters (task "
        "descriptor, source/target embodiments, target env, and every retargeting knob: hand_policy, "
        "num_interpolation_steps, init_robot_from_ik, replay_speed, retarget_frame, scene_translation, "
        "object_tracking, eef_offsets, ...). Almost all knobs are descriptor-only; runtime/IO flags stay "
        "on the CLI."
    ),
)
parser.add_argument(
    "--source_embodiment",
    type=str,
    default=None,
    help="Path to the source embodiment YAML (the embodiment the input dataset was recorded on).",
)
parser.add_argument(
    "--target_embodiment",
    type=str,
    default=None,
    help="Path to the target embodiment YAML (the embodiment to retarget the dataset onto).",
)
parser.add_argument(
    "--target_env_name",
    type=str,
    default=None,
    help="Environment id to instantiate for the target embodiment (the target robot's env).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help=(
        "Number of parallel environments. 1 (default) replays episodes sequentially. >1 replays them "
        "across that many envs on one sim (async workers, one episode per env at a time) for a large "
        "speedup. The per-step tracking report is single-env only and unavailable in parallel mode."
    ),
)
parser.add_argument("--input_file", type=str, required=True, help="Source dataset HDF5 file (must carry datagen_info).")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/retargeted_dataset.hdf5",
    help="Destination HDF5 for the retargeted episodes.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    metavar="IDX",
    help="Retarget only these source-episode indices (0-based, into the dataset); empty (default) = all.",
)
parser.add_argument(
    "--target_successes",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Stop once N replays have succeeded (early-stop). Unset (default) = run every source example. "
        "On a finite source that can't reach N, it stops when the source is exhausted."
    ),
)
parser.add_argument(
    "--target_runs",
    type=int,
    default=None,
    metavar="N",
    help="Stop once N replays have been attempted (successful or not); capped at the source size.",
)
parser.add_argument(
    "--keep_failed",
    action="store_true",
    help=(
        "Also export episodes whose replay did not satisfy the success term (overrides the task "
        "descriptor's keep_failed). Useful for inspecting the retargeted motion when success is "
        "expected to be low."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from autodata_retargeting.runner import run  # noqa: E402

if __name__ == "__main__":
    try:
        run(args_cli, simulation_app)
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
    simulation_app.close()
