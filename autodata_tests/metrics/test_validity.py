# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests of the episode checks and validity metrics, each on episodes with one known defect."""

import h5py
import json
import numpy as np
from pathlib import Path

import pytest

from autodata_metrics import REGISTERED_METRICS, build_report
from autodata_metrics.validity import FROZEN_FRAMES


def _write(path: Path, episodes: list[dict[str, np.ndarray]], entry: dict | None = None) -> str:
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        env_args = {"env_name": "Test-v0", "sim_args": {"dt": 0.01, "decimation": 5}}
        env_args.update({"autodata": {"version": 1, **entry}} if entry else {})
        data.attrs["env_args"] = json.dumps(env_args)
        for index, arrays in enumerate(episodes):
            group = data.create_group(f"demo_{index}")
            for key, array in arrays.items():
                group.create_dataset(key, data=array)
    return str(path)


def _metrics(*names: str) -> dict:
    return {name: REGISTERED_METRICS[name] for name in names}


def test_failing_episodes_are_counted_and_left_out_of_dataset_metrics(tmp_path):
    actions = np.zeros((3, 7))
    good = {"actions": actions, "step_success": np.array([False, False, True])}
    non_finite = {"actions": np.where(np.eye(3, 7) > 0, np.nan, 0.0), "step_success": np.array([True, True, False])}
    short = {"actions": actions, "step_success": np.array([False, False])}
    empty = {"actions": np.zeros((0, 7)), "step_success": np.zeros(0, dtype=bool)}

    report = build_report(
        _write(tmp_path / "d.hdf5", [good, non_finite, short, empty]),
        metrics=_metrics("correctness_pass_rate", "final_success_rate"),
    )

    assert report["metrics"]["correctness_pass_rate"]["value"] == pytest.approx(1 / 4)
    assert report["metrics"]["correctness_pass_rate"]["details"]["checks_skipped"] == [
        "valid_rotations",
        "pose_action_bounds",
    ]
    assert report["metrics"]["final_success_rate"]["value"] == 1.0 and report["metrics"]["final_success_rate"]["n"] == 1
    assert {name: entry["failed"] for name, entry in report["check_failures"].items() if entry["failed"]} == {
        "finite": 1,
        "aligned_lengths": 2,
    }


def test_rotation_and_bound_checks_flag_their_defects(tmp_path):
    pose = np.tile(np.eye(4), (3, 1, 1))
    entry = {"run_id": "r", "exported_episodes": 3, "pose_action": {"arm": {"action_slice": [0, 6], "bound": 1.0}}}
    clipped = {"actions": np.zeros((3, 7)), "obs/datagen_info/eef_pose/arm": pose}
    beyond_bound = {"actions": np.full((3, 7), 1.5), "obs/datagen_info/eef_pose/arm": pose}
    skewed = {"actions": np.zeros((3, 7)), "obs/datagen_info/eef_pose/arm": pose * [[1.1], [1.0], [1.0], [1.0]]}
    zero = {"actions": np.zeros((3, 7)), "obs/datagen_info/eef_pose/arm": np.zeros((3, 4, 4))}
    reflected = {"actions": np.zeros((3, 7)), "obs/datagen_info/eef_pose/arm": pose * [[-1.0], [1.0], [1.0], [1.0]]}
    episodes = [clipped, beyond_bound, skewed, zero, reflected]
    entry["exported_episodes"] = len(episodes)

    report = build_report(_write(tmp_path / "d.hdf5", episodes, entry), metrics={})

    failures = {name: sorted(entry["examples"]) for name, entry in report["check_failures"].items() if entry["failed"]}
    assert failures == {"pose_action_bounds": ["demo_1"], "valid_rotations": ["demo_2", "demo_3", "demo_4"]}


def test_image_checks_flag_black_frozen_and_missing_frames(tmp_path):
    steps = FROZEN_FRAMES + 2
    rng = np.random.default_rng(0)

    def frames(count: int) -> np.ndarray:
        return rng.integers(50, 200, size=(count, 4, 4, 3), dtype=np.uint8)

    black = frames(steps)
    black[1] = 0
    frozen = np.repeat(frames(1), steps, axis=0)
    episodes = [
        {"actions": np.zeros((steps, 7)), "obs/camera": camera}
        for camera in (frames(steps), black, frozen, frames(steps - 1))
    ]
    episodes.append({"actions": np.zeros((steps, 7))})

    report = build_report(_write(tmp_path / "d.hdf5", episodes), metrics=_metrics("image_stream_pass_rate"))

    failures = {name: sorted(entry["examples"]) for name, entry in report["check_failures"].items() if entry["failed"]}
    assert failures == {
        "no_black_frames": ["demo_1"],
        "no_frozen_frames": ["demo_2"],
        "one_frame_per_step": ["demo_3", "demo_4"],
    }
    assert report["metrics"]["image_stream_pass_rate"]["value"] == pytest.approx(1 / 5)


def test_final_success_rate_reports_the_source_baseline(tmp_path):
    ending = [{"actions": np.zeros((2, 7)), "step_success": np.array([True, held])} for held in (True, False)]
    broken = {"actions": np.full((2, 7), np.nan), "step_success": np.array([True, False])}
    source = _write(tmp_path / "source.hdf5", [ending[0], broken])

    report = build_report(_write(tmp_path / "d.hdf5", ending), source=source, metrics=_metrics("final_success_rate"))

    row = report["metrics"]["final_success_rate"]
    assert (row["value"], row["baseline"]) == (0.5, 1.0)
    assert (report["source_checks"]["num_valid_episodes"], report["source_checks"]["num_episodes"]) == (1, 2)


def test_action_rotation_error_measures_commanded_rotations(tmp_path):
    identity = np.tile(np.eye(4), (4, 1, 1))
    episode = {
        "actions": np.zeros((4, 7)),
        "obs/datagen_info/eef_pose/arm": identity,
        "obs/datagen_info/target_eef_pose/arm": identity * [[1.01], [1.0], [1.0], [1.0]],
        # A channel that commands no pose records an all-zero target; it is not a rotation and is not counted.
        "obs/datagen_info/target_eef_pose/body": np.zeros((4, 4, 4)),
    }

    report = build_report(_write(tmp_path / "d.hdf5", [episode]), metrics=_metrics("action_rotation_error"))

    row = report["metrics"]["action_rotation_error"]
    assert row["value"]["p95"] == pytest.approx(1.01**2 - 1) and row["n"] == 4
    assert report["num_valid_episodes"] == 1


def _poses(positions: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(positions), 1, 1))
    poses[:, :3, 3] = positions
    return poses


def test_tracking_error_measures_absolute_targets_against_the_next_step(tmp_path):
    reached = _poses(np.array([[0.01 * step, 0.0, 0.0] for step in range(5)]))
    targets = _poses(np.vstack([reached[1:, :3, 3] + [0.0, 0.02, 0.0], [[0.0, 0.0, 0.0]]]))
    targets[-1] = 0
    pose_action = {
        "arm": {"action_slice": [0, 6], "absolute": True},
        "wrist": {"action_slice": [6, 7], "absolute": False},
    }
    groups = ("obs/datagen_info/eef_pose", "obs/datagen_info/target_eef_pose")
    episode = {"actions": np.zeros((5, 7)), f"{groups[0]}/arm": reached, f"{groups[1]}/arm": targets}
    # A relative action and a channel that commands no pose action are not tracked.
    episode.update({f"{group}/{eef}": _poses(np.zeros((5, 3))) for group in groups for eef in ("wrist", "body")})
    one_step = {key: array[:1] for key, array in episode.items()}
    dataset = _write(
        tmp_path / "d.hdf5", [episode, one_step], {"run_id": "r", "exported_episodes": 2, "pose_action": pose_action}
    )

    row = build_report(dataset, metrics=_metrics("tracking_error"))["metrics"]["tracking_error"]

    assert row["value"]["p50"] == pytest.approx(20.0) and row["n"] == 4


def test_action_saturation_counts_steps_at_the_bound_and_borrows_the_run_layout_for_its_baseline(tmp_path):
    actions = np.zeros((4, 7))
    actions[1, 2] = -1.0
    entry = {"run_id": "r", "exported_episodes": 1, "pose_action": {"arm": {"action_slice": [0, 6], "bound": 1.0}}}
    source = _write(tmp_path / "source.hdf5", [{"actions": actions}])

    report = build_report(
        _write(tmp_path / "d.hdf5", [{"actions": actions}], entry), source=source, metrics=_metrics("action_saturation")
    )

    row = report["metrics"]["action_saturation"]
    assert (row["value"], row["baseline"]) == (0.25, 0.25)


def test_motion_smoothness_is_the_jerk_of_the_hand_labelled_with_the_noise(tmp_path):
    step = 0.01 * 5
    positions = np.array([[2.0 * (index * step) ** 3, 0.0, 0.0] for index in range(6)])
    entry = {"run_id": "r", "exported_episodes": 1, "action_noise": 0.03}
    episode = {"actions": np.zeros((6, 7)), "obs/datagen_info/eef_pose/arm": _poses(positions)}

    row = build_report(_write(tmp_path / "d.hdf5", [episode], entry), metrics=_metrics("motion_smoothness"))["metrics"][
        "motion_smoothness"
    ]

    assert row["value"]["p50"] == pytest.approx(12.0) and row["details"]["action_noise"] == 0.03


def test_joint_limit_ratio_compares_velocity_and_acceleration_with_the_datasheet(tmp_path):
    entry = {
        "run_id": "r",
        "exported_episodes": 3,
        "joints": {"asset": "robot", "names": ["j1", "j2"]},
        "hardware_limits": {"velocity": {"j[12]": 2.0}, "acceleration": {"j1": 10.0}},
    }
    # The first episode doubles j2's velocity limit and j1's acceleration limit; the others run at 96% and 94% of j1's.
    velocities = [[[0.0, 4.0], [1.0, 0.0], [1.0, 0.0]], [[1.92, 0.0]] * 3, [[1.88, 0.0]] * 3]
    episodes = [
        {"actions": np.zeros((3, 7)), "states/articulation/robot/joint_velocity": np.array(velocity)}
        for velocity in velocities
    ]

    row = build_report(_write(tmp_path / "d.hdf5", episodes, entry), metrics=_metrics("joint_limit_ratio"))["metrics"][
        "joint_limit_ratio"
    ]

    assert row["value"] == pytest.approx({"velocity": 2 / 3, "acceleration": 1 / 3})
    assert (row["details"]["acceleration_peak_ratio"], row["details"]["velocity_worst_joint"]) == (2.0, "j2")
