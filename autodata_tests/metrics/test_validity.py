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
        env_args = {"env_name": "Test-v0", **({"autodata": entry} if entry else {})}
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
