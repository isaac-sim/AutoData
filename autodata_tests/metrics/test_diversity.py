# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests of the diversity metrics."""

import h5py
import json
import numpy as np
from pathlib import Path

import pytest

from autodata_metrics import REGISTERED_METRICS, build_report

CUBE = np.array([0.5, 0.0, 0.02])


def _write(path: Path, episodes: list[dict[str, np.ndarray]], format_version: int | None = None) -> str:
    with h5py.File(path, "w") as h5_file:
        if format_version is not None:
            h5_file.attrs["format_version"] = format_version
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test-v0"})
        for index, arrays in enumerate(episodes):
            for key, array in arrays.items():
                data.create_dataset(f"demo_{index}/{key}", data=array)
    return str(path)


def _row(dataset: str, name: str, source: str | None = None) -> dict:
    return build_report(dataset, source=source, metrics={name: REGISTERED_METRICS[name]})["metrics"][name]


def _starts(**positions: list[float]) -> dict[str, np.ndarray]:
    """An episode whose objects start at the given xy."""
    poses = {
        f"initial_state/rigid_object/{name}/root_pose": [[*xy, 0.0, 1.0, 0.0, 0.0, 0.0]]
        for name, xy in positions.items()
    }
    return {"actions": np.zeros((2, 7)), **poses}


def _hand(track: list) -> dict[str, np.ndarray]:
    """An episode whose hand passes through ``track``."""
    poses = np.tile(np.eye(4), (len(track), 1, 1))
    poses[:, :3, 3] = track
    return {"actions": np.zeros((len(track), 7)), "obs/datagen_info/eef_pose/arm": poses}


def _lift(
    direction: list[float], hand_offset: float = 0.0, yaw: float = 0.0, xyzw: bool = False, resting_hand: bool = False
) -> dict[str, np.ndarray]:
    """An episode whose grasp point comes straight to the cube from 20 cm along ``direction`` in the cube's frame, then
    lifts it 2 to 4 cm.

    The cube is turned by ``yaw`` [rad], its quaternion stored as (x, y, z, w) when ``xyzw`` and as (w, x, y, z)
    otherwise. The recorded hand frame sits ``hand_offset`` below the grasp point. With ``resting_hand``, a second
    hand waits 20 cm from the cube.
    """
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    track = [CUBE + 0.2 * turn @ np.array(direction) * (1 - step / 9) for step in range(10)]
    track += [CUBE + [0.0, 0.0, rise] for rise in (0.02, 0.03, 0.04)]
    half = [np.cos(yaw / 2), np.sin(yaw / 2)]
    quaternion = [0.0, 0.0, half[1], half[0]] if xyzw else [half[0], 0.0, 0.0, half[1]]
    cube = np.tile([*CUBE, *quaternion], (len(track), 1))
    cube[10:, :3] = track[10:]
    episode = {
        **_hand([point - [0.0, 0.0, hand_offset] for point in track]),
        "states/rigid_object/cube/root_pose": cube,
    }
    if resting_hand:
        rest = _hand([CUBE + [0.0, -0.2, 0.0]] * len(track))
        episode["obs/datagen_info/eef_pose/other"] = rest["obs/datagen_info/eef_pose/arm"]
    return episode


def test_start_spread_averages_the_layout_distance_over_the_objects_that_move(tmp_path):
    # Over 8 episodes the cube steps 4 cm and the ball 3 cm, so two layouts are 3 × sqrt((4² + 3²) / 2) cm apart on
    # average. The 4 seeds move only the cube, 6.67 cm on average. The table never moves.
    seeds = [_starts(cube=[0.04 * i, 0.0], ball=[0.0, 0.0], table=[0.0, 0.0]) for i in range(4)]
    spread = [_starts(cube=[0.04 * i, 0.0], ball=[0.0, 0.03 * i], table=[0.0, 0.0]) for i in range(8)]
    pinned = [_starts(cube=[0.0, 0.0], ball=[0.0, 0.03 * i], table=[0.0, 0.0]) for i in range(8)]
    seeds = _write(tmp_path / "s.hdf5", seeds)

    row = _row(_write(tmp_path / "spread.hdf5", spread), "start_spread", seeds)
    pinned_row = _row(_write(tmp_path / "pinned.hdf5", pinned), "start_spread", seeds)

    assert row["value"] == pytest.approx(3000 * np.sqrt((0.04**2 + 0.03**2) / 2)) and row[
        "baseline"
    ] == pytest.approx(200 / 3)
    # A cube the seeds move but this dataset pins shows in the details beside the seeds' spread.
    assert pinned_row["value"] == pytest.approx(90.0)
    assert pinned_row["details"]["objects"]["cube"] == {"spread_mm": 0.0, "seed_spread_mm": pytest.approx(200 / 3)}


def test_workspace_coverage_averages_the_space_two_episodes_do_not_share(tmp_path):
    # Two hands sweep 4 voxels each, 2 in common: 4 of the 6 voxels are not shared, and the paths are 10 cm apart.
    across = [[0.525 + 0.05 * step, 0.025, 0.025] for step in range(6)]
    seeds = _write(tmp_path / "s.hdf5", [_hand(across[:4]), _hand(across[:4])])

    row = _row(_write(tmp_path / "d.hdf5", [_hand(across[:4]), _hand(across[2:])]), "workspace_coverage", seeds)

    assert row["value"] == pytest.approx(2 / 3) and row["baseline"] == 0.0
    assert row["details"]["path_distance_mm"] == pytest.approx(100.0)


@pytest.mark.parametrize("hand_offset, resting_hand", [(0.0, False), (0.1, False), (0.26, True)])
def test_approach_metrics_average_over_pairs_of_lifts_in_the_object_frame(tmp_path, hand_offset, resting_hand):
    # From ±x, +y and above: one opposite pair and five perpendicular ones. Straight approaches over their last
    # 15.6 cm are 0.0913 m × |d1 - d2| apart in root-mean-square, so 138.0 mm on average.
    # A hand frame away from the grasp point, as a flange or a wrist is, gives the same answer, also when the other
    # hand rests nearer the cube than the carrying wrist.
    sides = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1])
    dataset = _write(tmp_path / "d.hdf5", [_lift(side, hand_offset, resting_hand=resting_hand) for side in sides])
    seeds = _write(tmp_path / "s.hdf5", [_lift([1, 0, 0]), _lift([1, 0, 0])])

    spread = _row(dataset, "approach_spread", seeds)
    diversity = _row(dataset, "behavior_diversity", seeds)

    assert (spread["value"], spread["baseline"], spread["n"]) == (pytest.approx(105.0), 0.0, 4)
    assert diversity["value"] == pytest.approx(138.0, abs=0.1) and diversity["baseline"] == 0.0


@pytest.mark.parametrize("xyzw", [False, True])
def test_approach_metrics_read_the_quaternion_order_of_the_file(tmp_path, xyzw):
    # Every lift comes at the cube's own +x face while the cube turns, so in its frame all approaches are one.
    lifts = [_lift([1, 0, 0], yaw=yaw, xyzw=xyzw) for yaw in np.radians([0, 30, 60, 90])]
    dataset = _write(tmp_path / "d.hdf5", lifts, format_version=1 if xyzw else None)

    assert _row(dataset, "approach_spread")["value"] == pytest.approx(0.0, abs=1e-3)
    assert _row(dataset, "behavior_diversity")["value"] == pytest.approx(0.0, abs=1e-3)


def test_exact_duplicates_counts_episodes_that_repeat_an_earlier_one(tmp_path):
    # The last episode holds the same bytes as the first in another shape, which is not a repeat.
    actions = [np.zeros((3, 7)), np.ones((3, 7)), np.zeros((3, 7)), np.zeros((7, 3))]

    row = _row(_write(tmp_path / "d.hdf5", [{"actions": array} for array in actions]), "exact_duplicates")

    assert row["value"] == 0.25 and row["details"]["duplicate_of"] == {"demo_2": "demo_0"}


@pytest.mark.parametrize("name", ["workspace_coverage", "approach_spread", "behavior_diversity"])
def test_behavior_metrics_need_recorded_hand_poses(tmp_path, name):
    episodes = [_starts(cube=[0.0, 0.0]), _starts(cube=[0.1, 0.0])]
    assert _row(_write(tmp_path / "d.hdf5", episodes), name)["status"] == "not_applicable"
