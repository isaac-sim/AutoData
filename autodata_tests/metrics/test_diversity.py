# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests of the diversity metrics."""

import h5py
import json
import numpy as np
from pathlib import Path

import pytest

from autodata_metrics import REGISTERED_METRICS, build_report
from autodata_metrics.diversity import _digest, _sample
from autodata_metrics.evidence import Evidence


def _write(path: Path, episodes: list[dict[str, np.ndarray]]) -> str:
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test-v0"})
        for index, arrays in enumerate(episodes):
            for key, array in arrays.items():
                data.create_dataset(f"demo_{index}/{key}", data=array)
    return str(path)


def _row(dataset: str, name: str) -> dict:
    return build_report(dataset, metrics={name: REGISTERED_METRICS[name]})["metrics"][name]


def _starts(**positions: list[float]) -> dict[str, np.ndarray]:
    """An episode whose objects start at the given xy, or xyz."""
    poses = {
        f"initial_state/rigid_object/{name}/root_pose": [[*position, *[0.0] * (3 - len(position)), 1.0, 0.0, 0.0, 0.0]]
        for name, position in positions.items()
    }
    return {"actions": np.zeros((2, 7)), **poses}


def _poses(positions: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(positions), 1, 1))
    poses[:, :3, 3] = positions
    return poses


def _moving(**tracks: np.ndarray) -> dict[str, np.ndarray]:
    """An episode whose hands follow the given tracks."""
    first = next(iter(tracks.values()))
    return {
        "actions": np.pad(first, ((0, 0), (0, 4))),
        **{f"obs/datagen_info/eef_pose/{hand}": _poses(track) for hand, track in tracks.items()},
    }


def test_start_layout_spread_counts_every_object_so_pinning_one_lowers_it(tmp_path):
    # x moves 2 cm and rises 10 cm, which does not count, and y moves 10 cm per episode while the table stays; a pair's
    # distance is the root-sum-square of the objects' horizontal ones.
    varied = [_starts(x=[0.02 * i, 0.0, 0.1 * i], y=[0.0, 0.1 * i], table=[0.5, 0.5]) for i in range(4)]
    pinned = [_starts(x=[0.0, 0.0], y=[0.0, 0.1 * i], table=[0.5, 0.5]) for i in range(4)]
    mean_step = np.mean([j - i for i in range(4) for j in range(i + 1, 4)])

    varied_row = _row(_write(tmp_path / "varied.hdf5", varied), "start_layout_spread")
    pinned_row = _row(_write(tmp_path / "pinned.hdf5", pinned), "start_layout_spread")

    assert varied_row["value"] == pytest.approx(1000 * np.hypot(0.02, 0.1) * mean_step)
    assert pinned_row["value"] == pytest.approx(1000 * 0.1 * mean_step)


def test_a_deformable_object_starts_at_its_nodes_centroid(tmp_path):
    # A rope's first node stays at the origin while its second lies 0, 10 and 20 cm away: its centroid moves 5 cm a step.
    episodes = [
        {
            "actions": np.zeros((2, 7)),
            "initial_state/deformable_object/rope/nodal_position": [[[0.0] * 3, [0.1 * i, 0, 0]]],
        }
        for i in range(3)
    ]

    row = _row(_write(tmp_path / "d.hdf5", episodes), "start_layout_spread")

    assert row["value"] == pytest.approx((50.0 + 100.0 + 50.0) / 3)


_LINE = np.linspace([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 11)
_WITH_BALL, _WITHOUT_BALL = _starts(cube=[0.0, 0.0], ball=[0.1, 0.0]), _starts(cube=[0.1, 0.0])
_TWO_HANDS, _ONE_HAND = _moving(left=_LINE, right=_LINE), _moving(left=_LINE)


@pytest.mark.parametrize(
    "name, episodes, reason",
    [
        ("start_layout_spread", [_WITH_BALL], "needs two episodes"),
        ("start_layout_spread", [_WITH_BALL, _WITHOUT_BALL], "rigid_object/ball has no start pose in episode demo_1"),
        ("start_layout_spread", [_WITHOUT_BALL, _WITH_BALL], "rigid_object/ball has no start pose in episode demo_0"),
        ("workspace_spread", [_TWO_HANDS, _ONE_HAND], "episode demo_1 records no pose of right"),
        ("workspace_spread", [_ONE_HAND, _TWO_HANDS], "episode demo_0 records no pose of right"),
    ],
)
def test_a_spread_says_why_it_does_not_apply(tmp_path, name, episodes, reason):
    # Two episodes that hold the same actions are read in the opposite order once reversed, so in one of the two orders
    # a spread reads first the episode missing what the other records.
    row = _row(_write(tmp_path / "d.hdf5", episodes), name)

    assert row["status"] == "not_applicable" and row["reason"] == reason


def test_repeated_episodes_spread_through_the_episodes_a_spread_reads(tmp_path):
    # 40 distinct episodes, 30 copies each: the 1,000 read include every one.
    episodes = [{"actions": np.full((2, 7), float(k))} for k in range(40) for _ in range(30)]
    path = _write(tmp_path / "d.hdf5", episodes)

    with h5py.File(path, "r") as h5_file:
        sample = _sample(Evidence(h5_file))
        distinct = {_digest(episode) for episode in sample}

    assert len(sample) == 1000 and len(distinct) == 40


def test_the_workspace_refuses_a_hand_that_moves_more_than_8_cm_in_one_step(tmp_path):
    # The right hand moves 1 mm a step but once 9 cm, 0.5% of its steps; the left, which sorts first, stands still.
    steps = np.array([0.001] * 199 + [0.09])
    episodes = [
        _moving(right=np.stack([np.concatenate([[0.0], np.cumsum(steps)]), np.full(201, side), np.zeros(201)], axis=1))
        | _moving(left=np.zeros((201, 3)))
        for side in (0.0, 0.1)
    ]

    row = _row(_write(tmp_path / "d.hdf5", episodes), "workspace_spread")

    assert row["status"] == "not_applicable" and "right moves 9 cm in one recorded step" in row["reason"]


def test_workspace_counts_a_retraced_path_once(tmp_path):
    # Against a 30 cm line, the line with a 10 cm branch and the line traced five times with the same branch visit the
    # same space.
    line = np.linspace([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], 31)
    branch = np.linspace([0.3, 0.0, 0.0], [0.3, 0.1, 0.0], 11)[1:]
    once = np.concatenate([line, branch])
    retraced = np.concatenate([line, line[-2::-1], line[1:], line[-2::-1], line[1:], branch])

    values = [
        _row(_write(tmp_path / f"{i}.hdf5", [_moving(arm=line), _moving(arm=other)]), "workspace_spread")
        for i, other in enumerate((once, retraced))
    ]

    assert values[0]["value"] == pytest.approx(values[1]["value"], abs=0.01)


def test_the_workspace_is_how_far_apart_two_episodes_hands_run(tmp_path):
    # In one episode both hands sweep 10 cm along x, recorded at 0, 5 and 10 cm; in the other, the left hand rests
    # 10 cm beyond the sweep's end and the right 3 cm beside its middle. An offset of 1.25 mm keeps the sweep's 2.5 mm
    # samples off the 5 mm cell edges, so it keeps a point every 5 mm. The left sweep's points lie 150 mm from the
    # resting hand on average, which lies 100 mm from the nearest of them: 125 mm. The right sweep's points at x [mm] lie
    # hypot(30, x - 52.5) mm from the resting hand, which lies hypot(30, 2.5) mm from the nearest of them.
    offset = np.full(3, 0.00125)
    sweep = offset + [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]]
    beyond = np.tile(offset + [0.2, 0.0, 0.0], (3, 1))
    beside = np.tile(offset + [0.0525, 0.03, 0.0], (3, 1))
    right = 0.5 * (np.mean([np.hypot(30.0, x - 52.5) for x in range(0, 101, 5)]) + np.hypot(30.0, 2.5))
    episodes = [_moving(left=sweep, right=sweep), _moving(left=beyond, right=beside)]

    row = _row(_write(tmp_path / "d.hdf5", episodes), "workspace_spread")

    assert row["value"] == pytest.approx(0.5 * (125.0 + right), rel=1e-9)


def test_the_workspace_does_not_depend_on_the_order_of_episodes(tmp_path):
    # 64 episodes make 2,016 pairs, more than the workspace measures; which ones depends on content, not order.
    starts = np.random.default_rng(0).uniform(-0.3, 0.3, (64, 3))
    episodes = [_moving(arm=np.linspace(start, start + [0.1, 0.0, 0.0], 6)) for start in starts]

    values = [
        _row(_write(tmp_path / f"{label}.hdf5", ordered), "workspace_spread")["value"]
        for label, ordered in (("forward", episodes), ("reversed", episodes[::-1]))
    ]

    assert values[0] == pytest.approx(values[1], rel=1e-12)
