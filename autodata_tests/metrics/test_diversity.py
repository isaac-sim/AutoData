# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests of the diversity metrics."""

import h5py
import json
import numpy as np
from pathlib import Path

import pytest

from autodata_metrics import REGISTERED_METRICS, build_report
from autodata_metrics.diversity import MAX_PAIRS, _pairs

CUBE = np.array([0.5, 0.0, 0.02])
BALL = np.array([0.5, 0.5, 0.02])
SIDES = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1])


def _write(path: Path, episodes: list[dict[str, np.ndarray]]) -> str:
    with h5py.File(path, "w") as h5_file:
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


def _hand(track: np.ndarray, name: str = "arm") -> dict[str, np.ndarray]:
    """An episode whose hand ``name`` passes through ``track``."""
    poses = np.tile(np.eye(4), (len(track), 1, 1))
    poses[:, :3, 3] = track
    return {"actions": np.zeros((len(track), 7)), f"obs/datagen_info/eef_pose/{name}": poses}


def _still(name: str, at: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """An object resting at ``at`` for ``steps`` steps."""
    poses = np.tile(np.eye(4), (steps, 1, 1))
    poses[:, :3, 3] = at
    return {
        f"obs/datagen_info/object_pose/{name}": poses,
        f"states/rigid_object/{name}/root_pose": np.tile([*at, 1.0, 0.0, 0.0, 0.0], (steps, 1)),
    }


def _lift(
    direction: list[float],
    hand_offset: float = 0.0,
    yaw: float = 0.0,
    walk: float = 0.0,
    rise: float = 0.02,
    slip: float = 0.0,
    resting_hand: bool = False,
    hand: str = "arm",
    name: str = "cube",
    at: np.ndarray = CUBE,
) -> dict[str, np.ndarray]:
    """An episode whose grasp point comes straight to an object from 20 cm along ``direction`` in the object's frame,
    then lifts it by 1, 1.5 and 2 times ``rise``.

    The object is turned by ``yaw`` [rad]. Hand and object poses are observed before each step in a robot frame that
    moves ``walk`` along y per step; the states record the object in the env after each step. The recorded hand frame
    sits ``hand_offset`` below the grasp point, and while lifting the hand rises ``slip`` per step more than the
    object. With ``resting_hand``, a second hand waits 20 cm from the object.
    """
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    track = [at + 0.2 * turn @ np.array(direction) * (1 - step / 9) for step in range(10)]
    track = np.array(track + [at + [0.0, 0.0, rise * scale] for scale in (1.0, 1.5, 2.0)])
    robot = np.array([[0.0, walk * step, 0.0] for step in range(len(track))])
    world = np.tile(np.eye(4), (len(track), 1, 1))
    world[:, :3, :3] = turn
    world[:, :3, 3] = at
    world[10:, :3, 3] = track[10:]
    observed = world.copy()
    observed[:, :3, 3] -= robot
    after = np.concatenate([world[1:, :3, 3], world[-1:, :3, 3]])
    quaternion = np.tile([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], (len(track), 1))
    grasp = track + [[0.0, 0.0, slip * max(0, step - 9)] for step in range(len(track))]
    episode = {
        **_hand(grasp - [0.0, 0.0, hand_offset] - robot, hand),
        f"obs/datagen_info/object_pose/{name}": observed,
        f"states/rigid_object/{name}/root_pose": np.concatenate([after, quaternion], axis=1),
    }
    if resting_hand:
        episode.update(_hand(np.tile(at + [0.0, -0.2, 0.0], (len(track), 1)) - robot, "other"))
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
    # Two hands sweep 9.5 cm along x, 7 cm apart, past a cube nobody lifts. Of the 5 cm voxels they visit, x cells
    # 10-11 and 11-13, three of four are not shared, and their paths are 7 cm apart.
    sweep = np.array([[0.5025 + 0.005 * step, 0.0125, 0.0125] for step in range(20)])
    cube = _still("cube", CUBE, len(sweep))
    seeds = _write(tmp_path / "s.hdf5", [_hand(sweep) | cube, _hand(sweep) | cube])

    episodes = [_hand(sweep) | cube, _hand(sweep + [0.07, 0.0, 0.0]) | cube]
    row = _row(_write(tmp_path / "d.hdf5", episodes), "workspace_coverage", seeds)

    assert row["value"] == pytest.approx(3 / 4) and row["baseline"] == 0.0
    assert row["details"]["path_distance_mm"] == pytest.approx(70.0)


@pytest.mark.parametrize(
    "hand_offset, resting_hand, walk", [(0.0, False, 0.0), (0.1, False, 0.0), (0.26, True, 0.0), (0.0, False, 0.05)]
)
def test_approach_metrics_average_over_pairs_of_lifts_in_the_object_frame(tmp_path, hand_offset, resting_hand, walk):
    # From ±x, +y and above: one opposite pair and five perpendicular ones. Straight approaches over their last
    # 15.6 cm are 0.0913 m × |d1 - d2| apart in root-mean-square, so 138.0 mm on average.
    # A hand frame away from the grasp point, as a flange or a wrist is, gives the same answer, also when the other
    # hand rests nearer the cube than the carrying wrist, and so does a robot frame that walks.
    lifts = [_lift(side, hand_offset, walk=walk, resting_hand=resting_hand) for side in SIDES]
    dataset = _write(tmp_path / "d.hdf5", lifts)
    # Seeds from -x and -y, 90° and 129.1 mm apart: the nearest seed path is 0 mm from the -x lift and 129.1 mm from
    # the three others, whose median novelty is that.
    seeds = _write(tmp_path / "s.hdf5", [_lift([-1, 0, 0]), _lift([0, -1, 0])])

    spread = _row(dataset, "approach_spread", seeds)
    diversity = _row(dataset, "behavior_diversity", seeds)

    assert (spread["value"], spread["baseline"], spread["n"]) == (pytest.approx(105.0), pytest.approx(90.0), 4)
    assert diversity["value"] == pytest.approx(138.0, abs=0.1) and diversity["baseline"] == pytest.approx(
        129.1, abs=0.1
    )
    assert diversity["details"]["objects"]["cube"]["novelty_mm"] == pytest.approx(129.1, abs=0.1)


def test_lift_metrics_average_over_the_lifted_objects(tmp_path):
    # The left hand lifts the cube from the four sides above. The right hand lifts the ball in two of the episodes,
    # from one side and at the same pace, and rests beside it in the others. Each object counts once.
    episodes = []
    for index, side in enumerate(SIDES):
        ball = _lift([1, 0, 0], hand="right", name="ball", at=BALL)
        if index >= 2:
            ball = _hand(np.tile(BALL + [0.2, 0.0, 0.0], (13, 1)), "right") | _still("ball", BALL, 13)
        episodes.append(_lift(side, hand="left") | ball)
    dataset = _write(tmp_path / "d.hdf5", episodes)

    spread, diversity = _row(dataset, "approach_spread"), _row(dataset, "behavior_diversity")

    assert (spread["value"], spread["n"]) == (pytest.approx(105.0 / 2), 6)
    assert diversity["value"] == pytest.approx(138.0 / 2, abs=0.1)


def test_the_carrying_hand_is_near_the_object(tmp_path):
    # The cube slips in the hand while rising, 1 cm per step: in the frame of the wrist, 25 cm away like a humanoid's,
    # it moves more than the idle hand 55 cm away sees it move.
    idle = _hand(np.tile(CUBE + [0.0, -0.55, 0.0], (13, 1)), "idle")
    lifts = [_lift(side, hand_offset=0.25, rise=0.012, slip=0.01) | idle for side in SIDES]
    dataset = _write(tmp_path / "d.hdf5", lifts)

    assert _row(dataset, "approach_spread")["value"] == pytest.approx(105.0)


def test_approach_metrics_measure_in_the_object_frame(tmp_path):
    # Every lift comes at the cube's own +x face while the cube turns, so in its frame all approaches are one.
    dataset = _write(tmp_path / "d.hdf5", [_lift([1, 0, 0], yaw=yaw) for yaw in np.radians([0, 30, 60, 90])])

    assert _row(dataset, "approach_spread")["value"] == pytest.approx(0.0, abs=1e-3)
    assert _row(dataset, "behavior_diversity")["value"] == pytest.approx(0.0, abs=1e-3)


def test_objects_without_usable_poses_are_left_out(tmp_path):
    # Beside the cube, each episode records a ghost with an empty start pose and a lid observed but without a root
    # pose; the metrics read the cube alone.
    episodes = []
    for index, side in enumerate(SIDES[::2]):
        episode = _starts(cube=[0.04 * index, 0.0]) | _lift(side)
        episode["initial_state/rigid_object/ghost/root_pose"] = np.zeros((0, 7))
        episode["obs/datagen_info/object_pose/lid"] = episode["obs/datagen_info/object_pose/cube"]
        episode["states/rigid_object/lid/root_velocity"] = np.zeros((13, 6))
        episodes.append(episode)
    dataset = _write(tmp_path / "d.hdf5", episodes)

    rows = {name: _row(dataset, name) for name in ("start_spread", "workspace_coverage", "approach_spread")}

    assert {name: row["status"] for name, row in rows.items()} == dict.fromkeys(rows, "ok")
    assert rows["start_spread"]["value"] == pytest.approx(40.0) and rows["approach_spread"]["value"] == pytest.approx(
        90.0
    )


def test_pairs_drawn_beyond_the_cap_never_pair_an_item_with_itself():
    first, second = _pairs(65)  # 2,080 pairs, more than the cap

    assert len(first) == MAX_PAIRS and (first != second).all()


def test_exact_duplicates_counts_episodes_that_repeat_an_earlier_one(tmp_path):
    # The last episode holds the same bytes as the first in another shape, which is not a repeat.
    actions = [np.zeros((3, 7)), np.ones((3, 7)), np.zeros((3, 7)), np.zeros((7, 3))]

    row = _row(_write(tmp_path / "d.hdf5", [{"actions": array} for array in actions]), "exact_duplicates")

    assert row["value"] == 0.25 and row["details"]["duplicate_of"] == {"demo_2": "demo_0"}


@pytest.mark.parametrize("missing", ["obs/datagen_info/object_pose", "states"])
@pytest.mark.parametrize("name", ["workspace_coverage", "approach_spread", "behavior_diversity"])
def test_behavior_metrics_need_object_poses_and_states(tmp_path, name, missing):
    # Grasp points come from lifts, which need both.
    episodes = [{key: array for key, array in _lift(side).items() if not key.startswith(missing)} for side in SIDES]
    assert _row(_write(tmp_path / "d.hdf5", episodes), name)["status"] == "not_applicable"
