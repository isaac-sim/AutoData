# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.selection_strategy`."""

import torch

import pytest

from isaac_autodata_core.datagen_info import DatagenInfo
from isaac_autodata_core.selection_strategy import (
    REGISTERED_SELECTION_STRATEGIES,
    NearestNeighborObjectStrategy,
    NearestNeighborRobotDistanceStrategy,
    RandomStrategy,
    make_selection_strategy,
)


def _pose(pos) -> torch.Tensor:
    p = torch.eye(4)
    p[:3, 3] = torch.tensor(pos, dtype=torch.float32)
    return p


def test_registry_contents():
    assert set(REGISTERED_SELECTION_STRATEGIES) == {
        "random",
        "nearest_neighbor_object",
        "nearest_neighbor_robot_distance",
    }


def test_make_returns_registered_instance():
    assert isinstance(make_selection_strategy("random"), RandomStrategy)
    assert isinstance(make_selection_strategy("nearest_neighbor_object"), NearestNeighborObjectStrategy)


def test_make_unknown_error():
    with pytest.raises(KeyError, match="Unknown selection strategy"):
        make_selection_strategy("nope")


def test_random_returns_index_in_range():
    torch.manual_seed(0)
    strategy = RandomStrategy()
    infos = [DatagenInfo() for _ in range(4)]
    for _ in range(25):
        idx = strategy.select_source_demo(None, None, infos)
        assert 0 <= int(idx) < 4


def test_nearest_neighbor_object_picks_closest():
    positions = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]
    infos = [DatagenInfo(object_poses={"cube": _pose(p).unsqueeze(0)}) for p in positions]
    # nn_k=1 => deterministically the single closest source demo.
    idx = NearestNeighborObjectStrategy().select_source_demo(None, _pose([1.0, 1.0, 1.0]), infos, nn_k=1)
    assert int(idx) == 2


def test_nearest_neighbor_robot_distance_picks_closest():
    # All source demos share the same (identity) object pose, so the transform is a no-op and the
    # pick reduces to the source EEF pose closest to the current EEF pose.
    obj = _pose([0.0, 0.0, 0.0])
    eef_positions = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
    infos = [
        DatagenInfo(eef_pose=_pose(p).unsqueeze(0), object_poses={"cube": obj.unsqueeze(0)}) for p in eef_positions
    ]
    idx = NearestNeighborRobotDistanceStrategy().select_source_demo(_pose([2.0, 2.0, 2.0]), obj, infos, nn_k=1)
    assert int(idx) == 2
