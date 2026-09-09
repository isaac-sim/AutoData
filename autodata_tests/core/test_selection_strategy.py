# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`autodata_core.selection_strategy`."""

import torch

import pytest

from autodata_core.datagen_info import DatagenInfo
from autodata_core.selection_strategy import (
    REGISTERED_SELECTION_STRATEGIES,
    NearestNeighborObjectStrategy,
    NearestNeighborRobotDistanceStrategy,
    RandomStrategy,
    RegistrationCostStrategy,
    make_selection_strategy,
)


def _pose(pos, rot: torch.Tensor | None = None) -> torch.Tensor:
    p = torch.eye(4)
    if rot is not None:
        p[:3, :3] = rot
    p[:3, 3] = torch.tensor(pos, dtype=torch.float32)
    return p


# +90 degrees about z; a non-identity rotation makes the object-frame transform observable.
_RZ90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_registry_contents():
    assert set(REGISTERED_SELECTION_STRATEGIES) == {
        "random",
        "nearest_neighbor_object",
        "nearest_neighbor_robot_distance",
        "registration_cost",
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


def test_nearest_neighbor_robot_distance_transforms_source_eef_into_current_object_frame():
    # Test source-object -> current-object frame conversion and its multiplication order.
    #   - demo 0 wins under the correct object-frame transform,
    #   - demo 1 wins if the conversion is skipped (raw EEF distance),
    #   - demo 2 wins if the transform is applied in reversed order.
    current_object_pose = _pose([10.0, 0.0, 0.0], _RZ90)
    local_eef_offset = _pose([1.0, 0.0, 0.0])  # EEF pose expressed in its object's frame
    target_eef_pose = current_object_pose @ local_eef_offset

    # demo 0: a non-identity source object with same local offset.
    source_object_pose = _pose([1.0, 2.0, 0.0])
    correct_match = DatagenInfo(
        eef_pose=(source_object_pose @ local_eef_offset).unsqueeze(0),
        object_poses={"cube": source_object_pose.unsqueeze(0)},
    )
    # demo 1: raw EEF sits exactly on the target.
    raw_distance_decoy = DatagenInfo(
        eef_pose=target_eef_pose.unsqueeze(0),
        object_poses={"cube": torch.eye(4).unsqueeze(0)},
    )
    # demo 2: constructed so that only a reversed transform order (eef @ obj_inv @ current) would
    # land on the target.
    reversed_order_decoy = DatagenInfo(
        eef_pose=(target_eef_pose @ torch.inverse(current_object_pose)).unsqueeze(0),
        object_poses={"cube": torch.eye(4).unsqueeze(0)},
    )

    infos = [correct_match, raw_distance_decoy, reversed_order_decoy]
    index = NearestNeighborRobotDistanceStrategy().select_source_demo(
        target_eef_pose, current_object_pose, infos, nn_k=1
    )

    assert int(index) == 0


def test_registration_cost_picks_lowest_cost(monkeypatch):
    import autodata_core.deformable_transforms as deformable_transforms

    monkeypatch.setattr(
        deformable_transforms,
        "nodal_registration_cost",
        lambda source, target, **kwargs: float(torch.linalg.vector_norm(source - target)),
    )
    current = torch.zeros(6, 3)
    infos = [DatagenInfo(object_nodal_positions={"rope": torch.full((1, 6, 3), value)}) for value in (2.0, 0.0, 1.0)]
    index = RegistrationCostStrategy().select_source_demo(
        None,
        None,
        infos,
        object_nodal_positions=current,
        nn_k=1,
    )
    assert int(index) == 1


def test_registration_cost_skips_failed_and_nonfinite_candidates(monkeypatch):
    import autodata_core.deformable_transforms as deformable_transforms

    def registration_cost(source, target, **kwargs):
        del target, kwargs
        source_value = source[0, 0].item()
        if source_value == 0.0:
            raise ValueError("invalid TPS input")
        if source_value == 1.0:
            return float("nan")
        return 1.0

    monkeypatch.setattr(deformable_transforms, "nodal_registration_cost", registration_cost)
    current = torch.zeros(6, 3)
    infos = [DatagenInfo(object_nodal_positions={"rope": torch.full((1, 6, 3), value)}) for value in (0.0, 1.0, 2.0)]

    index = RegistrationCostStrategy().select_source_demo(
        None,
        None,
        infos,
        object_nodal_positions=current,
        nn_k=3,
    )

    assert index == 2


def test_registration_cost_rejects_all_invalid_candidates(monkeypatch):
    import autodata_core.deformable_transforms as deformable_transforms

    monkeypatch.setattr(deformable_transforms, "nodal_registration_cost", lambda source, target, **kwargs: float("inf"))
    current = torch.zeros(6, 3)
    infos = [DatagenInfo(object_nodal_positions={"rope": torch.zeros(1, 6, 3)})]

    with pytest.raises(AssertionError, match="could not compute a finite TPS cost"):
        RegistrationCostStrategy().select_source_demo(
            None,
            None,
            infos,
            object_nodal_positions=current,
            nn_k=1,
        )
