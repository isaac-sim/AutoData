# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SoftMimicGen deformable transforms."""

import torch

import pytest

from isaac_autodata_core.deformable_transforms import (
    nodal_registration_cost,
    transform_source_data_segment_using_nodal_registration,
)

_NODES = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)


def _poses() -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    poses[0, :3, 3] = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)
    poses[1, :3, 3] = torch.tensor([0.75, 0.25, 0.25], dtype=torch.float64)
    return poses


def test_nodal_tps_translation_transforms_positions_and_preserves_rotations():
    translation = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    source_poses = _poses()
    transformed = transform_source_data_segment_using_nodal_registration(
        source_poses,
        _NODES,
        _NODES + translation,
    )
    assert torch.allclose(transformed[:, :3, 3], source_poses[:, :3, 3] + translation, atol=1e-6)
    assert torch.allclose(transformed[:, :3, :3], source_poses[:, :3, :3], atol=1e-6)
    assert torch.allclose(torch.linalg.det(transformed[:, :3, :3]), torch.ones(2, dtype=torch.float64))


def test_nodal_registration_cost_is_near_zero_for_identical_nodes():
    assert nodal_registration_cost(_NODES, _NODES) == pytest.approx(0.0, abs=1e-8)


def test_nodal_tps_rejects_mismatched_node_counts_before_fitting():
    with pytest.raises(AssertionError, match="same node count"):
        transform_source_data_segment_using_nodal_registration(
            _poses(),
            _NODES,
            _NODES[:-1],
        )
