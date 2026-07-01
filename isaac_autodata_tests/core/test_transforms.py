# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.transforms`."""

import torch

import pytest
from isaaclab.envs import SubTaskConstraintCoordinationScheme

from isaac_autodata_core.transforms import (
    get_delta_pose_with_scheme,
    transform_source_data_segment_using_delta_object_pose,
    transform_source_data_segment_using_object_pose,
)


def _pose(pos) -> torch.Tensor:
    """Identity-rotation 4x4 pose at ``pos``."""
    p = torch.eye(4)
    p[:3, 3] = torch.tensor(pos, dtype=torch.float32)
    return p


def _constraint(scheme, pos_noise: float = 0.0, rot_noise: float = 0.0) -> dict:
    return {
        "coordination_scheme": scheme,
        "coordination_scheme_pos_noise_scale": pos_noise,
        "coordination_scheme_rot_noise_scale": rot_noise,
    }


# ---------------------------------------------------------------------------------------------------
# transform_source_data_segment_using_delta_object_pose
# ---------------------------------------------------------------------------------------------------
def test_delta_identity_is_noop():
    src = torch.stack([_pose([0.0, 0.0, 0.0]), _pose([1.0, 2.0, 3.0])])
    out = transform_source_data_segment_using_delta_object_pose(src, torch.eye(4))
    assert torch.allclose(out, src, atol=1e-6)


def test_delta_translation_shifts_all_poses():
    src = torch.stack([_pose([0.0, 0.0, 0.0]), _pose([1.0, 2.0, 3.0])])
    out = transform_source_data_segment_using_delta_object_pose(src, _pose([1.0, 0.0, 0.0]))
    assert torch.allclose(out[:, :3, 3], src[:, :3, 3] + torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------------------------------
# transform_source_data_segment_using_object_pose
# ---------------------------------------------------------------------------------------------------
def test_object_transform_is_noop_when_object_unmoved():
    src_obj = _pose([1.0, 1.0, 1.0])
    src_eef = torch.stack([_pose([2.0, 2.0, 2.0]), _pose([3.0, 3.0, 3.0])])
    out = transform_source_data_segment_using_object_pose(src_obj, src_eef, src_obj)
    assert torch.allclose(out, src_eef, atol=1e-5)


def test_object_transform_translates_eef_with_object():
    src_obj = _pose([1.0, 1.0, 1.0])
    cur_obj = _pose([2.0, 1.0, 1.0])  # object moved +x by 1
    src_eef = torch.stack([_pose([2.0, 2.0, 2.0]), _pose([3.0, 3.0, 3.0])])
    out = transform_source_data_segment_using_object_pose(cur_obj, src_eef, src_obj)
    assert torch.allclose(out[:, :3, 3], src_eef[:, :3, 3] + torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------------------------------
# get_delta_pose_with_scheme
# ---------------------------------------------------------------------------------------------------
def test_scheme_replay_is_identity():
    delta = get_delta_pose_with_scheme(
        _pose([0.0, 0.0, 0.0]), _pose([5.0, 5.0, 5.0]), _constraint(SubTaskConstraintCoordinationScheme.REPLAY)
    )
    assert torch.allclose(delta, torch.eye(4), atol=1e-6)


def test_scheme_translate_is_position_difference():
    delta = get_delta_pose_with_scheme(
        _pose([1.0, 2.0, 3.0]), _pose([4.0, 6.0, 8.0]), _constraint(SubTaskConstraintCoordinationScheme.TRANSLATE)
    )
    assert torch.allclose(delta[:3, 3], torch.tensor([3.0, 4.0, 5.0]), atol=1e-6)
    assert torch.allclose(delta[:3, :3], torch.eye(3), atol=1e-6)


def test_scheme_transform_identity_when_object_unmoved():
    obj = _pose([1.0, 2.0, 3.0])
    delta = get_delta_pose_with_scheme(obj, obj, _constraint(SubTaskConstraintCoordinationScheme.TRANSFORM))
    assert torch.allclose(delta, torch.eye(4), atol=1e-5)


def test_scheme_transform_delta_matches_object_relative_transform():
    # The TRANSFORM delta applied to a segment must equal re-expressing that segment for the current
    # object pose (this pins the get_delta_object_pose replacement to the right behavior).
    src_obj = _pose([1.0, 0.0, 0.0])
    cur_obj = _pose([0.0, 2.0, 0.0])
    src_eef = torch.stack([_pose([1.0, 1.0, 1.0]), _pose([2.0, 0.0, 3.0])])
    delta = get_delta_pose_with_scheme(src_obj, cur_obj, _constraint(SubTaskConstraintCoordinationScheme.TRANSFORM))
    via_delta = transform_source_data_segment_using_delta_object_pose(src_eef, delta)
    via_object = transform_source_data_segment_using_object_pose(cur_obj, src_eef, src_obj)
    assert torch.allclose(via_delta, via_object, atol=1e-5)


def test_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported coordination scheme"):
        get_delta_pose_with_scheme(_pose([0.0, 0.0, 0.0]), _pose([1.0, 0.0, 0.0]), _constraint(999))


def test_zero_noise_is_deterministic():
    args = (_pose([0.0, 0.0, 0.0]), _pose([1.0, 2.0, 3.0]), _constraint(SubTaskConstraintCoordinationScheme.TRANSLATE))
    assert torch.equal(get_delta_pose_with_scheme(*args), get_delta_pose_with_scheme(*args))
