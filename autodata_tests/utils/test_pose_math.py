# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose sampling, quaternion composition, and camera-frame checks."""

import numpy as np
import torch
from itertools import product
from scipy.spatial.transform import Rotation

import pytest

from autodata_utils import pose_math

# Local camera bases expressed in OpenGL axes, from the viewing-axis definitions.
_CAMERA_TO_OPENGL = {
    "opengl": np.eye(3),
    "ros": np.diag([1.0, -1.0, -1.0]),
    "world": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}


@pytest.mark.parametrize("value", [0.0, 1.3, -0.1])
def test_fixed_bounds_preserve_the_value_and_rng(value):
    rng = torch.get_rng_state().clone()
    assert pose_math.sample_uniform_scalar((value, value), "cpu") == value
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_composition_broadcasts_and_matches_scipy(dtype):
    first = Rotation.from_euler("XYZ", [[41, -12, 83], [-20, 45, 15]], degrees=True)
    second = Rotation.from_euler("XYZ", [3, -3, 2], degrees=True)
    left = torch.tensor(first.as_quat(), dtype=dtype)
    right = torch.tensor(second.as_quat(), dtype=dtype)
    before_left, before_right = left.clone(), right.clone()

    result = pose_math.quat_mul(left, right)

    assert result.shape == (2, 4) and result.dtype == dtype and result.device == left.device
    np.testing.assert_allclose(Rotation.from_quat(result.numpy()).as_matrix(), (first * second).as_matrix(), atol=1e-6)
    torch.testing.assert_close(left, before_left, rtol=0, atol=0)
    torch.testing.assert_close(right, before_right, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_intrinsic_xyz_handles_batched_angles_and_preserves_dtype(dtype):
    angles = np.array([
        [0, 0, 0],
        [90, 0, 0],
        [0, 0, 90],
        [2, -3, 1],
        [45, 30, -20],
        [90, 90, 0],
        [180, 0, 0],
        [-180, 90, 270],
    ])
    radians = torch.deg2rad(torch.tensor(angles, dtype=dtype))

    result = pose_math.quat_from_euler_xyz(radians)

    assert result.shape == (8, 4) and result.dtype == dtype and result.device == radians.device
    np.testing.assert_allclose(
        Rotation.from_quat(result.numpy()).as_matrix(),
        Rotation.from_euler("XYZ", angles, degrees=True).as_matrix(),
        atol=1e-6,
    )
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(8, dtype=dtype))
    torch.testing.assert_close(result[0], torch.tensor([0, 0, 0, 1], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("origin,target", list(product(_CAMERA_TO_OPENGL, repeat=2)))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_camera_conventions_preserve_the_physical_mount(origin, target, dtype):
    rotation = Rotation.from_euler("XYZ", [[41, -12, 83], [90, 0, -45]], degrees=True)
    orientation = torch.tensor(rotation.as_quat(), dtype=dtype)
    before = orientation.clone()

    result = pose_math.convert_camera_frame_orientation_convention(orientation, origin, target)

    expected = rotation.as_matrix() @ _CAMERA_TO_OPENGL[origin] @ _CAMERA_TO_OPENGL[target].T
    assert result.shape == orientation.shape and result.dtype == dtype and result.device == orientation.device
    np.testing.assert_allclose(Rotation.from_quat(result.numpy()).as_matrix(), expected, atol=1e-6)
    torch.testing.assert_close(orientation, before, rtol=0, atol=0)


@pytest.mark.parametrize("origin,target", [("unknown", "ros"), ("opengl", "unknown")])
def test_camera_conventions_reject_unknown_names(origin, target):
    with pytest.raises(AssertionError, match="Unsupported camera orientation convention"):
        pose_math.convert_camera_frame_orientation_convention(torch.tensor([0.0, 0.0, 0.0, 1.0]), origin, target)


def test_vector_sampling_consumes_draws_only_for_varying_components():
    torch.manual_seed(19)
    torch.rand(2)
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(19)
    first = pose_math.sample_uniform_vector((-0.3, 0.0, 1.5), (0.4, 0.0, 3.0), "cpu")
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert -0.3 <= first[0] <= 0.4 and first[1] == 0.0 and 1.5 <= first[2] <= 3.0

    torch.manual_seed(19)
    torch.testing.assert_close(
        pose_math.sample_uniform_vector((-0.3, 0.0, 1.5), (0.4, 0.0, 3.0), "cpu"), first, rtol=0, atol=0
    )
