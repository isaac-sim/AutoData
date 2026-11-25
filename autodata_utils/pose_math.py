# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose math utilities used by embodiment adapters and domain randomization.

Pure-torch implementations of pose composition, quaternion conversion, axis-angle
conversion, and sampling. Quaternions are ``(x, y, z, w)``. Axis-angle vectors
are ``axis * angle`` (magnitude is the rotation angle in radians). Rotation
matrices act on column vectors.
"""

from __future__ import annotations

import torch
from typing import Literal

_EPS = 1e-8


def sample_uniform_scalar(value_range: tuple[float, float], device: str) -> float:
    """Sample a uniform scalar, leaving the RNG untouched when the bounds are equal.

    Args:
        value_range: Lower and upper sampling bounds.
        device: Device whose Torch RNG supplies the sample.
    """
    low, high = value_range
    if low == high:
        return float(low)
    return torch.empty((), device=device).uniform_(low, high).item()


def sample_uniform_vector(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    device: str,
) -> torch.Tensor:
    """Sample a 3D offset, leaving fixed components out of the RNG draw sequence.

    Args:
        minimum: Lower sampling bounds for each component.
        maximum: Upper sampling bounds for each component.
        device: Device whose Torch RNG supplies the samples.

    Returns:
        A float64 CPU tensor of shape ``(3,)`` for pose and USD mount calculations.
    """
    return torch.tensor(
        [sample_uniform_scalar((low, high), device) for low, high in zip(minimum, maximum)],
        dtype=torch.float64,
        device="cpu",
    )


def make_pose(pos: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Assemble a 4x4 homogeneous pose from position and rotation matrix.

    Args:
        pos: Position of shape (..., 3).
        rot_mat: Rotation matrix of shape (..., 3, 3).

    Returns:
        Pose tensor of shape (..., 4, 4).
    """
    assert pos.shape[-1] == 3, f"pos last dim must be 3, got {pos.shape[-1]}"
    assert rot_mat.shape[-2:] == (3, 3), f"rot_mat last two dims must be (3, 3), got {rot_mat.shape[-2:]}"
    assert (
        pos.shape[:-1] == rot_mat.shape[:-2]
    ), f"pos batch {pos.shape[:-1]} must match rot_mat batch {rot_mat.shape[:-2]}"
    pose = torch.zeros(*pos.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot_mat
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def unmake_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a 4x4 homogeneous pose into (position, rotation matrix).

    Args:
        pose: Pose tensor of shape (..., 4, 4).

    Returns:
        Tuple of (position, rotation matrix) with shapes (..., 3) and (..., 3, 3).
    """
    assert pose.shape[-2:] == (4, 4), f"pose last two dims must be (4, 4), got {pose.shape[-2:]}"
    return pose[..., :3, 3], pose[..., :3, :3]


def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to rotation matrix.

    Args:
        quat: Quaternion of shape (..., 4) in (x, y, z, w) order.

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    assert quat.shape[-1] == 4, f"quat last dim must be 4, got {quat.shape[-1]}"
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return rot.reshape(*quat.shape[:-1], 3, 3)


def quat_from_matrix(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (x, y, z, w).

    Uses Shepperd's method: picks the largest of four candidate squared
    components for numerical stability. Stable at the 180-degree antipode.

    Args:
        rot_mat: Rotation matrix of shape (..., 3, 3).

    Returns:
        Quaternion of shape (..., 4) in (x, y, z, w) order.
    """
    assert rot_mat.shape[-2:] == (3, 3), f"rot_mat last two dims must be (3, 3), got {rot_mat.shape[-2:]}"
    m = rot_mat
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    sq_w = (1 + m00 + m11 + m22).clamp(min=0)
    sq_x = (1 + m00 - m11 - m22).clamp(min=0)
    sq_y = (1 - m00 + m11 - m22).clamp(min=0)
    sq_z = (1 - m00 - m11 + m22).clamp(min=0)
    branch = torch.stack([sq_w, sq_x, sq_y, sq_z], dim=-1).argmax(dim=-1)
    sw = 2.0 * torch.sqrt(sq_w + _EPS)
    sx = 2.0 * torch.sqrt(sq_x + _EPS)
    sy = 2.0 * torch.sqrt(sq_y + _EPS)
    sz = 2.0 * torch.sqrt(sq_z + _EPS)
    w_branch = torch.stack([(m21 - m12) / sw, (m02 - m20) / sw, (m10 - m01) / sw, 0.25 * sw], dim=-1)
    x_branch = torch.stack([0.25 * sx, (m01 + m10) / sx, (m02 + m20) / sx, (m21 - m12) / sx], dim=-1)
    y_branch = torch.stack([(m01 + m10) / sy, 0.25 * sy, (m12 + m21) / sy, (m02 - m20) / sy], dim=-1)
    z_branch = torch.stack([(m02 + m20) / sz, (m12 + m21) / sz, 0.25 * sz, (m10 - m01) / sz], dim=-1)
    branches = torch.stack([w_branch, x_branch, y_branch, z_branch], dim=-2)
    idx = branch.unsqueeze(-1).unsqueeze(-1).expand(*branch.shape, 1, 4)
    return torch.gather(branches, dim=-2, index=idx).squeeze(-2)


def axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to compact axis-angle vector ``axis * angle``.

    Returns the zero vector for the identity quaternion. The output is a
    single 3-vector whose magnitude is the rotation angle in radians.

    Args:
        quat: Quaternion of shape (..., 4) in (x, y, z, w) order.

    Returns:
        Axis-angle vector of shape (..., 3).
    """
    assert quat.shape[-1] == 4, f"quat last dim must be 4, got {quat.shape[-1]}"
    xyz = quat[..., 0:3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    safe = sin_half > _EPS
    axis = torch.where(safe, xyz / sin_half.clamp(min=_EPS), torch.zeros_like(xyz))
    return axis * angle


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Build a quaternion from separate angle and unit axis.

    The caller is responsible for ensuring ``axis`` is unit-norm. For compact
    axis-angle vectors use :func:`quat_from_axis_angle_vec`.

    Args:
        angle: Rotation angle in radians of shape (...,).
        axis: Unit rotation axis of shape (..., 3).

    Returns:
        Quaternion of shape (..., 4) in (x, y, z, w) order.
    """
    assert axis.shape[-1] == 3, f"axis last dim must be 3, got {axis.shape[-1]}"
    half = 0.5 * angle
    sin_half = torch.sin(half).unsqueeze(-1)
    cos_half = torch.cos(half).unsqueeze(-1)
    return torch.cat([axis * sin_half, cos_half], dim=-1)


def quat_from_axis_angle_vec(axis_angle: torch.Tensor) -> torch.Tensor:
    """Build a quaternion from a compact axis-angle vector ``axis * angle``.

    Splits the input into magnitude (angle) and direction (unit axis),
    handling the zero-rotation case so the division is safe.

    Args:
        axis_angle: Axis-angle vector of shape (..., 3) in radians.

    Returns:
        Quaternion of shape (..., 4) in (x, y, z, w) order.
    """
    assert axis_angle.shape[-1] == 3, f"axis_angle last dim must be 3, got {axis_angle.shape[-1]}"
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    safe = angle > _EPS
    default_axis = torch.zeros_like(axis_angle)
    default_axis[..., 0] = 1.0
    axis = torch.where(safe, axis_angle / angle.clamp(min=_EPS), default_axis)
    half = 0.5 * angle
    return torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)


def quat_unique(quat: torch.Tensor) -> torch.Tensor:
    """Canonicalize a quaternion so that ``w >= 0``.

    ``q`` and ``-q`` represent the same rotation. Canonicalizing prevents
    sign drift in recorded actions.

    Args:
        quat: Quaternion of shape (..., 4) in (x, y, z, w) order.

    Returns:
        Quaternion with non-negative ``w``, same shape.
    """
    assert quat.shape[-1] == 4, f"quat last dim must be 4, got {quat.shape[-1]}"
    sign = torch.where(quat[..., 3:4] < 0, -torch.ones_like(quat[..., 3:4]), torch.ones_like(quat[..., 3:4]))
    return quat * sign


def quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose quaternions, applying ``second`` in the body frame of ``first``.

    Args:
        first: Quaternion of shape (..., 4) in (x, y, z, w) order.
        second: Quaternion of shape (..., 4), broadcastable with ``first``.

    Returns:
        Quaternion product with the broadcast shape of the inputs.
    """
    assert first.shape[-1] == 4 and second.shape[-1] == 4, "Quaternions must have four components."
    first, second = torch.broadcast_tensors(first, second)
    xyz_first, w_first = first[..., :3], first[..., 3:4]
    xyz_second, w_second = second[..., :3], second[..., 3:4]
    xyz = w_first * xyz_second + w_second * xyz_first + torch.linalg.cross(xyz_first, xyz_second, dim=-1)
    w = w_first * w_second - (xyz_first * xyz_second).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_from_euler_xyz(euler_angles: torch.Tensor) -> torch.Tensor:
    """Convert intrinsic XYZ Euler angles to a quaternion.

    The rotation is ``Rx(roll) @ Ry(pitch) @ Rz(yaw)``; angles are about the moving axes.

    Args:
        euler_angles: Roll, pitch, and yaw [rad] of shape (..., 3).

    Returns:
        Quaternion of shape (..., 4) in (x, y, z, w) order.
    """
    assert euler_angles.shape[-1] == 3, "Euler angles must have three components."
    axes = torch.eye(3, dtype=euler_angles.dtype, device=euler_angles.device)
    rotations = quat_from_angle_axis(euler_angles, axes)
    return quat_mul(quat_mul(rotations[..., 0, :], rotations[..., 1, :]), rotations[..., 2, :])


def convert_camera_frame_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    """Express a camera orientation in another convention while preserving its physical mount.

    OpenGL looks along -Z with +Y up; ROS looks along +Z with -Y up; world looks along +X
    with +Z up. Conversion changes the local camera basis, leaving its world pose intact.

    Args:
        orientation: Quaternion of shape (..., 4) in (x, y, z, w) order.
        origin: Source camera convention.
        target: Destination camera convention.

    Returns:
        Converted quaternion with the input shape, dtype, and device.
    """
    assert orientation.shape[-1] == 4, "Quaternions must have four components."
    # Right-multiply by these fixed rotations to express each convention in OpenGL.
    to_opengl = {
        "opengl": (0.0, 0.0, 0.0, 1.0),
        "ros": (1.0, 0.0, 0.0, 0.0),
        "world": (0.5, -0.5, -0.5, 0.5),
    }
    assert origin in to_opengl and target in to_opengl, "Unsupported camera orientation convention."
    if origin == target:
        return orientation.clone()
    source_to_opengl = orientation.new_tensor(to_opengl[origin])
    opengl_to_target = orientation.new_tensor(to_opengl[target])
    opengl_to_target[:3] *= -1
    return quat_mul(quat_mul(orientation, source_to_opengl), opengl_to_target)
