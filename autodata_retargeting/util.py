# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic pose / quaternion math helpers shared across the retargeting pipeline."""

import torch
from typing import Any

import isaaclab.utils.math as math_utils

from autodata_utils.tensor_utils import as_torch


def se3_inverse(pose: torch.Tensor) -> torch.Tensor:
    """Inverse of a batched SE(3) homogeneous transform ``(..., 4, 4)`` (``R^T``, ``-R^T t``)."""
    rot = pose[..., :3, :3]
    trans = pose[..., :3, 3:]
    rot_t = rot.transpose(-1, -2)
    out = torch.zeros_like(pose)
    out[..., :3, :3] = rot_t
    out[..., :3, 3:] = -rot_t @ trans
    out[..., 3, 3] = 1.0
    return out


def pose_tracking_error(commanded: torch.Tensor, achieved: torch.Tensor) -> tuple[float, float]:
    """Return ``(position error [m], orientation error [deg])`` between two ``(4, 4)`` poses."""
    pos_err = float(torch.linalg.norm(commanded[:3, 3] - achieved[:3, 3]).item())
    quat_cmd = math_utils.quat_from_matrix(commanded[:3, :3].unsqueeze(0))
    quat_ach = math_utils.quat_from_matrix(achieved[:3, :3].unsqueeze(0))
    rot_err_deg = float(torch.rad2deg(math_utils.quat_error_magnitude(quat_cmd, quat_ach))[0].item())
    return pos_err, rot_err_deg


def quat_slerp_batch(q0: torch.Tensor, q1: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """Batched SLERP between quaternion rows ``q0``/``q1`` (N, 4) by ``frac`` (N, 1), in (x,y,z,w)."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # take the shorter arc
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    sin = torch.sin(angle)
    small = sin.abs() < 1e-6  # nearly parallel -> fall back to (normalized) linear
    w0 = torch.where(small, 1.0 - frac, torch.sin((1.0 - frac) * angle) / sin)
    w1 = torch.where(small, frac, torch.sin(frac * angle) / sin)
    return math_utils.normalize(w0 * q0 + w1 * q1)


def poses_from_root_pose(root_pose: torch.Tensor) -> torch.Tensor:
    """Convert a recorded ``root_pose`` ``(T, 7)`` = [pos(3), quat wxyz(4)] into ``(T, 4, 4)`` poses."""
    num_steps = root_pose.shape[0]
    poses = torch.eye(4, dtype=root_pose.dtype, device=root_pose.device).repeat(num_steps, 1, 1)
    poses[:, :3, :3] = math_utils.matrix_from_quat(root_pose[:, 3:7])
    poses[:, :3, 3] = root_pose[:, :3]
    return poses


def read_target_base_pose(env: Any, robot_asset_name: str, env_id: int = 0) -> torch.Tensor:
    """Read the target robot base (root) pose in the env-relative frame as a ``(4, 4)`` tensor."""
    robot = env.scene[robot_asset_name]
    pos_w = robot.data.root_pos_w
    quat_w = robot.data.root_quat_w
    pos_w = pos_w.torch if hasattr(pos_w, "torch") else as_torch(pos_w)
    quat_w = quat_w.torch if hasattr(quat_w, "torch") else as_torch(quat_w)
    origin = env.scene.env_origins
    origin = origin.torch if hasattr(origin, "torch") else as_torch(origin)
    pose = torch.eye(4, dtype=pos_w.dtype, device=pos_w.device)
    pose[:3, :3] = math_utils.matrix_from_quat(quat_w[env_id : env_id + 1])[0]
    pose[:3, 3] = pos_w[env_id] - origin[env_id]
    return pose


def reanchor_to_target_base(poses: torch.Tensor, source_base: torch.Tensor, target_base: torch.Tensor) -> torch.Tensor:
    """Re-express world ``poses`` relative to ``source_base`` (per step), then anchor to ``target_base``.

    ``poses`` and ``source_base`` are ``(T, 4, 4)``; ``target_base`` is ``(4, 4)``. Returns ``(T, 4, 4)``.
    """
    return target_base.unsqueeze(0) @ torch.linalg.inv(source_base) @ poses
