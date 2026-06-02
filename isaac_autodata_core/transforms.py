# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pose-transform helpers used to adapt source subtask segments to the current scene.

Three free functions live here so they can be unit-tested without instantiating the full
generator. They are pure tensor ops; no env or sim handles touched.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import SubTaskConstraintCoordinationScheme


def transform_source_data_segment_using_delta_object_pose(
    src_eef_poses: torch.Tensor,
    delta_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """Apply a single 4x4 delta pose to every EEF pose in a subtask segment.

    Args:
        src_eef_poses: ``[T, 4, 4]`` source EEF target poses [m, rad].
        delta_obj_pose: ``[4, 4]`` delta transform [m, rad].
    """
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=delta_obj_pose[None],
    )


def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,
    src_eef_poses: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """Re-express a subtask segment so EEF poses keep the same relation to the current object pose.

    Args:
        obj_pose: ``[4, 4]`` object pose in the current scene [m, rad].
        src_eef_poses: ``[T, 4, 4]`` source EEF target poses [m, rad].
        src_obj_pose: ``[4, 4]`` object pose at the start of the source segment [m, rad].
    """
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None]),
    )
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )


def get_delta_pose_with_scheme(
    src_obj_pose: torch.Tensor,
    cur_obj_pose: torch.Tensor,
    task_constraint: dict,
) -> torch.Tensor:
    """Compute the delta transform between two object poses under a coordination scheme.

    The scheme is read from ``task_constraint["coordination_scheme"]``. Optional uniform position /
    rotation noise is then layered on top using the constraint's noise scales.

    Args:
        src_obj_pose: ``[4, 4]`` object pose in the source scene [m, rad].
        cur_obj_pose: ``[4, 4]`` object pose in the current scene [m, rad].
        task_constraint: Runtime constraint dict; required keys: ``coordination_scheme``,
            ``coordination_scheme_pos_noise_scale``, ``coordination_scheme_rot_noise_scale``.
    """
    scheme = task_constraint["coordination_scheme"]
    device = src_obj_pose.device

    if scheme == SubTaskConstraintCoordinationScheme.TRANSFORM:
        delta_pose = PoseUtils.get_delta_object_pose(cur_obj_pose, src_obj_pose)
    elif scheme == SubTaskConstraintCoordinationScheme.TRANSLATE:
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = cur_obj_pose[:3, 3] - src_obj_pose[:3, 3]
    elif scheme == SubTaskConstraintCoordinationScheme.REPLAY:
        delta_pose = torch.eye(4, device=device)
    else:
        raise ValueError(
            f"Unsupported coordination scheme {scheme}; expected one of "
            f"{[e.value for e in SubTaskConstraintCoordinationScheme]}"
        )

    pos_noise_scale = task_constraint["coordination_scheme_pos_noise_scale"]
    rot_noise_scale = task_constraint["coordination_scheme_rot_noise_scale"]
    if pos_noise_scale != 0.0 or rot_noise_scale != 0.0:
        pos_new, rot_new = PoseUtils.add_uniform_noise_to_pose(
            delta_pose[:3, 3], delta_pose[:3, :3], pos_noise_scale, rot_noise_scale
        )
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = pos_new
        delta_pose[:3, :3] = rot_new
    return delta_pose
