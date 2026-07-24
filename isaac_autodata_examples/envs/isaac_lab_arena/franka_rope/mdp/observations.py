# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for Arena Franka rope manipulation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the end-effector position in the environment frame [m]."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w.torch[:, 0, :] - env.scene.env_origins


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the end-effector orientation quaternion in the world frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w.torch[:, 0, :]


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
) -> torch.Tensor:
    """Return the two Franka finger positions using the SoftMimicGen sign convention [m]."""

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    return torch.cat((joint_pos[:, 0:1], -joint_pos[:, 1:2]), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.015,
    gripper_open_value: float = 0.04,
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Return whether the gripper is closed around a nearby rope node.

    Args:
        env: Environment containing the robot and rope.
        robot_cfg: Franka articulation and finger-joint selection.
        ee_frame_cfg: End-effector frame sensor.
        object_cfg: Deformable rope entity.
        distance_threshold: Maximum end-effector-to-node distance [m].
        gripper_open_value: Finger position when fully open [m].
        gripper_threshold: Required deviation from the open position [m].

    Returns:
        Boolean tensor with one value per environment.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    rope: DeformableObject = env.scene[object_cfg.name]

    nodal_pos_w = rope.data.nodal_pos_w.torch
    end_effector_pos_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    minimum_distance = torch.linalg.vector_norm(nodal_pos_w - end_effector_pos_w.unsqueeze(1), dim=-1).amin(dim=1)

    finger_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    gripper_closed = (torch.abs(finger_pos - gripper_open_value) > gripper_threshold).all(dim=1)
    return (minimum_distance < distance_threshold) & gripper_closed


def object_nodal_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return rope simulation-node positions in the environment frame [m]."""

    rope: DeformableObject = env.scene[object_cfg.name]
    return rope.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
