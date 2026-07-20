# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch

from isaac_autodata_interfaces.embodiments.embodiment_types import PoseObsKeys
from isaac_autodata_interfaces.embodiments.single_arm_embodiment_adapter import DeltaPoseIKSingleArmAdapter


class _ActionManager:
    active_terms = ("arm_action",)

    def __init__(self, scale: float) -> None:
        term_class = type("DifferentialInverseKinematicsAction", (), {})
        self.term = term_class()
        self.term._scale = torch.full((1, 6), scale)

    def get_term(self, name: str):
        assert name == "arm_action"
        return self.term


class _Env:
    def __init__(self, scale: float) -> None:
        self.action_manager = _ActionManager(scale)
        self.obs_buf = {
            "policy": {
                "eef_pos": torch.zeros((1, 3), dtype=torch.float32),
                "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            }
        }


def _adapter(scale: float) -> DeltaPoseIKSingleArmAdapter:
    adapter = DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
    )
    adapter.bind_env(_Env(scale))
    return adapter


def test_action_to_pose_applies_live_ik_action_scale():
    adapter = _adapter(0.5)
    action = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    target = adapter.action_to_target_eef_pose(action)["franka"]

    assert torch.allclose(target[0, :3, 3], torch.tensor([0.1, 0.0, 0.0]))


def test_pose_to_action_divides_by_live_ik_action_scale():
    adapter = _adapter(0.5)
    target = torch.eye(4)
    target[0, 3] = 0.1

    action = adapter.target_eef_pose_to_action(
        target_eef_pose_dict={"franka": target},
        gripper_action_dict={"franka": torch.tensor([1.0])},
        env_id=0,
    )

    assert torch.allclose(action, torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_missing_action_manager_preserves_legacy_unit_scale():
    adapter = _adapter(1.0)
    adapter.env.action_manager = None
    target = torch.eye(4)
    target[0, 3] = 0.1

    action = adapter.target_eef_pose_to_action(
        target_eef_pose_dict={"franka": target},
        gripper_action_dict={"franka": torch.tensor([1.0])},
        env_id=0,
    )

    assert torch.allclose(action[:3], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)


def test_gripper_action_extraction_preserves_empty_trailing_dimension():
    adapter = DeltaPoseIKSingleArmAdapter(
        name="arm_without_gripper",
        eef_name="hand",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=0,
    )
    actions = torch.zeros((2, 3, 6))

    gripper_actions = adapter.actions_to_gripper_actions(actions)["hand"]

    assert gripper_actions.shape == (2, 3, 0)


def test_eef_observation_quaternion_uses_xyzw_scalar_last_order():
    adapter = _adapter(0.5)
    half_angle = math.pi / 4.0
    adapter.env.obs_buf["policy"]["eef_quat"][:] = torch.tensor(
        [[math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)]]
    )

    pose = adapter.get_eef_poses()["franka"][0]

    expected_rotation = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    assert torch.allclose(pose[:3, :3], expected_rotation, atol=1e-6)


def test_eef_offset_is_applied_along_the_rotated_observation_local_axis():
    adapter = DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
        eef_offset=(0.0, 0.0, -0.0036),
    )
    env = _Env(0.5)
    half_angle = math.pi / 4.0
    env.obs_buf["policy"]["eef_pos"][:] = torch.tensor([[1.0, 2.0, 3.0]])
    # FrameTransformer target_quat_w is XYZW; ``_w`` means world-frame coordinates.
    env.obs_buf["policy"]["eef_quat"][:] = torch.tensor([[0.0, math.sin(half_angle), 0.0, math.cos(half_angle)]])
    adapter.bind_env(env)

    pose = adapter.get_eef_poses()["franka"][0]

    assert torch.allclose(pose[:3, 3], torch.tensor([1.0036, 2.0, 3.0]), atol=1e-6)
