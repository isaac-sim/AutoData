# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the environment-delegating mimic embodiment adapter."""

import torch

from isaac_autodata_interfaces.embodiments import EnvironmentEmbodimentAdapter, embodiment_adapter_from_dict


class _MimicEnv:
    def __init__(self) -> None:
        self.pose = torch.eye(4).repeat(2, 1, 1)
        self.objects = {"pick_object": self.pose.clone()}

    def get_robot_eef_pose(self, eef_name, env_ids=None):
        del eef_name
        return self.pose if env_ids is None else self.pose[env_ids]

    def action_to_target_eef_pose(self, action):
        pose = self.pose[: action.shape[0]]
        return {"left": pose, "right": pose, "body": torch.zeros_like(pose)}

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        del target_eef_pose_dict, action_noise_dict, env_id
        return torch.cat((gripper_action_dict["left"], gripper_action_dict["right"], gripper_action_dict["body"]))

    def actions_to_gripper_actions(self, actions):
        return {"left": actions[..., 0], "right": actions[..., 1], "body": actions[..., 2:]}

    def get_object_poses(self, env_ids=None):
        if env_ids is None:
            return self.objects
        return {name: pose[env_ids] for name, pose in self.objects.items()}


def _adapter() -> EnvironmentEmbodimentAdapter:
    adapter = embodiment_adapter_from_dict({
        "type": "environment_mimic",
        "name": "arena_g1_pink",
        "eef_names": ["left", "right"],
        "robot_asset_name": "robot",
    })
    assert isinstance(adapter, EnvironmentEmbodimentAdapter)
    adapter.bind_env(_MimicEnv())
    return adapter


def test_environment_adapter_delegates_pose_action_and_passthrough_methods():
    adapter = _adapter()
    assert adapter.get_eef_names() == ("left", "right")
    assert set(adapter.get_eef_poses()) == {"left", "right"}

    actions = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    passthrough = adapter.actions_to_passthrough_actions(actions)
    assert set(passthrough) == {"left", "right", "body"}
    assert passthrough["body"].shape == (2, 2, 1)

    target_poses = adapter.action_to_target_eef_pose(torch.zeros(1, 3))
    action = adapter.target_eef_pose_to_action(
        target_poses,
        {"left": torch.tensor([0.0]), "right": torch.tensor([1.0]), "body": torch.tensor([2.0])},
    )
    assert torch.equal(action, torch.tensor([0.0, 1.0, 2.0]))


def test_environment_adapter_owns_controller_frame_object_poses():
    adapter = _adapter()
    poses = adapter.get_object_poses(env_ids=[1])
    assert set(poses) == {"pick_object"}
    assert poses["pick_object"].shape == (1, 4, 4)
