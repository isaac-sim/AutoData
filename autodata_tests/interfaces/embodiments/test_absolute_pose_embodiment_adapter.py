# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for absolute-pose embodiments with independently arranged action channels."""

import torch
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodata_interfaces.embodiments.factory import embodiment_adapter_from_dict, embodiment_adapter_from_yaml
from autodata_tests.interfaces.mocks import MockAsset, MockEnv, MockScene


def _adapter():
    return embodiment_adapter_from_yaml(Path(__file__).parents[3] / "autodata_examples/embodiments/g1_wbc_pink.yaml")


def test_pose_conversion_preserves_scalar_grippers_and_body_commands():
    adapter = _adapter()
    actions = torch.arange(46, dtype=torch.float32).reshape(2, 23)
    diagonal = 2.0**-0.5
    actions[:, 5:9] = torch.tensor([0.0, 0.0, diagonal, diagonal])
    actions[:, 12:16] = torch.tensor([diagonal, 0.0, 0.0, diagonal])
    poses = adapter.action_to_target_eef_pose(actions)
    channels = adapter.actions_to_passthrough_actions(actions)
    for i in range(2):
        reconstructed = adapter.target_eef_pose_to_action(
            {name: pose[i] for name, pose in poses.items()},
            {name: command[i] for name, command in channels.items()},
        )
        assert torch.allclose(reconstructed, actions[i])
        noisy_action = adapter.target_eef_pose_to_action(
            {name: pose[i] for name, pose in poses.items()},
            {name: command[i] for name, command in channels.items()},
            action_noise_dict={name: 0.01 for name in poses},
        )
        noisy_channels = adapter.actions_to_passthrough_actions(noisy_action)
        assert all(torch.equal(noisy_channels[name], command[i]) for name, command in channels.items())


def test_controller_state_queries_select_requested_environment():
    adapter = _adapter()
    state = torch.zeros(2, 2, 13)
    state[..., 6] = 1.0
    state[1, 1, :3] = torch.tensor([3.0, 4.0, 0.8])
    robot = MockAsset(SimpleNamespace(body_names=["other", "pelvis"], body_link_state_w=state))
    scene = MockScene(assets={"robot": robot}, env_origins=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]))
    adapter.bind_env(
        MockEnv(
            scene=scene,
            obs_buf={
                "wbc": {
                    "is_navigating": torch.tensor([False, True]),
                    "navigation_goal_reached": torch.tensor([True, False]),
                }
            },
        )
    )
    frame = adapter.get_reference_frame_pose([1])
    assert frame.shape == (1, 4, 4)
    assert torch.allclose(frame[0, :3, 3], torch.tensor([2.0, 2.0, 0.8]))
    assert adapter.get_navigation_state(1) == ("body", True, False)


def test_bad_action_layout_rejected():
    adapter = _adapter()
    adapter.eefs["right"]["pose_action_slice"] = [8, 15]
    with pytest.raises(AssertionError, match="Overlapping"):
        embodiment_adapter_from_dict({
            "type": "absolute_pose",
            "name": "overlapping_layout",
            "action_dim": adapter.action_dim,
            "eefs": adapter.eefs,
        })
