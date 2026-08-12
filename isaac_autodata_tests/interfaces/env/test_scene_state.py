# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel env-relative scene-state capture."""

from __future__ import annotations

import copy
import torch

from isaac_autodata_interfaces.env.recorders import (
    InitialStateRecorder,
    PostStepStatesRecorder,
    make_action_state_recorder_manager_cfg,
)
from isaac_autodata_interfaces.env.scene_state import get_scene_state


class _MockScene:
    def __init__(self) -> None:
        self.env_origins = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        self.state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.tensor(
                        [[1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0], [10.5, 20.5, 30.5, 0.0, 0.0, 0.0, 1.0]]
                    )
                }
            },
            "deformable_object": {
                "rope": {
                    "nodal_position": torch.tensor([
                        [[1.1, 2.2, 3.3], [1.4, 2.5, 3.6]],
                        [[10.1, 20.2, 30.3], [10.4, 20.5, 30.6]],
                    ])
                }
            },
            "rigid_object": {
                "cube": {
                    "root_pose": torch.tensor(
                        [[1.2, 2.3, 3.4, 0.0, 0.0, 0.0, 1.0], [10.2, 20.3, 30.4, 0.0, 0.0, 0.0, 1.0]]
                    )
                }
            },
        }

    def get_state(self, is_relative: bool = False) -> dict:
        assert is_relative is False
        return copy.deepcopy(self.state)


def test_scene_state_subtracts_origin_from_every_deformable_node():
    state = get_scene_state(_MockScene(), is_relative=True)
    assert torch.allclose(
        state["deformable_object"]["rope"]["nodal_position"],
        torch.tensor([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ]),
    )
    assert torch.allclose(
        state["articulation"]["robot"]["root_pose"][:, :3],
        torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
    )
    assert torch.allclose(
        state["rigid_object"]["cube"]["root_pose"][:, :3],
        torch.tensor([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]]),
    )


def test_scene_state_world_coordinates_are_unchanged():
    scene = _MockScene()
    state = get_scene_state(scene, is_relative=False)
    assert torch.equal(
        state["articulation"]["robot"]["root_pose"],
        scene.state["articulation"]["robot"]["root_pose"],
    )
    assert torch.equal(
        state["deformable_object"]["rope"]["nodal_position"],
        scene.state["deformable_object"]["rope"]["nodal_position"],
    )


def test_action_state_recorder_uses_autodata_state_terms():
    cfg = make_action_state_recorder_manager_cfg()
    assert cfg.record_initial_state.class_type is InitialStateRecorder
    assert cfg.record_post_step_states.class_type is PostStepStatesRecorder
