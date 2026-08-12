# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-state reads with correct per-environment coordinate conversion."""

from __future__ import annotations

from typing import Any


def get_scene_state(scene: Any, is_relative: bool = True) -> dict:
    """Return scene state in world or per-environment coordinates.

    Isaac Lab's relative-state conversion indexes deformable nodal positions as if their second
    dimension were XYZ. Read world state first and apply the environment origin along the actual
    ``(environment, node, xyz)`` layout here.

    Args:
        scene: Isaac Lab interactive scene.
        is_relative: Whether positions are relative to each environment origin.

    Returns:
        Nested scene-state mapping containing cloned tensors.
    """

    state = scene.get_state(is_relative=False)
    if not is_relative:
        return state

    env_origins = scene.env_origins
    for asset_state in state.get("articulation", {}).values():
        if "root_pose" in asset_state:
            asset_state["root_pose"][:, :3] -= env_origins
    for asset_state in state.get("deformable_object", {}).values():
        if "nodal_position" in asset_state:
            asset_state["nodal_position"] -= env_origins.unsqueeze(1)
    for asset_state in state.get("rigid_object", {}).values():
        if "root_pose" in asset_state:
            asset_state["root_pose"][:, :3] -= env_origins
    return state
