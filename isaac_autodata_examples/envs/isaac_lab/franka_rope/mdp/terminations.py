# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for Franka rope manipulation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rope_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return whether the rope center is below a minimum world height [m]."""

    rope: DeformableObject = env.scene[object_cfg.name]
    return rope.data.root_pos_w.torch[:, 2] < minimum_height


def rope_ends_close_tracked(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    distance_threshold_min: float = 0.02,
    num_end_nodes: int = 3,
) -> torch.Tensor:
    """Return whether the two tracked rope ends are close enough for success.

    Endpoint nodes are discovered from the farthest node pair at the start of each
    episode and then tracked by simulation-node index.

    Args:
        env: Environment containing the rope.
        object_cfg: Deformable rope entity.
        distance_threshold: Maximum distance between endpoint neighborhoods [m].
        distance_threshold_min: Minimum separation used to reject collapsed false positives [m].
        num_end_nodes: Number of nearby nodes used for each endpoint neighborhood.

    Returns:
        Boolean tensor with one value per environment.
    """

    rope: DeformableObject = env.scene[object_cfg.name]
    nodal_positions = rope.data.nodal_pos_w.torch
    num_envs, num_nodes, _ = nodal_positions.shape
    assert num_end_nodes <= num_nodes, f"Requested {num_end_nodes} endpoint nodes from a {num_nodes}-node rope."

    if not hasattr(env, "_rope_end_indices"):
        env._rope_end_indices = {}

    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(num_envs):
        env_nodal_pos = nodal_positions[env_id]
        if env_id not in env._rope_end_indices:
            farthest_flat_index = torch.cdist(env_nodal_pos, env_nodal_pos).argmax()
            env._rope_end_indices[env_id] = (
                int((farthest_flat_index // num_nodes).item()),
                int((farthest_flat_index % num_nodes).item()),
            )

        end_1_id, end_2_id = env._rope_end_indices[env_id]
        end_1_distances = torch.linalg.vector_norm(env_nodal_pos - env_nodal_pos[end_1_id], dim=1)
        end_2_distances = torch.linalg.vector_norm(env_nodal_pos - env_nodal_pos[end_2_id], dim=1)
        end_1_node_ids = torch.topk(end_1_distances, num_end_nodes, largest=False).indices
        end_2_node_ids = torch.topk(end_2_distances, num_end_nodes, largest=False).indices

        minimum_end_distance = torch.cdist(env_nodal_pos[end_1_node_ids], env_nodal_pos[end_2_node_ids]).amin()
        result[env_id] = distance_threshold_min < minimum_end_distance < distance_threshold

    return result
