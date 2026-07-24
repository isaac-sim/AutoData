# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset and settling events for Arena Franka rope manipulation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# Mesh-resolution-specific nodes covering one half of the 549-node rope simulation mesh.
# fmt: off
_ROPE_PARTIAL_NODE_IDS = (
    222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
    282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 296, 297,
    298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
    322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 334,
    335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347,
    348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
    372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
    384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
    396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
    432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443,
    444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
    456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
    492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
    504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
    516, 517, 518, 519, 520, 528, 529, 530, 531, 532, 533, 534,
    535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
    547, 548,
)
# fmt: on


def _sample_pose_delta(
    pose_range: dict[str, tuple[float, float]],
    count: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample translation and rotation deltas from named pose ranges."""

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    ranges = torch.tensor(range_list, device=device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (count, 6), device=device)
    quaternion = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    return samples[:, :3], quaternion


def reset_rope_end_tracking(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Discard cached rope endpoint indices for reset environments."""

    if not hasattr(env, "_rope_end_indices"):
        return
    if env_ids is None:
        env._rope_end_indices.clear()
        return
    for env_id in env_ids.tolist():
        env._rope_end_indices.pop(int(env_id), None)


def reset_rope_nodal_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    partial_pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset and randomize the rope, including a transform of one rope half.

    Args:
        env: Environment containing the deformable rope.
        env_ids: Environments to reset.
        pose_range: Translation [m] and rotation [rad] ranges applied to all nodes.
        partial_pose_range: Additional translation [m] and rotation [rad] ranges applied to one rope half.
        asset_cfg: Deformable rope entity.
    """

    rope: DeformableObject = env.scene[asset_cfg.name]
    nodal_state = rope.data.default_nodal_state_w.torch[env_ids].clone()

    position, quaternion = _sample_pose_delta(pose_range, len(env_ids), rope.device)
    nodal_state[..., :3] = rope.transform_nodal_pos(nodal_state[..., :3], position, quaternion)

    assert max(_ROPE_PARTIAL_NODE_IDS) < nodal_state.shape[1], (
        f"Rope mesh has {nodal_state.shape[1]} simulation nodes, but the reset mapping expects at least "
        f"{max(_ROPE_PARTIAL_NODE_IDS) + 1}."
    )
    node_ids = torch.tensor(_ROPE_PARTIAL_NODE_IDS, device=rope.device)
    position, quaternion = _sample_pose_delta(partial_pose_range, len(env_ids), rope.device)
    nodal_state[..., node_ids, :3] = rope.transform_nodal_pos(nodal_state[..., node_ids, :3], position, quaternion)

    rope.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


def settle_rope_after_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    settling_steps: int = 10,
) -> None:
    """Advance zero-action control iterations after nodal-state reset.

    PhysX advances the full simulation when any subset of environments resets, so
    zero actions are deliberately applied to all environments.

    Args:
        env: Arena manager-based environment.
        env_ids: Reset environment indices. Used to satisfy the reset-event interface.
        settling_steps: Number of control iterations to advance.
    """

    del env_ids
    if settling_steps <= 0:
        return

    zero_actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    for _ in range(settling_steps):
        env.action_manager.process_action(zero_actions)
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)
