# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena composition and Gym registration for Franka rope manipulation."""

from __future__ import annotations

import gymnasium as gym

from . import embodiment as _embodiment  # noqa: F401
from .environment import FRANKA_ROPE_ARENA_ENV_ID, FrankaRopeArenaEnvironment, FrankaRopeArenaEnvironmentCfg


def register_environments(enable_cameras: bool = True) -> list[str]:
    """Compose and register the AutoData-owned Arena Franka rope environment.

    Args:
        enable_cameras: Whether to include the fixed and wrist camera sensors.

    Returns:
        An empty list of callback-specific command-line arguments.
    """

    if FRANKA_ROPE_ARENA_ENV_ID in gym.registry:
        return []

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = FrankaRopeArenaEnvironment().build(
        FrankaRopeArenaEnvironmentCfg(enable_cameras=enable_cameras),
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=1,
            env_spacing=2.5,
            seed=7,
            solve_relations=False,
            device="cuda:0",
        ),
    )
    builder.build_registered()
    return []


__all__ = [
    "FRANKA_ROPE_ARENA_ENV_ID",
    "FrankaRopeArenaEnvironment",
    "FrankaRopeArenaEnvironmentCfg",
    "register_environments",
]
