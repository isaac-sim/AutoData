# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym registration for the Franka rope manipulation environment."""

import gymnasium as gym

FRANKA_ROPE_ENV_ID = "Isaac-Rope-Franka-IK-Rel-v0"

if FRANKA_ROPE_ENV_ID not in gym.registry:
    gym.register(
        id=FRANKA_ROPE_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                "isaac_autodata_examples.envs.isaac_lab.franka_rope.franka_rope_env_cfg:FrankaRopeEnvCfg"
            ),
        },
        disable_env_checker=True,
    )
