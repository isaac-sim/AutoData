# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka rope environment with deformable reset settling."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


class FrankaRopeEnv(ManagerBasedRLEnv):
    """Manager-based environment that settles the rope after each reset."""

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        settling_steps = self.cfg.settling_steps
        if settling_steps <= 0:
            return

        zero_actions = torch.zeros(
            (self.num_envs, self.action_manager.total_action_dim),
            device=self.device,
        )
        for _ in range(settling_steps):
            self.action_manager.process_action(zero_actions)
            for _ in range(self.cfg.decimation):
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=True)
                self.scene.update(dt=self.physics_dt)
