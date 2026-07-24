# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed Arena composition for the AutoData Franka rope environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab.devices import DevicesCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose

from .rope_asset import FrankaRopeAsset
from .task import FrankaRopeTask

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

FRANKA_ROPE_ARENA_ENV_ID = "Isaac-Rope-Franka-Arena-IK-Rel-v0"


@dataclass
class FrankaRopeArenaEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the AutoData-owned Arena Franka rope environment."""

    enable_cameras: bool = False
    embodiment: str = "franka_rope_ik"
    teleop_device: str | None = None


def _configure_franka_rope_env(env_cfg: Any) -> Any:
    """Apply rope-specific simulation timing and legacy teleop devices."""

    env_cfg.decimation = 5
    env_cfg.sim.dt = 0.01
    env_cfg.sim.render_interval = 2
    env_cfg.sim.render.antialiasing_mode = "DLSS"
    env_cfg.num_rerenders_on_reset = 1
    env_cfg.scene.replicate_physics = False
    env_cfg.teleop_devices = DevicesCfg(
        devices={
            "keyboard": Se3KeyboardCfg(
                pos_sensitivity=0.05,
                rot_sensitivity=0.2,
            ),
            "spacemouse": Se3SpaceMouseCfg(
                pos_sensitivity=0.2,
                rot_sensitivity=0.5,
            ),
        },
    )
    return env_cfg


class FrankaRopeArenaEnvironment(ArenaEnvironmentFactory[FrankaRopeArenaEnvironmentCfg]):
    """Compose the rope scene, Franka embodiment, and rope task through Arena."""

    name = FRANKA_ROPE_ARENA_ENV_ID
    _legacy_argparse_cfg_type = FrankaRopeArenaEnvironmentCfg

    def build(self, cfg: FrankaRopeArenaEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build an Arena environment description from ``cfg``."""

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        table = self.asset_registry.get_asset_by_name("table")()
        table.set_initial_pose(
            Pose(
                position_xyz=(0.5, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.707, 0.707),
            ),
        )
        rope = FrankaRopeAsset()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(3000.0)
        light.set_color((0.75, 0.75, 0.75))

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )

        scene = Scene(
            assets=[
                table,
                rope,
                ground_plane,
                light,
            ],
        )
        task = FrankaRopeTask()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_franka_rope_env,
        )
