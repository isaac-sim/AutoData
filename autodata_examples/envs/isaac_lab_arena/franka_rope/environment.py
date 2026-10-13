# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab.devices import DevicesCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose

from .rope_asset import FrankaRopeAsset, rope_newton_spawn_cfg
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
    """Apply rope timing and asset/solver settings for Arena's selected backend."""

    from isaaclab_newton.physics import NewtonCfg

    env_cfg.decimation = 5
    env_cfg.sim.dt = 0.01
    env_cfg.sim.render_interval = 2
    env_cfg.num_rerenders_on_reset = 1
    if isinstance(env_cfg.sim.physics, NewtonCfg):
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg

        from .newton_cfg import rope_newton_physics_cfg, spawn_rope_table_newton

        env_cfg.sim.physics = rope_newton_physics_cfg()
        # Refresh Kit once per control step; the solver still advances at 1 ms.
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.scene.object.spawn = rope_newton_spawn_cfg()
        # A discontinuous half-mesh reset can invert the connecting tetrahedra.
        # Blend the same sampled poses through the middle half of the rope.
        env_cfg.events.reset_object_position.params["transition_length"] = 0.4
        env_cfg.scene.replicate_physics = True
        env_cfg.scene.table.spawn.make_uninstanceable = True
        env_cfg.scene.table.spawn.func = spawn_rope_table_newton
        # A firm, critically damped table contact in MJWarp. Priority keeps the
        # table response from being softened by the gripper's default material.
        # These MuJoCo parameters do not change VBD's rope contact properties.
        env_cfg.scene.table.spawn.collision_props = MujocoCollisionCfg(priority=1, solref=(0.004, 1.0))
    else:
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


@register_environment
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
