# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton rigid--deformable physics for the shared Franka rope task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_rope_franka_newton(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Use analytic finger colliders for stable volumetric rope contacts.

    The boxes occupy the original collision meshes' bounds. Visual meshes and
    articulation joints remain those of the shared Franka.

    Args:
        prim_path: Prim path or pattern at which to spawn the robot.
        cfg: Robot USD spawner configuration.
        translation: Translation relative to the parent [m].
        orientation: Rotation quaternion in XYZW order.
        **kwargs: Additional arguments forwarded to the USD spawner.

    Returns:
        The spawned robot prim.
    """
    from pxr import Gf, UsdGeom

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        collider = prim.GetStage().GetPrimAtPath(f"{prim.GetPath()}/{finger}/collisions/collisions")
        assert collider.IsValid() and collider.GetTypeName() == "Mesh", "Expected the legacy Franka finger collider"
        points = UsdGeom.Mesh(collider).GetPointsAttr().Get()
        lower = Gf.Vec3d(*(min(point[axis] for point in points) for axis in range(3)))
        upper = Gf.Vec3d(*(max(point[axis] for point in points) for axis in range(3)))
        collider.SetTypeName("Cube")
        cube = UsdGeom.Cube(collider)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp(opSuffix="rope_contact_center").Set((lower + upper) * 0.5)
        cube.AddScaleOp(opSuffix="rope_contact_size").Set(upper - lower)
    return prim


@clone
def spawn_rope_table_newton(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Use an analytic box for the Seattle table's existing unit-cube collider.

    Args:
        prim_path: Prim path or pattern at which to spawn the table.
        cfg: Table USD spawner configuration.
        translation: Translation relative to the parent [m].
        orientation: Rotation quaternion in XYZW order.
        **kwargs: Additional arguments forwarded to the USD spawner.

    Returns:
        The spawned table prim.
    """
    from pxr import UsdGeom

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    collider = prim.GetStage().GetPrimAtPath(f"{prim.GetPath()}/Collisions/Cube")
    assert collider.IsValid() and collider.GetTypeName() == "Mesh", "Expected the Seattle table's cube mesh collider"
    collider.SetTypeName("Cube")
    UsdGeom.Cube(collider).CreateSizeAttr(1.0)
    return prim


def rope_newton_physics_cfg() -> NewtonCfg:
    """Couple the Franka's rigid dynamics to the rope's volumetric VBD solve."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            class_type="autodata_examples.envs.isaac_lab_arena.franka_rope.newton_visuals:NewtonRopeManager",
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=20, integrator="implicitfast"),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                    include_static_shapes=True,
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(
                        iterations=10,
                        rigid_body_particle_contact_buffer_size=4096,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=0.002,
                        particle_self_contact_margin=0.003,
                    ),
                    all_particles=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_[^/]+/Robot/.*panda_hand",
                        r"/World/envs/env_[^/]+/Robot/.*panda_(left|right)finger",
                    ],
                    mode="lagged",
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=True),
                ),
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            # Resolve contacts around the narrow rope, with enough friction to carry it.
            soft_contact_ke=1.0e5,
            soft_contact_kd=10.0,
            soft_contact_mu=10.0,
        ),
        num_substeps=10,
    )
