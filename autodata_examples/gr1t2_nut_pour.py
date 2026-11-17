# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the GR1T2 nut-pouring example with an explicit physics backend."""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab_tasks.contrib.pick_place.nutpour_gr1t2_pink_ik_env_cfg import NutPourGR1T2PinkIKEnvCfg
    from pxr import Usd

ENV_NAME = "AutoData-NutPour-GR1T2-ScheduleGen-v0"


def _mimic_joints_newton(root: Usd.Prim, solver_dt: float) -> None:
    """Preserve the robot's composed mimic compliance at the solver timestep [s]."""
    from pxr import Sdf, Usd, UsdPhysics, Vt

    assert solver_dt > 0
    for prim in Usd.PrimRange(root):
        for schema in prim.GetPrimTypeInfo().GetAppliedAPISchemas():
            if not schema.startswith("PhysxMimicJointAPI:"):
                continue
            assert prim.IsA(UsdPhysics.RevoluteJoint), f"Expected an angular mimic joint: {prim.GetPath()}."
            prefix = f"physxMimicJoint:{schema.rsplit(':', 1)[1]}"
            frequency = prim.GetAttribute(f"{prefix}:naturalFrequency").Get()
            damping_ratio = prim.GetAttribute(f"{prefix}:dampingRatio").Get()
            assert frequency is not None and damping_ratio is not None, f"Missing mimic compliance: {prim.GetPath()}."
            assert frequency > 0 and damping_ratio > 0, f"Missing compliant mimic settings: {prim.GetPath()}."
            targets = prim.GetRelationship(f"{prefix}:referenceJoint").GetTargets()
            assert len(targets) == 1, f"Expected one mimic reference: {prim.GetPath()}."

            # Match the implicit spring update without triggering MuJoCo's 2*dt reference clamp.
            damping = 2 * frequency * damping_ratio + solver_dt * frequency**2
            denominator = 1 + solver_dt * damping
            reference = (2 * denominator / damping, damping / (2 * frequency * math.sqrt(denominator)))
            impedance = 1 - 1 / denominator
            prim.AddAppliedSchema("MjcEqualityJointAPI")
            prim.AddAppliedSchema("NewtonMimicAPI")
            prim.CreateRelationship("newton:mimicJoint").SetTargets(targets)
            prim.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(
                -prim.GetAttribute(f"{prefix}:offset").Get()
            )
            prim.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(
                -prim.GetAttribute(f"{prefix}:gearing").Get()
            )
            prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(reference))
            prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set(
                Vt.DoubleArray((impedance, impedance, 0.001, 0.5, 2.0))
            )


def _robot_spawn_newton(prim_path: str, cfg: UsdFileCfg, **kwargs: Any) -> Usd.Prim:
    """Spawn the robot with native Newton joint-equality settings."""
    from isaaclab.sim import SimulationContext, clone, spawn_from_usd

    @clone
    def spawn(prim_path, cfg, **kwargs):
        """Spawn the USD and configure mimic joints at the Newton solver timestep."""
        root = spawn_from_usd(prim_path, cfg, **kwargs)
        sim = SimulationContext.instance()
        _mimic_joints_newton(root, sim.get_physics_dt() / sim.cfg.physics.num_substeps)
        return root

    return spawn(prim_path, cfg, **kwargs)


def _fixture_spawn_newton(prim_path: str, cfg: UsdFileCfg, **kwargs: Any) -> Usd.Prim:
    """Preserve fixture visuals while using solid collision hulls and thin planar proxies."""
    import numpy as np

    from isaaclab.sim import clone, spawn_from_usd
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    @clone
    def spawn(prim_path, cfg, **kwargs):
        """Spawn fixture collision proxies while retaining the original visual meshes."""
        root = spawn_from_usd(prim_path, cfg, **kwargs)
        meshes = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Mesh)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ]
        assert meshes, f"The fixture {prim_path} must contain collision meshes."
        xforms = UsdGeom.XformCache()
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(root.GetStage())
        planar_count = 0
        for prim in meshes:
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            assert points is not None and len(points) >= 3, f"Missing fixture mesh points {prim.GetPath()}."
            transform = xforms.GetLocalToWorldTransform(prim)
            assert transform.GetDeterminant() != 0, f"Singular fixture transform {prim.GetPath()}."
            vertices = np.asarray([transform.Transform(Gf.Vec3d(point)) for point in points], dtype=np.float64)
            vertices *= meters_per_unit
            axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)[2]
            axes[-1] *= np.sign(np.linalg.det(axes))
            projected = vertices @ axes.T
            lower, upper = projected.min(axis=0), projected.max(axis=0)
            size = upper - lower
            if size.min() < max(2e-6 * np.linalg.norm(size), 1e-8):
                # MuJoCo requires volume; retain thin fixture surfaces with 0.5 mm box colliders.
                box = UsdGeom.Cube.Define(root.GetStage(), prim.GetPath().AppendChild("collision_box"))
                box.CreateSizeAttr(1.0)
                box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                dimensions = np.maximum(size, 0.0005) / meters_per_unit
                box_world = np.eye(4)
                box_world[:3, :3] = axes * dimensions[:, None]
                box_world[3, :3] = ((lower + upper) * 0.5) @ axes / meters_per_unit
                # USD composes row-vector transforms; preserve the exact source mesh frame.
                box_local = Gf.Matrix4d(*box_world.ravel().tolist()) * transform.GetInverse()
                # Keep physical extents explicit for the planner's USD primitive reader.
                scale = Gf.Matrix4d(1.0)
                scale.SetScale(Gf.Vec3d(*dimensions))
                box_xform = UsdGeom.Xformable(box)
                box_xform.AddTransformOp().Set(scale.GetInverse() * box_local)
                box_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*dimensions))
                assert np.allclose(
                    np.asarray(xforms.GetLocalToWorldTransform(box.GetPrim())), box_world, atol=1e-8, rtol=1e-8
                ), f"Fixture proxy transform mismatch {prim.GetPath()}."
                UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                planar_count += 1
                continue
            applied = apply_mesh_collision_properties(
                str(prim.GetPath()),
                [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")],
                stage=root.GetStage(),
            )
            assert applied, f"Could not configure fixture collider {prim.GetPath()}."
        print(f"[INFO]: Newton fixture {prim_path} uses {len(meshes)} solid colliders ({planar_count} thin boxes).")
        return root

    return spawn(prim_path, cfg, **kwargs)


def _configure_robot_newton(robot: ArticulationCfg) -> None:
    """Configure the GR1T2 hands and mimic joints for Newton.

    Args:
        robot: Robot configuration to update before spawning.
    """
    robot.spawn.func = _robot_spawn_newton
    robot.spawn.make_uninstanceable = True
    # Compliant drives avoid fighting the finger contact and mimic constraints.
    driven = "[LR]_((index|middle|pinky|ring)_proximal|thumb_proximal_(yaw|pitch))_joint"
    passive = "[LR]_((index|middle|pinky|ring)_intermediate|thumb_distal)_joint"
    for name in ("left-hand", "right-hand"):
        hand = robot.actuators[name]
        hand.armature = 2e-4
        hand.stiffness = {driven: 174.53292, passive: 0.0}
        hand.damping = {driven: 5.5848, passive: 0.0}


def _configure_newton(config: NutPourGR1T2PinkIKEnvCfg) -> None:
    """Apply Newton solver, robot and fixture settings without changing control timing.

    Args:
        config: Nut-pouring environment configuration with the Newton preset selected.
    """
    # Resolve hand contacts at 0.1 ms without changing the control period.
    config.sim.physics.num_substeps = 100
    # The humanoid scene exceeds Arena's small-hand contact budget.
    config.sim.physics.solver_cfg.nconmax = 2048
    config.sim.physics.solver_cfg.njmax = 8192
    config.sim.physics.solver_cfg.use_mujoco_contacts = True
    config.sim.physics.solver_cfg.ccd_iterations = 35
    # Resolve the small finger/object contacts at a 2 ms response time.
    config.sim.physics.default_shape_cfg.ke = 2.5e5
    config.sim.physics.default_shape_cfg.kd = 1000.0
    _configure_robot_newton(config.scene.robot)
    for fixture in (config.scene.table, config.scene.sorting_scale):
        fixture.spawn.func = _fixture_spawn_newton
        fixture.spawn.make_uninstanceable = True


def _environment_config(physics_backend: str, env_profile: str | None = None) -> NutPourGR1T2PinkIKEnvCfg:
    """Create the existing nut-pouring task with the selected Arena physics preset.

    Args:
        physics_backend: Physics preset, either ``physx`` or ``newton``.
        env_profile: Optional AutoData environment profile, including when recording with Isaac Lab tools.

    Returns:
        Task configuration with backend-specific settings applied.
    """
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg
    from isaaclab_tasks.contrib.pick_place.nutpour_gr1t2_pink_ik_env_cfg import NutPourGR1T2PinkIKEnvCfg

    from autodata_interfaces.env import EnvironmentProfile, apply_env_profile
    from autodata_interfaces.env.pink_ik_action import PinkInverseKinematicsActionSharedUrdf

    config = NutPourGR1T2PinkIKEnvCfg()
    # The table is a fixed fixture, not a dynamic object included in episode resets.
    config.scene.table.spawn.rigid_props.kinematic_enabled = True
    config.actions.gr1_action.class_type = PinkInverseKinematicsActionSharedUrdf
    config.sim.physics = getattr(ArenaPhysicsCfg(), physics_backend)
    if physics_backend == "newton":
        _configure_newton(config)
    if env_profile is not None:
        apply_env_profile(config, EnvironmentProfile.from_yaml(env_profile), ENV_NAME)
    return config


def register() -> list[str]:
    """Register the example through AutoData's external-environment callback.

    The callback runs after simulator startup. ``--physics_backend`` selects Arena's PhysX or
    Newton preset; the task's existing control period and source annotations remain applicable.

    Returns:
        Arguments not consumed by this callback.
    """
    import gymnasium as gym

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--physics_backend", choices=("physx", "newton"), default="physx")
    parser.add_argument("--env_profile", default=None)
    args, remaining = parser.parse_known_args()
    assert args.task == ENV_NAME, f"This callback registers only {ENV_NAME}."

    def environment_config() -> NutPourGR1T2PinkIKEnvCfg:
        """Build the selected config through Isaac Lab's function-based registry."""
        return _environment_config(args.physics_backend, args.env_profile)

    gym.register(
        id=ENV_NAME,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": environment_config},
        disable_env_checker=True,
    )
    return remaining
