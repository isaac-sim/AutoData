# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ScheduleStream TAMP planner grounded in an Isaac AutoData env.

Subclasses cuStream2's :class:`Planner` (``custream2.policy``) in the style of
``robolab/policy.py``: the observation hooks read the live IsaacLab scene, ``extract_action``
emits Isaac AutoData :class:`Waypoint`\\ s, and the base class supplies world/state sync,
planning (timeout, animation, error tally), and the open-loop command rollout.

Import only after the Isaac app is launched (the isaaclab imports require the running app);
``schedulestream_algorithm`` imports this lazily when a run selects ``--alg schedulestream``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from curobo._src.util.usd_scene_parser import UsdSceneParser
from curobo.types import JointState, Pose
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.sim import find_matching_prims
from isaaclab.utils.math import axis_angle_from_quat, convert_quat, quat_from_matrix
from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked

from schedulestream.applications.custream2.animate import animate_commands
from schedulestream.applications.custream2.command import Commands
from schedulestream.applications.custream2.example import Attached
from schedulestream.applications.custream2.franka import load_franka_config
from schedulestream.applications.custream2.object import GraspConfig, MeshObject
from schedulestream.applications.custream2.policy import Planner
from schedulestream.applications.custream2.scene import CAMERA_POSE
from schedulestream.applications.custream2.utils import (
    autograd_enabled,
    multiply_poses,
    position_from_pose,
    to_cpu,
    to_pose,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import apply_mapping, profiler

from isaac_autodata_core.waypoint import Waypoint

if TYPE_CHECKING:
    from isaac_autodata_interfaces.datastream.datastream import Datastream

# Binary gripper action for the trailing action dim, inferred from world attachments (closed
# while the arm holds an object). Flip OPEN_ACTION if the env's convention is inverted.
OPEN_ACTION = 1.0
CLOSE_ACTION = -OPEN_ACTION


def _reorder_positions(src_names: list[str], src_positions: Any, dst_names: list[str]) -> Any:
    """Reorder ``src_positions`` (indexed by ``src_names``) into ``dst_names`` order."""
    mapping = dict(zip(src_names, to_cpu(src_positions)))
    return np.asarray(apply_mapping(mapping, dst_names), dtype=np.float32)


class ScheduleStreamPlanner(Planner):
    """cuStream2 :class:`Planner` grounded in the Isaac AutoData env for one ``env_id``."""

    def __init__(
        self,
        datastream: Datastream,
        success_term: Any,
        *,
        env_id: int = 0,
        batch_size: int = 128,
        collisions: bool = True,
        max_time: float = 60.0,
        profile: bool = False,
        hold: int | None = None,
        animate: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.datastream = datastream
        self.env = datastream.get_env()
        self.env_id = env_id
        self.batch_size = batch_size
        self.profile = profile
        self.hold = hold
        self.verbose = verbose
        world = self._create_world()
        # collisions/profile and any extra kwargs flow through **solve_kwargs into solve_tamp.
        super().__init__(world, max_time=max_time, animate=animate, collisions=collisions, profile=profile, **kwargs)
        self.set_goal(self.create_goal(success_term))
        # The IK action's control link + body offset are invariant, so resolve them once.
        self.body_name, self.body_offset = self._eef_action_info()
        # Calibrated per get_waypoints() call, at the synced start configuration.
        self._correction: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Env / scene accessors
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> Any:
        if isinstance(self.env, ManagerBasedEnv):
            return self.env
        return self.env.env

    @property
    def scene(self) -> Any:
        return self.base_env.scene

    @property
    def robot(self) -> str:
        [robot] = self.scene.articulations
        return robot

    @property
    def articulation(self) -> Any:
        return self.scene.articulations[self.robot]

    # ------------------------------------------------------------------
    # Observation hooks (Planner interface)
    # ------------------------------------------------------------------

    def _to_pose(self, root_pose: Any) -> Pose:
        """Convert a batched IsaacLab ``root_pose`` (``[pos(3), quat]``) row to a cuRobo Pose.

        IsaacLab's warp ``root_pose_w`` quaternion is xyzw; cuRobo wants wxyz.
        """
        root_pose = self.world.to_device(root_pose[self.env_id : self.env_id + 1])
        root_pose[..., 3:7] = convert_quat(root_pose[..., 3:7], to="wxyz")
        return Pose(position=root_pose[:, :3], quaternion=root_pose[:, 3:])

    def get_env_robot_pose(self) -> Pose:
        """Return the robot's world root pose."""
        return self._to_pose(self.scene.state["articulation"][self.robot]["root_pose"])

    def get_env_object_pose(self, obj: str) -> Pose:
        """Return the named object's world pose."""
        for body_type, bodies in self.scene.state.items():
            if (body_type != "articulation") and (obj in bodies):
                return self._to_pose(bodies[obj]["root_pose"])
        raise KeyError(f"Object {obj!r} not in the env scene state")

    def get_env_joint_state(self) -> JointState:
        """Return the robot's current joint state in world joint order."""
        positions = self.scene.state["articulation"][self.robot]["joint_position"][self.env_id]
        positions = _reorder_positions(list(self.articulation.joint_names), positions, self.world.all_joints)
        positions = torch.tensor(positions, dtype=torch.float32, device=self.world.device)
        return JointState.from_position(positions.unsqueeze(0), joint_names=self.world.all_joints)

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _convert_objects(self, scene_cfg: Any) -> list[Any]:
        name_from_path: dict[str, str] = {}
        for name, rigid_object in self.scene.rigid_objects.items():
            for prim in find_matching_prims(rigid_object.cfg.prim_path):
                name_from_path[prim.GetPath().pathString] = name

        objects = []
        for i, obstacle in enumerate(scene_cfg.objects):
            path = obstacle.name
            name = path
            for _path, _name in name_from_path.items():
                if path.startswith(_path):
                    name = _name
                    break

            floating = True
            if name in self.scene.rigid_objects:
                rigid_object = self.scene.rigid_objects[name]
                if (rigid_object.cfg.spawn is not None) and (rigid_object.cfg.spawn.rigid_props is not None):
                    floating = not rigid_object.cfg.spawn.rigid_props.kinematic_enabled

            mesh = obstacle.get_trimesh_mesh()
            pose = to_pose(obstacle.pose)
            # Floating objects (cubes) get a top-down grasp config (custream2/scene.py idiom);
            # static objects (e.g. the table) are collision-only.
            grasp_config = GraspConfig(roll_interval="top") if floating else None
            objects.append(MeshObject(name, mesh, pose=pose, grasp_config=grasp_config, surface_config=None))
            if self.verbose:
                print(
                    f"{i}/{len(scene_cfg.objects)}) Name: {name} | Path: {path} | Floating: {floating} "
                    f"| Position: {np.round(position_from_pose(pose), 2)}"
                )
        return objects

    def _create_objects(self) -> list[Any]:
        usd_parser = UsdSceneParser()
        usd_parser.load_stage(self.scene.stage)

        env_path = self.scene.env_regex_ns.replace(".*", f"{self.env_id}")
        robot_path = f"{env_path}/Robot"
        ignore_list = [
            f"{env_path}/Robot",
            f"{env_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        scene_cfg = usd_parser.get_obstacles_from_stage(
            only_paths=[env_path],
            reference_prim_path=robot_path,
            ignore_substring=ignore_list,
            timecode=0,
        )
        obstacle_names = [obj.name for obj in scene_cfg.objects]
        assert obstacle_names, f"no obstacles parsed under {env_path}"
        if self.verbose:
            print(f"Obstacles ({len(obstacle_names)}): {obstacle_names}")
        return self._convert_objects(scene_cfg)

    def _create_world(self) -> World:
        objects = self._create_objects()

        usd_name = os.path.basename(self.articulation.cfg.spawn.usd_path)

        # Franka-stack scope: only the standard panda is supported.
        if usd_name != "panda_instanceable.usd":
            raise NotImplementedError(
                f"schedulestream currently supports only the Franka panda, got robot USD {usd_name!r}"
            )
        robot_config = load_franka_config(base_poses=None)

        # The World and ALL cuRobo operations must run with inference mode OFF. The Isaac AutoData
        # env_loop runs the whole generation under torch.inference_mode(); if the World is built in
        # that context, cuRobo allocates "inference tensors" (e.g. kinematics link_spheres) that can
        # neither be backward()'d (IK) nor updated in-place outside inference mode (grasp sphere
        # context) -- producing "Inplace update to inference tensor outside InferenceMode" errors.
        # Building everything under autograd_enabled() makes them normal tensors throughout.
        #
        # World creation is profiled separately from solving (see plan()): cProfile when
        # self.profile is set, else a no-op.
        if self.profile:
            print(f"{'=' * 30} PROFILE: world creation {'=' * 30}")
        with profiler(field="cumtime" if self.profile else None, num=25), autograd_enabled():
            # ik_batch: the IK stream batch size (custream2's -b/--batch; its example defaults to 128).
            world = World(robot_config, objects, ik_batch=self.batch_size)

            state_dict = self.scene.state
            positions = state_dict["articulation"][self.robot]["joint_position"][self.env_id]
            # Reorder env joint positions into the world's joint ordering by name.
            env_joint_names = list(self.articulation.joint_names)
            ordered = _reorder_positions(env_joint_names, positions, world.all_joints)
            world.set_joint_positions(world.all_joints, ordered)
            world.set_camera_pose(CAMERA_POSE)

            # Warm up the IK / motion-planning solvers, which CAPTURES cuRobo's CUDA graphs. Without
            # this, the first solve_tamp replays a never-captured graph and crashes in seed_ik_solver
            # ('NoneType' has no attribute 'replay'). custream2's example.py and the upstream isaaclab
            # planner both warm up before solving (the latter via its now-removed v1 initialize()).
            world.warmup()

        # Diagnostic (outside autograd_enabled — just toggles collision-active state, no backward):
        # every movable object must be registered in the cuRobo scene_collision_checker, else grasp
        # sampling (grasp.py active_context -> set_object_active) raises KeyError. Toggling active
        # forces the strict container walk (is_object_active defaults True, so a no-op set wouldn't
        # probe membership). Restores state and never raises.
        missing = []
        for name in world.movable_names:
            try:
                world.set_object_active(name, False)
                world.set_object_active(name, True)
            except KeyError:
                missing.append(name)
        if missing:
            print(
                f"[schedulestream] WARNING: movable objects missing from scene_collision_checker: "
                f"{missing} | movable_names={world.movable_names} | fixed_names={world.fixed_names}"
            )
        return world

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def create_goal(self, success_term: Any) -> Any:
        """Derive the symbolic ScheduleStream goal from the task success term (cube stacking)."""
        if success_term is None:
            # Fall back to stacking the two highest-indexed movable objects.
            self.world.dump()
            obj1, obj2 = sorted(self.world.movable_names, reverse=True)[:2]
            return Attached(obj1) == obj2

        if success_term.func == cubes_stacked:
            cubes: list[str | None] = [f"cube_{i}" for i in range(1, 3 + 1)]
            for i, cube in enumerate(cubes):
                cube_cfg = f"{cube}_cfg"
                if cube_cfg not in success_term.params:
                    continue
                if success_term.params[cube_cfg] is None:
                    cubes[i] = None
                else:
                    cubes[i] = success_term.params[cube_cfg].name
            cube1, cube2, cube3 = cubes
            goal = Attached(cube2) == cube1
            if cube3 is not None:
                goal = goal & (Attached(cube3) == cube2)
            return goal
        raise NotImplementedError(f"schedulestream goal not implemented for {success_term.func}")

    # ------------------------------------------------------------------
    # EEF frame calibration + action extraction
    # ------------------------------------------------------------------

    def _eef_action_info(self) -> tuple[str, Any]:
        """Return ``(ik_body_name, body_offset_pose_or_None)`` for the EEF's IK action term."""
        action_manager = getattr(self.base_env, "action_manager", None)
        assert action_manager is not None, "env has no action_manager"
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            if isinstance(term, DifferentialInverseKinematicsAction):
                offset = None
                if term.cfg.body_offset is not None:
                    offset = Pose(
                        position=self.world.to_device(list(term.cfg.body_offset.pos)),
                        quaternion=self.world.to_device(list(term.cfg.body_offset.rot)),
                    )
                return term.cfg.body_name, offset
        raise NotImplementedError("schedulestream requires a DifferentialInverseKinematicsAction (IK env)")

    def _raw_eef_pose(self) -> torch.Tensor:
        """World-frame EEF target as cuRobo sees it: ``robot_root * node_pose(body) * body_offset``.

        This is in cuRobo's ``panda_hand`` frame convention; :meth:`_eef_frame_correction` maps it
        into the env's ee_frame convention.
        """
        link_pose = self.world.get_node_pose(self.body_name)
        if self.body_offset is not None:
            link_pose = link_pose.multiply(self.body_offset)
        link_pose = multiply_poses(self.get_env_robot_pose(), link_pose)
        return link_pose.get_matrix().squeeze(0).to(device=self.datastream.device, dtype=torch.float32)

    def _eef_frame_correction(self) -> torch.Tensor:
        """Constant body-frame transform mapping cuRobo's EEF frame to the env's ee_frame.

        Measured once with the world at the current config: ``inv(raw_eef) @ obs_eef``.

        Root cause (IsaacLab-version asset difference, NOT a quaternion-convention bug here):
        cuStream2 builds the World from its own Franka URDF (``custream2.franka.load_franka_config``),
        whose ``panda_hand`` link frame is defined ~180 deg about z relative to the ``panda_hand``
        frame in the Franka USD shipped with *this* IsaacLab version. The env's ee_frame observation
        (FrameTransformer on ``panda_hand`` + pos [0,0,0.1034], identity rot) therefore differs from
        cuRobo's ``panda_hand``-derived target by that constant 180-deg-z link-frame redefinition
        (plus the 0.107-vs-0.1034 z offset). The standalone ``applications/isaaclab`` agent does not
        need this correction only because it runs against an IsaacLab whose Franka USD ``panda_hand``
        happens to match custream2's URDF.

        Because the mismatch is a constant BODY-frame (link-local) transform, ``inv(raw0) @ obs0``
        measured at one config reproduces the true EEF pose at EVERY config (obs(q) = raw(q) @ C), so
        this is correct for full trajectories, not just a static hold. (The base ``reference_pose``
        is computed correctly: ``_to_pose`` converts the xyzw warp ``root_pose_w`` to wxyz for cuRobo,
        so there is no world-frame error that a body-frame correction would fail to cancel.) The
        ``[schedulestream] EEF frame check`` print validates the result reads ~0 after correction.
        """
        eef_name = self.datastream.get_eef_names()[0]
        obs_eef = self.datastream.get_robot_eef_pose(env_ids=[self.env_id], eef_name=eef_name)[0].to(
            device=self.datastream.device, dtype=torch.float32
        )
        return torch.linalg.inv(self._raw_eef_pose()) @ obs_eef

    def _debug_eef_frame_check(self, waypoints: list[Waypoint]) -> None:
        """Compare the first computed waypoint pose to the adapter's CURRENT EEF pose.

        For a hold (and the first step of any plan, which starts at the current config) these should
        be identical. Any nonzero delta — especially a pure z-axis rotation — exposes a frame /
        quaternion-convention mismatch between cuRobo's ``panda_hand`` and the env's ``ee_frame``
        observation. Prints translation delta and the relative rotation as axis*angle (rad).
        """
        if not waypoints:
            return
        eef_name = self.datastream.get_eef_names()[0]
        current = self.datastream.get_robot_eef_pose(env_ids=[self.env_id], eef_name=eef_name)[0]
        target = waypoints[0].pose.to(device=current.device, dtype=current.dtype)
        cur_rot, tgt_rot = current[:3, :3], target[:3, :3]
        rel = tgt_rot @ cur_rot.transpose(-1, -2)
        axis_angle = axis_angle_from_quat(quat_from_matrix(rel.unsqueeze(0)))[0]
        delta_pos = target[:3, 3] - current[:3, 3]
        print(
            "[schedulestream] EEF frame check (first waypoint vs current obs EEF; hold => ~0):\n"
            f"  delta_pos = {[round(v, 4) for v in delta_pos.tolist()]}\n"
            f"  delta_rot axis*angle (rad) = {[round(v, 4) for v in axis_angle.tolist()]}"
            f"  | |angle| = {float(axis_angle.norm()):.4f}"
        )

    def extract_action(self) -> Waypoint:
        """Extract the :class:`Waypoint` (EEF pose target + binary gripper) implied by the world state.

        The gripper is inferred from the world's arm attachments (closed while holding an object),
        not from the command stream — RoboLabPolicy style. ``noise`` must be a float (not the
        Waypoint default ``None``): the embodiment adapter does ``noise > 0.0``.
        """
        assert self._correction is not None, "get_waypoints() calibrates the frame correction first"
        pose = self._raw_eef_pose() @ self._correction
        gripper = CLOSE_ACTION if self.get_world_arm_attachments() else OPEN_ACTION
        gripper_action = torch.tensor([gripper], dtype=torch.float32, device=self.datastream.device)
        return Waypoint(pose=pose, gripper_action=gripper_action, noise=0.0)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self) -> list[Any]:
        """Base planning under autograd (cuRobo IK/trajopt call ``backward()``); raise on failure.

        The base returns None on failure, but returning an empty result to the data generator would
        livelock ``env_loop`` (a failed attempt enqueues no action, so the attempt-count stop check
        is never reached) — see ``ScheduleStream.plan_subtask_trajectory``. The base has already
        printed the solver traceback and animated the initial state when ``self.animate``.
        """
        if self.profile:
            print(f"{'=' * 30} PROFILE: solve_tamp {'=' * 30}")
        with autograd_enabled():
            commands = super().plan()
        if commands is None:
            raise RuntimeError(f"solve_tamp found no plan for env {self.env_id}.")
        return commands

    def get_waypoints(self) -> list[Waypoint]:
        """Sync the world, plan (or hold), and roll the plan out open loop into Waypoints."""
        self.update_state()
        # Calibrate the constant cuRobo->ee_frame correction at the synced start configuration.
        self._correction = self._eef_frame_correction()

        if self.hold is not None:
            # Null plan: hold the current configuration (mirrors the isaaclab planner's hold).
            commands = self.hold * [self.world.configuration()]
            if self.animate or self.record:
                self.frames.extend(animate_commands(self.state, Commands(self.world, commands), record=self.record))
        else:
            commands = self.plan()

        # Open-loop rollout (update=False): step the plan purely in the world model.
        waypoints = list(self.get_controller(commands))
        self._debug_eef_frame_check(waypoints)
        return waypoints
