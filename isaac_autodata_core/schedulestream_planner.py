# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ScheduleStream TAMP planner grounded in an Isaac AutoData env.

Subclasses cuStream2's :class:`Planner`: the observation hooks read the live
IsaacLab scene and ``extract_action`` emits Isaac AutoData :class:`Waypoint`\\ s.

Import only after the Isaac app is launched (the isaaclab imports require it).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from curobo.types import JointState, Pose
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import convert_quat
from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked

from schedulestream.applications.custream2.animate import animate_commands
from schedulestream.applications.custream2.command import Commands
from schedulestream.applications.custream2.franka import load_franka_config
from schedulestream.applications.custream2.tamp import Attached, Holding, movable_from_goal
from schedulestream.applications.custream2.policy import Planner
from schedulestream.applications.custream2.scene import CAMERA_POSE
from schedulestream.applications.custream2.utils import (
    autograd_enabled,
    multiply_poses,
    to_cpu,
)
from schedulestream.applications.custream2.world import World
from schedulestream.common.utils import apply_mapping, profiler

# Side effect: patches curobo's USD parser to fan-triangulate quad/n-gon faces;
# Arena/RoboLab assets otherwise fail obstacle extraction.
import schedulestream.applications.robolab.usd_utils  # noqa: F401

from isaac_autodata_core.schedulestream_utils import create_objects, destination_from_contact_sensor
from isaac_autodata_core.waypoint import Waypoint

if TYPE_CHECKING:
    from isaac_autodata_interfaces.datastream.datastream import Datastream

# Binary gripper action; flip OPEN_ACTION if the env's convention is inverted.
OPEN_ACTION = 1.0
CLOSE_ACTION = -OPEN_ACTION


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
        noise: float = 0.0,
        animate: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.datastream = datastream
        self.env = datastream.get_env()
        self.env_id = env_id
        self.profile = profile
        self.hold = hold
        self.noise = noise
        self.verbose = verbose
        goal = self.create_goal(success_term)
        world = self._create_world(goal, ik_batch=batch_size)
        # collisions/profile and extra kwargs flow through **solve_kwargs into solve_tamp.
        super().__init__(world, goal=goal, max_time=max_time, animate=animate, collisions=collisions, profile=profile, **kwargs)
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
        """Convert a batched IsaacLab ``root_pose`` row to a cuRobo Pose (xyzw -> wxyz)."""
        root_pose = self.world.to_device(root_pose[self.env_id : self.env_id + 1])
        root_pose[..., 3:7] = convert_quat(root_pose[..., 3:7], to="wxyz")
        return Pose(position=root_pose[:, :3], quaternion=root_pose[:, 3:])

    def get_env_robot_pose(self) -> Pose:
        """Return the robot's world root pose."""
        return self._to_pose(self.scene.state["articulation"][self.robot]["root_pose"])

    def get_env_object_pose(self, obj: str) -> Pose:
        """Return the named object's world pose."""
        for body_type, bodies in self.scene.state.items():
            if (body_type == "articulation") or (obj not in bodies):
                continue
            body = bodies[obj]
            if "root_pose" in body:
                return self._to_pose(body["root_pose"])
            if "nodal_position" in body:
                # Deformables expose nodal state only: use the mean nodal position and keep
                # the world model's current orientation (they carry no root orientation).
                position = self.world.to_device(body["nodal_position"][self.env_id]).mean(dim=0, keepdim=True)
                world_pose = multiply_poses(self.get_env_robot_pose(), self.world.get_object_pose(obj))
                return Pose(position=position, quaternion=world_pose.quaternion)
        raise KeyError(f"Object {obj!r} not in the env scene state")

    def get_env_joint_state(self) -> JointState:
        """Return the robot's current joint state in world joint order."""
        positions = self.scene.state["articulation"][self.robot]["joint_position"][self.env_id]
        mapping = dict(zip(self.articulation.joint_names, to_cpu(positions)))
        positions = torch.tensor(
            apply_mapping(mapping, self.world.all_joints), dtype=torch.float32, device=self.world.device
        )
        return JointState.from_position(positions.unsqueeze(0), joint_names=self.world.all_joints)

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _create_world(self, goal: Any = None, **kwargs: Any) -> World:
        # Only the goal's Attached/Holding objects need to float; an empty set
        # falls back to the env's default movability.
        objects = create_objects(
            self.scene, env_id=self.env_id, floating=movable_from_goal(goal) or None, verbose=self.verbose
        )

        usd_name = os.path.basename(self.articulation.cfg.spawn.usd_path)

        # The Arena stand is part of the robot USD but NOT in cuRobo's collision model
        # (custream2's Franka URDF has no stand link).
        if usd_name not in ("panda_instanceable.usd", "franka_panda_hand_on_stand.usd"):
            raise NotImplementedError(
                f"schedulestream currently supports only the Franka panda, got robot USD {usd_name!r}"
            )
        robot_config = load_franka_config(base_poses=None)

        # env_loop runs under torch.inference_mode(); building the World there allocates
        # cuRobo "inference tensors" that can't be backward()'d (IK) or updated in-place.
        # autograd_enabled() makes them normal tensors throughout.
        with profiler(field="cumtime" if self.profile else None, num=25), autograd_enabled():
            world = World(robot_config, objects, debug=True, **kwargs)

            # The observation hooks read self.world (Planner.__init__ re-assigns the same World).
            self.world = world
            self.update_joint_state()
            world.set_camera_pose(CAMERA_POSE)

            # Captures cuRobo's CUDA graphs; without it the first solve replays a
            # never-captured graph and crashes in seed_ik_solver.
            world.warmup()

        return world

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def create_goal(self, success_term: Any) -> Any:
        """Derive the symbolic goal from the task success term."""
        if success_term is None:
            return self._create_default_goal()
        if success_term.func == cubes_stacked:
            return self._create_stack_goal(success_term)
        arena_goal = self._create_arena_goal(success_term)
        if arena_goal is not None:
            return arena_goal
        raise NotImplementedError(f"schedulestream goal not implemented for {success_term.func}")

    def _create_default_goal(self) -> Any:
        """No success term: one movable -> hold it (lift); several -> stack the two highest-indexed."""
        movable = self.world.movable_names
        if len(movable) == 1:
            # [arm] = self.world.arms
            arm = World.ARM
            return Holding(arm) <= movable[0]
        obj1, obj2 = sorted(movable, reverse=True)[:2]
        return Attached(obj1) == obj2

    def _create_stack_goal(self, success_term: Any) -> Any:
        """Goal for isaaclab_tasks' ``cubes_stacked``: cube2 on cube1, then cube3 on cube2."""
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

    def _create_arena_goal(self, success_term: Any) -> Any:
        """Map IsaacLab-Arena success terms to symbolic goals; None if not an Arena term."""
        try:
            from isaaclab_arena.tasks import terminations as arena_terminations
        except ImportError:
            return None
        params = success_term.params

        if success_term.func is arena_terminations.object_on_destination:
            destination = destination_from_contact_sensor(
                self.scene, params["contact_sensor_cfg"].name, env_id=self.env_id
            )
            return Attached(params["object_cfg"].name) == destination

        if success_term.func is arena_terminations.objects_on_destinations:
            goal = None
            for object_cfg, sensor_cfg in zip(params["object_cfg_list"], params["contact_sensor_cfg_list"]):
                clause = Attached(object_cfg.name) == destination_from_contact_sensor(
                    self.scene, sensor_cfg.name, env_id=self.env_id
                )
                goal = clause if goal is None else (goal & clause)
            return goal

        if success_term.func is arena_terminations.lift_object_il_success:
            # [arm] = self.world.arms
            arm = World.ARM
            return Holding(arm) == params["object_cfg"].name

        return None

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
        """World-frame EEF target as cuRobo sees it: ``robot_root * node_pose(body) * body_offset``
        (cuRobo's ``panda_hand`` convention; :meth:`_eef_frame_correction` maps to the env's ee_frame)."""
        link_pose = self.world.get_node_pose(self.body_name)
        if self.body_offset is not None:
            link_pose = link_pose.multiply(self.body_offset)
        link_pose = multiply_poses(self.get_env_robot_pose(), link_pose)
        return link_pose.get_matrix().squeeze(0).to(device=self.datastream.device, dtype=torch.float32)

    def _eef_frame_correction(self) -> torch.Tensor:
        """Constant body-frame transform mapping cuRobo's EEF frame to the env's ee_frame,
        measured once at the current config: ``inv(raw_eef) @ obs_eef``.

        custream2's Franka URDF defines ``panda_hand`` ~180 deg about z (and a slightly
        different z offset) relative to this IsaacLab version's Franka USD. The mismatch is a
        constant link-local transform, so one measurement reproduces the true EEF pose at
        every config (obs(q) = raw(q) @ C) — valid for full trajectories, not just holds.
        """
        eef_name = self.datastream.get_eef_names()[0]
        obs_eef = self.datastream.get_robot_eef_pose(env_ids=[self.env_id], eef_name=eef_name)[0].to(
            device=self.datastream.device, dtype=torch.float32
        )
        return torch.linalg.inv(self._raw_eef_pose()) @ obs_eef

    def extract_action(self) -> Waypoint:
        """Extract the :class:`Waypoint` (EEF pose target + binary gripper) implied by the world state.

        The gripper closes while the arm holds an attachment. ``noise`` must be a
        float, not None: the embodiment adapter does ``noise > 0.0``.
        """
        assert self._correction is not None, "get_waypoints() calibrates the frame correction first"
        pose = self._raw_eef_pose() @ self._correction
        gripper = CLOSE_ACTION if self.get_world_arm_attachments() else OPEN_ACTION
        gripper_action = torch.tensor([gripper], dtype=torch.float32, device=self.datastream.device)
        return Waypoint(pose=pose, gripper_action=gripper_action, noise=self.noise)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self) -> list[Any]:
        """Base planning under autograd (cuRobo IK/trajopt call ``backward()``); raise on failure.

        Returning an empty result would livelock ``env_loop``: a failed attempt
        enqueues no action, so its attempt-count stop check is never reached.
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
        # Calibrate the cuRobo->ee_frame correction at the synced start configuration.
        self._correction = self._eef_frame_correction()

        if self.hold is not None:
            # Null plan: hold the current configuration.
            commands = self.hold * [self.world.configuration()]
        else:
            commands = self.plan()

        # Open-loop rollout: step the plan purely in the world model. Task-space
        # commands run IK during execute, so this also needs autograd.
        with autograd_enabled():
            waypoints = list(self.get_controller(commands))
        return waypoints
