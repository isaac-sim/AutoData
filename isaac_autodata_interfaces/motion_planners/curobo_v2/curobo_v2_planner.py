# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion planner backed by cuRobo v2.

Implements :class:`MotionPlannerBase` on :class:`curobo.motion_planner.MotionPlanner`, letting
:mod:`isaac_autodata_core.algorithms.SkillGen` plan collision-free transits between skill
segments.

Each planner instance serves one environment and covers:

* Collision geometry extracted once from the USD stage and kept in sync per plan.
* Attaching a grasped object to the robot so it is collision-checked as part of the arm.
* Planning to a target end-effector pose in three phases: retreat, approach, goal.

All world state is read through the :class:`Datastream` facade, so the planner stays
independent of the simulator wiring.

Two conventions are translated at the cuRobo boundary. Callers exchange poses in the
env-relative frame while cuRobo plans in the robot base frame, and Isaac Lab stores
quaternions as ``(x, y, z, w)`` while cuRobo stores ``(w, x, y, z)``.
"""

from __future__ import annotations

import contextlib
import logging
import numpy as np
import torch
from typing import TYPE_CHECKING, Any

# The sphere-fit enum and USD scene parser have no public alias, but the public
# ``curobo.motion_planner.MotionPlanner`` API consumes both directly.
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.util.usd_scene_parser import UsdSceneParser
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

from isaac_autodata_interfaces.motion_planners.curobo_v2.curobo_v2_planner_cfg import CuroboV2PlannerCfg
from isaac_autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

if TYPE_CHECKING:
    from isaac_autodata_interfaces.datastream.datastream import Datastream


class CuroboV2Planner(MotionPlannerBase):
    """cuRobo v2 backend implementing :class:`MotionPlannerBase`.

    Args:
        datastream: Read facade over the env, task, and embodiment. All world state (collision
            geometry source, object poses, joint configuration) is read through it.
        config: Backend-specific configuration; see :class:`CuroboV2PlannerCfg`.
        env_id: Index of the env this planner instance serves in a vectorized setup.
        debug: Whether to print detailed debug information during planning.
    """

    # Each planner owns a single-world MotionPlanner (multi_env=False), so every collision-world
    # operation targets slot 0. ``env_id`` selects which Isaac env to read state from.
    _COLLISION_ENV_IDX = 0

    def __init__(
        self,
        datastream: Datastream,
        config: CuroboV2PlannerCfg,
        env_id: int = 0,
        debug: bool = False,
    ) -> None:
        super().__init__(datastream=datastream, env_id=env_id, debug=debug)
        self.config = config
        # Tag the logger with the env id: planners for every env run concurrently and their
        # messages interleave, so an untagged line cannot be traced back to one.
        self._logger = logging.getLogger(f"CuroboV2Planner_{env_id}")

        self._logger.info("Building cuRobo MotionPlanner for %r", config.robot_name)
        planner_kwargs = config.to_v2_kwargs()
        planner_kwargs["robot"] = self._load_robot_config_dict()
        self.motion_planner = MotionPlanner(MotionPlannerCfg.create(**planner_kwargs))
        self.motion_planner.warmup()
        self._logger.info("MotionPlanner ready. tool_frames=%s", self.motion_planner.tool_frames)

        if self.config.visualize_spheres:
            self._logger.warning("visualize_spheres is not implemented; use visualize_plan instead.")

        # Gripper kinematics for the open and closed poses, plus the pose currently loaded into
        # the planner. See :meth:`_set_gripper_state`.
        self._gripper_kinematics: dict[bool, Any] | None = self._build_gripper_kinematics()
        self._gripper_is_closed: bool | None = None

        # Current plan. The end-effector poses are computed once when the plan is stored so
        # waypoint iteration does not re-run forward kinematics for every waypoint.
        self._planned_joint_trajectory: JointState | None = None
        self._planned_eef_poses: list[torch.Tensor] = []
        self._waypoint_index: int = 0
        self._currently_attached: str | None = None

        # Transforms between the env-relative frame callers use and the robot base frame cuRobo
        # plans in, refreshed each plan by :meth:`_refresh_base_frame_transform`.
        self._base_pose_in_env: torch.Tensor | None = None
        self._env_to_base_transform: torch.Tensor | None = None
        self._base_frame_is_identity: bool = True

        # Joints the solvers optimize over. A robot config may lock joints (e.g. Franka's
        # fingers) that still appear in ``MotionPlanner.joint_names`` but are excluded from the
        # optimization, so states passed to :meth:`plan_pose` must omit them.
        self._active_joint_names: list[str] = self._compute_active_joint_names()
        self._logger.info(
            "Active DoF: %d / %d %s",
            len(self._active_joint_names),
            len(self.motion_planner.joint_names),
            self._active_joint_names,
        )

        # Collision world, built once from the USD stage. ``_object_mapping`` relates scene
        # object names to cuRobo obstacle names so per-plan pose updates know what to move;
        # objects matching ``_static_object_substrings`` are treated as fixed and never moved.
        self._object_mapping: dict[str, str] = {}
        self._world_obstacle_names: list[str] = []
        self._static_object_substrings: list[str] = [s.lower() for s in self.config.static_objects]
        self._obstacle_storage_cache: dict[str, Any] = {}
        self._attached_curobo_name: str | None = None
        self._initialize_static_world()

        # Imported lazily so runs without visualization do not require the rerun-sdk dependency.
        self.plan_visualizer: Any = None
        if self.config.visualize_plan:
            from isaac_autodata_interfaces.motion_planners.curobo_v2.plan_visualizer import PlanVisualizer

            self.plan_visualizer = PlanVisualizer(
                robot_name=self.config.robot_name or "robot",
                recording_id=f"curobo_v2_plan_{self.env_id}",
                save_path=f"curobo_v2_plan_env{self.env_id}.rrd",
                debug=self.debug,
            )
            self.plan_visualizer.set_motion_planner_reference(self.motion_planner)

    def _load_robot_config_dict(self, lock_joints: dict[str, float] | None = None) -> dict[str, Any]:
        """Load the robot config, applying the sphere allocation and locked-joint overrides.

        cuRobo sizes each link's collision-sphere buffer when the config loads, and an attach
        asking for more spheres than the link was allocated fails. Since
        :meth:`MotionPlannerCfg.create` accepts a config dict as readily as a path, the overrides
        are applied here instead of maintaining a separate robot YAML.

        Args:
            lock_joints: Values for the robot's locked joints, or None to keep the config's own.

        Returns:
            The robot config with the overrides applied.
        """
        from curobo.config_io import join_path, load_yaml
        from curobo.content import get_robot_configs_path

        assert self.config.robot_config_file, "CuroboV2PlannerCfg.robot_config_file must be set."
        robot_dict = load_yaml(join_path(get_robot_configs_path(), self.config.robot_config_file))
        kinematics = robot_dict["robot_cfg"]["kinematics"]
        if self.config.extra_collision_spheres:
            kinematics["extra_collision_spheres"] = {
                **(kinematics.get("extra_collision_spheres") or {}),
                **self.config.extra_collision_spheres,
            }
        if lock_joints:
            kinematics["lock_joints"] = dict(lock_joints)
        return robot_dict

    def _build_gripper_kinematics(self) -> dict[bool, Any] | None:
        """Build the robot kinematics for the open and closed gripper poses.

        cuRobo folds locked-joint values into the robot's fixed transforms when a config loads,
        so changing the gripper width means loading a separate kinematics config. Both are built
        once here, since building one per plan would re-parse the robot description each time.

        Returns:
            Kinematics keyed by whether the gripper is closed, or None if the config does not
            define both poses, in which case the gripper keeps its configured width.
        """
        if not (self.config.gripper_open_positions and self.config.gripper_closed_positions):
            return None

        from curobo._src.types.robot import RobotCfg

        device_cfg = self.motion_planner.device_cfg
        variants = {False: self.config.gripper_open_positions, True: self.config.gripper_closed_positions}
        return {
            is_closed: RobotCfg.create(
                self._load_robot_config_dict(lock_joints=positions), device_cfg, num_envs=1
            ).kinematics.kinematics_config
            for is_closed, positions in variants.items()
        }

    def _set_gripper_state(self, is_closed: bool) -> None:
        """Set the gripper width the planner collision-checks against.

        A gripper holding an object is closed around it, so planning with the fingers modeled
        open overstates the gripper's width and rejects valid plans. The inverse kinematics,
        trajectory optimization, and graph planner share one kinematics config, so this update
        reaches all three.

        Call before attaching an object: the update overwrites the link spheres that hold the
        fitted attachment, so calling it afterwards would discard that attachment.

        Args:
            is_closed: Whether the gripper is holding an object.
        """
        if self._gripper_kinematics is None or self._gripper_is_closed == is_closed:
            return
        self.motion_planner.kinematics.update_kinematics_config(self._gripper_kinematics[is_closed])
        self._gripper_is_closed = is_closed

    def _compute_active_joint_names(self) -> list[str]:
        """Return the planner's joint names with any locked joints removed."""
        kinematics_config = self.motion_planner.kinematics.config.kinematics_config
        locked_state = getattr(kinematics_config, "lock_jointstate", None)
        locked_names = set(getattr(locked_state, "joint_names", None) or [])
        return [name for name in self.motion_planner.joint_names if name not in locked_names]

    # ------------------------------------------------------------------
    # Frame conversion
    # ------------------------------------------------------------------

    def _refresh_base_frame_transform(self) -> None:
        """Re-read the robot base pose and cache the transforms derived from it.

        cuRobo plans in the robot base frame while callers exchange poses in the env-relative
        frame. Caching both directions here keeps the conversion off the per-pose path.
        """
        base_pose_in_env = self.datastream.get_robot_root_pose(env_ids=[self.env_id])[0]
        identity = torch.eye(4, dtype=base_pose_in_env.dtype, device=base_pose_in_env.device)
        self._base_pose_in_env = base_pose_in_env
        self._env_to_base_transform = torch.linalg.inv(base_pose_in_env)
        self._base_frame_is_identity = bool(torch.allclose(base_pose_in_env, identity, atol=1e-6))

    def _pose_env_to_base(self, pose_env: torch.Tensor) -> torch.Tensor:
        """Express an env-relative pose in the robot base frame.

        Returns the pose unchanged when the robot base sits at the env origin.

        Args:
            pose_env: Pose as a 4x4 transformation matrix in the env-relative frame.

        Returns:
            The same pose as a 4x4 transformation matrix in the robot base frame.
        """
        if self._base_frame_is_identity or self._env_to_base_transform is None:
            return pose_env
        return self._env_to_base_transform.to(device=pose_env.device, dtype=pose_env.dtype) @ pose_env

    def _pose_base_to_env(self, pose_base: torch.Tensor) -> torch.Tensor:
        """Express a robot-base-frame pose in the env-relative frame.

        Inverse of :meth:`_pose_env_to_base`, applied to planned poses before returning them.

        Args:
            pose_base: Pose as a 4x4 transformation matrix in the robot base frame.

        Returns:
            The same pose as a 4x4 transformation matrix in the env-relative frame.
        """
        if self._base_frame_is_identity or self._base_pose_in_env is None:
            return pose_base
        return self._base_pose_in_env.to(device=pose_base.device, dtype=pose_base.dtype) @ pose_base

    def _pose_matrix_to_curobo(self, pose_mat: torch.Tensor) -> Pose:
        """Convert a 4x4 transformation matrix into a cuRobo :class:`Pose`.

        Reorders the quaternion from Isaac Lab's ``(x, y, z, w)`` to cuRobo's ``(w, x, y, z)``.

        Args:
            pose_mat: Pose as a 4x4 transformation matrix in the robot base frame.

        Returns:
            The equivalent batched (``[1, ...]``) cuRobo pose.
        """
        import isaaclab.utils.math as PoseUtils  # deferred so the module imports without a sim

        position_xyz, rot_mat = PoseUtils.unmake_pose(pose_mat)
        quat_xyzw = PoseUtils.quat_from_matrix(rot_mat)
        return self._make_pose(
            position_xyz=position_xyz.unsqueeze(0),
            quaternion_wxyz=torch.roll(quat_xyzw, shifts=1, dims=-1).unsqueeze(0),
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        **kwargs: Any,
    ) -> bool:
        """Sync the collision world, handle attachments, and plan to ``target_pose``.

        Args:
            target_pose: Target end-effector pose as a 4x4 transformation matrix [m, rad] in the
                env-relative frame.
            expected_attached_object: Object the gripper is holding, which is collision-checked
                as part of the robot. Resolved against the obstacles found on the USD stage at
                construction. ``None`` plans with nothing attached.
            env_id: Environment index, which must be the env this planner serves. Every other
                world read uses :attr:`env_id`, so a mismatch would mix state across envs.

        Returns:
            ``True`` if planning succeeded and a trajectory is now stored, else ``False``.
        """
        del kwargs
        assert env_id == self.env_id, (
            f"Planner for env {self.env_id} was asked to plan for env {env_id}. Every other world"
            " read in this call uses self.env_id, so the two must agree."
        )

        # Trajectory optimization, attachment, and forward kinematics all build tensors that
        # track gradients. The env loop runs under ``torch.inference_mode()``, which forbids
        # that, so lift it for the duration of the call.
        with torch.inference_mode(False), torch.enable_grad():
            self.reset_plan()

            # Move obstacles to their current poses. Updating them individually, rather than
            # rebuilding the scene, preserves attachment and obstacle-disable state.
            self._refresh_base_frame_transform()
            self._sync_obstacle_poses()

            current_state = self._get_current_joint_state(env_id=env_id)

            # Close the gripper before attaching: the gripper update rewrites the link spheres
            # that the attachment is fitted into.
            self._set_gripper_state(is_closed=expected_attached_object is not None)
            self._update_attachment(expected_attached_object, current_state)

            goal_pose = self._pose_matrix_to_curobo(self._pose_env_to_base(target_pose))
            tool_frame = self.motion_planner.tool_frames[0]
            full_trajectory = self._plan_three_phase(
                current_state=current_state,
                goal_pose=goal_pose,
                tool_frame=tool_frame,
            )

            if full_trajectory is not None:
                self._planned_joint_trajectory = full_trajectory
                self._planned_eef_poses = self._joint_trajectory_to_eef_poses(full_trajectory)
                if self.plan_visualizer is not None:
                    try:
                        self._visualize_plan(target_pose=target_pose, current_state=current_state)
                    except Exception as exc:  # noqa: BLE001  (visualization must not break planning)
                        self._logger.warning("Plan visualization failed: %s", exc)

            # Release the object. Its spheres were fitted at the grasp pose of this call, so
            # holding them into the next one would collision-check the object where it no longer
            # is. Each plan re-fits from the object's current pose instead.
            self._update_attachment(None, current_state)

        if full_trajectory is None:
            if self.plan_visualizer is not None:
                with contextlib.suppress(Exception):
                    self.plan_visualizer.mark_idle()
            return False
        return True

    # ------------------------------------------------------------------
    # Plan visualization
    # ------------------------------------------------------------------

    def _visualize_plan(self, target_pose: torch.Tensor, current_state: JointState) -> None:
        """Send the stored plan to the Rerun visualizer.

        Draws the end-effector path, the goal, the obstacles, and the robot's collision spheres
        split into the robot's own and those of the object it holds. Everything is drawn in the
        robot base frame, the frame cuRobo collision-checks in, so the spheres, obstacles, and
        goal line up.

        Args:
            target_pose: Goal end-effector pose as a 4x4 matrix in the env-relative frame.
            current_state: Joint state the collision spheres are evaluated at.
        """
        with torch.inference_mode(False), torch.enable_grad():
            active_q = current_state.position
            if active_q.ndim == 1:
                active_q = active_q.unsqueeze(0)
            # Identify the held object's spheres by the attached link's sphere indices rather
            # than by position in the list. ``filter_valid=False`` keeps absolute indices so they
            # line up with those, leaving inactive (radius <= 0) slots to drop here.
            spheres = self.motion_planner.kinematics.get_robot_as_spheres(active_q.contiguous(), filter_valid=False)[0]
            attached_idx: set[int] = set()
            try:
                _idx = self._attachment_manager().kinematics_params.get_sphere_index_from_link_name(
                    self.config.attached_object_link_name
                )
                attached_idx = {int(j) for j in _idx.detach().cpu().tolist()}
            except Exception:  # noqa: BLE001
                attached_idx = set()
            robot_spheres = [s for i, s in enumerate(spheres) if i not in attached_idx and float(s.radius) > 0.0]
            attached_spheres = [s for i, s in enumerate(spheres) if i in attached_idx and float(s.radius) > 0.0]

            ee_positions = None
            planned_poses = [self._pose_env_to_base(pose) for pose in self._planned_eef_poses]
            if planned_poses:
                ee_positions = np.array([p.detach().cpu().numpy().reshape(4, 4)[:3, 3] for p in planned_poses])
            plan_active = self._planned_trajectory_active()
            world_scene = self._build_world_trimesh_scene()

        self.plan_visualizer.visualize_plan(
            plan=plan_active,
            target_pose=self._pose_env_to_base(target_pose),
            robot_spheres=robot_spheres,
            attached_spheres=attached_spheres,
            ee_positions=ee_positions,
            world_scene=world_scene,
        )
        if ee_positions is not None:
            self.plan_visualizer.animate_plan(ee_positions)
        self.plan_visualizer.animate_spheres_along_path(plan=plan_active, robot_sphere_count=len(robot_spheres))

    def _planned_trajectory_active(self) -> JointState:
        """Return the stored plan reduced to the planner's active joints."""
        traj = self._planned_joint_trajectory
        assert traj is not None, "No plan stored; a successful plan is required."
        full_dof = traj.position.shape[-1]
        flat = traj.position.reshape(-1, full_dof).contiguous()
        names = list(traj.joint_names) if traj.joint_names else list(self.motion_planner.joint_names)
        active = JointState.from_position(flat, joint_names=names).reorder(self._active_joint_names)
        if not active.position.is_contiguous():
            active = JointState.from_position(active.position.contiguous(), joint_names=active.joint_names)
        return active

    def _build_world_trimesh_scene(self):
        """Build a :class:`trimesh.Scene` of the obstacles the planner collides against.

        Triangles come from the scene model, so concave shapes such as a sorting bin are drawn
        with their true geometry rather than a bounding box. Each obstacle is placed at the pose
        held in the collision data, which is kept current each plan, so moving objects follow.
        Together these reproduce the geometry cuRobo collision-checks against.

        Returns:
            The scene, or None if it cannot be built, so visualization never breaks planning.
        """
        try:
            import trimesh

            checker = getattr(self.motion_planner, "scene_collision_checker", None)
            scene_model = getattr(checker, "scene_model", None) if checker is not None else None
            data = getattr(checker, "data", None) if checker is not None else None
            if data is None:
                return None

            # Current pose and bounding-box dimensions per obstacle, from the collision data.
            e = self._COLLISION_ENV_IDX
            world_pose: dict[str, Any] = {}
            world_dims: dict[str, Any] = {}
            for attr in ("cuboids", "meshes"):
                arr = getattr(data, attr, None)
                if arr is None:
                    continue
                for i in range(int(arr.count[e].item())):
                    if int(arr.enable[e, i].item()) == 0:
                        continue
                    name = str(arr.names[e][i])
                    inv_pose = arr.inv_pose[e, i, :7].detach().cpu().numpy()  # [x, y, z, qw, qx, qy, qz]
                    world_pose[name] = np.linalg.inv(self._pose_vec_to_matrix(inv_pose))
                    world_dims[name] = arr.dims[e, i, :3].detach().cpu().numpy()

            scene = trimesh.Scene()
            # Mesh obstacles, drawn from their own triangles at their current pose.
            for mesh_obs in (getattr(scene_model, "mesh", None) or []) if scene_model is not None else []:
                name = str(getattr(mesh_obs, "name", ""))
                verts, faces = getattr(mesh_obs, "vertices", None), getattr(mesh_obs, "faces", None)
                if verts is None or faces is None:
                    continue
                verts = np.asarray(verts, dtype=float).reshape(-1, 3)
                faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
                if verts.size == 0 or faces.size == 0:
                    continue
                scene.add_geometry(
                    trimesh.Trimesh(vertices=verts, faces=faces, process=False),
                    node_name=name.replace("/", "_"),
                    transform=world_pose.get(name, np.eye(4)),
                )
            # Cuboid obstacles, drawn as boxes from their dimensions.
            for cub in (getattr(scene_model, "cuboid", None) or []) if scene_model is not None else []:
                name = str(getattr(cub, "name", ""))
                dims = world_dims.get(name)
                if dims is None or float(np.min(dims)) <= 0.0:
                    continue
                scene.add_geometry(
                    trimesh.creation.box(extents=dims.tolist()),
                    node_name=name.replace("/", "_"),
                    transform=world_pose.get(name, np.eye(4)),
                )
            return scene if len(scene.geometry) else None
        except Exception as exc:  # noqa: BLE001
            self._LOGGER.debug("world scene build for visualization failed: %s", exc)
            return None

    @staticmethod
    def _pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
        """Convert a ``[x, y, z, qw, qx, qy, qz]`` pose vector to a 4x4 homogeneous matrix."""
        x, y, z, qw, qx, qy, qz = (float(v) for v in pose_vec[:7])
        norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
        if norm > 0:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        mat = np.eye(4, dtype=float)
        mat[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        mat[0, 1] = 2 * (qx * qy - qz * qw)
        mat[0, 2] = 2 * (qx * qz + qy * qw)
        mat[1, 0] = 2 * (qx * qy + qz * qw)
        mat[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        mat[1, 2] = 2 * (qy * qz - qx * qw)
        mat[2, 0] = 2 * (qx * qz - qy * qw)
        mat[2, 1] = 2 * (qy * qz + qx * qw)
        mat[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        mat[:3, 3] = (x, y, z)
        return mat

    def _plan_three_phase(
        self,
        current_state: JointState,
        goal_pose: Pose,
        tool_frame: str,
    ) -> JointState | None:
        """Plan retreat, approach, and goal as three chained :meth:`plan_pose` calls.

        Each phase starts from the last waypoint of the one before. Retreat backs the gripper
        away from whatever it currently touches, approach crosses free space to a pose short of
        the goal, and goal closes the remaining distance. Retreat and goal expect contact, so the
        gripper links are collision-disabled for them. Phases with zero distance are skipped.

        Args:
            current_state: Joint state the first phase starts from.
            goal_pose: Target end-effector pose in the robot base frame.
            tool_frame: Robot frame the goal pose applies to.

        Returns:
            The concatenated trajectory, or None if any phase failed.
        """
        retreat_dist = float(self.config.retreat_distance or 0.0)
        approach_dist = float(self.config.approach_distance or 0.0)

        phases: list[tuple[str, GoalToolPose, bool]] = []

        if retreat_dist > 0:
            ee_pose_cu = self._eef_pose_from_state(current_state)
            retreat_pose = ee_pose_cu.multiply(self._local_offset_pose(0.0, 0.0, -retreat_dist))
            phases.append(("retreat", GoalToolPose.from_poses({tool_frame: retreat_pose}), True))

        if approach_dist > 0:
            approach_pose = goal_pose.multiply(self._local_offset_pose(0.0, 0.0, -approach_dist))
            phases.append(("approach", GoalToolPose.from_poses({tool_frame: approach_pose}), False))

        phases.append(("goal", GoalToolPose.from_poses({tool_frame: goal_pose}), True))

        # Links to ignore during contact phases. A held object is included so it does not
        # register a collision against the surface it is being placed on.
        base_disable_links = list(self.config.contact_disable_collision_links)
        if self._currently_attached is not None and self.config.attached_object_link_name:
            base_disable_links.append(self.config.attached_object_link_name)

        phase_trajectories: list[JointState] = []
        state = current_state
        for name, goal_tool_poses, contact in phases:
            phase_traj = self._plan_single_phase(
                name=name,
                goal_tool_poses=goal_tool_poses,
                start_state=state,
                contact=contact,
                disable_links=base_disable_links,
            )
            if phase_traj is None:
                return None
            phase_trajectories.append(phase_traj)
            # Start the next phase where this one ended. Trajectory positions can carry extra
            # leading dimensions, so flatten to ``[N, dof]`` and take the last row. The result
            # holds every joint, while :meth:`plan_pose` accepts only the active ones.
            full_dof = phase_traj.position.shape[-1]
            last_pos_full = phase_traj.position.reshape(-1, full_dof)[-1:].contiguous()  # [1, full_dof]
            state_full = JointState.from_position(last_pos_full, joint_names=phase_traj.joint_names)
            state = state_full.reorder(self._active_joint_names)
            if not state.position.is_contiguous():
                state = JointState.from_position(state.position.contiguous(), joint_names=state.joint_names)

        return self._concat_phase_trajectories(phase_trajectories)

    def _plan_single_phase(
        self,
        name: str,
        goal_tool_poses: GoalToolPose,
        start_state: JointState,
        contact: bool,
        disable_links: list[str],
    ) -> JointState | None:
        """Run :meth:`plan_pose` for one phase, disabling contact links around the call.

        Args:
            name: Phase name, used in failure messages.
            goal_tool_poses: Goal pose for the phase.
            start_state: Joint state the phase starts from.
            contact: Whether the phase ends in contact, allowing collisions on ``disable_links``.
            disable_links: Robot links whose collisions are ignored during a contact phase.

        Returns:
            The trajectory for this phase, or None if planning failed.
        """
        toggled = contact and bool(disable_links)
        hand_links: list[str] = []
        saved_attached = None
        if toggled:
            attach_link = self.config.attached_object_link_name
            # Re-enabling a link restores its spheres from the robot config, where the attached
            # link has none. Doing that would erase the fitted object, so its spheres are saved
            # and restored here instead of going through enable/disable_link_collision.
            hand_links = [link for link in disable_links if link != attach_link]
            if hand_links:
                self.motion_planner.disable_link_collision(hand_links)
            if attach_link in disable_links:
                saved_attached = self._save_disable_attached_spheres(attach_link)
        try:
            result = self.motion_planner.plan_pose(goal_tool_poses, start_state)
        finally:
            if toggled:
                if hand_links:
                    self.motion_planner.enable_link_collision(hand_links)
                if saved_attached is not None:
                    self._restore_attached_spheres(*saved_attached)

        if result is None or getattr(result, "success", None) is None:
            self._logger.warning("Phase %r failed: no result returned.", name)
            return None
        if not bool(result.success.any().item()):
            self._logger.warning(
                "Phase %r failed: every seed failed (status=%s).", name, getattr(result, "status", None)
            )
            return None

        traj = result.interpolated_trajectory
        if traj is None:
            traj = result.js_solution
        if traj is None or traj.position is None:
            self._logger.warning("Phase %r failed: the returned trajectory was empty.", name)
            return None

        last_tstep = self._extract_last_tstep_from(getattr(result, "interpolated_last_tstep", None))
        trimmed = self._trim_inclusive(traj, last_tstep)
        if trimmed is None or trimmed.position is None or trimmed.position.shape[-2] == 0:
            return None
        return trimmed

    def _save_disable_attached_spheres(self, link_name: str):
        """Disable the attached link's spheres by clearing their radii, keeping a copy.

        Args:
            link_name: Link holding the attached object's spheres.

        Returns:
            The sphere indices and their saved values, for :meth:`_restore_attached_spheres`.
        """
        kp = self._attachment_manager().kinematics_params
        idx = kp.get_sphere_index_from_link_name(link_name)
        saved = kp.link_spheres[:, idx, :].clone()
        kp.link_spheres[:, idx, 3] = -100.0
        return idx, saved

    def _restore_attached_spheres(self, idx, saved) -> None:
        """Restore the spheres saved by :meth:`_save_disable_attached_spheres`."""
        self._attachment_manager().kinematics_params.link_spheres[:, idx, :] = saved

    def _eef_pose_from_state(self, state: JointState) -> Pose:
        """Compute the end-effector pose for a ``[batch, dof]`` joint state."""
        position = state.position
        if position.ndim == 2:  # Add the horizon dimension compute_kinematics expects.
            position = position.unsqueeze(1)
        js = JointState.from_position(position, joint_names=state.joint_names)
        kin_state = self.motion_planner.compute_kinematics(js)
        link_pose = kin_state.tool_poses.get_link_pose(self.motion_planner.tool_frames[0])
        return self._make_pose(
            position_xyz=link_pose.position,
            quaternion_wxyz=link_pose.quaternion,
        )

    def _local_offset_pose(self, x: float, y: float, z: float) -> Pose:
        """Build a :class:`Pose` translating by ``(x, y, z)`` [m] with no rotation."""
        return self._make_pose(
            position_xyz=torch.tensor([[x, y, z]]),
        )

    def _make_pose(
        self,
        position_xyz: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
    ) -> Pose:
        """Build a cuRobo :class:`Pose` with contiguous float32 tensors on the planner's device.

        cuRobo's pose kernels reject non-contiguous tensors and tensors of the wrong dtype, both
        of which arise from slicing and from autograd outputs. Normalizing here keeps that
        handling in one place. The quaternion defaults to identity.
        """
        device = self.motion_planner.device_cfg.device
        position = position_xyz.to(device=device, dtype=torch.float32).contiguous()
        if quaternion_wxyz is None:
            return Pose(position=position)
        quaternion = quaternion_wxyz.to(device=device, dtype=torch.float32).contiguous()
        return Pose(position=position, quaternion=quaternion)

    @staticmethod
    def _extract_last_tstep_from(last_tstep_tensor) -> int | None:
        """Read a scalar final time step out of a per-batch tensor, or None if unavailable."""
        if last_tstep_tensor is None:
            return None
        try:
            return int(last_tstep_tensor.flatten()[0].item())
        except (RuntimeError, ValueError, IndexError):
            return None

    @staticmethod
    def _trim_inclusive(traj: JointState, last_tstep: int | None) -> JointState | None:
        """Drop the padding an interpolated trajectory carries past ``last_tstep``."""
        if traj is None or traj.position is None:
            return None
        total = traj.position.shape[-2]
        if last_tstep is None or last_tstep <= 0:
            return traj
        end_idx = min(last_tstep + 1, total)
        if end_idx >= total:
            return traj
        from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory

        return trim_joint_state_trajectory(traj, start_idx=0, end_idx=end_idx)

    @staticmethod
    def _concat_phase_trajectories(phases: list[JointState]) -> JointState:
        """Join per-phase trajectories into one along the time dimension."""
        if len(phases) == 1:
            return phases[0]
        position = torch.cat([p.position for p in phases], dim=-2)
        velocity = (
            torch.cat([p.velocity for p in phases], dim=-2) if all(p.velocity is not None for p in phases) else None
        )
        acceleration = (
            torch.cat([p.acceleration for p in phases], dim=-2)
            if all(p.acceleration is not None for p in phases)
            else None
        )
        jerk = torch.cat([p.jerk for p in phases], dim=-2) if all(p.jerk is not None for p in phases) else None
        head = phases[0]
        return JointState(
            position=position,
            velocity=velocity if velocity is not None else position * 0.0,
            acceleration=acceleration if acceleration is not None else position * 0.0,
            jerk=jerk if jerk is not None else position * 0.0,
            joint_names=head.joint_names,
        )

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Return the plan as 4x4 end-effector pose matrices [m, rad] in the env-relative frame."""
        return list(self._planned_eef_poses)

    def has_next_waypoint(self) -> bool:
        return self._waypoint_index < len(self._planned_eef_poses)

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        assert self.has_next_waypoint(), "No more waypoints in the current plan."
        pose = self._planned_eef_poses[self._waypoint_index]
        self._waypoint_index += 1
        return pose

    def reset_plan(self) -> None:
        """Discard the stored plan and rewind the waypoint iterator."""
        self._planned_joint_trajectory = None
        self._planned_eef_poses = []
        self._waypoint_index = 0

    # ------------------------------------------------------------------
    # Collision world
    # ------------------------------------------------------------------

    def _initialize_static_world(self) -> None:
        """Build the collision world from the USD stage.

        Walks the env's subtree, expresses each obstacle relative to the robot base, and skips
        the prims named in ``world_ignore_substrings``. The world is built once here; later
        plans only move obstacles within it, which keeps attachment and obstacle-disable state
        intact between plans.
        """
        if getattr(self.motion_planner, "scene_collision_checker", None) is None:
            self._logger.warning(
                "MotionPlanner has no scene collision checker; skipping world setup. "
                "Set CuroboV2PlannerCfg.collision_cache or scene_model."
            )
            return

        env_prim = self.datastream.get_env_prim_path(self.env_id)
        robot_prim = self.datastream.get_robot_prim_path(self.env_id)
        ignore = list(self.config.world_ignore_substrings)

        parser = UsdSceneParser()
        parser.load_stage(self.datastream.get_usd_stage())
        scene = parser.get_obstacles_from_stage(
            only_paths=[env_prim],
            reference_prim_path=robot_prim,
            ignore_substring=ignore,
        )
        if self.config.obstacle_representation == "obb":
            scene = self._scene_as_cuboids(scene)
        else:
            scene = scene.get_collision_check_world()
        self.motion_planner.update_world(scene)

        checker = self.motion_planner.scene_collision_checker
        self._world_obstacle_names = list(checker.get_obstacle_names(env_idx=self._COLLISION_ENV_IDX))
        self._object_mapping = self._discover_object_mapping(self._world_obstacle_names)

        moving = [k for k in self._object_mapping if not self._is_static_object(k)]
        fixed = [k for k in self._object_mapping if self._is_static_object(k)]
        self._logger.info(
            "Found %d obstacles under %s (relative to %s); moving=%s fixed=%s",
            len(self._world_obstacle_names),
            env_prim,
            robot_prim,
            moving,
            fixed,
        )

    def _scene_as_cuboids(self, scene) -> Scene:
        """Replace every mesh in ``scene`` with its bounding box, leaving cuboids untouched.

        Box collision is exact for box-shaped obstacles and cheaper than a mesh query. Any other
        shape becomes a conservative enclosure, so concave obstacles such as a bin are better
        served by the mesh representation.

        Args:
            scene: Scene of obstacles read from the stage.

        Returns:
            A scene containing only cuboids.
        """
        cuboids: list[Cuboid] = list(scene.cuboid or [])
        for mesh in scene.mesh or []:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if verts.size == 0:
                continue
            scale = mesh.scale
            if scale is not None:
                verts = verts * np.asarray(scale, dtype=np.float64).reshape(1, -1)
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            dims = (hi - lo).tolist()
            center_local = ((lo + hi) / 2.0).tolist()
            # Offset the mesh's own pose by the box center to place the cuboid.
            mesh_pose = Pose.from_list(list(mesh.pose), self.motion_planner.device_cfg)
            center_offset = self._make_pose(position_xyz=torch.tensor([center_local]))
            cuboid_pose = mesh_pose.multiply(center_offset).tolist()
            cuboids.append(Cuboid(name=str(mesh.name), dims=dims, pose=cuboid_pose))
        return Scene(cuboid=cuboids)

    def _is_static_object(self, name: str) -> bool:
        """Return whether ``name`` is configured as a fixed obstacle that is never moved."""
        return any(s in name.lower() for s in self._static_object_substrings)

    def _discover_object_mapping(self, world_obstacle_names: list[str]) -> dict[str, str]:
        """Relate scene object names to the obstacle names cuRobo assigned them.

        cuRobo names obstacles after their USD prim path while the scene keys objects by short
        name, so the two are matched on substring. Only objects the scene reports poses for can
        be matched; geometry that exists solely on the stage stays where it was loaded.

        Args:
            world_obstacle_names: Obstacle names present in the collision world.

        Returns:
            Map of scene object name to cuRobo obstacle name.
        """
        scene_object_names = list(self.datastream.get_object_poses(env_ids=[self.env_id]).keys())
        mapping: dict[str, str] = {}
        for obj_name in scene_object_names:
            key = obj_name.lower().replace("_", "")
            for path in world_obstacle_names:
                if key in str(path).lower().replace("_", ""):
                    mapping[obj_name] = path
                    break
        return mapping

    def _sync_obstacle_poses(self) -> None:
        """Move every non-fixed obstacle to the pose its scene object currently has."""
        checker = getattr(self.motion_planner, "scene_collision_checker", None)
        if checker is None or not self._object_mapping:
            return

        object_poses = self.datastream.get_object_poses(env_ids=[self.env_id])
        for obj_name, curobo_name in self._object_mapping.items():
            if self._is_static_object(obj_name):
                continue
            pose_mat = object_poses.get(obj_name)
            if pose_mat is None:
                continue
            self._set_obstacle_pose(curobo_name, self._pose_matrix_to_curobo(self._pose_env_to_base(pose_mat[0])))

    # ------------------------------------------------------------------
    # Obstacle updates
    # ------------------------------------------------------------------
    # ``SceneCollisionChecker.update_obstacle_pose`` and ``enable_obstacle`` search the cuboid
    # list first and raise before reaching the mesh list, so a mesh obstacle cannot be updated
    # through them. The methods below find the array holding each obstacle and update it there.

    def _obstacle_storage(self, curobo_name: str):
        """Return the collision-data array holding ``curobo_name``, or None if absent."""
        cached = self._obstacle_storage_cache.get(curobo_name)
        if cached is not None:
            return cached
        data = self.motion_planner.scene_collision_checker.data
        for attr in ("cuboids", "meshes", "voxels"):
            arr = getattr(data, attr, None)
            if arr is not None and curobo_name in arr.get_names(self._COLLISION_ENV_IDX):
                self._obstacle_storage_cache[curobo_name] = arr
                return arr
        return None

    def _set_obstacle_pose(self, curobo_name: str, w_obj_pose: Pose) -> None:
        """Move one obstacle to ``w_obj_pose``."""
        arr = self._obstacle_storage(curobo_name)
        if arr is None:
            self._logger.warning("Obstacle %r is not in the collision world; skipping its pose update.", curobo_name)
            return
        arr.update_pose(curobo_name, w_obj_pose=w_obj_pose, env_idx=self._COLLISION_ENV_IDX)

    def _set_obstacle_enabled(self, curobo_name: str, enabled: bool) -> None:
        """Include or exclude one obstacle from collision checking."""
        arr = self._obstacle_storage(curobo_name)
        if arr is None:
            self._logger.warning("Obstacle %r is not in the collision world; skipping enable=%s.", curobo_name, enabled)
            return
        arr.set_enabled(curobo_name, enabled, self._COLLISION_ENV_IDX)

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def _attachment_manager(self):
        """Return the attachment manager.

        The ``MotionPlanner.attachment_manager`` property looks the manager up at a path where
        it does not live, so fall back to its actual location when the property fails.
        """
        try:
            mgr = self.motion_planner.attachment_manager
            if mgr is not None:
                return mgr
        except AttributeError:
            pass
        return self.motion_planner.trajopt_solver.core.attachment_manager

    def _update_attachment(self, expected: str | None, current_state: JointState) -> None:
        """Attach ``expected`` to the robot, releasing whatever was held before.

        Args:
            expected: Object the gripper holds, or None to release without attaching.
            current_state: Joint state the object's spheres are fitted at.
        """
        if expected == self._currently_attached:
            return

        # Release the previous object and restore it as a world obstacle. cuRobo's own re-enable
        # path cannot reach mesh obstacles, so it is skipped in favour of _set_obstacle_enabled.
        if self._currently_attached is not None:
            self._attachment_manager().detach(
                link_name=self.config.attached_object_link_name,
                enable_obstacle_names=None,
            )
            if self._attached_curobo_name is not None:
                self._set_obstacle_enabled(self._attached_curobo_name, True)
            self._currently_attached = None
            self._attached_curobo_name = None

        if expected is None:
            return

        curobo_name = self._object_mapping.get(expected)
        if curobo_name is None:
            self._logger.warning("Object %r is not in the collision world; planning unattached.", expected)
            return

        checker = self.motion_planner.scene_collision_checker
        obstacle = checker.scene_model.get_obstacle(curobo_name) if checker.scene_model is not None else None
        if obstacle is None:
            self._logger.warning(
                "Object %r (%s) is not in the scene model; planning unattached.", expected, curobo_name
            )
            return

        # Fitting spheres to an obstacle bakes in the pose the obstacle carries, and the scene
        # model keeps the pose the object was loaded at rather than where it is now. Passing the
        # offset between the two cancels the stale pose so the spheres land on the object.
        world_pose_offset = None
        current_world_pose = self._object_world_pose(expected)
        if current_world_pose is not None:
            obstacle_pose = Pose.from_list(list(obstacle.pose), self.motion_planner.device_cfg)
            world_pose_offset = current_world_pose.multiply(obstacle_pose.inverse())

        try:
            # The object is excluded from the world below rather than through cuRobo's own
            # auto-disable, which cannot reach mesh obstacles. Without that it would be counted
            # twice: once as attached spheres and once as an obstacle.
            self._attachment_manager().attach(
                joint_states=current_state,
                obstacles=[obstacle],
                link_name=self.config.attached_object_link_name,
                num_spheres=self.config.attached_object_num_spheres,
                surface_radius=self.config.surface_sphere_radius,
                sphere_fit_type=self._resolve_sphere_fit_type(),
                world_objects_pose_offset=world_pose_offset,
                disable_obstacle_names=None,
            )
            self._set_obstacle_enabled(curobo_name, False)
            self._currently_attached = expected
            self._attached_curobo_name = curobo_name
        except Exception as exc:  # noqa: BLE001  (a failed attach should not abort planning)
            self._logger.warning("Attaching %r failed: %s. Planning unattached.", expected, exc)
            self._currently_attached = None
            self._attached_curobo_name = None

    def _object_world_pose(self, obj_name: str) -> Pose | None:
        """Return a scene object's current pose in the robot base frame, or None if unknown."""
        pose_mat = self.datastream.get_object_poses(env_ids=[self.env_id]).get(obj_name)
        if pose_mat is None:
            return None
        return self._pose_matrix_to_curobo(self._pose_env_to_base(pose_mat[0]))

    def _resolve_sphere_fit_type(self) -> SphereFitType:
        """Convert the configured sphere-fit name into its :class:`SphereFitType` value."""
        name = (self.config.sphere_fit_type or "").upper()
        try:
            return SphereFitType[name]
        except KeyError:
            self._logger.warning(
                "Unknown sphere_fit_type %r; using SURFACE. Valid names: %s",
                self.config.sphere_fit_type,
                [m.name for m in SphereFitType],
            )
            return SphereFitType.SURFACE

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def _get_current_joint_state(self, env_id: int) -> JointState:
        """Read the robot's joint positions, keeping only the planner's active joints.

        The solvers operate on the active joints alone, so passing the full set, locked joints
        included, fails downstream on a dimension mismatch.
        """
        joint_pos_isaac = self.datastream.get_robot_joint_positions(env_ids=[env_id])[0]
        position = (
            joint_pos_isaac.unsqueeze(0)
            .to(device=self.motion_planner.device_cfg.device, dtype=torch.float32)
            .contiguous()
        )
        state = JointState.from_position(position, joint_names=self.datastream.get_robot_joint_names())
        reordered = state.reorder(self._active_joint_names)
        # Reordering can return a view, which the CUDA kernels reject.
        if not reordered.position.is_contiguous():
            reordered = JointState.from_position(reordered.position.contiguous(), joint_names=reordered.joint_names)
        return reordered

    # ------------------------------------------------------------------
    # End-effector poses
    # ------------------------------------------------------------------

    def _joint_trajectory_to_eef_poses(self, traj: JointState) -> list[torch.Tensor]:
        """Convert a joint trajectory into end-effector poses with one batched call.

        Args:
            traj: Planned joint trajectory.

        Returns:
            One 4x4 pose matrix per waypoint, in the env-relative frame.
        """
        import isaaclab.utils.math as PoseUtils  # deferred so the module imports without a sim

        positions = traj.position
        full_dof = positions.shape[-1]
        flat = positions.reshape(-1, full_dof)

        # The trajectory holds every joint, while forward kinematics takes the active ones.
        traj_joint_names = traj.joint_names if traj.joint_names else self.motion_planner.joint_names
        js_position = flat.unsqueeze(0).to(self.motion_planner.device_cfg.device)
        full_state = JointState.from_position(js_position, joint_names=list(traj_joint_names))
        single_state = full_state.reorder(self._active_joint_names)
        if not single_state.position.is_contiguous():
            single_state = JointState.from_position(
                single_state.position.contiguous(), joint_names=single_state.joint_names
            )

        # Forward kinematics builds tensors that track gradients, which the env loop forbids.
        with torch.inference_mode(False), torch.enable_grad():
            kin_state = self.motion_planner.compute_kinematics(single_state)

        tool_frame = self.motion_planner.tool_frames[0]
        link_pose = kin_state.tool_poses.get_link_pose(tool_frame)
        positions_w = link_pose.position  # [T, 3]
        quaternions_wxyz = link_pose.quaternion  # [T, 4]
        quaternions_xyzw = torch.roll(quaternions_wxyz, shifts=-1, dims=-1)
        rotations = PoseUtils.matrix_from_quat(quaternions_xyzw)  # [T, 3, 3]
        return [
            self._pose_base_to_env(PoseUtils.make_pose(positions_w[t], rotations[t]))
            for t in range(positions_w.shape[0])
        ]
