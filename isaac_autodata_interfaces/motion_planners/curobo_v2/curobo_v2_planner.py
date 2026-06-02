# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo v2 motion-planner implementation.

Implements :class:`MotionPlannerBase` against cuRobo v2's redesigned public API
(:class:`curobo.motion_planner.MotionPlanner`). The constructor signature and the methods
consumed by the SkillGen algorithm match the v1 backend so
:mod:`isaac_autodata_core.algorithms.SkillGen` can use either backend without modification.

Feature surface implemented here:

* Single goal-pose planning via :meth:`MotionPlanner.plan_pose`.
* Per-call scene synchronization: poses of dynamic obstacles declared in
  :attr:`CuroboV2PlannerCfg.dynamic_object_dims` are read from the live Isaac Lab scene each
  call and pushed into the planner's collision world via :meth:`MotionPlanner.update_world`.
* Object attachment / detachment via :attr:`MotionPlanner.attachment_manager`.
* Joint-state ordering between the live articulation and the planner's internal order.
* Quaternion convention bridge: Isaac Lab uses ``(x, y, z, w)``; cuRobo internals use
  ``(w, x, y, z)``. Conversion is localized to the cuRobo boundary.
* Trajectory extraction: each interpolated joint waypoint is mapped to a 4x4 EEF pose
  through :meth:`MotionPlanner.compute_kinematics`.

Plan visualization is not yet hooked up; :attr:`CuroboV2PlannerCfg.visualize_plan` is accepted
for API parity but unused.
"""

from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING, Any

import warp as wp

# Sphere-fit algorithm enum used by the attachment manager. Lives under ``_src``; safe to
# import directly since the public ``curobo.motion_planner.MotionPlanner`` uses it natively.
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose
from isaac_autodata_interfaces.motion_planners.curobo_v2.curobo_v2_planner_cfg import CuroboV2PlannerCfg
from isaac_autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs.manager_based_env import ManagerBasedEnv


def _as_torch(arr):
    """Materialize a torch view of a warp-or-tensor handle.

    Recent Isaac Lab releases expose articulation-data properties as :class:`warp.array`;
    warp arrays do not support Python-style indexing, so callers must convert before slicing.
    This helper is a no-op for tensors.
    """
    return wp.to_torch(arr) if isinstance(arr, wp.array) else arr


class CuroboV2Planner(MotionPlannerBase):
    """cuRobo v2 backend implementing :class:`MotionPlannerBase`.

    Args:
        env: The Isaac Lab environment instance the robot lives in.
        robot: Robot articulation handle.
        config: Backend-specific configuration; see :class:`CuroboV2PlannerCfg`.
        env_id: Index of the env this planner instance serves in a vectorized setup.
        debug: Whether to print detailed debug information during planning.
    """

    _LOGGER = logging.getLogger("CuroboV2Planner")

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboV2PlannerCfg,
        env_id: int = 0,
        debug: bool = False,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=debug)
        self.config = config

        self._LOGGER.info("Constructing cuRobo v2 MotionPlannerCfg for %r", config.robot_name)
        v2_cfg = MotionPlannerCfg.create(**config.to_v2_kwargs())

        self._LOGGER.info("Instantiating MotionPlanner; warming up...")
        self.motion_planner = MotionPlanner(v2_cfg)
        self.motion_planner.warmup()
        self._LOGGER.info("MotionPlanner ready. tool_frames=%s", self.motion_planner.tool_frames)

        # Per-plan state.
        self._planned_joint_trajectory: JointState | None = None
        self._waypoint_index: int = 0
        # Track the currently attached object so we can detach cleanly between subtasks.
        self._currently_attached: str | None = None

        # Active (non-locked) joint names. The robot YAML may declare joints under
        # ``lock_joints`` (e.g. Franka's two finger joints) that count toward
        # ``MotionPlanner.joint_names`` but are *not* part of the optimization DoF the IK /
        # trajopt solvers operate on. Filter them out once so the state we feed
        # :meth:`plan_pose` matches the solver's active-dof dimension.
        self._active_joint_names: list[str] = self._compute_active_joint_names()
        self._LOGGER.info(
            "Active DoF: %d / %d  (planner.joint_names=%s)",
            len(self._active_joint_names),
            len(self.motion_planner.joint_names),
            self._active_joint_names,
        )

        # Register every obstacle (static + dynamic at their current poses) ONCE up front,
        # then per-call updates only touch dynamic obstacle poses. Mirrors v1's pattern.
        self._registered_dynamic_objects: list[str] = []
        self._register_scene_obstacles()

    def _compute_active_joint_names(self) -> list[str]:
        """Return planner joint names minus any joints listed under ``lock_joints``."""
        all_names = list(self.motion_planner.joint_names)
        kinematics = self.motion_planner.kinematics
        locked_state = getattr(kinematics.config.kinematics_config, "lock_jointstate", None)
        locked_names: set[str] = set()
        if locked_state is not None and getattr(locked_state, "joint_names", None):
            locked_names = set(locked_state.joint_names)
        return [n for n in all_names if n not in locked_names]

    # ------------------------------------------------------------------
    # Surface consumed by SkillGen (matches the v1 backend)
    # ------------------------------------------------------------------

    @property
    def step_size(self) -> float | None:
        """Retiming step size in radians [rad]; ``None`` disables retiming."""
        return self.config.motion_step_size

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Sync the collision world, handle attachments, and plan to ``target_pose``.

        Args:
            target_pose: Target end-effector pose as a 4x4 transformation matrix [m, rad]
                in the same frame as the planner's robot base.
            expected_attached_object: Name of an object that should be considered attached to
                the gripper during planning. Resolved against the dynamic obstacle names in
                :attr:`CuroboV2PlannerCfg.dynamic_object_dims`. ``None`` releases any prior
                attachment.
            env_id: Vectorized environment index.
            step_size: Retiming step size [rad]. Accepted for protocol parity; the v2 planner
                returns an interpolated trajectory at its own configured dt and does not
                currently re-retime here.
            enable_retiming: Accepted for protocol parity; currently unused.

        Returns:
            ``True`` if planning succeeded and a trajectory is now stored, else ``False``.
        """
        del kwargs, step_size, enable_retiming

        # cuRobo v2's trajopt uses autograd (``cost.backward``) and parts of the attachment +
        # FK path also create grad-tracking tensors. The upstream env loop wraps everything in
        # ``torch.inference_mode()``, which forbids both. Locally lift inference mode for the
        # whole planning entrypoint.
        with torch.inference_mode(False), torch.enable_grad():
            # Reset prior plan before attempting a fresh one.
            self._planned_joint_trajectory = None
            self._waypoint_index = 0

            # 1. Sync dynamic obstacle poses from the live scene. Per-obstacle update (no
            # scene rebuild) — preserves attachment state and prior obstacle-disable flags.
            self._sync_dynamic_obstacle_poses()

            # 2. Read current joint state in the planner's expected order.
            current_state = self._get_current_joint_state(env_id=env_id)

            # 3. Handle attachment state — attach the requested object, or detach any prior one.
            self._update_attachment(expected_attached_object, current_state)

            # 4. Build the goal pose. cuRobo v2 ``Pose`` stores quaternions in (w, x, y, z);
            # Isaac Lab works in (x, y, z, w), so we reorder at this single boundary.
            import isaaclab.utils.math as PoseUtils  # deferred so the module imports sim-free

            target_pos_xyz, target_rot_mat = PoseUtils.unmake_pose(target_pose)
            target_quat_xyzw = PoseUtils.quat_from_matrix(target_rot_mat)
            target_quat_wxyz = torch.roll(target_quat_xyzw, shifts=1, dims=-1)
            goal_pose = self._make_pose(
                position_xyz=target_pos_xyz.unsqueeze(0),
                quaternion_wxyz=target_quat_wxyz.unsqueeze(0),
            )
            tool_frame = self.motion_planner.tool_frames[0]
            _ = GoalToolPose.from_poses({tool_frame: goal_pose})

            # 5. Plan via the same three-phase pattern the v1 backend uses in
            # ``_plan_to_contact``: retreat (lift from the current EEF), approach (lift to
            # above the goal), and goal (descend to target). Each is a separate
            # :meth:`MotionPlanner.plan_pose` call whose start state is the last waypoint of
            # the preceding phase. Hand links (and the attached-object link, if any) are
            # collision-disabled during the retreat and goal phases so contact is allowed.
            full_trajectory = self._plan_three_phase(
                current_state=current_state,
                goal_pose=goal_pose,
                tool_frame=tool_frame,
            )

        if full_trajectory is None:
            return False
        self._planned_joint_trajectory = full_trajectory
        self._waypoint_index = 0
        return True

    def _plan_three_phase(
        self,
        current_state: JointState,
        goal_pose: Pose,
        tool_frame: str,
    ) -> JointState | None:
        """Plan retreat → approach → goal as three chained :meth:`plan_pose` calls.

        Replicates the v1 backend's ``_plan_to_contact`` flow. Phases with a zero distance
        are skipped. The contact flag for each phase mirrors v1: ``True`` for retreat and
        goal (hand links collision-disabled so the gripper can be in contact with the
        target), ``False`` for the free-space approach above the goal.

        Returns ``None`` if any phase fails to produce a usable trajectory.
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

        # Hand links to drop during contact phases. When an object is attached, also include
        # the attached-object link so the held geometry doesn't trigger false collisions
        # against the target / stack.
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
            # Chain: start of next phase = last waypoint of this phase. v2 trajectory
            # positions can carry extra leading dims (e.g. ``[B, seeds, T, dof]``), so
            # collapse to ``[N, dof]`` and take the final row — the trimmed trajectory
            # ends at the phase goal, which is the only waypoint we need. The output
            # trajectory carries *all* the robot's joints (active + locked passthrough);
            # plan_pose only accepts active-DoF states, so we drop the locked columns
            # here via :meth:`JointState.reorder` against :attr:`_active_joint_names`.
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
        """Run :meth:`plan_pose` for one phase, bracketed by contact-mode link toggling."""
        toggled = contact and bool(disable_links)
        if toggled:
            self.motion_planner.disable_link_collision(disable_links)
        try:
            result = self.motion_planner.plan_pose(goal_tool_poses, start_state)
        finally:
            if toggled:
                self.motion_planner.enable_link_collision(disable_links)

        if result is None or getattr(result, "success", None) is None:
            self._LOGGER.debug("phase=%s plan_pose returned no result.", name)
            return None
        if not bool(result.success.any().item()):
            self._LOGGER.debug("phase=%s plan_pose reported failure across all seeds.", name)
            return None

        traj = result.interpolated_trajectory
        if traj is None:
            traj = result.js_solution
        if traj is None or traj.position is None:
            self._LOGGER.debug("phase=%s plan_pose succeeded but trajectory was empty.", name)
            return None

        last_tstep = self._extract_last_tstep_from(getattr(result, "interpolated_last_tstep", None))
        trimmed = self._trim_inclusive(traj, last_tstep)
        if trimmed is None or trimmed.position is None or trimmed.position.shape[-2] == 0:
            return None
        return trimmed

    def _eef_pose_from_state(self, state: JointState) -> Pose:
        """Compute the EEF :class:`Pose` for a 2D ``[batch, dof]`` :class:`JointState`."""
        position = state.position
        if position.ndim == 2:  # [B, dof] → [B, 1, dof] so compute_kinematics sees a horizon dim.
            position = position.unsqueeze(1)
        # ``state`` is already in active-joint order (see :meth:`_get_current_joint_state`);
        # rebuild a JointState in that same order — compute_kinematics operates on active DoF.
        js = JointState.from_position(position, joint_names=state.joint_names)
        kin_state = self.motion_planner.compute_kinematics(js)
        tool_frame = self.motion_planner.tool_frames[0]
        link_pose = kin_state.tool_poses.get_link_pose(tool_frame)
        # ``get_link_pose`` returns ``[B*H, 3]``/``[B*H, 4]``; with B=H=1 that's ``[1, ...]``.
        return self._make_pose(
            position_xyz=link_pose.position,
            quaternion_wxyz=link_pose.quaternion,
        )

    def _local_offset_pose(self, x: float, y: float, z: float) -> Pose:
        """Build a :class:`Pose` representing a translation in the local frame (identity rotation)."""
        return self._make_pose(
            position_xyz=torch.tensor([[x, y, z]]),
        )

    def _make_pose(
        self,
        position_xyz: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
    ) -> Pose:
        """Build a cuRobo :class:`Pose` with contiguous, float32, on-device tensors.

        cuRobo's CUDA pose ops (``BatchTransformPose``, ``pose_multiply``) reject tensors
        that are non-contiguous in element-strides or that have the wrong dtype. Views
        produced by upstream slicing / ``unsqueeze`` and tensors returned from autograd
        functions can fail one or both checks, so we normalize at this single boundary.
        Quaternion defaults to identity ``(w=1, 0, 0, 0)`` when omitted.
        """
        device = self.motion_planner.device_cfg.device
        position = position_xyz.to(device=device, dtype=torch.float32).contiguous()
        if quaternion_wxyz is None:
            return Pose(position=position)
        quaternion = quaternion_wxyz.to(device=device, dtype=torch.float32).contiguous()
        return Pose(position=position, quaternion=quaternion)

    @staticmethod
    def _extract_last_tstep_from(last_tstep_tensor) -> int | None:
        """Pull a scalar ``last_tstep`` out of a per-batch tensor (or ``None`` if absent)."""
        if last_tstep_tensor is None:
            return None
        try:
            return int(last_tstep_tensor.flatten()[0].item())
        except (RuntimeError, ValueError, IndexError):
            return None

    @staticmethod
    def _trim_inclusive(traj: JointState, last_tstep: int | None) -> JointState | None:
        """Trim a padded interpolated trajectory at ``last_tstep`` (treated as inclusive)."""
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
        """Concatenate phase JointStates along the horizon dim."""
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
        """Return the planned trajectory as a list of 4x4 EEF pose matrices [m, rad]."""
        if self._planned_joint_trajectory is None:
            return []
        return self._joint_trajectory_to_eef_poses(self._planned_joint_trajectory)

    # ------------------------------------------------------------------
    # Surface required by :class:`MotionPlannerBase`
    # ------------------------------------------------------------------

    def has_next_waypoint(self) -> bool:
        if self._planned_joint_trajectory is None or self._planned_joint_trajectory.position is None:
            return False
        return self._waypoint_index < self._planned_joint_trajectory.position.shape[-2]

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        assert self.has_next_waypoint(), "No more waypoints in the current plan."
        pose = self._waypoint_eef_pose(self._planned_joint_trajectory, self._waypoint_index)
        self._waypoint_index += 1
        return pose

    def reset_plan(self) -> None:
        self._waypoint_index = 0

    # ------------------------------------------------------------------
    # World / scene sync
    # ------------------------------------------------------------------

    def _register_scene_obstacles(self) -> None:
        """One-time scene registration at construction.

        Builds a :class:`Scene` from the cfg's static cuboids plus the *current* dynamic
        obstacle poses, then calls :meth:`MotionPlanner.update_world` once. After this, the
        scene-collision instance is stable: subsequent plan calls only push per-obstacle pose
        updates via :meth:`_sync_dynamic_obstacle_poses` (mirrors v1's update cadence and
        avoids wiping attachment / enable state between plans).
        """
        if getattr(self.motion_planner, "scene_collision_checker", None) is None:
            self._LOGGER.warning(
                "MotionPlanner has no scene_collision_checker; skipping world registration. "
                "Ensure CuroboV2PlannerCfg.collision_cache (or scene_model) is set."
            )
            return

        cuboids: list[Cuboid] = []
        registered_dynamic: list[str] = []

        # Static obstacles from cfg.
        for name, dims, pose in self.config.static_cuboids:
            cuboids.append(Cuboid(name=name, dims=list(dims), pose=list(pose)))

        # Dynamic obstacles: read each one's pose now so the registered Scene is consistent
        # with the live state at init time. Subsequent calls only refresh poses, not dims.
        for name, dims in self.config.dynamic_object_dims.items():
            pose = self._read_object_pose_wxyz(name)
            if pose is None:
                self._LOGGER.warning(
                    "Dynamic object %r not found in scene at init; obstacle not registered.",
                    name,
                )
                continue
            cuboids.append(Cuboid(name=name, dims=list(dims), pose=pose))
            registered_dynamic.append(name)

        scene_cfg = Scene(cuboid=cuboids) if cuboids else Scene()
        self.motion_planner.update_world(scene_cfg)
        self._registered_dynamic_objects = registered_dynamic
        # Print the pose we just registered for each obstacle so the user can sanity-check
        # against the Isaac Lab scene without instrumenting the run.
        for c in cuboids:
            print(
                f"[CuroboV2Planner] registered obstacle {c.name:<12s} "
                f"dims={[round(v, 4) for v in c.dims]} "
                f"pose(xyz,wxyz)={[round(v, 4) for v in c.pose]}",
                flush=True,
            )
        print(
            f"[CuroboV2Planner] Registered {len(cuboids)} obstacles "
            f"({len(self.config.static_cuboids)} static + {len(registered_dynamic)} dynamic) "
            "with the collision checker.",
            flush=True,
        )
        # Flag so we only readback-verify on the first sync to keep logs quiet.
        self._verified_obstacle_sync: bool = False

    def _sync_dynamic_obstacle_poses(self) -> None:
        """Per-plan-call: push the current pose of every dynamic obstacle to the checker.

        Uses :meth:`SceneCollisionChecker.update_obstacle_pose` per-obstacle so the
        scene-collision instance stays stable (preserves any prior ``enable_obstacle`` flags
        set by :class:`AttachmentManager`).
        """
        checker = getattr(self.motion_planner, "scene_collision_checker", None)
        if checker is None:
            return

        verify = not getattr(self, "_verified_obstacle_sync", True)
        for name in self._registered_dynamic_objects:
            pose_wxyz = self._read_object_pose_wxyz(name)
            if pose_wxyz is None:
                continue
            position = torch.tensor([pose_wxyz[:3]])
            quaternion = torch.tensor([pose_wxyz[3:]])
            checker.update_obstacle_pose(
                name=name,
                w_obj_pose=self._make_pose(position_xyz=position, quaternion_wxyz=quaternion),
                env_idx=self.env_id,
            )
            if verify:
                print(
                    f"[CuroboV2Planner] first-sync {name:<12s} "
                    f"xyz=({pose_wxyz[0]:+.4f},{pose_wxyz[1]:+.4f},{pose_wxyz[2]:+.4f}) "
                    f"wxyz=({pose_wxyz[3]:+.4f},{pose_wxyz[4]:+.4f},{pose_wxyz[5]:+.4f},{pose_wxyz[6]:+.4f})",
                    flush=True,
                )
        if verify:
            self._verified_obstacle_sync = True

    def _read_object_pose_wxyz(self, name: str) -> list[float] | None:
        """Read a rigid object's world pose from the env and return it as ``[x, y, z, qw, qx, qy, qz]``.

        Returns ``None`` if the named entity isn't in the scene.
        """
        scene = self.env.scene
        if name not in (scene.keys() if hasattr(scene, "keys") else []):
            return None
        obj = scene[name]
        pos = _as_torch(obj.data.root_pos_w)[self.env_id]
        quat_xyzw = _as_torch(obj.data.root_quat_w)[self.env_id]
        # Convert obs convention (xyzw) → cuRobo pose convention (xyz, wxyz).
        quat_wxyz = torch.roll(quat_xyzw, shifts=1, dims=-1)
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ]

    # ------------------------------------------------------------------
    # Attachment / detachment
    # ------------------------------------------------------------------

    def _attachment_manager(self):
        """Resolve the v2 :class:`AttachmentManager` instance.

        The convenience property :attr:`MotionPlanner.attachment_manager` in this v2 release
        looks it up at ``trajopt_solver.attachment_manager``, but the manager actually lives
        on ``trajopt_solver.core.attachment_manager``. We try the property first (so the code
        keeps working when upstream fixes the alias) and fall back to the canonical path.
        """
        try:
            mgr = self.motion_planner.attachment_manager  # may raise AttributeError upstream
            if mgr is not None:
                return mgr
        except AttributeError:
            pass
        return self.motion_planner.trajopt_solver.core.attachment_manager

    def _update_attachment(self, expected: str | None, current_state: JointState) -> None:
        """Bring the attachment manager in sync with ``expected``.

        * If ``expected`` matches the currently attached object: no-op.
        * If a different object is currently attached: detach first.
        * If ``expected`` is set: attach by name from the scene.
        """
        if expected == self._currently_attached:
            return

        if self._currently_attached is not None:
            self._attachment_manager().detach(link_name=self.config.attached_object_link_name)
            self._currently_attached = None

        if expected is None:
            return

        if expected not in self.config.dynamic_object_dims:
            self._LOGGER.warning(
                "Attachment requested for %r but it is not registered in dynamic_object_dims; "
                "planning without attachment.",
                expected,
            )
            return

        try:
            self._attachment_manager().attach_from_scene(
                joint_states=current_state,
                obstacle_names=[expected],
                link_name=self.config.attached_object_link_name,
                surface_radius=self.config.surface_sphere_radius,
                sphere_fit_type=self._resolve_sphere_fit_type(),
            )
            self._currently_attached = expected
        except Exception as exc:  # noqa: BLE001  (attachment is best-effort)
            self._LOGGER.warning(
                "attach_from_scene(%r) failed: %s — planning without attachment.",
                expected,
                exc,
            )
            self._currently_attached = None

    def _resolve_sphere_fit_type(self) -> SphereFitType:
        """Translate the cfg's string into a v2 :class:`SphereFitType` enum value."""
        name = (self.config.sphere_fit_type or "").upper()
        try:
            return SphereFitType[name]
        except KeyError:
            valid = [m.name for m in SphereFitType]
            self._LOGGER.warning(
                "Unknown sphere_fit_type %r; falling back to SURFACE. Valid names: %s",
                self.config.sphere_fit_type,
                valid,
            )
            return SphereFitType.SURFACE

    # ------------------------------------------------------------------
    # Joint state helpers
    # ------------------------------------------------------------------

    def _get_current_joint_state(self, env_id: int) -> JointState:
        """Read the current joint state and project it onto the planner's *active* joints.

        The output ``JointState`` carries exactly :attr:`_active_joint_names` joints — the
        planner's full joint_names minus any joints declared in ``lock_joints``. cuRobo v2's
        IK / trajopt solvers operate on the active DoF only, so feeding the full set (e.g.
        Franka's 9 joints including the two locked fingers) would fail a downstream concat
        with ``Expected size N but got size active_dof``.
        """
        joint_pos_isaac = _as_torch(self.robot.data.joint_pos)[env_id]
        position = (
            joint_pos_isaac.unsqueeze(0)
            .to(device=self.motion_planner.device_cfg.device, dtype=torch.float32)
            .contiguous()
        )
        state = JointState.from_position(position, joint_names=list(self.robot.data.joint_names))
        reordered = state.reorder(self._active_joint_names)
        # ``reorder`` may produce a permuted view; rebuild with a contiguous tensor so
        # downstream CUDA ops (which require element-stride contiguity) don't reject it.
        if not reordered.position.is_contiguous():
            reordered = JointState.from_position(reordered.position.contiguous(), joint_names=reordered.joint_names)
        return reordered

    # ------------------------------------------------------------------
    # Waypoint → EEF pose extraction
    # ------------------------------------------------------------------

    def _joint_trajectory_to_eef_poses(self, traj: JointState) -> list[torch.Tensor]:
        """Compute EEF poses for every waypoint in ``traj`` in one batched FK call.

        ``traj.position`` can carry extra batch / seed dimensions in v2; we flatten to
        ``[T, dof]`` so :meth:`MotionPlanner.compute_kinematics` sees a single
        ``[batch=1, horizon=T, dof]`` tensor. The returned :class:`ToolPose` is then
        unpacked into ``T`` 4x4 EEF pose matrices.
        """
        import isaaclab.utils.math as PoseUtils  # deferred so the module imports sim-free

        positions = traj.position
        full_dof = positions.shape[-1]
        flat = positions.reshape(-1, full_dof)  # [N, full_dof]

        # The trajectory carries the *full* DoF (active + locked passthrough). FK / the IK
        # pipeline operate on active DoF only, so wrap with the trajectory's own joint_names
        # and project down to :attr:`_active_joint_names` via reorder.
        traj_joint_names = traj.joint_names if traj.joint_names else self.motion_planner.joint_names
        js_position = flat.unsqueeze(0).to(self.motion_planner.device_cfg.device)
        full_state = JointState.from_position(js_position, joint_names=list(traj_joint_names))
        single_state = full_state.reorder(self._active_joint_names)
        if not single_state.position.is_contiguous():
            single_state = JointState.from_position(
                single_state.position.contiguous(), joint_names=single_state.joint_names
            )

        # ``compute_kinematics`` creates grad-tracking tensors; lift inference mode so
        # callers reading planned poses from inside the env loop don't crash.
        with torch.inference_mode(False), torch.enable_grad():
            kin_state = self.motion_planner.compute_kinematics(single_state)

        # v2 ``KinematicsState.tool_poses`` is a ``ToolPose`` with shape
        # ``[batch, horizon, num_links, 3/4]``. ``get_link_pose(link)`` returns a flat
        # ``Pose`` ``[B*H, 3/4]`` — exactly the per-step list we want.
        tool_frame = self.motion_planner.tool_frames[0]
        link_pose = kin_state.tool_poses.get_link_pose(tool_frame)
        positions_w = link_pose.position  # [T, 3]
        quaternions_wxyz = link_pose.quaternion  # [T, 4]
        quaternions_xyzw = torch.roll(quaternions_wxyz, shifts=-1, dims=-1)
        rotations = PoseUtils.matrix_from_quat(quaternions_xyzw)  # [T, 3, 3]
        return [PoseUtils.make_pose(positions_w[t], rotations[t]) for t in range(positions_w.shape[0])]

    def _waypoint_eef_pose(self, traj: JointState, t: int) -> torch.Tensor:
        """Return the EEF pose for waypoint ``t`` (used by the abstract base-class iterator)."""
        return self._joint_trajectory_to_eef_poses(traj)[t]
