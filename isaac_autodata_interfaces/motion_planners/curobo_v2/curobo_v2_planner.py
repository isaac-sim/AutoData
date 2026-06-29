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

Feature surface implemented here (parity with the v1 backend):

* Static collision geometry is extracted once from the live USD stage via cuRobo's
  :class:`UsdSceneParser`, scoped to the env subtree and expressed relative to the robot base.
  Geometry is read through the :class:`Datastream` facade — the planner never touches the env
  or robot articulation directly.
* Per-plan obstacle pose synchronization: each dynamic obstacle's current (env-relative) pose
  is read from :meth:`Datastream.get_object_poses` and pushed into the collision world via
  :meth:`SceneCollisionChecker.update_obstacle_pose`. Static obstacles are skipped.
* Object attachment / detachment via the v2 :class:`AttachmentManager`.
* Three-phase planning (retreat → approach → goal) mirroring the v1 contact flow.
* Quaternion convention bridge: Isaac Lab uses ``(x, y, z, w)``; cuRobo internals use
  ``(w, x, y, z)``. Conversion is localized to the cuRobo boundary.
* Active-vs-full DoF handling for robots with locked joints (e.g. Franka's fingers).

Plan visualization is not yet hooked up; :attr:`CuroboV2PlannerCfg.visualize_plan` is accepted
for API parity but unused.
"""

from __future__ import annotations

import contextlib
import logging
import numpy as np
import torch
from typing import TYPE_CHECKING, Any

# Sphere-fit enum + USD scene parser live under ``_src``; safe to import directly since the
# public ``curobo.motion_planner.MotionPlanner`` uses them natively.
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

    _LOGGER = logging.getLogger("CuroboV2Planner")

    # Each per-env planner owns its own single-world MotionPlanner (multi_env=False), so every
    # collision-world operation (registration, pose sync, attachment) targets env slot 0.
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

        self._LOGGER.info("Constructing cuRobo v2 MotionPlannerCfg for %r", config.robot_name)
        v2_cfg = MotionPlannerCfg.create(**config.to_v2_kwargs())

        self._LOGGER.info("Instantiating MotionPlanner; warming up...")
        self.motion_planner = MotionPlanner(v2_cfg)
        self.motion_planner.warmup()
        self._LOGGER.info("MotionPlanner ready. tool_frames=%s", self.motion_planner.tool_frames)

        # Per-plan state.
        self._planned_joint_trajectory: JointState | None = None
        self._waypoint_index: int = 0
        # TEMP DIAGNOSTIC: gate the one-shot collision-world dump (see _debug_dump_collision_world).
        self._debug_dumped: bool = False
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

        # Extract static collision geometry from the USD stage ONCE up front (mirrors v1).
        # ``_object_mapping`` maps Isaac scene object names -> cuRobo obstacle names; the
        # per-plan sync iterates it to push current poses. ``_static_object_substrings`` names
        # obstacles that should never be pose-synced (treated as fixed furniture).
        self._object_mapping: dict[str, str] = {}
        self._world_obstacle_names: list[str] = []
        self._static_object_substrings: list[str] = [s.lower() for s in self.config.static_objects]
        self._verified_obstacle_sync: bool = False
        # cuRobo name -> owning collision-data array (cuboids/meshes/voxels). Resolved lazily to
        # route obstacle pose/enable updates around an upstream dispatcher bug (see
        # :meth:`_obstacle_storage`). The cuRobo obstacle name of the currently attached object,
        # tracked so we can re-enable it on detach.
        self._obstacle_storage_cache: dict[str, Any] = {}
        self._attached_curobo_name: str | None = None
        self._initialize_static_world()

        # Optional Rerun plan visualization (debug aid; mirrors the v1 backend). Imported lazily so
        # non-visualized runs never require the rerun-sdk dependency. ``_robot_self_sphere_count``
        # caches the robot's own active-sphere count (captured when nothing is attached) so we can
        # peel off the attached-object spheres — which cuRobo appends last — for separate coloring.
        self.plan_visualizer: Any = None
        self._robot_self_sphere_count: int | None = None
        if self.config.visualize_plan:
            from isaac_autodata_interfaces.motion_planners.curobo_v2.plan_visualizer import PlanVisualizer

            self.plan_visualizer = PlanVisualizer(
                robot_name=self.config.robot_name or "robot",
                recording_id=f"curobo_v2_plan_{self.env_id}",
                save_path=f"curobo_v2_plan_env{self.env_id}.rrd",
                debug=getattr(self, "debug", False),
            )
            self.plan_visualizer.set_motion_planner_reference(self.motion_planner)

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
                the gripper during planning. Resolved against the obstacles discovered from the
                USD stage at construction. ``None`` releases any prior attachment.
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
            self._sync_obstacle_poses()

            # 2. Read current joint state in the planner's expected order.
            current_state = self._get_current_joint_state(env_id=env_id)

            # 3. Handle attachment state — attach the requested object, or detach any prior one.
            self._update_attachment(expected_attached_object, current_state)

            # TEMP DIAGNOSTIC: one-shot dump of fetched joint state + loaded collision geometry +
            # whether the current configuration reads as in-collision. Triages why every plan fails.
            # Remove once the v2 mesh-collision issue is resolved.
            if not self._debug_dumped:
                self._debug_dumped = True
                self._debug_dump_collision_world(current_state)

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
            if self.plan_visualizer is not None:
                with contextlib.suppress(Exception):
                    self.plan_visualizer.mark_idle()
            return False
        self._planned_joint_trajectory = full_trajectory
        self._waypoint_index = 0
        if self.plan_visualizer is not None:
            try:
                self._visualize_plan(target_pose=target_pose, current_state=current_state)
            except Exception as exc:  # noqa: BLE001  (viz must never break planning)
                self._LOGGER.warning("plan visualization failed: %s", exc)
        return True

    # ------------------------------------------------------------------
    # TEMP DIAGNOSTIC (remove once v2 mesh-collision is fixed)
    # ------------------------------------------------------------------

    def _debug_dump_collision_world(self, current_state: JointState) -> None:
        """One-shot triage dump for the "every plan fails" symptom.

        Prints three independent things so we can tell a fetch problem from a collision-world
        problem in a single run:

        1. The fetched active joint vector — proves whether sim/USD state reads are sane.
        2. Each loaded obstacle's geometry (mesh AABB ``dims`` + stored pose, or cuboid dims) —
           reveals wrong-sized / mis-placed obstacles (e.g. a 5 cm cube loaded as a 1 m mesh).
        3. Whether the *current* configuration is in collision against that world — if the home
           pose collides, the world is globally poisoned and no plan (not even a lift) can succeed.
        """
        mp = self.motion_planner
        q = [round(float(x), 4) for x in current_state.position.flatten().tolist()]
        print(f"[CVDBG] env={self.env_id} fetched active q ({len(q)} dof) = {q}", flush=True)
        print(f"[CVDBG] active_joint_names = {current_state.joint_names}", flush=True)

        checker = getattr(mp, "scene_collision_checker", None)
        if checker is None:
            print("[CVDBG] no scene_collision_checker on planner.", flush=True)
            return

        # (2) Obstacle geometry actually loaded into the collision world.
        try:
            data = checker.data
            meshes = getattr(data, "meshes", None)
            cuboids = getattr(data, "cuboids", None)
            e = self._COLLISION_ENV_IDX
            n_mesh = int(meshes.count[e].item()) if meshes is not None else 0
            n_cub = int(cuboids.count[e].item()) if cuboids is not None else 0
            print(f"[CVDBG] collision world: meshes={n_mesh} cuboids={n_cub}", flush=True)
            for i in range(n_mesh):
                name = meshes.names[e][i]
                dims = [round(float(x), 4) for x in meshes.dims[e, i, :3].tolist()]
                inv_t = [round(float(x), 4) for x in meshes.inv_pose[e, i, :3].tolist()]
                en = int(meshes.enable[e, i].item())
                print(f"[CVDBG]   mesh[{i}] name={name} enabled={en} AABB_dims={dims} inv_pose_t={inv_t}", flush=True)
            for i in range(n_cub):
                name = cuboids.names[e][i]
                dims = [round(float(x), 4) for x in cuboids.dims[e, i, :3].tolist()]
                print(f"[CVDBG]   cuboid[{i}] name={name} dims={dims}", flush=True)
            print(
                f"[CVDBG] checker.max_distance(SDF query) = {getattr(checker.checker, 'max_distance', '?')}", flush=True
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[CVDBG] obstacle-geometry dump FAILED: {exc!r}\n{traceback.format_exc()}", flush=True)

        # (3) Is the *current* configuration in collision against the loaded world?
        try:
            from curobo._src.geom.collision.buffer_collision import CollisionBuffer

            dcfg = checker.device_cfg
            pos = current_state.position
            if pos.ndim == 2:  # [B, dof] -> [B, 1, dof] so compute_kinematics gets a horizon dim.
                pos = pos.unsqueeze(1)
            js = JointState.from_position(pos.contiguous(), joint_names=current_state.joint_names)
            kin = mp.compute_kinematics(js)
            spheres = kin.robot_spheres
            act_val = float(self.config.optimizer_collision_activation_distance)
            weight = torch.tensor([1.0], device=dcfg.device)
            act = torch.tensor([act_val], device=dcfg.device)
            buf = CollisionBuffer.from_shape(spheres.shape, dcfg)
            coll = checker.get_sphere_collision(
                state=kin, collision_buffer=buf, weight=weight, activation_distance=act, env_query_idx=None
            )
            buf.zero_()
            dist = checker.get_sphere_distance(
                state=kin, collision_buffer=buf, weight=weight, activation_distance=act, env_query_idx=None
            )
            n_spheres = spheres.shape[-2]
            n_hit = int((coll.flatten().abs() > 1e-6).sum().item())
            print(
                f"[CVDBG] home-state collision (act={act_val}): n_spheres={n_spheres} "
                f"in_collision={n_hit} max_coll_cost={float(coll.max().item()):.4f} "
                f"max_dist_cost={float(dist.max().item()):.4f}",
                flush=True,
            )
            print("[CVDBG] >> if in_collision is high for the HOME pose, the mesh world is poisoned.", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[CVDBG] collision query FAILED: {exc!r}\n{traceback.format_exc()}", flush=True)

    # ------------------------------------------------------------------
    # Plan visualization (Rerun; optional, gated by config.visualize_plan)
    # ------------------------------------------------------------------

    def _visualize_plan(self, target_pose: torch.Tensor, current_state: JointState) -> None:
        """Push the current plan to the Rerun visualizer: EE path, target, and split spheres.

        Computes the robot's collision spheres at ``current_state`` and splits them into robot-self
        vs attached-object spheres (the attached ones cuRobo appends last). The end-effector path
        comes from the just-stored plan, and the obstacle scene from the live collision world.
        """
        with torch.inference_mode(False), torch.enable_grad():
            active_q = current_state.position
            if active_q.ndim == 1:
                active_q = active_q.unsqueeze(0)
            # Split robot-self vs attached-object spheres by the attached link's own sphere indices
            # (ground truth), not a positional heuristic. ``filter_valid=False`` keeps absolute
            # indices so they line up with the link index map; we drop inactive (radius<=0) slots.
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
            planned_poses = self.get_planned_poses()
            if planned_poses:
                ee_positions = np.array([p.detach().cpu().numpy().reshape(4, 4)[:3, 3] for p in planned_poses])
            plan_active = self._planned_trajectory_active()
            world_scene = self._build_world_trimesh_scene()

        self.plan_visualizer.visualize_plan(
            plan=plan_active,
            target_pose=target_pose,
            robot_spheres=robot_spheres,
            attached_spheres=attached_spheres,
            ee_positions=ee_positions,
            world_scene=world_scene,
        )
        if ee_positions is not None:
            self.plan_visualizer.animate_plan(ee_positions)
        self.plan_visualizer.animate_spheres_along_path(plan=plan_active, robot_sphere_count=len(robot_spheres))

    def _planned_trajectory_active(self) -> JointState:
        """Return the stored plan reduced to active-DoF positions (for sphere animation)."""
        traj = self._planned_joint_trajectory
        full_dof = traj.position.shape[-1]
        flat = traj.position.reshape(-1, full_dof).contiguous()
        names = list(traj.joint_names) if traj.joint_names else list(self.motion_planner.joint_names)
        active = JointState.from_position(flat, joint_names=names).reorder(self._active_joint_names)
        if not active.position.is_contiguous():
            active = JointState.from_position(active.position.contiguous(), joint_names=active.joint_names)
        return active

    def _build_world_trimesh_scene(self):
        """``trimesh.Scene`` of the real obstacle geometry the planner collides against.

        Geometry (vertices/faces) comes from the cuRobo scene model — the *actual* triangle meshes,
        so concave shapes (e.g. a hollow sorting bin) render with their true geometry, matching the
        v1 backend instead of a bounding box. Each mesh is placed at its current world pose,
        recovered from the live collision data (``inv_pose``-inverse, kept in sync each plan), so
        dynamic cubes track their motion. Local vertices + current pose reproduce exactly the world
        geometry cuRobo collides against (its warp mesh is the same local verts + ``inv_pose``).
        ``Cuboid`` obstacles, if any, fall back to exact boxes. Returns ``None`` on any failure so
        visualization never breaks planning.
        """
        try:
            import trimesh

            checker = getattr(self.motion_planner, "scene_collision_checker", None)
            scene_model = getattr(checker, "scene_model", None) if checker is not None else None
            data = getattr(checker, "data", None) if checker is not None else None
            if data is None:
                return None

            # Current world pose + AABB dims per obstacle name, from the synced collision data.
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
            # Real mesh obstacles (hollow bin, cubes): local vertices/faces from the scene model,
            # placed at the current synced world pose.
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
            # Cuboid obstacles (rare in mesh mode): exact boxes from their AABB dims.
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
        hand_links: list[str] = []
        saved_attached = None
        if toggled:
            attach_link = self.config.attached_object_link_name
            # Hand links: cuRobo's reference-based enable/disable is correct (their reference IS the
            # robot config). The attached_object link is dynamic — cuRobo's ``enable_link_spheres``
            # restores radii from the reference config (where attached_object has no spheres), which
            # would WIPE the fitted attachment after this phase. So we save/disable/restore that
            # link's spheres ourselves, keeping the held object alive across all phases.
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
            print(f"[CVDBG] phase={name} FAIL: plan_pose returned no result.", flush=True)
            return None
        if not bool(result.success.any().item()):
            print(
                f"[CVDBG] phase={name} FAIL: all seeds failed. "
                f"success={getattr(result, 'success', None)} status={getattr(result, 'status', None)}",
                flush=True,
            )
            return None

        traj = result.interpolated_trajectory
        if traj is None:
            traj = result.js_solution
        if traj is None or traj.position is None:
            print(f"[CVDBG] phase={name} FAIL: plan_pose succeeded but trajectory was empty.", flush=True)
            return None

        last_tstep = self._extract_last_tstep_from(getattr(result, "interpolated_last_tstep", None))
        trimmed = self._trim_inclusive(traj, last_tstep)
        if trimmed is None or trimmed.position is None or trimmed.position.shape[-2] == 0:
            return None
        return trimmed

    def _save_disable_attached_spheres(self, link_name: str):
        """Save the attached link's spheres, then disable them by zeroing radii (no fit lost).

        Returns ``(sphere_index, saved_spheres)`` for :meth:`_restore_attached_spheres`. Used
        instead of cuRobo's ``enable/disable_link_collision`` for the dynamic attached link, whose
        ``enable_link_spheres`` would restore radii from the (empty) reference config and wipe the
        attachment.
        """
        kp = self._attachment_manager().kinematics_params
        idx = kp.get_sphere_index_from_link_name(link_name)
        saved = kp.link_spheres[:, idx, :].clone()
        kp.link_spheres[:, idx, 3] = -100.0
        return idx, saved

    def _restore_attached_spheres(self, idx, saved) -> None:
        """Restore the exact attached-link spheres saved by :meth:`_save_disable_attached_spheres`."""
        self._attachment_manager().kinematics_params.link_spheres[:, idx, :] = saved

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

    def _initialize_static_world(self) -> None:
        """Extract static collision geometry from the USD stage once, at construction.

        Mirrors the v1 backend: cuRobo's :class:`UsdSceneParser` traverses the env subtree
        (``only_paths=[env_prim]``), expresses everything relative to the robot base
        (``reference_prim_path=robot_prim``), and drops the ignore-listed prims (robot, ground
        plane, cuRobo debug prims). All stage / prim-path access goes through the Datastream so
        the planner stays agnostic to the simulator wiring.

        The resulting collision world is loaded once; subsequent plans only push per-obstacle
        pose updates via :meth:`_sync_obstacle_poses`, keeping the scene-collision instance
        stable (so attachment / obstacle-disable state survives between plans).
        """
        if getattr(self.motion_planner, "scene_collision_checker", None) is None:
            self._LOGGER.warning(
                "MotionPlanner has no scene_collision_checker; skipping world registration. "
                "Ensure CuroboV2PlannerCfg.collision_cache (or scene_model) is set."
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
        # Choose the collision representation. ``"mesh"`` keeps the triangulated USD geometry —
        # parity with the cuRobo v1 interface — so concave objects (e.g. a sorting bin) are
        # represented faithfully and the gripper can reach inside; it is the principled default.
        # ``"obb"`` converts each obstacle to an exact oriented bounding box (analytic SDF, no
        # mesh query): exact for boxes and slightly cheaper, kept as a fallback for box-only
        # scenes or debugging. The mesh path relies on the cuRobo v2 mesh-collision fix (PR #682,
        # >= 0.8.0.post1.dev34), which floors the per-query mesh search distance by the query
        # sphere's radius; on earlier builds the distance was capped at ``||AABB|| * 0.5`` and
        # produced false-positive collisions for spheres larger than that cap.
        if self.config.obstacle_representation == "obb":
            scene = self._scene_as_cuboids(scene)
        else:
            scene = scene.get_collision_check_world()
        self.motion_planner.update_world(scene)

        # Each per-env planner owns a single-world MotionPlanner (multi_env=False), so every
        # collision operation targets env slot 0 within this planner; ``self.env_id`` selects
        # which Isaac env to *read* state from, not a cuRobo collision slot.
        checker = self.motion_planner.scene_collision_checker
        self._world_obstacle_names = list(checker.get_obstacle_names(env_idx=self._COLLISION_ENV_IDX))
        self._object_mapping = self._discover_object_mapping(self._world_obstacle_names)

        print(
            f"[CuroboV2Planner] env={self.env_id} extracted {len(self._world_obstacle_names)} "
            f"obstacles from {env_prim} (ref={robot_prim}): {self._world_obstacle_names}",
            flush=True,
        )
        dynamic = {k: v for k, v in self._object_mapping.items() if not self._is_static_object(k)}
        static = {k: v for k, v in self._object_mapping.items() if self._is_static_object(k)}
        print(
            f"[CuroboV2Planner] env={self.env_id} dynamic (pose-synced) objects: {list(dynamic)}; "
            f"static (fixed) objects: {list(static)}",
            flush=True,
        )

    def _scene_as_cuboids(self, scene) -> Scene:
        """Return a cuboid-only :class:`Scene`, representing each extracted mesh by its OBB.

        cuRobo v2 cuboid (OBB) collision is exact for boxes, fast, and — crucially — robust where
        v2's mesh path is not (its winding-based mesh-SDF sign test yields false positives on the
        USD-extracted box meshes). Real :class:`Cuboid` obstacles pass through unchanged; each
        :class:`Mesh` becomes a cuboid sized to its vertex axis-aligned bounding box, posed at the
        box center in the mesh's frame. For a true box this is exact; for any other mesh it is a
        conservative enclosure (collision-safe). Genuinely concave geometry (e.g. a bin) is *not*
        served well by a single OBB and should instead use ``obstacle_representation="mesh"`` with
        a watertight mesh (or be convex-decomposed) — see :class:`CuroboV2PlannerCfg`.
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
            # Compose the mesh's frame with the local box-center offset to get the cuboid pose.
            mesh_pose = Pose.from_list(list(mesh.pose), self.motion_planner.device_cfg)
            center_offset = self._make_pose(position_xyz=torch.tensor([center_local]))
            cuboid_pose = mesh_pose.multiply(center_offset).tolist()
            cuboids.append(Cuboid(name=str(mesh.name), dims=dims, pose=cuboid_pose))
        return Scene(cuboid=cuboids)

    def _is_static_object(self, name: str) -> bool:
        """True if ``name`` matches a configured static-obstacle substring (never pose-synced)."""
        return any(s in name.lower() for s in self._static_object_substrings)

    def _discover_object_mapping(self, world_obstacle_names: list[str]) -> dict[str, str]:
        """Map live scene object names -> cuRobo obstacle names by normalized substring match.

        cuRobo names obstacles by their USD prim path (e.g. ``/World/envs/env_0/Cube_1/...``)
        while the Datastream keys objects by short name (``cube_1``). Mirrors the v1 discovery.
        Only objects that appear in :meth:`Datastream.get_object_poses` (i.e. live rigid bodies)
        are mappable; static furniture extracted from USD has no entry and is never pose-synced.
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
        """Per-plan-call: push the current (env-relative) pose of every dynamic obstacle.

        Reads poses through :meth:`Datastream.get_object_poses` (env-origin-relative 4x4
        matrices — the robot-base frame for a fixed-base robot at the env origin) and updates
        each mapped obstacle via :meth:`SceneCollisionChecker.update_obstacle_pose`. Obstacles
        whose name matches :attr:`_static_object_substrings` are skipped.
        """
        checker = getattr(self.motion_planner, "scene_collision_checker", None)
        if checker is None or not self._object_mapping:
            return

        import isaaclab.utils.math as PoseUtils  # deferred so the module imports sim-free

        object_poses = self.datastream.get_object_poses(env_ids=[self.env_id])
        verify = not self._verified_obstacle_sync
        for obj_name, curobo_name in self._object_mapping.items():
            if any(s in obj_name.lower() for s in self._static_object_substrings):
                continue
            pose_mat = object_poses.get(obj_name)
            if pose_mat is None:
                continue
            pos_xyz, rot_mat = PoseUtils.unmake_pose(pose_mat[0])
            quat_xyzw = PoseUtils.quat_from_matrix(rot_mat)
            quat_wxyz = torch.roll(quat_xyzw, shifts=1, dims=-1)
            self._set_obstacle_pose(
                curobo_name,
                self._make_pose(
                    position_xyz=pos_xyz.unsqueeze(0),
                    quaternion_wxyz=quat_wxyz.unsqueeze(0),
                ),
            )
            if verify:
                p = pos_xyz.tolist()
                print(
                    f"[CuroboV2Planner] env={self.env_id} first-sync {obj_name:<10s} "
                    f"-> {curobo_name.split('/')[-1]}  xyz=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})",
                    flush=True,
                )
        self._verified_obstacle_sync = True

    # ------------------------------------------------------------------
    # Obstacle update routing (works around an upstream cuRobo dispatcher bug)
    # ------------------------------------------------------------------
    # ``SceneCollisionChecker.update_obstacle_pose`` / ``enable_obstacle`` probe the cuboid list
    # first; its index lookup raises ``ValueError`` (which the dispatcher does not catch) before
    # ever reaching the mesh list, so a *mesh* obstacle (e.g. an Isaac cube spawned as a USD Mesh)
    # can never be updated through the public API. We resolve the owning collision-data array once
    # and call its typed ``update_pose`` / ``set_enabled`` directly.

    def _obstacle_storage(self, curobo_name: str):
        """Return the collision-data array (cuboids/meshes/voxels) that owns ``curobo_name``."""
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
        """Update one obstacle's pose, routed to its owning collision-data array."""
        arr = self._obstacle_storage(curobo_name)
        if arr is None:
            self._LOGGER.warning("Obstacle %r not found in collision world; skipping pose sync.", curobo_name)
            return
        arr.update_pose(curobo_name, w_obj_pose=w_obj_pose, env_idx=self._COLLISION_ENV_IDX)

    def _set_obstacle_enabled(self, curobo_name: str, enabled: bool) -> None:
        """Enable/disable one obstacle for collision, routed to its owning collision-data array."""
        arr = self._obstacle_storage(curobo_name)
        if arr is None:
            self._LOGGER.warning("Obstacle %r not found in collision world; skipping enable=%s.", curobo_name, enabled)
            return
        arr.set_enabled(curobo_name, enabled, self._COLLISION_ENV_IDX)

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

        # Detach any prior object: reset its link spheres and re-enable its world collision.
        # ``enable_obstacle_names=None`` skips cuRobo's own re-enable path (which hits the mesh
        # dispatcher bug); we re-enable via the type-correct routing instead.
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
            self._LOGGER.warning(
                "Attachment requested for %r but it was not discovered in the extracted "
                "collision world; planning without attachment.",
                expected,
            )
            return

        checker = self.motion_planner.scene_collision_checker
        obstacle = checker.scene_model.get_obstacle(curobo_name) if checker.scene_model is not None else None
        if obstacle is None:
            self._LOGGER.warning(
                "Attached object %r (%s) not found in scene_model; planning without attachment.",
                expected,
                curobo_name,
            )
            return

        # Place the attached object at its CURRENT world pose so cuRobo collision-checks the held
        # object where it actually is. Two cuRobo behaviors interact here:
        #   * ``fit_spheres`` bakes the obstacle's own pose into the fitted sphere centers
        #     (``_obstacles_to_trimesh`` -> ``get_trimesh_mesh(transform_with_pose=True)``), and
        #   * ``attach`` then applies ``world_objects_pose_offset`` on top.
        # Our scene_model obstacle carries only its stale load-time pose (``_sync_obstacle_poses``
        # updates the collision arrays, not scene_model). So we pass
        # ``offset = current_pose · inv(obstacle_pose)``, which cancels the baked stale pose and
        # lands the spheres exactly at the object's current world pose. (v1 sidesteps this by
        # attaching the object *by name*, letting cuRobo read the live pose from the world model.)
        world_pose_offset = None
        current_world_pose = self._object_world_pose(expected)
        if current_world_pose is not None:
            obstacle_pose = Pose.from_list(list(obstacle.pose), self.motion_planner.device_cfg)
            world_pose_offset = current_world_pose.multiply(obstacle_pose.inverse())

        try:
            # Use attach(...) with ``disable_obstacle_names=None`` rather than attach_from_scene:
            # the latter's auto-disable hits the same mesh-dispatcher bug. We disable the carried
            # obstacle's world collision ourselves via the type-correct routing below, so the held
            # cube doesn't double-count as both an attached sphere set and a world obstacle.
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
            self._verify_attachment(expected, current_state, current_world_pose)
        except Exception as exc:  # noqa: BLE001  (attachment is best-effort)
            self._LOGGER.warning(
                "attach(%r) failed: %s — planning without attachment.",
                expected,
                exc,
            )
            self._currently_attached = None
            self._attached_curobo_name = None

    def _object_world_pose(self, obj_name: str) -> Pose | None:
        """Current world (robot-base-frame) pose of a live scene object as a cuRobo :class:`Pose`."""
        import isaaclab.utils.math as PoseUtils  # deferred so the module imports sim-free

        pose_mat = self.datastream.get_object_poses(env_ids=[self.env_id]).get(obj_name)
        if pose_mat is None:
            return None
        pos_xyz, rot_mat = PoseUtils.unmake_pose(pose_mat[0])
        quat_xyzw = PoseUtils.quat_from_matrix(rot_mat)
        quat_wxyz = torch.roll(quat_xyzw, shifts=1, dims=-1)
        return self._make_pose(position_xyz=pos_xyz.unsqueeze(0), quaternion_wxyz=quat_wxyz.unsqueeze(0))

    def _verify_attachment(self, expected: str, current_state: JointState, object_world_pose: Pose | None) -> None:
        """Ground-truth check of what cuRobo *actually* sees for the attached object.

        Independent of any visualization: reads the attached-link collision spheres straight from
        cuRobo's kinematics (FK at the grasp state, via :meth:`get_robot_as_spheres`) using the
        link's own sphere indices, then compares their world centroid to the gripper (EE) and to
        the object's world pose. Centroid at the gripper/object ⇒ cuRobo collision-checks the held
        object in the right place; centroid far away (e.g. near the base) ⇒ the attach offset is
        wrong. Best-effort; never raises into the planning path.
        """
        try:
            import numpy as _np

            with torch.inference_mode(False), torch.enable_grad():
                pos = current_state.position
                if pos.ndim == 1:
                    pos = pos.unsqueeze(0)
                ee_xyz = _np.asarray(
                    self._eef_pose_from_state(current_state).position.flatten()[:3].detach().cpu().tolist()
                )
                spheres = self.motion_planner.kinematics.get_robot_as_spheres(pos.contiguous(), filter_valid=False)[0]
                idx = self._attachment_manager().kinematics_params.get_sphere_index_from_link_name(
                    self.config.attached_object_link_name
                )
                idx_list = [int(j) for j in idx.detach().cpu().tolist()]
            att = [
                _np.asarray(spheres[j].position, dtype=float).reshape(-1)
                for j in idx_list
                if 0 <= j < len(spheres) and float(spheres[j].radius) > 0.0
            ]
            if not att:
                print(f"[ATTACH-CHK] env={self.env_id} {expected!r}: NO active attached spheres in cuRobo.", flush=True)
                return
            cen = _np.mean(att, axis=0)
            d_ee = float(_np.linalg.norm(cen - ee_xyz))
            obj = (
                _np.asarray(object_world_pose.position.flatten()[:3].detach().cpu().tolist())
                if object_world_pose is not None
                else None
            )
            obj_str = "(%+.3f,%+.3f,%+.3f)" % tuple(obj) if obj is not None else "n/a"
            print(
                f"[ATTACH-CHK] env={self.env_id} {expected!r}: {len(att)} attached spheres | "
                f"centroid=({cen[0]:+.3f},{cen[1]:+.3f},{cen[2]:+.3f}) "
                f"EE=({ee_xyz[0]:+.3f},{ee_xyz[1]:+.3f},{ee_xyz[2]:+.3f}) obj={obj_str} | "
                f"centroid->EE={d_ee:.3f}m  {'OK (held at gripper)' if d_ee < 0.2 else 'FAR — attach offset WRONG'}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ATTACH-CHK] env={self.env_id} verification failed: {exc!r}", flush=True)

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
        joint_pos_isaac = self.datastream.get_robot_joint_positions(env_ids=[env_id])[0]
        position = (
            joint_pos_isaac.unsqueeze(0)
            .to(device=self.motion_planner.device_cfg.device, dtype=torch.float32)
            .contiguous()
        )
        state = JointState.from_position(position, joint_names=self.datastream.get_robot_joint_names())
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
