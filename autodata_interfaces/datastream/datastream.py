# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import torch
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from isaaclab.utils.datasets import EpisodeData

from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_interfaces.embodiments.embodiment_types import ActionTargetFrame, NamedJointTrajectory
from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from autodata_interfaces.tasks.subtask_constraint_spec import SubtaskConstraint
from autodata_interfaces.tasks.subtask_spec import Subtask, SubtaskAlgoParams
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from autodata_utils import pose_math
from autodata_utils.tensor_utils import as_torch


def scene_object_poses(
    scene: Any,
    index: slice | Sequence[int],
    robot_asset_name: str,
    reference_inverse: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Non-robot rigid and articulated object poses (SE(3)) read live from ``scene``.

    The single implementation shared by :meth:`Datastream.get_object_poses` and
    :mod:`autodata_retargeting`'s per-step datagen-info recorder, which need the same live-scene
    reading (env-local root poses, optionally expressed in an embodiment's reference frame) and must
    not be allowed to drift apart again (see git history of this function).

    Args:
        scene: The env's ``InteractiveScene``.
        index: Environment selector into each asset's batched root-pose ``(num_envs, ...)`` tensors
            -- a slice (``slice(None)`` for every env) or a sequence of env indices.
        robot_asset_name: Scene key of the robot articulation, excluded from the result.
        reference_inverse: Optional inverse reference-frame pose(s), left-multiplied into the result;
            ``None`` leaves poses in the scene's env-local frame.

    Returns:
        ``{object_name: (*, 4, 4)}`` SE(3) poses, batch shape matching ``index``.
    """
    env_origins = as_torch(scene.env_origins)[index]
    assets = {**scene.rigid_objects, **getattr(scene, "articulations", {})}
    poses: dict[str, torch.Tensor] = {}
    for name, obj in assets.items():
        if name == robot_asset_name:
            continue
        pos_rel = as_torch(obj.data.root_pos_w)[index] - env_origins
        quat = as_torch(obj.data.root_quat_w)[index]
        pose = pose_math.make_pose(pos_rel, pose_math.matrix_from_quat(quat))
        poses[name] = reference_inverse @ pose if reference_inverse is not None else pose
    return poses


class Datastream:
    """Aggregates the env, task descriptor, embodiment adapter, and DataGenInfoPool
    behind one interface. The data generator uses this interface to access all static
    and runtime information needed.
    """

    def __init__(
        self,
        env: Any,
        task_descriptor: TaskDescriptor,
        embodiment_adapter: EmbodimentAdapter,
        source_pool: DataGenInfoPool | None = None,
        source_dataset_path: str | None = None,
        select_demo_keys: list[str] | None = None,
        asyncio_lock: asyncio.Lock | None = None,
        uses_start_signals: bool = False,
    ) -> None:
        """Compose the three sources and build (or adopt) the source-demo pool.

        Args:
            env: Simulation environment.
            task_descriptor: Task semantics (subtask order, signals, object refs, algo params).
            embodiment_adapter: Robot kinematics adapter.
            source_pool: A pre-built DataGenInfoPool to use. Mutually exclusive
                with ``source_dataset_path``.
            source_dataset_path: HDF5 path to load source demos from. Mutually exclusive with
                ``source_pool``.
            asyncio_lock: (Optional) Lock guarding concurrent pool growth across async env tasks.
                Pass ``None`` (the default) for static pools that are fully loaded before
                generation starts (no locking overhead is incurred).
            select_demo_keys: (Optional) Subset of episode keys to load when reading from HDF5.
            uses_start_signals: (For SkillGen) Whether the algorithm reads subtask start signals.
        """
        # TODO: Decide on if uses_start_signals parameter should be refactored. Keeping it explicit for now as it's
        # needed for DataGenInfoPool construction.

        self.env = env
        self.task_descriptor = task_descriptor
        self.embodiment_adapter = embodiment_adapter

        # Ensure same env is bound to the task descriptor and embodiment adapter.
        task_env = getattr(task_descriptor, "env", None)
        if task_env is None:
            task_descriptor.bind_env(env)
        else:
            assert task_env is env, (
                "The task descriptor is bound to a different env than the Datastream. "
                "Either use the same env for the task descriptor and the Datastream, or leave "
                "the task descriptor env unset before creating the Datastream."
            )
        embodiment_adapter_env = getattr(embodiment_adapter, "env", None)
        if embodiment_adapter_env is None:
            embodiment_adapter.bind_env(env)
        else:
            assert embodiment_adapter_env is env, (
                "The embodiment adapter is bound to a different env than the Datastream. "
                "Either use the same env for the embodiment adapter and the Datastream, or leave "
                "the embodiment adapter env unset before creating the Datastream."
            )

        task_descriptor.validate_embodiment(embodiment_adapter)

        # Initialize the source-demo pool.
        assert (source_pool is None) != (
            source_dataset_path is None
        ), "Must provide exactly one of source_pool or source_dataset_path."
        if source_pool is not None:
            self._pool = source_pool
        else:
            self._pool = DataGenInfoPool.from_hdf5(
                file_path=source_dataset_path,
                task_descriptor=task_descriptor,
                embodiment_adapter=embodiment_adapter,
                device=env.device,
                uses_start_signals=uses_start_signals,
                select_demo_keys=select_demo_keys,
                asyncio_lock=asyncio_lock,
            )

    @property
    def device(self) -> torch.device:
        """Compute device used by the environment."""

        return self.env.device

    def get_env(self) -> Any:
        """Return the environment.

        Catch-all for algorithms that need access to specific env state not included in the
        task descriptor or embodiment adapter interfaces. E.g. SkillGen's expected-attached-object computation.
        """

        return self.env

    def get_eef_names(self) -> list[str]:
        """Return the ordered end-effector names declared by the task."""

        return self.task_descriptor.get_eef_names()

    # ------------------------------------------------------------------
    # Task queries
    # ------------------------------------------------------------------

    def get_subtasks(self, eef_name: str) -> list[Subtask]:
        """Get the subtask list for the given EEF."""

        return self.task_descriptor.get_subtasks(eef_name)

    def get_subtask(self, eef_name: str, subtask_index: int) -> Subtask:
        """Get the individual subtask at the given index for the given EEF."""

        return self.task_descriptor.get_subtasks(eef_name)[subtask_index]

    def num_subtasks(self, eef_name: str) -> int:
        """Return the number of subtasks declared for the EEF."""

        return len(self.task_descriptor.get_subtasks(eef_name))

    def get_task_constraints(self) -> list[SubtaskConstraint]:
        """Return the tasks's cross-subtask constraints."""

        return self.task_descriptor.get_task_constraints()

    def get_generation_policy(self) -> GenerationPolicy:
        """Return the cross-cutting generation flags (source-demo selection, interpolation seeding)."""

        return self.task_descriptor.get_generation_policy()

    def get_object_refs(self, eef_name: str) -> list[str]:
        """Return per-subtask object reference names for the EEF, in subtask order."""

        return self.task_descriptor.get_object_refs(eef_name)

    def get_term_signal_names(self, eef_name: str) -> list[str]:
        """Return per-subtask termination-signal names for the EEF, in subtask order."""

        return self.task_descriptor.get_term_signal_names(eef_name)

    def get_start_signal_names(self, eef_name: str) -> list[str]:
        """Return per-subtask start-signal names for the EEF, in subtask order."""

        return self.task_descriptor.get_start_signal_names(eef_name)

    def get_subtask_algo_params(self, eef_name: str) -> list[SubtaskAlgoParams]:
        """Return per-subtask algorithm parameters for the EEF, in subtask order."""

        return self.task_descriptor.get_subtask_algo_params(eef_name)

    def get_subtask_descriptions(self, eef_name: str) -> list[str]:
        """Return per-subtask human-readable descriptions for the EEF, in subtask order."""

        return self.task_descriptor.get_subtask_descriptions(eef_name)

    def get_expected_attached_object(self, eef_name: str, subtask_index: int) -> str | None:
        """Return the object the EEF is expected to carry during a subtask, or ``None``.

        Delegates to :meth:`TaskDescriptor.get_expected_attached_object`. SkillGen reads this to
        attach the carried object to the planner's collision model before planning transit motion.
        """

        return self.task_descriptor.get_expected_attached_object(eef_name, subtask_index)

    # ------------------------------------------------------------------
    # Embodiment queries
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self, env_ids: Sequence[int] | None, eef_name: str) -> torch.Tensor:
        """Read the current pose of the given EEF from the embodiment adapter."""

        return self.embodiment_adapter.get_eef_poses(env_ids=env_ids)[eef_name]

    def get_robot_joint_positions(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        joint_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Read joint positions [rad, m] through the embodiment adapter.

        Args:
            env_ids: Environment indices; None selects all environments.
            joint_names: Ordered subset of articulation joints. None preserves native order.

        Returns:
            Positions shaped ``(num_selected_envs, num_selected_joints)``.
        """

        if joint_names is None:
            return self.embodiment_adapter.get_joint_positions(env_ids=env_ids)
        return self.embodiment_adapter.get_joint_positions(env_ids=env_ids, joint_names=joint_names)

    def get_robot_joint_position_limits(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read live lower and upper joint position limits [rad, m] via the embodiment adapter.

        Args:
            env_ids: Environment indices; None selects all environments.

        Returns:
            Limits shaped ``(num_selected_envs, num_joints, 2)`` in articulation order.
        """

        return self.embodiment_adapter.get_joint_position_limits(env_ids=env_ids)

    def get_robot_joint_velocity_limits(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read live maximum joint speeds [rad/s, m/s] through the embodiment adapter.

        Args:
            env_ids: Environment indices; None selects all environments.

        Returns:
            Maximum speeds shaped ``(num_selected_envs, num_joints)`` in articulation order.
        """
        return self.embodiment_adapter.get_joint_velocity_limits(env_ids=env_ids)

    def get_robot_joint_names(self) -> list[str]:
        """Return articulation joint names in native order."""

        return self.embodiment_adapter.get_joint_names()

    def get_robot_eef_joint_names(self, eef_name: str) -> tuple[str, ...]:
        """Return positional passthrough joints owned by one end effector.

        Args:
            eef_name: End effector to query.

        Returns:
            Joint names in passthrough-action order.
        """

        return self.embodiment_adapter.get_eef_joint_names(eef_name)

    def get_eef_passthrough_preset(self, eef_name: str, preset_name: str) -> tuple[float, ...]:
        """Return an embodiment-owned passthrough preset.

        Args:
            eef_name: End effector to query.
            preset_name: Configured posture name.

        Returns:
            Joint positions in the end effector's passthrough-action order [rad, m].
        """

        return self.embodiment_adapter.get_eef_passthrough_preset(eef_name, preset_name)

    def get_passthrough_actions(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return current passthrough actions when the embodiment can recover them.

        Args:
            env_ids: Environment indices, or None for all environments.

        Returns:
            Batched tensors per action channel, or None if a required channel is unavailable.
        """

        return self.embodiment_adapter.get_passthrough_actions(env_ids)

    def get_robot_action_target_frames(self) -> tuple[ActionTargetFrame, ...]:
        """Return exact controlled frames in pose-action block order, when available."""

        return self.embodiment_adapter.get_action_target_frames()

    def get_robot_planning_model_path(self) -> Path | None:
        """Return the exact local model used by the live embodiment controller, when exposed."""

        return self.embodiment_adapter.get_planning_model_path()

    def solve_robot_joint_trajectory(
        self,
        target_eef_poses: dict[str, torch.Tensor],
        joint_seed_trajectories: dict[str, NamedJointTrajectory],
        initial_joint_positions: torch.Tensor,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        joint_state_is_valid: Callable[[tuple[str, ...], torch.Tensor, int], bool] | None = None,
        env_id: int = 0,
    ) -> NamedJointTrajectory | None:
        """Solve shared-joint end-effector targets through the embodiment controller.

        Args:
            target_eef_poses: Target poses per end effector with shape (T, 4, 4) [m].
            joint_seed_trajectories: Named planner joint proposals per end effector.
            initial_joint_positions: Initial state in live articulation joint order [rad, m].
            position_tolerance_m: Required tool-position accuracy [m].
            rotation_tolerance_rad: Required tool-rotation accuracy [rad].
            joint_state_is_valid: Optional feasibility check for each joint sample.
            env_id: Environment controller to use.

        Returns:
            Named joint trajectory, or None when the embodiment does not support coupled solving.
        """

        return self.embodiment_adapter.solve_joint_trajectory(
            target_eef_poses,
            joint_seed_trajectories,
            initial_joint_positions,
            position_tolerance_m,
            rotation_tolerance_rad,
            joint_state_is_valid,
            env_id,
        )

    def set_robot_joint_posture_target(
        self,
        joint_names: tuple[str, ...],
        joint_positions: torch.Tensor,
        env_id: int = 0,
    ) -> None:
        """Set the embodiment controller's posture reference for the next step.

        Args:
            joint_names: Names aligned with the position values.
            joint_positions: Posture target with shape (J,) [rad, m].
            env_id: Environment controller to update.
        """

        self.embodiment_adapter.set_joint_posture_target(joint_names, joint_positions, env_id)

    def reset_robot_joint_posture_target(self, env_id: int = 0) -> None:
        """Restore the embodiment controller's default posture reference.

        Args:
            env_id: Environment controller to update.
        """

        self.embodiment_adapter.reset_joint_posture_target(env_id)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses and passthrough actions to an env action."""

        return self.embodiment_adapter.target_eef_pose_to_action(
            target_eef_pose_dict=target_eef_pose_dict,
            passthrough_action_dict=passthrough_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert an env action to target EEF poses."""

        return self.embodiment_adapter.action_to_target_eef_pose(action)

    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract passthrough actions (eef grippers + any non-eef channels) from an env action."""

        return self.embodiment_adapter.actions_to_passthrough_actions(actions)

    def passthrough_actions_to_joint_positions(
        self, passthrough_action_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply passthrough commands as externally controlled articulation positions."""

        return self.embodiment_adapter.passthrough_actions_to_joint_positions(passthrough_action_dict)

    def joint_positions_to_passthrough_actions(
        self,
        joint_position_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Project named planned joints into positional passthrough channels."""

        return self.embodiment_adapter.joint_positions_to_passthrough_actions(
            joint_position_dict,
            passthrough_action_dict,
        )

    # ------------------------------------------------------------------
    # Other runtime queries
    # ------------------------------------------------------------------

    def get_navigation_state(self, env_id: int) -> tuple[str, bool, bool] | None:
        """Return the navigation channel and its controller's active and goal-reached flags.

        Args:
            env_id: Environment index to query.

        Returns:
            Channel name, active flag, and goal-reached flag, or None when unavailable.
        """
        return self.embodiment_adapter.get_navigation_state(env_id)

    def _reference_frame_inverse(self, env_ids: Sequence[int] | None) -> torch.Tensor | None:
        reference = self.embodiment_adapter.get_reference_frame_pose(env_ids)
        return None if reference is None else torch.linalg.inv(reference)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get non-robot rigid and articulated object poses in the embodiment's reference frame."""

        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot_name = getattr(self.embodiment_adapter, "robot_asset_name", "robot")
        return scene_object_poses(self.env.scene, index, robot_name, self._reference_frame_inverse(env_ids))

    def get_object_nodal_positions(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get all deformable-object nodal positions in the embodiment's reference frame.

        Without a configured controller frame, positions remain environment-relative.

        Args:
            env_ids: Environments to read; ``None`` reads all.

        Returns:
            Mapping from deformable-object name to a tensor shaped
            ``(len(env_ids), num_nodes, 3)`` containing positions [m].
        """

        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        scene = self.env.scene
        env_origins = scene.env_origins[index]
        reference_inverse = self._reference_frame_inverse(env_ids)
        object_nodal_positions: dict[str, torch.Tensor] = {}
        for obj_name, obj in scene.deformable_objects.items():
            nodal_pos_w = obj.data.nodal_pos_w
            nodal_pos_w = nodal_pos_w.torch if hasattr(nodal_pos_w, "torch") else as_torch(nodal_pos_w)
            nodes = nodal_pos_w[index] - env_origins.unsqueeze(1)
            if reference_inverse is not None:
                rotation = reference_inverse[:, :3, :3]
                translation = reference_inverse[:, :3, 3]
                nodes = nodes @ rotation.transpose(-1, -2) + translation.unsqueeze(1)
            object_nodal_positions[obj_name] = nodes
        return object_nodal_positions

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None, obs_group: str = "subtask_terms"
    ) -> dict[str, torch.Tensor]:
        """Read the current per-subtask termination signals from the env observation buffer.

        For every non-empty termination-signal name declared by the task descriptor (across all
        EEFs), reads the identically named boolean observation term from
        ``env.obs_buf[obs_group]``. The env must therefore publish one observation term per
        term-signal name in a non-concatenated group. Auto annotation samples this each replay
        step to locate subtask boundaries without a human in the loop.

        Args:
            env_ids: Environments to read; ``None`` reads all.
            obs_group: Observation-buffer group holding the per-subtask boolean terms.

        Returns:
            Mapping of term-signal name to a boolean tensor of shape ``(len(env_ids),)``.
        """

        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        obs_buf = self.env.obs_buf
        assert obs_group in obs_buf, (
            f"Observation group {obs_group!r} not found in the env observation buffer. "
            f"Available groups: {sorted(obs_buf.keys())}"
        )
        group = obs_buf[obs_group]
        assert isinstance(group, dict), (
            f"Observation group {obs_group!r} is concatenated into a single tensor. Per-name "
            "signal reads require the group to set 'concatenate_terms = False'."
        )
        signals: dict[str, torch.Tensor] = {}
        for eef_name in self.get_eef_names():
            for signal_name in self.get_term_signal_names(eef_name):
                if not signal_name:
                    continue
                assert signal_name in group, (
                    f"Termination signal {signal_name!r} has no matching observation term in "
                    f"group {obs_group!r}. Available terms: {sorted(group.keys())}"
                )
                signals[signal_name] = as_torch(group[signal_name])[index].reshape(-1).bool()
        return signals

    def get_robot_root_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get the robot articulation root pose in the embodiment's reference frame.

        Returns a tensor of shape ``(len(env_ids), 4, 4)`` using the same frame convention as
        :meth:`get_object_poses`. Without a configured controller frame, both queries return
        environment-relative poses. Motion planners use this pose to transform between the
        embodiment frame and the robot base frame their solver plans in.
        Planning-root queries use :meth:`get_robot_planning_root_pose` for named links.
        """

        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        scene = self.env.scene
        robot_name = getattr(self.embodiment_adapter, "robot_asset_name", "robot")
        robot = scene[robot_name]
        pos_rel = as_torch(robot.data.root_pos_w)[index] - scene.env_origins[index]
        quat = as_torch(robot.data.root_quat_w)[index]
        pose = pose_math.make_pose(pos_rel, pose_math.matrix_from_quat(quat))
        reference_inverse = self._reference_frame_inverse(env_ids)
        return reference_inverse @ pose if reference_inverse is not None else pose

    def get_robot_planning_root_pose(
        self,
        root_pose_source_frame_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Get the named kinematic planning-root pose in the env-relative frame.

        This is the frame-safe query for motion planning and ScheduleGen. It delegates to the
        embodiment adapter so a controller-specific adapter can return the exact transform used
        by its live controller. :meth:`get_robot_root_pose` remains the articulation-root query
        used by legacy callers and is not assumed to equal this frame.

        Args:
            root_pose_source_frame_name: Exact live controller/articulation lookup key declared by
                the robot planning description.
            env_ids: Environments to read; ``None`` reads all.

        Returns:
            Planning-root transforms with shape ``(N, 4, 4)`` [m].
        """

        poses = self.embodiment_adapter.get_planning_root_pose(root_pose_source_frame_name, env_ids)
        assert isinstance(poses, torch.Tensor), "planning-root poses must be a torch.Tensor"
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4), "planning-root poses must have shape (N, 4, 4)"
        return poses

    def get_scene_state(self, is_relative: bool = True) -> dict:
        """Return the raw scene-state snapshot from the underlying env.

        Used by the data generator to record the env's initial state at episode start. Prefer
        :meth:`get_object_poses` for object pose math; this method is the raw scene-state escape
        hatch for callers that need the full dict (e.g. recorder ``initial_state``).
        """

        return self.env.scene.get_state(is_relative=is_relative)

    # ------------------------------------------------------------------
    # Collision-world source
    # ------------------------------------------------------------------
    # Motion planners build their collision world from the live scene. The Datastream is the
    # single owner of that simulator-facing access so the planners stay backend-agnostic: a
    # cuRobo planner feeds the stage + prim scoping below into its own USD parser, and per-step
    # obstacle pose sync goes through :meth:`get_object_poses` (embodiment reference frame).

    def get_usd_stage(self) -> Any:
        """Return the live USD stage backing the scene.

        Sanctioned collision-geometry source for motion planners that extract obstacles via a USD
        parser. Planners scope extraction with :meth:`get_env_prim_path`; frame-sensitive poses
        are reconciled with the live planning root rather than authored robot-prim transforms.
        """

        return self.env.scene.stage

    def get_env_prim_path(self, env_id: int) -> str:
        """Return the root USD prim path of the given environment's subtree (e.g. ``/World/envs/env_0``)."""

        return f"/World/envs/env_{env_id}"

    def get_object_prim_path(self, object_ref: str, env_id: int) -> str:
        """Return the exact environment-specific root prim for one rigid scene object.

        Args:
            object_ref: Exact key in the live scene's rigid-object mapping.
            env_id: Environment index.
        """

        scene = self.env.scene
        assert type(env_id) is int and 0 <= env_id < len(scene.env_prim_paths), "env_id is out of range"
        if object_ref not in scene.rigid_objects:
            raise KeyError(f"unknown rigid scene object {object_ref!r}")
        prim_pattern = scene.rigid_objects[object_ref].cfg.prim_path
        assert isinstance(prim_pattern, str) and prim_pattern, "rigid-object prim path must be non-empty"
        if prim_pattern == scene.env_regex_ns or prim_pattern.startswith(f"{scene.env_regex_ns}/"):
            prim_path = scene.env_prim_paths[env_id] + prim_pattern[len(scene.env_regex_ns) :]
        else:
            prim_path = prim_pattern
        assert scene.stage.GetPrimAtPath(prim_path).IsValid(), f"rigid-object prim does not exist: {prim_path}"
        return prim_path

    def get_robot_prim_path(self, env_id: int) -> str:
        """Return the USD prim path of the robot articulation root for the given environment.

        Used as the reference frame for obstacle extraction (obstacles are expressed relative to
        the robot base, which sits at the environment origin for a fixed-base robot). Derived from
        the live articulation when available, falling back to the standard ``{env}/Robot`` layout.
        """

        robot_name = getattr(self.embodiment_adapter, "robot_asset_name", "robot")
        try:
            prim_paths = self.env.scene[robot_name].root_view.prim_paths
            return prim_paths[env_id]
        except (KeyError, AttributeError, IndexError):
            return f"{self.get_env_prim_path(env_id)}/Robot"

    # ------------------------------------------------------------------
    # Source-demo pool access
    # ------------------------------------------------------------------

    @property
    def source_pool(self) -> DataGenInfoPool:
        """Get the source-demo pool."""

        return self._pool

    @property
    def datagen_infos(self) -> list:
        """Get the source-demo pool's datagen infos."""

        return self._pool.datagen_infos

    @property
    def subtask_boundaries(self) -> dict[str, list[list[tuple[int, int]]]]:
        """Get the source-demo pool's subtask boundaries."""
        return self._pool.subtask_boundaries

    @property
    def num_source_demos(self) -> int:
        """Get the number of source demos currently in the pool."""

        return self._pool.num_datagen_infos

    @property
    def asyncio_lock(self) -> asyncio.Lock | None:
        """Get the lock guarding the source-demo pool's growth across async env tasks."""

        return self._pool.asyncio_lock

    async def add_episode(self, episode: EpisodeData) -> None:
        """Add an episode to the source-demo pool."""

        await self._pool.add_episode(episode)
