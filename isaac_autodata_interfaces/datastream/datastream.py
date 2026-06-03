# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import torch
from collections.abc import Sequence
from typing import Any

from isaaclab.utils.datasets import EpisodeData

from isaac_autodata_core.pool import DataGenInfoPool
from isaac_autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from isaac_autodata_interfaces.tasks.subtask_constraint_spec import SubtaskConstraint
from isaac_autodata_interfaces.tasks.subtask_spec import Subtask, SubtaskAlgoParams
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from isaac_autodata_utils import pose_math


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
            asyncio_lock: Lock guarding concurrent pool growth across async env tasks. A fresh lock
                is created when not supplied and the pool is built here.
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

        # Ensure that task descriptor and embodiment adapter declare the same EEFs.
        task_eefs = set(task_descriptor.get_eef_names())
        adapter_eefs = set(embodiment_adapter.get_eef_names())
        assert task_eefs == adapter_eefs, (
            f"EEF mismatch: task descriptor declares {sorted(task_eefs)}, "
            f"embodiment adapter exposes {sorted(adapter_eefs)}"
        )

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
                asyncio_lock=asyncio_lock if asyncio_lock is not None else asyncio.Lock(),
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

    # ------------------------------------------------------------------
    # Embodiment queries
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self, env_ids: Sequence[int] | None, eef_name: str) -> torch.Tensor:
        """Read the current pose of the given EEF from the embodiment adapter."""

        return self.embodiment_adapter.get_eef_poses(env_ids=env_ids)[eef_name]

    def get_robot_joint_positions(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read the robot's current joint positions [rad] via the embodiment adapter.

        Returns a tensor of shape ``(len(env_ids), num_dof)`` in the articulation's native joint
        order (see :meth:`get_robot_joint_names`). Consumed by motion planners as a planning
        start state.
        """

        return self.embodiment_adapter.get_joint_positions(env_ids=env_ids)

    def get_robot_joint_names(self) -> list[str]:
        """Return the robot articulation's joint names, ordered to match :meth:`get_robot_joint_positions`."""

        return self.embodiment_adapter.get_joint_names()

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses to an env action."""

        return self.embodiment_adapter.target_eef_pose_to_action(
            target_eef_pose_dict=target_eef_pose_dict,
            gripper_action_dict=gripper_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert an env action to target EEF poses."""

        return self.embodiment_adapter.action_to_target_eef_pose(action)

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from an env action."""

        return self.embodiment_adapter.actions_to_gripper_actions(actions)

    # ------------------------------------------------------------------
    # Other runtime queries
    # ------------------------------------------------------------------

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get all rigid object poses from the environment as env-relative 4x4 matrices.

        Reads each rigid object's root pose directly from its cached articulation data rather
        than snapshotting the whole scene via ``env.scene.get_state()`` (which also gathers every
        articulation's joint state and velocities). The env-relative position is
        ``root_pos_w - env_origin`` — exactly what ``get_state(is_relative=True)`` computes — and
        the orientation is unchanged by the env-origin translation. Returns a map
        ``object_name -> (len(env_ids), 4, 4)``.
        """
        import warp as wp

        def _as_torch(arr):
            return wp.to_torch(arr) if isinstance(arr, wp.array) else arr

        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        scene = self.env.scene
        env_origins = scene.env_origins[index]
        object_pose_matrix: dict[str, torch.Tensor] = {}
        for obj_name, obj in scene.rigid_objects.items():
            pos_rel = _as_torch(obj.data.root_pos_w)[index] - env_origins
            quat = _as_torch(obj.data.root_quat_w)[index]
            object_pose_matrix[obj_name] = pose_math.make_pose(pos_rel, pose_math.matrix_from_quat(quat))
        return object_pose_matrix

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
    # cuRobo planner feeds the stage + prim scoping below into its own USD parser, while
    # per-step obstacle pose sync goes through :meth:`get_object_poses` (env-relative frame).

    def get_usd_stage(self) -> Any:
        """Return the live USD stage backing the scene.

        This is the sanctioned collision-geometry source for motion planners that extract
        obstacles via a USD parser. Planners scope extraction with :meth:`get_env_prim_path`
        and :meth:`get_robot_prim_path`.
        """

        return self.env.scene.stage

    def get_env_prim_path(self, env_id: int) -> str:
        """Return the root USD prim path of the given environment's subtree (e.g. ``/World/envs/env_0``)."""

        return f"/World/envs/env_{env_id}"

    def get_robot_prim_path(self, env_id: int) -> str:
        """Return the USD prim path of the robot articulation root for the given environment.

        Used as the reference frame for obstacle extraction (obstacles are expressed relative to
        the robot base, which sits at the environment origin for a fixed-base robot). Derived from
        the live articulation when available, falling back to the standard ``{env}/Robot`` layout.
        """

        robot_name = getattr(self.embodiment_adapter, "robot_asset_name", "robot")
        try:
            prim_paths = self.env.scene[robot_name].root_physx_view.prim_paths
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
