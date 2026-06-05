# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Modular data generator for different generation algorithms.

Behavior that varies between algorithms is routed through :class:`GenerationAlgorithm` so future
algorithms plug in without editing this file.
"""

from __future__ import annotations

import asyncio
import contextlib
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from isaaclab.managers import TerminationTermCfg

from isaac_autodata_core.algorithms import GenerationAlgorithm
from isaac_autodata_core.datagen_info import DatagenInfo
from isaac_autodata_core.selection_strategy import make_selection_strategy
from isaac_autodata_core.transforms import (
    get_delta_pose_with_scheme,
    transform_source_data_segment_using_delta_object_pose,
    transform_source_data_segment_using_object_pose,
)
from isaac_autodata_core.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory
from isaac_autodata_interfaces.datastream.datastream import Datastream
from isaac_autodata_interfaces.tasks.subtask_constraint_spec import (
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)


@contextlib.asynccontextmanager
async def _optional_lock(lock: asyncio.Lock | None):
    """Async context manager that acquires lock only when it is not None."""

    if lock is not None:
        async with lock:
            yield
    else:
        yield


@dataclass(frozen=True)
class GenerationResult:
    """Output of one :meth:`DataGenerator.generate` invocation."""

    initial_state: dict
    success: bool


@dataclass
class _GenerationBuffers:
    """Per-call success accumulator."""

    success: bool = False


@dataclass
class _EEFGenerationState:
    """Per-EEF state machine. Mutated in place by :class:`DataGenerator` and by
    :class:`GenerationAlgorithm` subclasses (for the SkillGen-style two-phase flow)."""

    current_subtask_index: int = 0
    current_trajectory: list[Waypoint] = field(default_factory=list)
    subtask_step_index: int | None = None
    subtasks_done: bool = False
    constraint_hold_waypoint: Waypoint | None = None
    is_currently_paused: bool = False
    _was_paused_prev_iter: bool = False
    # Two-phase transit support: when an algorithm returns an MP transit it stashes the actual
    # subtask trajectory here so the follow-up call splices it in.
    pending_subtask_trajectory: WaypointTrajectory | None = None
    is_in_motion_plan: bool = False


class DataGenerator:
    """Generates new demonstrations by stitching transformed source-demo subtask segments.

    The MimicGen-style pipeline (subtask boundary randomization → source-demo selection →
    object-centric transform → interpolation merge → step execution) is shared by all algorithms.
    The :class:`GenerationAlgorithm` instance gates the variants: EEF count, coordination
    constraints, subtask-start-signal expectations.
    """

    def __init__(
        self,
        datastream: Datastream,
        algorithm: GenerationAlgorithm,
        *,
        demo_keys: list[str] | None = None,
    ) -> None:
        """
        Args:
            datastream: Composed task descriptor + embodiment adapter + source pool + env handle.
                Datastream has already validated that the task descriptor and embodiment adapter
                agree on EEF names by construction.
            algorithm: Generation algorithm plug-in (Mimic, DexMimicGen, SkillGen, ...).
            demo_keys: Optional subset of source-demo keys to consider; ``None`` uses every demo
                in ``datastream.source_pool``.
        """
        self.datastream = datastream
        self.algorithm = algorithm
        self.src_demo_datagen_info_pool = datastream.source_pool
        self.demo_keys = demo_keys

        self._validate_eef_count_for_algorithm()
        self._validate_coordination_for_algorithm()
        self._validate_terminal_subtask_offsets()
        self.algorithm.validate_setup(self.datastream)

    def __repr__(self) -> str:
        return f"DataGenerator(algorithm={self.algorithm.name!r}, demo_keys={self.demo_keys})"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_eef_count_for_algorithm(self) -> None:
        num_eefs = len(self.datastream.get_eef_names())
        expected = self.algorithm.expected_eef_count
        ok = num_eefs == expected if isinstance(expected, int) else num_eefs in expected
        if not ok:
            raise ValueError(
                f"Algorithm {self.algorithm.name!r} expects {expected} EEF(s), task descriptor defines {num_eefs}"
            )

    def _validate_coordination_for_algorithm(self) -> None:
        has_coord_cfg = bool(self.datastream.get_task_constraints())
        if has_coord_cfg and not self.algorithm.supports_coordination:
            raise ValueError(
                "task descriptor declares constraints but algorithm "
                f"{self.algorithm.name!r} does not support coordination"
            )

    def _validate_terminal_subtask_offsets(self) -> None:
        # Upstream invariant: the last subtask of each EEF has no termination signal to offset.
        for eef_name in self.datastream.get_eef_names():
            last_subtask = self.datastream.get_subtasks(eef_name)[-1]
            assert last_subtask.subtask_term_offset_range[0] == 0
            assert last_subtask.subtask_term_offset_range[1] == 0

    # ------------------------------------------------------------------
    # Subtask boundary randomization
    # ------------------------------------------------------------------

    def randomize_subtask_boundaries(self) -> dict[str, np.ndarray]:
        """Sample per-demo subtask boundaries by jittering the recorded ones.

        For algorithms that read subtask start signals (SkillGen-family), both starts and ends are
        randomized. Otherwise the start of each subtask is pinned to the end of the previous one
        and only ends move.
        """
        randomize_starts = self.algorithm.uses_subtask_start_signals
        randomized: dict[str, np.ndarray] = {}

        for eef_name, raw_boundaries in self.src_demo_datagen_info_pool.subtask_boundaries.items():
            boundaries = np.array(raw_boundaries)
            eef_subtasks = self.datastream.get_subtasks(eef_name)

            first_start_offsets = np.random.randint(
                low=eef_subtasks[0].first_subtask_start_offset_range[0],
                high=eef_subtasks[0].first_subtask_start_offset_range[0] + 1,
                size=boundaries.shape[0],
            )
            boundaries[:, 0, 0] += first_start_offsets

            for i in range(boundaries.shape[1]):
                if randomize_starts:
                    # subtask_start_offset_range lives on the algorithm-specific algo_params
                    # (SkillGen today). Fall back to (0, 0) if the active algorithm does not
                    # declare it — mirrors the defensive lookup in DataGenInfoPool.
                    start_range = getattr(eef_subtasks[i].algo_params, "subtask_start_offset_range", (0, 0))
                    start_offset = np.random.randint(
                        low=start_range[0],
                        high=start_range[1] + 1,
                        size=boundaries.shape[0],
                    )
                    boundaries[:, i, 0] += start_offset
                elif i > 0:
                    boundaries[:, i, 0] = boundaries[:, i - 1, 1]

                end_offsets = np.random.randint(
                    low=eef_subtasks[i].subtask_term_offset_range[0],
                    high=eef_subtasks[i].subtask_term_offset_range[1] + 1,
                    size=boundaries.shape[0],
                )
                boundaries[:, i, 1] = boundaries[:, i, 1] + end_offsets

            assert np.all((boundaries[:, :, 1] - boundaries[:, :, 0]) > 0), "empty subtask after randomization"
            assert np.all((boundaries[:, 1:, :] - boundaries[:, :-1, :]) > 0), "subtask indices do not increase"
            flat = boundaries.reshape(boundaries.shape[0], -1)
            assert np.all((flat[:, 1:] - flat[:, :-1]) >= 0), "subtask indices out of order"

            randomized[eef_name] = boundaries

        return randomized

    # ------------------------------------------------------------------
    # Source-demo selection
    # ------------------------------------------------------------------

    def select_source_demo(
        self,
        eef_name: str,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor | None,
        src_demo_current_subtask_boundaries: np.ndarray,
        subtask_object_name: str | None,
        selection_strategy_name: str,
        selection_strategy_kwargs: dict | None = None,
    ) -> int:
        """Pick a source demonstration index for a subtask via the configured strategy."""
        if subtask_object_name is None:
            assert selection_strategy_name == "random", selection_strategy_name

        src_subtask_datagen_infos: list[DatagenInfo] = []
        for i in range(len(self.src_demo_datagen_info_pool.datagen_infos)):
            src_ep = self.src_demo_datagen_info_pool.datagen_infos[i]
            start_ind = src_demo_current_subtask_boundaries[i][0]
            end_ind = src_demo_current_subtask_boundaries[i][1]
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep.eef_pose[eef_name][start_ind:end_ind],
                    object_poses=(
                        {subtask_object_name: src_ep.object_poses[subtask_object_name][start_ind:end_ind]}
                        if subtask_object_name is not None
                        else None
                    ),
                    subtask_term_signals=None,
                    target_eef_pose=src_ep.target_eef_pose[eef_name][start_ind:end_ind],
                    gripper_action=src_ep.gripper_action[eef_name][start_ind:end_ind],
                )
            )

        strategy = make_selection_strategy(selection_strategy_name)
        kwargs = selection_strategy_kwargs or {}
        return strategy.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Subtask trajectory generation
    # ------------------------------------------------------------------

    def generate_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_ind: int,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> WaypointTrajectory:
        """Build a transformed :class:`WaypointTrajectory` for ``subtask_ind`` of ``eef_name``."""
        subtasks = self.datastream.get_subtasks(eef_name)
        policy = self.datastream.get_generation_policy()
        # Subtask.object_ref is empty string when no object is involved; normalize to None so the
        # rest of the pipeline can keep using the upstream `is not None` convention.
        subtask_object_name = subtasks[subtask_ind].object_ref or None
        subtask_object_pose = (
            self.datastream.get_object_poses(env_ids=[env_id])[subtask_object_name][0]
            if subtask_object_name is not None
            else None
        )

        is_first_subtask = subtask_ind == 0
        need_selection = is_first_subtask or policy.select_src_per_subtask
        if not policy.select_src_per_arm:
            need_selection = need_selection and selected_src_demo_inds[eef_name] is None

        use_delta_transform, coord_transform_scheme = self._resolve_coordination_for_subtask(
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            subtask_object_name=subtask_object_name,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        if use_delta_transform is not None:
            need_selection = False

        if need_selection:
            selected_src_demo_inds[eef_name] = self.select_source_demo(
                eef_name=eef_name,
                eef_pose=self.datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0],
                object_pose=subtask_object_pose,
                src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=subtasks[subtask_ind].selection_strategy,
                selection_strategy_kwargs=subtasks[subtask_ind].selection_strategy_kwargs,
            )

        assert selected_src_demo_inds[eef_name] is not None
        selected_src_demo_ind = selected_src_demo_inds[eef_name]

        if not policy.select_src_per_arm and need_selection:
            for other_eef_name in self.datastream.get_eef_names():
                selected_src_demo_inds[other_eef_name] = selected_src_demo_ind

        selected_boundary = all_randomized_subtask_boundaries[eef_name][selected_src_demo_ind, subtask_ind]
        self._propagate_coordination_selection(
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            selected_src_demo_ind=selected_src_demo_ind,
            selected_boundary=selected_boundary,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
        )

        src_ep = self.src_demo_datagen_info_pool.datagen_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep.eef_pose[eef_name][selected_boundary[0] : selected_boundary[1]]
        src_subtask_target_poses = src_ep.target_eef_pose[eef_name][selected_boundary[0] : selected_boundary[1]]
        src_subtask_gripper_actions = src_ep.gripper_action[eef_name][selected_boundary[0] : selected_boundary[1]]
        src_subtask_object_pose = (
            src_ep.object_poses[subtask_object_name][selected_boundary[0]] if subtask_object_name is not None else None
        )

        if is_first_subtask or policy.transform_first_robot_pose:
            # Prepending the recorded EEF pose makes interpolation seed from the robot's actual
            # pose rather than the first target pose.
            src_eef_poses = torch.cat([src_subtask_eef_poses[0:1], src_subtask_target_poses], dim=0)
            src_subtask_gripper_actions = torch.cat(
                [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], dim=0
            )
        else:
            src_eef_poses = src_subtask_target_poses.clone()
            src_subtask_gripper_actions = src_subtask_gripper_actions.clone()

        transformed_eef_poses = self._apply_subtask_transform(
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            subtask_object_name=subtask_object_name,
            subtask_object_pose=subtask_object_pose,
            src_subtask_object_pose=src_subtask_object_pose,
            src_eef_poses=src_eef_poses,
            use_delta_transform=use_delta_transform,
            coord_transform_scheme=coord_transform_scheme,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
        )

        seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtasks[subtask_ind].action_noise,
        )
        traj = WaypointTrajectory()
        traj.add_waypoint_sequence(seq)
        return traj

    def _resolve_coordination_for_subtask(
        self,
        eef_name: str,
        subtask_ind: int,
        subtask_object_name: str | None,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> tuple[torch.Tensor | None, Any]:
        """Inspect coordination constraints; return ``(delta_transform_or_None, scheme_or_None)``."""
        if not self.algorithm.supports_coordination:
            return None, None
        key = (eef_name, subtask_ind)
        if key not in runtime_subtask_constraints_dict:
            return None, None
        constraint = runtime_subtask_constraints_dict[key]
        if constraint["type"] != SubTaskConstraintType.COORDINATION:
            return None, None

        concurrent_key = constraint["concurrent_task_spec_key"]
        concurrent_ind = constraint["concurrent_subtask_ind"]
        concurrent_constraint = runtime_subtask_constraints_dict[(concurrent_key, concurrent_ind)]
        concurrent_selected = concurrent_constraint["selected_src_demo_ind"]

        if concurrent_selected is not None:
            # Concurrent task has already chosen a source demo — adopt the same demo + transform.
            selected_src_demo_inds[eef_name] = concurrent_selected
            return concurrent_constraint["transform"], None

        assert "transform" not in constraint, "transform should not be set for concurrent task"
        scheme = constraint["coordination_scheme"]
        if scheme != SubTaskConstraintCoordinationScheme.REPLAY:
            assert subtask_object_name is not None, f"object reference required for {scheme} coordination scheme"
        return None, scheme

    def _propagate_coordination_selection(
        self,
        eef_name: str,
        subtask_ind: int,
        selected_src_demo_ind: int,
        selected_boundary: np.ndarray,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
    ) -> None:
        """Record the selected source demo + sync window on the active coordination constraint."""
        if not self.algorithm.supports_coordination:
            return
        key = (eef_name, subtask_ind)
        if key not in runtime_subtask_constraints_dict:
            return
        constraint = runtime_subtask_constraints_dict[key]
        if constraint["type"] != SubTaskConstraintType.COORDINATION:
            return

        constraint["selected_src_demo_ind"] = selected_src_demo_ind
        concurrent_key = constraint["concurrent_task_spec_key"]
        concurrent_ind = constraint["concurrent_subtask_ind"]
        concurrent_bounds = all_randomized_subtask_boundaries[concurrent_key][selected_src_demo_ind, concurrent_ind]
        own_len = selected_boundary[1] - selected_boundary[0]
        concurrent_len = concurrent_bounds[1] - concurrent_bounds[0]
        constraint["synchronous_steps"] = min(own_len, concurrent_len)

    def _apply_subtask_transform(
        self,
        eef_name: str,
        subtask_ind: int,
        subtask_object_name: str | None,
        subtask_object_pose: torch.Tensor | None,
        src_subtask_object_pose: torch.Tensor | None,
        src_eef_poses: torch.Tensor,
        use_delta_transform: torch.Tensor | None,
        coord_transform_scheme: Any,
        runtime_subtask_constraints_dict: dict,
    ) -> torch.Tensor:
        """Apply object-centric or coordination-supplied transform to ``src_eef_poses``."""
        if use_delta_transform is not None:
            return transform_source_data_segment_using_delta_object_pose(src_eef_poses, use_delta_transform)

        if coord_transform_scheme is not None:
            delta = get_delta_pose_with_scheme(
                src_subtask_object_pose,
                subtask_object_pose,
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)],
            )
            transformed = transform_source_data_segment_using_delta_object_pose(src_eef_poses, delta)
            runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["transform"] = delta
            return transformed

        if subtask_object_name is not None:
            return transform_source_data_segment_using_object_pose(
                subtask_object_pose,
                src_eef_poses,
                src_subtask_object_pose,
            )
        # No reference object — segment is used verbatim.
        return src_eef_poses

    # ------------------------------------------------------------------
    # Merge subtask trajectory with interpolation segment
    # ------------------------------------------------------------------

    def merge_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_index: int,
        prev_executed_traj: list[Waypoint] | None,
        subtask_trajectory: WaypointTrajectory,
        force_use_prev_traj: bool = False,
    ) -> list[Waypoint]:
        """Prepend an interpolation segment to the subtask trajectory and flatten.

        ``force_use_prev_traj=True`` makes the seed waypoint be ``prev_executed_traj[-1]``
        unconditionally. Used by SkillGen so the skill segment interpolates from the motion-plan
        end pose rather than from the live robot pose.
        """
        is_first_subtask = subtask_index == 0
        traj_to_execute = WaypointTrajectory()
        subtask = self.datastream.get_subtask(eef_name, subtask_index)

        use_prev_traj = force_use_prev_traj or (
            self.datastream.get_generation_policy().interpolate_from_last_target_pose and not is_first_subtask
        )

        if use_prev_traj:
            assert prev_executed_traj is not None
            init_sequence = WaypointSequence(sequence=[prev_executed_traj[-1]])
        else:
            init_sequence = WaypointSequence.from_poses(
                poses=self.datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=subtask_trajectory[0].gripper_action.unsqueeze(0),
                action_noise=subtask.action_noise,
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        traj_to_execute.merge(
            subtask_trajectory,
            num_steps_interp=subtask.num_interpolation_steps,
            num_steps_fixed=subtask.num_fixed_steps,
            action_noise=(float(subtask.apply_noise_during_interpolation) * subtask.action_noise),
        )

        # Drop the seed waypoint used only to anchor interpolation.
        traj_to_execute.pop_first()
        return traj_to_execute.get_full_sequence().sequence

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    async def generate(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        *,
        env_reset_queue: asyncio.Queue,
        env_action_queue: asyncio.Queue,
        pause_subtask: bool = False,
        export_demo: bool = True,
    ) -> GenerationResult:
        """Generate one demonstration for ``env_id`` using the configured algorithm."""
        env_id_tensor, initial_state = await self._reset_environment_for_generation(
            env_id=env_id, env_reset_queue=env_reset_queue
        )

        runtime_constraints = self._build_runtime_subtask_constraints()
        eef_states = self._initialize_eef_states()
        selected_src_demo_inds: dict[str, int | None] = {name: None for name in self.datastream.get_eef_names()}
        buffers = _GenerationBuffers()

        randomized_subtask_boundaries: dict[str, np.ndarray] | None = None
        prev_pool_size = 0

        while True:
            await asyncio.sleep(0)
            pool_lock = self.src_demo_datagen_info_pool.asyncio_lock
            async with _optional_lock(pool_lock):
                randomized_subtask_boundaries, prev_pool_size = self._maybe_refresh_randomized_subtask_boundaries(
                    randomized_subtask_boundaries=randomized_subtask_boundaries,
                    prev_pool_size=prev_pool_size,
                )
                assert randomized_subtask_boundaries is not None

                for eef_name, eef_state in eef_states.items():
                    if eef_state.subtasks_done or eef_state.subtask_step_index is not None:
                        continue

                    result = self.algorithm.plan_subtask_trajectory(
                        data_generator=self,
                        env_id=env_id,
                        eef_name=eef_name,
                        eef_state=eef_state,
                        all_randomized_subtask_boundaries=randomized_subtask_boundaries,
                        runtime_subtask_constraints_dict=runtime_constraints,
                        selected_src_demo_inds=selected_src_demo_inds,
                    )
                    if result is None:
                        # Planning failure (e.g. SkillGen motion planner). Abort with no success.
                        return GenerationResult(initial_state=initial_state, success=False)
                    next_trajectory, _is_motion_plan = result
                    eef_state.current_trajectory = next_trajectory
                    eef_state.subtask_step_index = 0

            eef_waypoints = self._collect_eef_waypoints(
                runtime_constraints=runtime_constraints,
                eef_states=eef_states,
            )
            multi_waypoint = MultiWaypoint(eef_waypoints)

            await asyncio.sleep(0)

            exec_success = await multi_waypoint.execute(
                datastream=self.datastream,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            buffers.success = buffers.success or exec_success
            self._advance_subtask_progress(
                eef_states=eef_states,
                runtime_constraints=runtime_constraints,
                pause_subtask=pause_subtask,
            )

            if self._all_subtasks_completed(eef_states):
                break

        # Recorder lifecycle stays on env by design: the Datastream is
        # a read interface; controller-side mutation is reached through the get_env() escape hatch.
        env = self.datastream.get_env()
        env.recorder_manager.set_success_to_episodes(
            env_id_tensor,
            torch.tensor([[buffers.success]], dtype=torch.bool, device=self.datastream.device),
        )
        if export_demo:
            env.recorder_manager.export_episodes(env_id_tensor)

        return GenerationResult(
            initial_state=initial_state,
            success=buffers.success,
        )

    # ------------------------------------------------------------------
    # Generation loop helpers
    # ------------------------------------------------------------------

    async def _reset_environment_for_generation(
        self,
        env_id: int,
        env_reset_queue: asyncio.Queue,
    ) -> tuple[torch.Tensor, dict]:
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.datastream.device)
        # Recorder + reset queue stay on env. The initial scene state
        # snapshot is read through the Datastream interface.
        self.datastream.get_env().recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        return env_id_tensor, self.datastream.get_scene_state(is_relative=True)

    def _build_runtime_subtask_constraints(self) -> dict:
        runtime_constraints: dict = {}
        for subtask_constraint in self.datastream.get_task_constraints():
            runtime_constraints.update(subtask_constraint.generate_runtime_subtask_constraints())
        return runtime_constraints

    def _initialize_eef_states(self) -> dict[str, _EEFGenerationState]:
        return {name: _EEFGenerationState() for name in self.datastream.get_eef_names()}

    def _maybe_refresh_randomized_subtask_boundaries(
        self,
        randomized_subtask_boundaries: dict[str, np.ndarray] | None,
        prev_pool_size: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        current_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)
        if randomized_subtask_boundaries is None or current_pool_size > prev_pool_size:
            return self.randomize_subtask_boundaries(), current_pool_size
        return randomized_subtask_boundaries, prev_pool_size

    def _collect_eef_waypoints(
        self,
        runtime_constraints: dict,
        eef_states: dict[str, _EEFGenerationState],
    ) -> dict[str, Waypoint]:
        eef_waypoint_dict: dict[str, Waypoint] = {}
        for eef_name in sorted(self.datastream.get_eef_names()):
            eef_state = eef_states[eef_name]
            if eef_state.subtask_step_index is None:
                continue

            is_paused = self._apply_constraint_progression_rules(
                eef_name=eef_name,
                eef_state=eef_state,
                runtime_constraints=runtime_constraints,
                eef_states=eef_states,
            )

            just_paused = is_paused and not eef_state._was_paused_prev_iter
            eef_state._was_paused_prev_iter = is_paused
            eef_state.is_currently_paused = is_paused

            if is_paused:
                if just_paused or eef_state.constraint_hold_waypoint is None:
                    eef_state.constraint_hold_waypoint = deepcopy(
                        eef_state.current_trajectory[eef_state.subtask_step_index]
                    )
                waypoint = deepcopy(eef_state.constraint_hold_waypoint)
            else:
                waypoint = eef_state.current_trajectory[eef_state.subtask_step_index]
                eef_state.constraint_hold_waypoint = deepcopy(waypoint)

            eef_waypoint_dict[eef_name] = waypoint
        return eef_waypoint_dict

    def _apply_constraint_progression_rules(
        self,
        eef_name: str,
        eef_state: _EEFGenerationState,
        runtime_constraints: dict,
        eef_states: dict[str, _EEFGenerationState],
    ) -> bool:
        """Return True if this EEF should hold (use the cached hold waypoint) this tick."""
        # Motion-planned transits are constraint-free (Arena models this with a transient
        # subtask_index=-1 dict-miss; we use an explicit flag).
        if eef_state.is_in_motion_plan:
            return False
        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key not in runtime_constraints:
            return False

        step_index = eef_state.subtask_step_index
        if step_index is None:
            return False

        constraint = runtime_constraints[subtask_key]

        if constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
            if constraint["fulfilled"]:
                return False
            min_time_diff = constraint["min_time_diff"]
            traj_len = len(eef_state.current_trajectory)
            if traj_len == 0:
                return False
            if min_time_diff < 0:
                return True
            return step_index >= traj_len - min_time_diff

        if constraint["type"] != SubTaskConstraintType.COORDINATION:
            return False

        synchronous_steps = constraint["synchronous_steps"]
        concurrent_key = constraint["concurrent_task_spec_key"]
        concurrent_ind = constraint["concurrent_subtask_ind"]
        concurrent_constraint = runtime_constraints[(concurrent_key, concurrent_ind)]
        concurrent_state = eef_states[concurrent_key]

        if constraint["coordination_synchronize_start"] and concurrent_state.current_subtask_index < concurrent_ind:
            eef_state.subtask_step_index = 0
            return True

        if (
            not concurrent_constraint["fulfilled"]
            and step_index >= len(eef_state.current_trajectory) - synchronous_steps
        ):
            concurrent_constraint["fulfilled"] = True

        if not constraint["fulfilled"] and step_index >= len(eef_state.current_trajectory) - synchronous_steps:
            return True
        return False

    def _advance_subtask_progress(
        self,
        eef_states: dict[str, _EEFGenerationState],
        runtime_constraints: dict,
        pause_subtask: bool,
    ) -> None:
        for eef_name, eef_state in eef_states.items():
            if eef_state.subtask_step_index is None:
                continue
            if eef_state.is_currently_paused:
                continue
            eef_state.subtask_step_index += 1
            if eef_state.subtask_step_index == len(eef_state.current_trajectory):
                self._handle_subtask_completion(
                    eef_name=eef_name,
                    eef_state=eef_state,
                    eef_states=eef_states,
                    runtime_constraints=runtime_constraints,
                    pause_subtask=pause_subtask,
                )

    def _handle_subtask_completion(
        self,
        eef_name: str,
        eef_state: _EEFGenerationState,
        eef_states: dict[str, _EEFGenerationState],
        runtime_constraints: dict,
        pause_subtask: bool,
    ) -> None:
        if eef_state.current_trajectory:
            eef_state.constraint_hold_waypoint = deepcopy(eef_state.current_trajectory[-1])

        # When an MP transit completes (SkillGen), the actual subtask hasn't been executed yet —
        # don't fire constraint progression and don't advance the subtask index. Just reset
        # step_index so the next iteration triggers the algorithm's resume path.
        if eef_state.is_in_motion_plan:
            eef_state.subtask_step_index = None
            return

        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key in runtime_constraints:
            constraint = runtime_constraints[subtask_key]
            if constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                constrained_key = constraint["constrained_task_spec_key"]
                constrained_ind = constraint["constrained_subtask_ind"]
                runtime_constraints[(constrained_key, constrained_ind)]["fulfilled"] = True
            elif constraint["type"] == SubTaskConstraintType.COORDINATION:
                concurrent_key = constraint["concurrent_task_spec_key"]
                concurrent_ind = constraint["concurrent_subtask_ind"]
                constraint["finished"] = True
                concurrent_constraint = runtime_constraints[(concurrent_key, concurrent_ind)]
                concurrent_state = eef_states[concurrent_key]
                assert concurrent_constraint["finished"] or (
                    concurrent_state.subtask_step_index is not None
                    and concurrent_state.subtask_step_index >= len(concurrent_state.current_trajectory) - 1
                )

        if pause_subtask:
            input(f"Paused after subtask {eef_state.current_subtask_index} of {eef_name}. Press Enter to continue...")

        last_subtask_index = self.datastream.num_subtasks(eef_name) - 1
        if eef_state.current_subtask_index == last_subtask_index:
            eef_state.subtasks_done = True
            # Repeat the final waypoint to keep this EEF stationary while others finish.
            eef_state.current_trajectory.append(eef_state.current_trajectory[-1])
            return

        eef_state.subtask_step_index = None
        eef_state.current_subtask_index += 1

    @staticmethod
    def _all_subtasks_completed(eef_states: dict[str, _EEFGenerationState]) -> bool:
        return all(state.subtasks_done for state in eef_states.values())
