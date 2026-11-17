# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, schedule, validate and replay complete episodes with ScheduleGen."""

from __future__ import annotations

import math
import time
import torch
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

from autodata_core.algorithms import AlgorithmEpisodeResult, GenerationAlgorithm
from autodata_core.schedulegen.types import (
    ArmPath,
    ArmWaypointConflict,
    EpisodeGenerationError,
    EpisodePaths,
    ObjectStateSample,
    ScheduledObjectEvent,
    ScheduledObjectPose,
    ScheduledRobotContact,
    ScheduledWorldContact,
)
from autodata_core.waypoint import MultiWaypoint, Waypoint
from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory
from autodata_utils import pose_math

if TYPE_CHECKING:
    from isaaclab.managers import TerminationTermCfg

    from autodata_core.data_generator import DataGenerator
    from autodata_core.scheduling import Schedule
    from autodata_interfaces.datastream.datastream import Datastream
    from autodata_interfaces.motion_planners.curobo.collision_spheres import ObjectCollisionSpheres
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
    from autodata_interfaces.motion_planners.curobo.planner_factory import CuroboPlanningContext
    from autodata_interfaces.tasks.schedule_constraint_spec import ScheduleSpec, SegmentId


@dataclass(frozen=True)
class _ValidatedEpisode:
    """Paths and a schedule that passed the configured checks before replay."""

    arm_paths: Mapping[str, ArmPath]
    schedule: Schedule
    replay_joint_trajectory: NamedJointTrajectory | None
    diagnostics_before_success: tuple[tuple[str, str], ...]
    diagnostics_after_success: tuple[tuple[str, str], ...]


@dataclass
class _PreparedEpisode:
    """One reset scene and its paths, independent of the selected replay environment."""

    planning_env_id: int
    initial_state: dict
    initial_object_poses: Mapping[str, torch.Tensor]
    initial_joint_positions: torch.Tensor
    selected_source: dict[str, int | None]
    source_candidate_diagnostics: tuple[tuple[str, str], ...] = ()
    paths: EpisodePaths | None = None
    conflicts: tuple[ArmWaypointConflict, ...] = ()
    validated: _ValidatedEpisode | None = None
    result: AlgorithmEpisodeResult | None = None
    start_time_s: float = field(default_factory=time.monotonic)


class ScheduleGen(GenerationAlgorithm):
    """Build complete paths, schedule them and replay collision-checked episodes.

    Planning contexts and path-builder caches are isolated per environment. Scene reset,
    recording and dataset export remain owned by :class:`DataGenerator`.
    """

    name = "schedulegen"
    expected_eef_count = None
    requires_motion_planner = True
    uses_subtask_start_signals = True
    supports_coordination = False
    plans_complete_episode = True

    def __init__(
        self,
        planning_contexts: Mapping[int, CuroboPlanningContext],
        *,
        collision_densify_factor: int = 1,
        attempt_diagnostics: Callable[[Datastream, int, TerminationTermCfg], tuple[tuple[str, str], ...]] | None = None,
    ) -> None:
        """Create the algorithm with one planning context per environment.

        Args:
            planning_contexts: Arm planners and a full-state collision model per environment.
            collision_densify_factor: Linear collision samples per waypoint interval.
            attempt_diagnostics: Optional task-owned terminal diagnostics callback.
        """

        from autodata_core.schedulegen.path_builder import PathBuilder

        assert planning_contexts, "ScheduleGen requires at least one planning context"
        assert all(
            type(env_id) is int and env_id >= 0 for env_id in planning_contexts
        ), "planning context keys must be non-negative environment indices"
        assert attempt_diagnostics is None or callable(attempt_diagnostics)
        assert type(collision_densify_factor) is int and collision_densify_factor >= 1
        self.planning_contexts = dict(planning_contexts)
        self.collision_densify_factor = collision_densify_factor
        self.attempt_diagnostics = attempt_diagnostics
        self._path_builders: dict[int, PathBuilder] = {}
        for env_id, context in self.planning_contexts.items():
            assert context.group_planners, "planners must not be empty"
            if context.full_state is not None:
                assert not context.full_state.config.planning_enabled, "full-state context must disable planning"
            self._path_builders[env_id] = PathBuilder(context.group_planners)

    def validate_setup(self, datastream: Datastream) -> None:
        """Check that the task defines named segments and a schedule.

        Args:
            datastream: Composed task, embodiment and environment interfaces.

        Raises:
            AssertionError: A required segment or schedule setting is missing.
        """

        schedule_spec = datastream.get_schedule_spec()
        assert schedule_spec is not None, "ScheduleGen requires a task schedule specification"
        assert datastream.get_eef_names(), "ScheduleGen requires at least one end effector"
        for eef_name in datastream.get_eef_names():
            for subtask_index, subtask in enumerate(datastream.get_subtasks(eef_name)):
                assert (
                    subtask.name and subtask.segments
                ), f"ScheduleGen subtask {eef_name}[{subtask_index}] requires a stable name and segments"

    async def generate_and_execute_episode(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        success_term: TerminationTermCfg,
        env_action_queue: Any,
        pause_subtask: bool,
    ) -> AlgorithmEpisodeResult:
        """Build all paths, schedule them, restore the scene and replay one attempt.

        Path-building, scheduling and validation failures are returned as failed attempts.
        Scene reset, recording and dataset export remain owned by :class:`DataGenerator`.

        Args:
            data_generator: Generator providing the source data and environment interfaces.
            env_id: Environment used for this attempt.
            success_term: Task success condition evaluated during and after replay.
            env_action_queue: Queue consumed by the shared environment stepping loop.
            pause_subtask: Common algorithm option; unused for complete-episode scheduling.

        Returns:
            Attempt outcome with source choices, failure stage and schedule diagnostics.
        """

        del pause_subtask
        assert env_id in self.planning_contexts, f"no ScheduleGen planning context configured for env {env_id}"
        prepared = self._episode_prepare(data_generator, env_id, success_term)
        if prepared.result is not None:
            return prepared.result
        try:
            prepared.validated = self._episode_validate(
                data_generator=data_generator,
                episode_paths=prepared.paths,
                initial_object_poses=prepared.initial_object_poses,
                initial_joint_positions=prepared.initial_joint_positions,
                schedule_spec=data_generator.datastream.get_schedule_spec(),
                env_id=env_id,
            )
        except EpisodeGenerationError as failure:
            self._state_restore(data_generator.datastream, prepared.initial_state, env_id)
            return self._failure_result(
                data_generator=data_generator,
                env_id=env_id,
                success_term=success_term,
                selected_source=prepared.selected_source,
                source_candidate_diagnostics=prepared.source_candidate_diagnostics,
                failure=failure,
            )
        self._state_restore(data_generator.datastream, prepared.initial_state, env_id)
        return await self._episode_execute(data_generator, prepared, env_id, success_term, env_action_queue)

    def _episode_prepare(
        self, data_generator: DataGenerator, env_id: int, success_term: TerminationTermCfg
    ) -> _PreparedEpisode:
        """Capture the selected reset scene and build complete unscheduled paths.

        Args:
            data_generator: Generator providing source selection and the datastream.
            env_id: Environment whose reset scene is used for planning.
            success_term: Task success predicate used in failure diagnostics.

        Returns:
            Reset state and complete paths, or a recorded preparation failure.
        """

        planners = self.planning_contexts[env_id].group_planners
        datastream = data_generator.datastream
        eef_names = tuple(datastream.get_eef_names())
        assert set(planners) == set(eef_names), "planner keys must match task EEF names"
        schedule_spec = datastream.get_schedule_spec()
        assert schedule_spec is not None, "ScheduleGen requires a schedule specification"

        prepared = _PreparedEpisode(
            planning_env_id=env_id,
            initial_state=datastream.get_scene_state(is_relative=True, env_ids=[env_id]),
            initial_object_poses={
                object_ref: poses[0].detach().clone()
                for object_ref, poses in datastream.get_object_poses(env_ids=[env_id]).items()
            },
            initial_joint_positions=datastream.get_robot_joint_positions(env_ids=[env_id])[0].detach().clone(),
            selected_source={name: None for name in eef_names},
        )
        boundaries = data_generator.randomize_subtask_boundaries()
        try:
            prepared.paths, prepared.source_candidate_diagnostics = self._path_builders[env_id].build(
                data_generator,
                env_id,
                eef_names,
                prepared.selected_source,
                boundaries=boundaries,
            )
        except EpisodeGenerationError as failure:
            self._state_restore(datastream, prepared.initial_state, env_id)
            prepared.result = self._failure_result(
                data_generator=data_generator,
                env_id=env_id,
                success_term=success_term,
                selected_source=prepared.selected_source,
                source_candidate_diagnostics=prepared.source_candidate_diagnostics,
                failure=failure,
            )

        return prepared

    async def _episode_execute(
        self,
        data_generator: DataGenerator,
        prepared: _PreparedEpisode,
        env_id: int,
        success_term: TerminationTermCfg,
        env_action_queue: Any,
    ) -> AlgorithmEpisodeResult:
        """Replay a validated episode using the target environment's controller.

        Args:
            data_generator: Generator providing the live datastream.
            prepared: Validated paths and the planning context used to build them.
            env_id: Replay environment, which may differ from the planning environment.
            success_term: Task success predicate evaluated during replay.
            env_action_queue: Queue consumed by the shared simulator stepping loop.

        Returns:
            Replay outcome with task, schedule and tracking diagnostics.
        """

        validated_episode = prepared.validated
        assert validated_episode is not None
        datastream = data_generator.datastream
        planners = self.planning_contexts[prepared.planning_env_id].group_planners
        try:
            success, first_success_tick = await self._schedule_replay(
                data_generator=data_generator,
                env_id=env_id,
                success_term=success_term,
                env_action_queue=env_action_queue,
                arm_paths=dict(validated_episode.arm_paths),
                schedule=validated_episode.schedule,
                joint_trajectory=validated_episode.replay_joint_trajectory,
                planning_env_id=prepared.planning_env_id,
            )
        except EpisodeGenerationError as failure:
            return self._failure_result(
                data_generator=data_generator,
                env_id=env_id,
                success_term=success_term,
                selected_source=prepared.selected_source,
                source_candidate_diagnostics=prepared.source_candidate_diagnostics,
                failure=failure,
                validated_episode=validated_episode,
                tracking_paths=validated_episode.arm_paths,
                planning_env_id=prepared.planning_env_id,
            )
        finally:
            if validated_episode.replay_joint_trajectory is not None:
                datastream.reset_robot_joint_posture_target(env_id)

        return AlgorithmEpisodeResult(
            success=success,
            diagnostics=(
                ("stage", "complete" if success else "task_evaluation"),
                ("replayed", "true"),
                *_source_demo_diagnostics(data_generator, prepared.selected_source),
                *prepared.source_candidate_diagnostics,
                *_validated_episode_diagnostics(validated_episode, first_success_tick),
                *_terminal_tracking_diagnostics(
                    datastream,
                    planners,
                    validated_episode.arm_paths,
                    env_id,
                ),
                *self._attempt_diagnostics_collect(data_generator, env_id, success_term),
            ),
        )

    def _failure_result(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        success_term: TerminationTermCfg,
        selected_source: Mapping[str, int | None],
        source_candidate_diagnostics: tuple[tuple[str, str], ...],
        failure: EpisodeGenerationError,
        validated_episode: _ValidatedEpisode | None = None,
        tracking_paths: Mapping[str, ArmPath] | None = None,
        planning_env_id: int | None = None,
    ) -> AlgorithmEpisodeResult:
        """Collect failure diagnostics without resetting an attempted replay.

        Args:
            data_generator: Generator providing the live scene and source pool.
            env_id: Environment whose terminal state is inspected.
            success_term: Task predicate used by task-specific diagnostics.
            selected_source: Selected source indices keyed by end effector.
            source_candidate_diagnostics: Candidate choices and earlier source rejections.
            failure: Expected preparation, validation or replay error.
            validated_episode: Validated schedule when replay was attempted, or None.
            tracking_paths: Paths used to compare final actual and target poses.
            planning_env_id: Planning context to reuse, or None for the replay environment.

        Returns:
            Failed attempt with stage, source, motion and task diagnostics.
        """

        planners = self.planning_contexts[env_id if planning_env_id is None else planning_env_id].group_planners
        validation_diagnostics = (
            _validated_episode_diagnostics(validated_episode, None) if validated_episode is not None else ()
        )
        tracking_diagnostics = (
            _terminal_tracking_diagnostics(data_generator.datastream, planners, tracking_paths, env_id)
            if tracking_paths is not None
            else ()
        )
        return AlgorithmEpisodeResult(
            success=False,
            diagnostics=(
                ("stage", failure.stage),
                ("replayed", "true" if validated_episode is not None else "false"),
                ("message", str(failure)),
                *_source_demo_diagnostics(data_generator, selected_source),
                *source_candidate_diagnostics,
                *validation_diagnostics,
                *failure.diagnostics,
                *tracking_diagnostics,
                *self._attempt_diagnostics_collect(data_generator, env_id, success_term),
            ),
        )

    def _episode_validate(
        self,
        *,
        data_generator: DataGenerator,
        episode_paths: EpisodePaths,
        initial_object_poses: Mapping[str, torch.Tensor],
        initial_joint_positions: torch.Tensor,
        schedule_spec: ScheduleSpec,
        env_id: int = 0,
    ) -> _ValidatedEpisode:
        """Find conflicts, solve a schedule and check the combined paths.

        Args:
            data_generator: Generator providing task and embodiment interfaces.
            episode_paths: Complete paths and carried-object geometry.
            initial_object_poses: Object transforms at reset [m].
            initial_joint_positions: Reset joint positions in articulation order [rad, m].
            schedule_spec: Collision margin, timing strategy and task constraints.
            env_id: Planning context used for collision checks.

        Returns:
            Checked paths, schedule and optional shared-joint trajectory.
        """

        conflicts = self._conflicts_find(dict(episode_paths.arm_paths), schedule_spec.collision_margin_m, env_id)
        schedule = self._episode_schedule_solve(
            data_generator=data_generator,
            episode_paths=episode_paths,
            conflicts=conflicts,
            schedule_spec=schedule_spec,
        )
        return self._episode_validate_scheduled(
            data_generator=data_generator,
            episode_paths=episode_paths,
            initial_object_poses=initial_object_poses,
            initial_joint_positions=initial_joint_positions,
            schedule_spec=schedule_spec,
            conflicts=conflicts,
            schedule=schedule,
            env_id=env_id,
        )

    def _episode_schedule_solve(
        self,
        *,
        data_generator: DataGenerator,
        episode_paths: EpisodePaths,
        conflicts: tuple[ArmWaypointConflict, ...],
        schedule_spec: ScheduleSpec,
    ) -> Schedule:
        """Solve a CPU schedule after collision checks identify conflicting waypoints.

        Args:
            data_generator: Generator providing segment resource requirements.
            episode_paths: Complete unscheduled paths.
            conflicts: Cross-arm waypoint pairs that cannot overlap.
            schedule_spec: Timing strategy and task constraints.

        Returns:
            Schedule satisfying collision and task constraints.

        Raises:
            EpisodeGenerationError: No valid schedule is returned by the solver.
        """

        from autodata_core.schedulegen.scheduling import schedule_arm_paths
        from autodata_core.scheduling import ScheduleFailure

        arm_paths = episode_paths.arm_paths
        schedule = schedule_arm_paths(
            arm_paths,
            conflicts,
            schedule_spec,
            segment_resource_ids=data_generator.datastream.get_segment_resource_ids(),
        )
        if isinstance(schedule, ScheduleFailure):
            raise EpisodeGenerationError(
                "scheduling",
                schedule.message,
                (
                    (
                        "object_collision_spheres",
                        _object_collision_spheres_format(episode_paths.object_collision_spheres),
                    ),
                    ("collision_conflicts", str(len(conflicts))),
                    ("path_layout", _path_layout_format(arm_paths)),
                    ("initial_conflicts", _endpoint_conflicts_format(arm_paths, conflicts, terminal=False)),
                    ("terminal_conflicts", _endpoint_conflicts_format(arm_paths, conflicts, terminal=True)),
                    ("deepest_conflicts", _deepest_conflicts_format(conflicts)),
                ),
            )
        return schedule

    def _episode_validate_scheduled(
        self,
        *,
        data_generator: DataGenerator,
        episode_paths: EpisodePaths,
        initial_object_poses: Mapping[str, torch.Tensor],
        initial_joint_positions: torch.Tensor,
        schedule_spec: ScheduleSpec,
        conflicts: tuple[ArmWaypointConflict, ...],
        schedule: Schedule,
        env_id: int = 0,
    ) -> _ValidatedEpisode:
        """Check a solved schedule without repeating collision discovery or optimization.

        Project object attachments and contact permissions onto execution ticks, then validate
        synchronized robot/object states. Return replay targets and diagnostics only after all
        configured geometric checks pass; expected failures raise :class:`EpisodeGenerationError`.

        Args:
            data_generator: Generator providing task and embodiment interfaces.
            episode_paths: Complete paths and carried-object geometry.
            initial_object_poses: Object transforms at reset [m].
            initial_joint_positions: Reset joint positions in articulation order [rad, m].
            schedule_spec: Collision margin and timing requirements.
            conflicts: Cross-arm waypoint pairs used by the scheduler.
            schedule: Solved execution ticks for all segments.
            env_id: Planning context used for collision checks.

        Returns:
            Checked paths, schedule, replay targets and diagnostics.
        """

        from autodata_core.schedulegen.collision import (
            scheduled_object_spheres,
            scheduled_robot_object_validate,
            scheduled_robot_validate,
            scheduled_world_validate,
        )
        from autodata_core.schedulegen.scheduling import (
            project_object_states,
            schedule_object_events,
            schedule_object_poses,
            schedule_robot_contacts,
            schedule_world_contacts,
        )
        from autodata_interfaces.tasks.segment_spec import SegmentEventType

        context = self.planning_contexts[env_id]
        planners = context.group_planners
        full_state_planner = context.full_state
        datastream = data_generator.datastream
        arm_paths = episode_paths.arm_paths
        diagnostics_before_success = [
            ("object_collision_spheres", _object_collision_spheres_format(episode_paths.object_collision_spheres)),
            ("collision_conflicts", str(len(conflicts))),
            ("makespan_ticks", str(schedule.makespan_ticks)),
            ("schedule_segments", _schedule_segments_format(schedule)),
        ]
        present_segments = {segment_id for path in arm_paths.values() for segment_id in path.segment_bounds}
        full_state_diagnostics: tuple[tuple[str, str], ...] = ()
        object_state_diagnostics: tuple[tuple[str, str], ...] = ()
        try:
            # Retain stage identifiers used by saved reports; these operations map paths to ticks.
            try:
                schedule_stage = "object_event_lowering"
                scheduled_object_events = schedule_object_events(arm_paths, schedule)
                diagnostics_before_success.append(
                    ("scheduled_object_events", _scheduled_object_events_format(scheduled_object_events))
                )
                schedule_stage = "object_pose_lowering"
                scheduled_object_poses = schedule_object_poses(arm_paths, schedule)
                diagnostics_before_success.append(
                    ("scheduled_object_poses", _scheduled_object_poses_format(scheduled_object_poses))
                )
                schedule_stage = "world_contact_lowering"
                scheduled_world_contacts = schedule_world_contacts(
                    arm_paths,
                    schedule,
                    {
                        segment_id: contacts
                        for segment_id, contacts in datastream.get_segment_world_contacts().items()
                        if segment_id in present_segments
                    },
                )
                diagnostics_before_success.append(
                    ("scheduled_world_contacts", _scheduled_world_contacts_format(scheduled_world_contacts))
                )
                schedule_stage = "robot_contact_lowering"
                scheduled_robot_contacts = schedule_robot_contacts(
                    arm_paths,
                    schedule,
                    {
                        segment_id: contacts
                        for segment_id, contacts in datastream.get_segment_robot_contacts().items()
                        if segment_id in present_segments
                    },
                    scheduled_object_events,
                )
                diagnostics_before_success.append(
                    ("scheduled_robot_contacts", _scheduled_robot_contacts_format(scheduled_robot_contacts))
                )
            except AssertionError as error:
                raise EpisodeGenerationError(schedule_stage, str(error)) from error

            tool_poses_by_eef: Mapping[str, torch.Tensor] | None = None
            full_state_visualization_index: int | None = None
            scheduled_robot = None
            joint_trajectory: NamedJointTrajectory | None = None
            if full_state_planner is not None:
                joint_trajectory = _scheduled_joint_trajectory_solve(
                    datastream,
                    planners,
                    full_state_planner,
                    arm_paths,
                    schedule,
                    initial_joint_positions,
                    collision_margin_m=schedule_spec.collision_margin_m,
                    env_id=env_id,
                )
                scheduled_robot = scheduled_robot_validate(
                    datastream,
                    planners,
                    full_state_planner,
                    arm_paths,
                    schedule,
                    initial_joint_positions,
                    joint_trajectory=joint_trajectory,
                    collision_margin_m=schedule_spec.collision_margin_m,
                    densify_factor=self.collision_densify_factor,
                )
                tool_poses_by_eef = scheduled_robot.tool_poses_by_eef
                full_state_visualization_index = scheduled_robot.visualization_index
                full_state_diagnostics = (
                    ("full_state_self_collision_samples", str(scheduled_robot.self_collision_sample_count)),
                )
                if joint_trajectory is not None:
                    full_state_diagnostics += (("whole_body_realized_joints", ",".join(joint_trajectory.joint_names)),)
            elif scheduled_world_contacts or scheduled_robot_contacts:
                raise EpisodeGenerationError(
                    "full_state_contact_resolution",
                    "task contact policy requires the designated full-state collision context",
                )

            object_spheres: Mapping[str, torch.Tensor] = {}
            if episode_paths.object_collision_spheres:
                if tool_poses_by_eef is None or scheduled_robot is None:
                    raise EpisodeGenerationError(
                        "object_state_projection",
                        "scheduled object state requires the designated full-state kinematic context",
                    )
                try:
                    object_states = project_object_states(
                        initial_object_poses,
                        tool_poses_by_eef,
                        tuple(episode_paths.object_collision_spheres),
                        scheduled_object_events,
                        scheduled_object_poses,
                        makespan_ticks=schedule.makespan_ticks,
                    )
                except AssertionError as error:
                    raise EpisodeGenerationError("object_state_projection", str(error)) from error
                object_state_diagnostics = (("scheduled_object_states", _object_states_format(object_states)),)
                object_spheres = scheduled_object_spheres(
                    object_states,
                    episode_paths.object_collision_spheres,
                    makespan_ticks=schedule.makespan_ticks,
                )
                if full_state_visualization_index is not None:
                    assert full_state_planner is not None
                    _scheduled_object_spheres_animate(
                        object_spheres,
                        full_state_planner.plan_visualizer,
                        full_state_visualization_index,
                    )

            if scheduled_robot is not None:
                assert full_state_planner is not None
                robot_object_samples, waived_robot_contact_samples = scheduled_robot_object_validate(
                    full_state_planner,
                    scheduled_robot,
                    object_spheres,
                    collision_margin_m=schedule_spec.collision_margin_m,
                    robot_contacts=scheduled_robot_contacts,
                )
                robot_world_samples, object_world_samples, waived_world_contact_samples = scheduled_world_validate(
                    full_state_planner,
                    scheduled_robot,
                    object_spheres,
                    episode_paths.object_collision_spheres,
                    collision_margin_m=schedule_spec.collision_margin_m,
                    object_manipulation_starts={
                        object_ref: min(
                            (
                                schedule.ticks_for(event.path_event.segment_id)[0]
                                for event in scheduled_object_events
                                if event.path_event.spec.event_type == SegmentEventType.ATTACH
                                and event.path_event.spec.object_ref == object_ref
                            ),
                            default=schedule.makespan_ticks + 1,
                        )
                        for object_ref in object_spheres
                    },
                    world_contacts=scheduled_world_contacts,
                )
                full_state_diagnostics += (
                    ("full_state_robot_world_collision_samples", str(robot_world_samples)),
                    ("full_state_robot_object_collision_samples", str(robot_object_samples)),
                    ("full_state_robot_object_contact_samples", str(waived_robot_contact_samples)),
                    ("full_state_object_world_collision_samples", str(object_world_samples)),
                    ("full_state_object_world_contact_samples", str(waived_world_contact_samples)),
                )
        except EpisodeGenerationError as failure:
            raise EpisodeGenerationError(
                failure.stage,
                str(failure),
                (
                    *diagnostics_before_success,
                    *full_state_diagnostics,
                    *object_state_diagnostics,
                    *failure.diagnostics,
                ),
            ) from failure

        return _ValidatedEpisode(
            arm_paths=arm_paths,
            schedule=schedule,
            replay_joint_trajectory=(
                NamedJointTrajectory(tuple(full_state_planner.active_joint_names), scheduled_robot.joint_positions)
                if scheduled_robot is not None and joint_trajectory is not None
                else None
            ),
            diagnostics_before_success=tuple(diagnostics_before_success),
            diagnostics_after_success=(*full_state_diagnostics, *object_state_diagnostics),
        )

    def _attempt_diagnostics_collect(
        self,
        data_generator: DataGenerator,
        env_id: int,
        success_term: TerminationTermCfg,
    ) -> tuple[tuple[str, str], ...]:
        """Collect task diagnostics without hiding the attempt outcome."""

        if self.attempt_diagnostics is None:
            return ()
        try:
            diagnostics = self.attempt_diagnostics(data_generator.datastream, env_id, success_term)
            assert isinstance(diagnostics, tuple), "attempt diagnostics callback must return a tuple"
            assert all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], str)
                for item in diagnostics
            ), "attempt diagnostics must contain string name/value pairs"
            names = tuple(name for name, _ in diagnostics)
            assert len(names) == len(set(names)), "attempt diagnostic names must be unique"
            assert all(name.startswith("final_") for name in names), "attempt diagnostic names must start with final_"
            return diagnostics
        except Exception as error:
            return (("attempt_diagnostics_error", f"{type(error).__name__}: {error}"),)

    def _conflicts_find(
        self, arm_paths: dict[str, ArmPath], margin_m: float, env_id: int
    ) -> tuple[ArmWaypointConflict, ...]:
        """Find cross-arm conflicts using this environment's planning models."""

        from autodata_core.schedulegen.collision import find_waypoint_conflicts

        planners = self.planning_contexts[env_id].group_planners
        conflicts: list[ArmWaypointConflict] = []
        for first_name, second_name in combinations(sorted(arm_paths), 2):
            conflicts.extend(
                find_waypoint_conflicts(
                    arm_paths[first_name],
                    arm_paths[second_name],
                    planners[first_name],
                    planners[second_name],
                    collision_margin_m=margin_m,
                    densify_factor=self.collision_densify_factor,
                )
            )
        return tuple(conflicts)

    async def _schedule_replay(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        success_term: TerminationTermCfg,
        env_action_queue: Any,
        arm_paths: dict[str, ArmPath],
        schedule: Schedule,
        joint_trajectory: NamedJointTrajectory | None = None,
        planning_env_id: int | None = None,
    ) -> tuple[bool, int | None]:
        """Replay scheduled commands and accept only terminal task success.

        Args:
            data_generator: Owner of the datastream and shared waypoint execution.
            env_id: Environment to execute in.
            success_term: Task success predicate evaluated after each command.
            env_action_queue: Optional queue used by asynchronous environment stepping.
            arm_paths: Complete unscheduled paths keyed by end-effector name.
            schedule: Execution ticks, including waits and terminal holds on each lane.
            joint_trajectory: Optional coupled posture targets in named joint order.
            planning_env_id: Kinematic context to reuse, or None for the replay environment's context.

        Returns:
            Terminal task success and the first successful tick, if any. A transient success
            alone is not accepted; configured final holds and convergence commands still run.

        Raises:
            EpisodeGenerationError: Terminal tool targets do not converge within the step limit.
            AssertionError: Replay produces non-finite robot joint positions.
        """

        planners = self.planning_contexts[env_id if planning_env_id is None else planning_env_id].group_planners
        timelines = {name: _scheduled_waypoints_build(path, schedule) for name, path in arm_paths.items()}
        datastream = data_generator.datastream
        policy = datastream.get_generation_policy()
        final_hold = policy.final_hold_steps
        terminal_success = False
        first_success_tick = None
        for tick in range(schedule.makespan_ticks + final_hold):
            if joint_trajectory is not None:
                posture_index = min(tick, schedule.makespan_ticks - 1)
                datastream.set_robot_joint_posture_target(
                    joint_trajectory.joint_names,
                    joint_trajectory.positions[posture_index],
                    env_id,
                )
            waypoints = {name: timeline[min(tick, schedule.makespan_ticks - 1)] for name, timeline in timelines.items()}
            tick_success = await MultiWaypoint(waypoints).execute(
                datastream=datastream,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            assert bool(
                torch.isfinite(datastream.get_robot_joint_positions(env_ids=[env_id])).all()
            ), f"Physics produced non-finite robot joints at replay tick {tick}."
            if tick_success and first_success_tick is None:
                first_success_tick = tick
            terminal_success = tick_success

        # Export only a task state that survives the complete scheduled/post-hold replay. A
        # transient success remains observable through first_success_tick but is not accepted.
        if terminal_success or policy.terminal_convergence_steps == 0:
            return terminal_success, first_success_tick
        terminal_waypoints = {name: timeline[-1] for name, timeline in timelines.items()}
        for correction_step in range(policy.terminal_convergence_steps + 1):
            errors = {
                name: _terminal_pose_error(datastream, planners[name], env_id, waypoint.pose)
                for name, waypoint in terminal_waypoints.items()
            }
            converged = all(
                position_error <= planners[name].config.position_threshold
                and rotation_error <= planners[name].config.rotation_threshold
                for name, (position_error, rotation_error) in errors.items()
            )
            if converged:
                return terminal_success, first_success_tick
            if correction_step == policy.terminal_convergence_steps:
                error_text = ", ".join(
                    f"{name}=({position_error:.6f}m,{rotation_error:.6f}rad)"
                    for name, (position_error, rotation_error) in sorted(errors.items())
                )
                raise EpisodeGenerationError(
                    "terminal_convergence",
                    f"terminal targets did not converge after {policy.terminal_convergence_steps} steps: {error_text}",
                )
            if joint_trajectory is not None:
                datastream.set_robot_joint_posture_target(
                    joint_trajectory.joint_names,
                    joint_trajectory.positions[-1],
                    env_id,
                )
            tick_success = await MultiWaypoint(terminal_waypoints).execute(
                datastream=datastream,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            if tick_success and first_success_tick is None:
                first_success_tick = schedule.makespan_ticks + final_hold + correction_step
            terminal_success = tick_success
        return terminal_success, first_success_tick

    @staticmethod
    def _state_restore(datastream: Any, initial_state: dict, env_id: int) -> None:
        """Restore scene state and actuator targets before replay or a planning failure."""

        env_ids = torch.tensor([env_id], dtype=torch.int64, device=datastream.device)
        env = datastream.get_env()
        env.scene.reset_to(initial_state, env_ids=env_ids, is_relative=True)


def _validated_episode_diagnostics(
    episode: _ValidatedEpisode,
    first_success_tick: int | None,
) -> tuple[tuple[str, str], ...]:
    """Combine path-check and replay diagnostics for the attempt result."""

    return (
        *episode.diagnostics_before_success,
        ("first_success_tick", "none" if first_success_tick is None else str(first_success_tick)),
        *episode.diagnostics_after_success,
    )


def _path_layout_format(arm_paths: Mapping[str, ArmPath]) -> str:
    """Summarize path lengths and half-open segment waypoint bounds."""

    return "; ".join(
        f"{eef_name}[{len(path.waypoints)}]:"
        + ",".join(f"{segment_id}={start}:{end}" for segment_id, (start, end) in path.segment_bounds.items())
        for eef_name, path in sorted(arm_paths.items())
    )


def _object_collision_spheres_format(geometries: Mapping[str, ObjectCollisionSpheres]) -> str:
    """Summarize each object's sphere count and geometry checksum."""

    return ";".join(
        f"{name}:{len(geometry.spheres)}:{geometry.geometry_digest}" for name, geometry in sorted(geometries.items())
    )


def _source_demo_diagnostics(
    data_generator: DataGenerator,
    selected_source: Mapping[str, int | None],
) -> tuple[tuple[str, str], ...]:
    """Report the selected demonstration key, or unselected status, for each arm."""

    demo_keys = data_generator.src_demo_datagen_info_pool.demo_keys
    return tuple(
        (
            f"source_demo.{eef_name}",
            "unselected" if source_index is None else demo_keys[source_index],
        )
        for eef_name, source_index in sorted(selected_source.items())
    )


def _schedule_segments_format(schedule: Schedule) -> str:
    """Format the first and last command tick of each scheduled segment."""

    return ";".join(
        f"{segment_id}={ticks[0]}:{ticks[-1]}"
        for segment_id, ticks in sorted(schedule.waypoint_ticks.items(), key=lambda item: str(item[0]))
    )


def _scheduled_world_contacts_format(contacts: tuple[ScheduledWorldContact, ...]) -> str:
    """Format exact logical pairs and half-open boundary intervals for diagnostics."""

    return (
        ";".join(
            f"{contact.segment_id}:{contact.spec.object_ref}->{contact.spec.fixture_ref}"
            f"@{contact.start_tick}:{contact.stop_tick}"
            for contact in contacts
        )
        or "none"
    )


def _scheduled_robot_contacts_format(contacts: tuple[ScheduledRobotContact, ...]) -> str:
    """Format exact robot/object permissions and half-open boundary intervals."""

    return (
        ";".join(
            f"{contact.segment_id}:{contact.spec.object_ref}<-{','.join(contact.spec.robot_link_names)}"
            f"@{contact.start_tick}:{contact.stop_tick}"
            for contact in contacts
        )
        or "none"
    )


def _scheduled_object_events_format(events: tuple[ScheduledObjectEvent, ...]) -> str:
    """Format attachment transitions with their boundary ticks and source annotations."""

    if not events:
        return "none"
    formatted = []
    for scheduled in events:
        event = scheduled.path_event
        source = event.spec.annotation_key or event.spec.point.value
        formatted.append(
            f"{scheduled.tick}:{event.segment_id}[{event.local_boundary_index}]:"
            f"{event.spec.event_type.value}:{event.spec.object_ref}@{source}"
        )
    return ";".join(formatted)


def _scheduled_object_poses_format(poses: tuple[ScheduledObjectPose, ...]) -> str:
    """Summarize object-pose sample counts and tick ranges by segment."""

    if not poses:
        return "none"
    grouped: dict[tuple[SegmentId, str], list[int]] = {}
    for scheduled in poses:
        path_pose = scheduled.path_pose
        grouped.setdefault((path_pose.segment_id, path_pose.object_ref), []).append(scheduled.sample_tick)
    return ";".join(
        f"{segment_id}:{object_ref}:{len(ticks)}@{ticks[0]}:{ticks[-1]}"
        for (segment_id, object_ref), ticks in grouped.items()
    )


def _object_states_format(states: tuple[ObjectStateSample, ...]) -> str:
    """Summarize attachment intervals and pose jumps at consecutive object boundaries."""

    if not states:
        return "none"
    by_object: dict[str, list[ObjectStateSample]] = {}
    for state in states:
        by_object.setdefault(state.object_ref, []).append(state)
    formatted = []
    for object_ref, samples in sorted(by_object.items()):
        assert tuple(sample.boundary_tick for sample in samples) == tuple(range(len(samples)))
        spans: list[str] = []
        jumps: list[str] = []
        start = 0
        owner = samples[0].owner_eef_name
        for sample in samples:
            if sample.attachment_pose_jump is not None:
                position_jump, rotation_jump = sample.attachment_pose_jump
                jumps.append(f"{sample.boundary_tick}={position_jump:.6f}m/{rotation_jump:.6f}rad")
        for index, sample in enumerate((*samples[1:], None), start=1):
            next_owner = None if sample is None else sample.owner_eef_name
            if next_owner != owner:
                if owner is not None:
                    spans.append(f"{owner}@{start}:{index}")
                start = index
                owner = next_owner
        formatted.append(
            f"{object_ref}:{len(samples)}:{','.join(spans) or 'world'}:attach_jump@{','.join(jumps) or 'none'}"
        )
    return ";".join(formatted)


def _terminal_tracking_diagnostics(
    datastream: Any,
    planners: Mapping[str, CuroboPlanner],
    arm_paths: Mapping[str, ArmPath],
    env_id: int,
) -> tuple[tuple[str, str], ...]:
    """Report final tool errors [m, rad] from measured joints and planner FK."""

    diagnostics: list[tuple[str, str]] = []
    for eef_name, path in sorted(arm_paths.items()):
        position_error, rotation_error = _terminal_pose_error(
            datastream,
            planners[eef_name],
            env_id,
            path.waypoints[-1].pose,
        )
        diagnostics.extend((
            (f"final_tracking.{eef_name}.position_error_m", f"{position_error:.6f}"),
            (f"final_tracking.{eef_name}.rotation_error_rad", f"{rotation_error:.6f}"),
        ))
    return tuple(diagnostics)


def _conflict_format(conflict: ArmWaypointConflict) -> str:
    """Format a conflicting arm-waypoint pair and its sphere clearance [m]."""

    return (
        f"{conflict.first_eef_name}[{conflict.first_waypoint_index}]<->"
        f"{conflict.second_eef_name}[{conflict.second_waypoint_index}]={conflict.clearance_m:.4f}m"
    )


def _endpoint_conflicts_format(
    arm_paths: Mapping[str, ArmPath],
    conflicts: tuple[ArmWaypointConflict, ...],
    *,
    terminal: bool,
) -> str:
    """Count endpoint conflicts and format at most twelve representative pairs."""

    endpoints = {eef_name: len(path.waypoints) - 1 if terminal else 0 for eef_name, path in arm_paths.items()}
    endpoint_conflicts = [
        conflict
        for conflict in conflicts
        if conflict.first_waypoint_index == endpoints[conflict.first_eef_name]
        or conflict.second_waypoint_index == endpoints[conflict.second_eef_name]
    ]
    samples = "; ".join(_conflict_format(conflict) for conflict in endpoint_conflicts[:12])
    return f"{len(endpoint_conflicts)}: {samples}" if endpoint_conflicts else "0"


def _deepest_conflicts_format(conflicts: tuple[ArmWaypointConflict, ...]) -> str:
    """Format up to eight waypoint conflicts with the smallest sphere clearances."""

    return "; ".join(
        _conflict_format(conflict) for conflict in sorted(conflicts, key=lambda item: item.clearance_m)[:8]
    )


def _terminal_pose_error(
    datastream: Any,
    planner: CuroboPlanner,
    env_id: int,
    target_pose: torch.Tensor,
) -> tuple[float, float]:
    """Compare the target tool pose with FK of measured planner joints.

    Args:
        datastream: Source of measured, named robot joint positions.
        planner: Arm model defining joint order and environment-relative tool poses.
        env_id: Environment whose joint state is queried.
        target_pose: Environment-relative tool transform with shape ``(4, 4)`` [m].

    Returns:
        Position and rotation error [m, rad], both infinite if either pose is non-finite.
    """

    selected = datastream.get_robot_joint_positions(env_ids=[env_id], joint_names=planner.active_joint_names)
    actual_pose = planner.joint_positions_to_ee_poses(selected)[0]
    target_pose = target_pose.to(device=actual_pose.device, dtype=actual_pose.dtype)
    if not bool(torch.isfinite(actual_pose).all()) or not bool(torch.isfinite(target_pose).all()):
        return math.inf, math.inf
    position_error, rotation_error = pose_math.pose_errors(actual_pose, target_pose)
    return float(position_error.item()), float(rotation_error.item())


def _scheduled_waypoints_build(path: ArmPath, schedule: Schedule) -> tuple[Waypoint, ...]:
    """Expand a path to one command per tick, holding its initial and final waypoints.

    Args:
        path: Unscheduled waypoints and segment bounds.
        schedule: Execution ticks, including waits between waypoints.

    Returns:
        One waypoint for each execution tick.
    """

    from autodata_core.schedulegen.scheduling import get_scheduled_path_indices

    return tuple(path.waypoints[index] for index in get_scheduled_path_indices(path, schedule))


def _scheduled_object_spheres_animate(
    object_spheres: Mapping[str, torch.Tensor],
    visualizer: Any,
    schedule_index: int,
) -> None:
    """Add exact projected object spheres to an existing full-state Rerun timeline."""

    timelines = tuple(object_spheres[name][:-1] for name in sorted(object_spheres))
    assert timelines and len({len(timeline) for timeline in timelines}) == 1
    visualizer.animate_object_collision_spheres(torch.cat(timelines, dim=1), schedule_index=schedule_index)


def _scheduled_joint_trajectory_solve(
    datastream: Any,
    planners: Mapping[str, CuroboPlanner],
    full_state_planner: CuroboPlanner,
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
    initial_joint_positions: torch.Tensor,
    *,
    collision_margin_m: float,
    env_id: int,
) -> NamedJointTrajectory | None:
    """Use the embodiment controller to solve paths that share active joints.

    Args:
        datastream: Embodiment interface owning coupled trajectory solving.
        planners: Arm planners keyed by end-effector name.
        full_state_planner: Combined robot model used to check self-collision.
        arm_paths: Unscheduled paths containing tool targets and joint seeds.
        schedule: Execution ticks used to align each arm's targets and seeds.
        initial_joint_positions: Initial robot state in live articulation order [rad, m].
        collision_margin_m: Minimum self-collision clearance required by the validity callback [m].
        env_id: Environment whose controller solves the trajectory.

    Returns:
        One named joint configuration per schedule tick, or ``None`` for disjoint planning groups.

    Raises:
        EpisodeGenerationError: Coupled solving fails, is unsupported, or returns incomplete targets.
    """

    from autodata_core.schedulegen.scheduling import get_scheduled_path_indices

    owners: dict[str, list[str]] = {}
    for eef_name, planner in planners.items():
        for joint_name in getattr(planner, "active_joint_names", ()):
            owners.setdefault(joint_name, []).append(eef_name)
    shared_joint_names = {name for name, eef_names in owners.items() if len(eef_names) > 1}
    if not shared_joint_names:
        return None

    live_names = tuple(datastream.get_robot_joint_names())
    full_names = tuple(full_state_planner.active_joint_names)
    live_index = {name: index for index, name in enumerate(live_names)}
    full_index = {name: index for index, name in enumerate(full_names)}
    assert not set(full_names) - set(live_names), "full-state joints must exist in the live articulation"
    state_templates = initial_joint_positions[[live_index[name] for name in full_names]].repeat(
        schedule.makespan_ticks, 1
    )
    target_eef_poses: dict[str, torch.Tensor] = {}
    joint_seed_trajectories: dict[str, NamedJointTrajectory] = {}
    for eef_name, path in arm_paths.items():
        indices = get_scheduled_path_indices(path, schedule)
        source_indices = torch.tensor(indices, device=path.joint_positions.device)
        lane_positions = path.joint_positions.index_select(0, source_indices).to(
            device=state_templates.device,
            dtype=state_templates.dtype,
        )
        target_eef_poses[eef_name] = torch.stack([path.waypoints[index].pose for index in indices])
        joint_seed_trajectories[eef_name] = NamedJointTrajectory(
            tuple(planners[eef_name].active_joint_names),
            path.joint_positions.index_select(0, source_indices),
        )
        for lane_index, joint_name in enumerate(planners[eef_name].active_joint_names):
            state_templates[:, full_index[joint_name]] = lane_positions[:, lane_index]

    def joint_state_is_valid(joint_names: tuple[str, ...], positions: torch.Tensor, sample: int) -> bool:
        """Check a coupled candidate in the full robot state for its scheduled sample."""

        missing = set(joint_names) - set(full_names)
        assert not missing, f"coupled trajectory contains joints outside the full-state context: {sorted(missing)}"
        state = state_templates[sample].clone()
        values = positions.to(device=state.device, dtype=state.dtype)
        state[[full_index[name] for name in joint_names]] = values
        return bool(full_state_planner.joint_positions_self_collision_free(state.unsqueeze(0), collision_margin_m)[0])

    try:
        trajectory = datastream.solve_robot_joint_trajectory(
            target_eef_poses,
            joint_seed_trajectories,
            initial_joint_positions,
            min(planner.config.position_threshold for planner in planners.values()),
            min(planner.config.rotation_threshold for planner in planners.values()),
            joint_state_is_valid,
            env_id,
        )
    except Exception as error:
        # Keep the stage name used by saved generation reports.
        raise EpisodeGenerationError(
            "whole_body_realization",
            f"{type(error).__name__}: {error}",
        ) from error
    if trajectory is None:
        raise EpisodeGenerationError(
            "whole_body_realization",
            f"planning groups share joints {sorted(shared_joint_names)}, but the embodiment has no shared-joint solver",
        )
    if len(trajectory.positions) != schedule.makespan_ticks:
        raise EpisodeGenerationError(
            "whole_body_realization",
            f"coupled trajectory has {len(trajectory.positions)} samples for a {schedule.makespan_ticks}-tick schedule",
        )
    missing = shared_joint_names - set(trajectory.joint_names)
    if missing:
        raise EpisodeGenerationError(
            "whole_body_realization",
            f"coupled trajectory does not own shared joints: {sorted(missing)}",
        )
    return trajectory


__all__ = ["ScheduleGen"]
