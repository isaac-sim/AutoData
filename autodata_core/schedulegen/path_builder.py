# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build complete unscheduled paths using AutoData trajectory and planner interfaces."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from autodata_core.schedulegen.collision import arm_path_fk_validate
from autodata_core.schedulegen.types import (
    ArmPath,
    EpisodeGenerationError,
    EpisodePaths,
    PathObjectEvent,
    PathObjectPose,
)
from autodata_core.waypoint import Waypoint, WaypointSequence, WaypointTrajectory
from autodata_interfaces.motion_planners.curobo.collision_spheres import ObjectCollisionSpheres
from autodata_interfaces.tasks.schedule_constraint_spec import SegmentId
from autodata_interfaces.tasks.segment_spec import (
    DemonstrationSourceWindowSpec,
    SegmentEventPoint,
    SegmentEventSpec,
    SegmentEventType,
    SegmentKind,
    SegmentProvider,
    SegmentSpec,
)
from autodata_utils import pose_math

if TYPE_CHECKING:
    from autodata_core.data_generator import DataGenerator
    from autodata_interfaces.datastream.datastream import Datastream
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner


@dataclass(frozen=True)
class _CarriedObject:
    """Rigid object pose relative to the planner/controller tool frame."""

    object_name: str
    tool_from_object: torch.Tensor

    def pose_at_tool(self, tool_pose: torch.Tensor) -> torch.Tensor:
        """Compose the tool-relative object transform with an environment-relative tool pose."""

        return tool_pose @ self.tool_from_object


@dataclass(frozen=True)
class _PreparedSegmentGoal:
    """Resolved tool target consumed by a motion-planned segment."""

    segment_id: SegmentId | None
    entry_waypoint: Waypoint
    object_pose_overrides: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class _SegmentBuildRequest:
    """Inputs for building one segment on an end-effector path."""

    data_generator: DataGenerator
    planner: CuroboPlanner
    env_id: int
    eef_name: str
    subtask_index: int
    segment_id: SegmentId
    segment_spec: SegmentSpec
    current_joints: torch.Tensor
    passthrough: dict[str, torch.Tensor]
    previous_waypoints: tuple[Waypoint, ...]
    goal: _PreparedSegmentGoal | None
    carried_object: _CarriedObject | None
    object_collision_spheres: dict[str, ObjectCollisionSpheres]
    source_skill: WaypointTrajectory | None
    source_boundary: tuple[int, int] | None
    source_demo_index: int | None


@dataclass(frozen=True)
class _SegmentPath:
    """Waypoints, joint states and object events produced by one segment."""

    waypoints: tuple[Waypoint, ...]
    joint_positions: tuple[torch.Tensor, ...]
    carried_object: _CarriedObject | None
    object_events: tuple[PathObjectEvent, ...] = ()
    object_pose_keyframes: tuple[PathObjectPose, ...] = ()

    def __post_init__(self) -> None:
        """Require a nonempty segment with one joint configuration per waypoint."""

        assert self.waypoints, "a segment must contain at least one waypoint"
        assert len(self.waypoints) == len(self.joint_positions), "segment waypoints and joint positions must align"


@dataclass(frozen=True)
class PathBuilder:
    """Build complete arm paths before scheduling or simulator execution.

    Reuse :class:`DataGenerator` for source selection, transforms and trajectory merging.
    Add planner joint paths, object events and collision geometry needed by the scheduler.
    Object geometry is cached across attempts; path and attachment state are per attempt.

    Args:
        planners: Arm planners keyed by end-effector name.
    """

    planners: Mapping[str, CuroboPlanner]
    _object_geometry_cache: dict[str, ObjectCollisionSpheres] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def build(
        self,
        data_generator: DataGenerator,
        env_id: int,
        eef_names: tuple[str, ...],
        selected_source: dict[str, int | None],
        *,
        boundaries: dict | None = None,
    ) -> tuple[EpisodePaths, tuple[tuple[str, str], ...]]:
        """Build all arm paths, trying source candidates in their selected order.

        Args:
            data_generator: Owner of source selection, transformation and trajectory merging.
            env_id: Environment whose initial state defines the paths.
            eef_names: End effectors whose paths must be built.
            selected_source: Source-demo index per end effector, updated in place.
            boundaries: Optional source subtask boundaries already randomized for this attempt.

        Returns:
            Complete unscheduled paths and source-selection diagnostics.

        Raises:
            EpisodeGenerationError: No selected source can produce valid paths.
        """
        if boundaries is None:
            boundaries = data_generator.randomize_subtask_boundaries()
        candidates = data_generator.select_episode_source_candidates(env_id, boundaries)
        if not candidates:
            return (
                self._arm_paths_build(
                    data_generator,
                    env_id,
                    eef_names,
                    selected_source,
                    boundaries,
                ),
                (),
            )

        attempted: list[int] = []
        rejections: list[tuple[int, EpisodeGenerationError]] = []
        for candidate_index, candidate in enumerate(candidates):
            attempted.append(candidate)
            selected_source.update(dict.fromkeys(eef_names, candidate))
            try:
                episode = self._arm_paths_build(
                    data_generator,
                    env_id,
                    eef_names,
                    selected_source,
                    boundaries,
                )
            except EpisodeGenerationError as failure:
                rejections.append((candidate, failure))
                diagnostics = _source_candidate_diagnostics(data_generator, attempted, rejections)
                if candidate_index == len(candidates) - 1:
                    raise EpisodeGenerationError(failure.stage, str(failure), diagnostics) from failure
                continue
            return episode, _source_candidate_diagnostics(data_generator, attempted, rejections)

        raise AssertionError("source candidate loop must return or raise")

    def _arm_paths_build(
        self,
        data_generator: DataGenerator,
        env_id: int,
        eef_names: tuple[str, ...],
        selected_source: dict[str, int | None],
        boundaries: dict,
    ) -> EpisodePaths:
        """Build every arm path and clear temporary planner attachments afterward."""

        planners = tuple(self.planners.values())
        for planner in planners:
            planner.object_collision_spheres_reset()
        try:
            object_collision_spheres: dict[str, ObjectCollisionSpheres] = {}
            object_effect_poses: dict[tuple[SegmentId, str], torch.Tensor] = {}
            arm_paths = {}
            path_eef_names = data_generator.datastream.get_path_build_order()
            assert set(path_eef_names) == set(eef_names), "path construction order must cover every EEF"
            for eef_name in path_eef_names:
                arm_paths[eef_name] = self._arm_path_build(
                    data_generator,
                    env_id,
                    eef_name,
                    boundaries,
                    selected_source,
                    object_collision_spheres,
                    object_effect_poses,
                )
            episode_paths = EpisodePaths(arm_paths, object_collision_spheres)
        finally:
            for planner in planners:
                planner.object_collision_spheres_reset()
        return episode_paths

    def _arm_path_build(
        self,
        data_generator: DataGenerator,
        env_id: int,
        eef_name: str,
        boundaries: dict,
        selected_source: dict[str, int | None],
        object_collision_spheres: dict[str, ObjectCollisionSpheres],
        object_effect_poses: dict[tuple[SegmentId, str], torch.Tensor] | None = None,
    ) -> ArmPath:
        """Join planned, demonstrated and scripted segments for one arm.

        Args:
            data_generator: Owner of source selection and trajectory transformation.
            env_id: Environment supplying the initial joint and object state.
            eef_name: End effector whose ordered segments are built.
            boundaries: Randomized source subtask boundaries for this attempt.
            selected_source: Source-demo indices, updated during demonstration selection.
            object_collision_spheres: Shared object geometry populated by attachment events.
            object_effect_poses: Object poses produced by earlier segments, updated in place.

        Returns:
            Complete arm path with aligned joint states, segment bounds and object events.
        """

        if object_effect_poses is None:
            object_effect_poses = {}
        datastream = data_generator.datastream
        planner = self.planners[eef_name]
        planner.planning_root_pose_refresh()
        subtasks = datastream.get_subtasks(eef_name)
        assert subtasks, f"ScheduleGen requires at least one subtask for {eef_name}"
        first_segments = subtasks[0].segments
        assert first_segments, f"ScheduleGen subtask {eef_name}.{subtasks[0].name} has no segments"
        policy = datastream.get_generation_policy()
        current_joints = datastream.get_robot_joint_positions(env_ids=[env_id], joint_names=planner.active_joint_names)[
            0
        ].clone()
        if first_segments[0].provider == SegmentProvider.MOTION_PLANNER:
            current_joints = _initial_auxiliary_joint_positions_apply(planner, current_joints)
        initial_joint_positions = planner.joint_positions_to_named_positions(current_joints)
        waypoints: list[Waypoint] = []
        joint_positions: list[torch.Tensor] = []
        segment_bounds: dict[SegmentId, tuple[int, int]] = {}
        object_events: list[PathObjectEvent] = []
        object_pose_keyframes: list[PathObjectPose] = []
        previous_passthrough: dict[str, torch.Tensor] | None = None
        carried_object: _CarriedObject | None = None
        passthrough_smoothed = False
        previous_segment_waypoints: tuple[Waypoint, ...] = ()

        for subtask_index, subtask in enumerate(subtasks):
            assert subtask.segments, f"ScheduleGen subtask {eef_name}.{subtask.name} has no segments"
            demonstration_segments = tuple(
                (index, segment)
                for index, segment in enumerate(subtask.segments)
                if segment.provider == SegmentProvider.DEMONSTRATION
            )
            assert (
                len(demonstration_segments) <= 1
            ), f"ScheduleGen subtask {eef_name}.{subtask.name} supports at most one demonstration segment"
            source_skill = None
            source_boundary = None
            selected_source_index = None
            prepared_goals: dict[SegmentId, _PreparedSegmentGoal] = {}
            if demonstration_segments:
                skill_index, skill_spec = demonstration_segments[0]
                assert skill_spec.kind == SegmentKind.SKILL
                source_skill, source_boundary, selected_source_index = self._demonstration_prepare(
                    data_generator=data_generator,
                    env_id=env_id,
                    eef_name=eef_name,
                    subtask_index=subtask_index,
                    segment_index=skill_index,
                    boundaries=boundaries,
                    selected_source=selected_source,
                )
                skill_id = SegmentId(eef_name, subtask.name, skill_spec.name)
                prepared_goals[skill_id] = _PreparedSegmentGoal(skill_id, deepcopy(source_skill[0]))
                if previous_passthrough is None:
                    previous_passthrough = datastream.joint_positions_to_passthrough_actions(
                        initial_joint_positions,
                        prepared_goals[skill_id].entry_waypoint.passthrough_action,
                    )
            else:
                if previous_passthrough is None:
                    live_passthrough = datastream.get_passthrough_actions(env_ids=[env_id])
                    if live_passthrough is None:
                        raise EpisodeGenerationError(
                            "passthrough_state",
                            f"embodiment cannot recover live passthrough actions for first motion-only lane {eef_name}",
                        )
                    assert all(
                        isinstance(values, torch.Tensor) and values.ndim == 2 and len(values) == 1
                        for values in live_passthrough.values()
                    ), "live passthrough actions must contain one batched row per channel"
                    passthrough_template = {
                        name: values[0].detach().clone() for name, values in live_passthrough.items()
                    }
                    previous_passthrough = datastream.joint_positions_to_passthrough_actions(
                        initial_joint_positions,
                        passthrough_template,
                    )
                else:
                    assert (
                        previous_segment_waypoints
                    ), f"motion-only subtask {eef_name}.{subtask.name} requires a preceding segment path"

            for segment_spec in subtask.segments:
                if segment_spec.provider == SegmentProvider.SCRIPTED and not passthrough_smoothed:
                    _passthrough_transitions_smooth(
                        waypoints,
                        eef_name,
                        delay_steps=policy.skill_passthrough_delay_steps,
                        interpolation_steps=policy.skill_passthrough_interpolation_steps,
                    )
                    passthrough_smoothed = True
                    previous_passthrough = deepcopy(waypoints[-1].passthrough_action)

                segment_id = SegmentId(eef_name, subtask.name, segment_spec.name)
                goal = prepared_goals.get(segment_spec.goal_segment_id)
                if segment_spec.motion_goal is not None:
                    assert previous_passthrough is not None
                    pose_template = (
                        previous_segment_waypoints[-1].pose
                        if previous_segment_waypoints
                        else planner.joint_positions_to_ee_poses(current_joints.unsqueeze(0))[0]
                    )
                    goal = self._motion_goal_prepare(
                        data_generator=data_generator,
                        env_id=env_id,
                        segment_id=segment_id,
                        segment_spec=segment_spec,
                        pose_template=pose_template,
                        passthrough=previous_passthrough,
                        object_effect_poses=object_effect_poses,
                    )
                segment_path = self._segment_build(
                    _SegmentBuildRequest(
                        data_generator=data_generator,
                        planner=planner,
                        env_id=env_id,
                        eef_name=eef_name,
                        subtask_index=subtask_index,
                        segment_id=segment_id,
                        segment_spec=segment_spec,
                        current_joints=current_joints,
                        passthrough=previous_passthrough,
                        previous_waypoints=previous_segment_waypoints,
                        goal=goal,
                        carried_object=carried_object,
                        object_collision_spheres=object_collision_spheres,
                        source_skill=source_skill,
                        source_boundary=source_boundary,
                        source_demo_index=selected_source_index,
                    )
                )
                segment_start = len(waypoints)
                waypoints.extend(segment_path.waypoints)
                joint_positions.extend(segment_path.joint_positions)
                segment_bounds[segment_id] = segment_start, len(waypoints)
                object_events.extend(segment_path.object_events)
                object_pose_keyframes.extend(segment_path.object_pose_keyframes)
                for keyframe in segment_path.object_pose_keyframes:
                    object_effect_poses[(segment_id, keyframe.object_ref)] = keyframe.env_from_object
                current_joints = segment_path.joint_positions[-1]
                carried_object = segment_path.carried_object
                previous_passthrough = deepcopy(segment_path.waypoints[-1].passthrough_action)
                previous_segment_waypoints = segment_path.waypoints

        if not passthrough_smoothed:
            _passthrough_transitions_smooth(
                waypoints,
                eef_name,
                delay_steps=policy.skill_passthrough_delay_steps,
                interpolation_steps=policy.skill_passthrough_interpolation_steps,
            )
        joint_positions = [
            self._joint_positions_match_passthrough(
                datastream,
                planner,
                positions,
                waypoint.passthrough_action,
            )
            for waypoint, positions in zip(waypoints, joint_positions, strict=True)
        ]
        joint_position_path = torch.stack(joint_positions)
        arm_path_fk_validate(eef_name, waypoints, joint_position_path, planner, segment_bounds)
        path = ArmPath(
            eef_name=eef_name,
            waypoints=tuple(waypoints),
            joint_positions=joint_position_path,
            segment_bounds=segment_bounds,
            object_events=tuple(object_events),
            object_pose_keyframes=tuple(object_pose_keyframes),
        )
        if policy.max_joint_step_rad is None:
            return path
        return _arm_path_refine(
            path,
            planner=planner,
            datastream=datastream,
            max_joint_step_rad=policy.max_joint_step_rad,
        )

    def _demonstration_prepare(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        eef_name: str,
        subtask_index: int,
        segment_index: int,
        boundaries: dict,
        selected_source: dict[str, int | None],
    ) -> tuple[WaypointTrajectory, tuple[int, int], int]:
        """Select, transform and trim the demonstration for one skill segment.

        Args:
            data_generator: Owner of source selection and trajectory transformation.
            env_id: Environment whose object poses define the transformed skill.
            eef_name: End effector executing the skill.
            subtask_index: Subtask containing the demonstrated segment.
            segment_index: Position of the demonstration within the subtask.
            boundaries: Randomized source subtask boundaries for this attempt.
            selected_source: Source indices updated by the generator during selection.

        Returns:
            Transformed trajectory, retained source boundaries and selected demo index.
        """

        datastream = data_generator.datastream
        subtask = datastream.get_subtasks(eef_name)[subtask_index]
        skill_spec = subtask.segments[segment_index]
        source_skill = data_generator.generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=subtask_index,
            all_randomized_subtask_boundaries=boundaries,
            runtime_subtask_constraints_dict={},
            selected_src_demo_inds=selected_source,
        )
        assert len(source_skill), f"source skill {eef_name}.{subtask.name} is empty"
        has_observation_prefix = subtask_index == 0 or datastream.get_generation_policy().transform_first_robot_pose
        if has_observation_prefix and segment_index:
            # A preceding segment replaces the observed-EEF approach seed, not a demonstrated command.
            assert len(source_skill) > 1, f"source skill {eef_name}.{subtask.name} has no target commands"
            source_skill.pop_first()
        source_demo_index = selected_source[eef_name]
        assert source_demo_index is not None
        selected_boundary = boundaries[eef_name][source_demo_index, subtask_index]
        source_skill, source_boundary = self._demonstration_source_window_apply(
            data_generator=data_generator,
            eef_name=eef_name,
            subtask_index=subtask_index,
            events=skill_spec.events,
            source_demo_index=source_demo_index,
            source_boundary=(int(selected_boundary[0]), int(selected_boundary[1])),
            source_skill=source_skill,
            source_window=skill_spec.demonstration_source_window,
        )
        return source_skill, source_boundary, source_demo_index

    @staticmethod
    def _motion_goal_prepare(
        *,
        data_generator: DataGenerator,
        env_id: int,
        segment_id: SegmentId,
        segment_spec: SegmentSpec,
        pose_template: torch.Tensor,
        passthrough: dict[str, torch.Tensor],
        object_effect_poses: Mapping[tuple[SegmentId, str], torch.Tensor] | None = None,
    ) -> _PreparedSegmentGoal:
        """Resolve a configured tool target in the environment or object frame.

        Args:
            data_generator: Generator providing current object poses.
            env_id: Environment whose object frames define the target.
            segment_id: Segment receiving the target.
            segment_spec: Segment declaring the motion goal.
            pose_template: Pose tensor supplying the device and numeric type.
            passthrough: Hand and other non-pose commands retained at the target.
            object_effect_poses: Object poses predicted after earlier segments [m].

        Returns:
            Resolved tool target and any predicted object-pose overrides.
        """

        goal = segment_spec.motion_goal
        assert goal is not None and segment_spec.goal_segment_id is None
        position = pose_template.new_tensor(goal.position_m)
        quaternion = pose_template.new_tensor(goal.rotation_xyzw)
        reference_from_tool = pose_math.make_pose(position, pose_math.matrix_from_quat(quaternion))
        env_from_tool = reference_from_tool
        object_pose_overrides = {}
        if goal.object_ref:
            if goal.object_pose_after_segment_id is None:
                object_poses = data_generator.datastream.get_object_poses(env_ids=[env_id])
                if goal.object_ref not in object_poses:
                    raise EpisodeGenerationError(
                        "motion_goal",
                        f"motion goal for {segment_id} references unknown object {goal.object_ref!r}",
                    )
                env_from_object = object_poses[goal.object_ref][0]
            else:
                effect_key = goal.object_pose_after_segment_id, goal.object_ref
                if object_effect_poses is None or effect_key not in object_effect_poses:
                    raise EpisodeGenerationError(
                        "task_effect",
                        f"motion goal for {segment_id} has no terminal object pose from "
                        f"{goal.object_pose_after_segment_id}",
                    )
                env_from_object = object_effect_poses[effect_key]
                object_pose_overrides[goal.object_ref] = env_from_object
            env_from_tool = env_from_object.to(pose_template) @ reference_from_tool
        return _PreparedSegmentGoal(
            None,
            Waypoint(env_from_tool, deepcopy(passthrough)),
            MappingProxyType(object_pose_overrides),
        )

    def _segment_build(
        self,
        request: _SegmentBuildRequest,
    ) -> _SegmentPath:
        """Build a segment using its configured trajectory source."""

        builder_by_provider = {
            SegmentProvider.MOTION_PLANNER: self._motion_segment_build,
            SegmentProvider.DEMONSTRATION: self._demonstration_segment_build,
            SegmentProvider.SCRIPTED: self._scripted_segment_build,
        }
        builder = builder_by_provider.get(request.segment_spec.provider)
        if builder is None:
            raise EpisodeGenerationError(
                "segment_provider",
                f"segment provider {request.segment_spec.provider.value!r} is not integrated for {request.segment_id}",
            )
        return builder(request)

    def _motion_segment_build(
        self,
        request: _SegmentBuildRequest,
    ) -> _SegmentPath:
        """Plan a transit to the configured segment goal."""

        assert request.goal is not None, f"unresolved segment goal for {request.segment_id}"
        assert request.segment_spec.goal_segment_id == request.goal.segment_id
        carried_object_pose = None
        if request.carried_object is not None:
            assert request.previous_waypoints, "a carried-object transit requires a preceding segment"
            carried_object_pose = request.carried_object.pose_at_tool(request.previous_waypoints[-1].pose)
        waypoints, joint_positions = self._transit_plan(
            request.data_generator,
            request.planner,
            request.env_id,
            request.goal.entry_waypoint.pose,
            request.passthrough,
            request.current_joints,
            request.carried_object.object_name if request.carried_object is not None else None,
            carried_object_pose,
            request.goal.object_pose_overrides,
        )
        return _SegmentPath(
            waypoints=tuple(waypoints),
            joint_positions=tuple(joint_positions),
            carried_object=request.carried_object,
        )

    def _demonstration_segment_build(
        self,
        request: _SegmentBuildRequest,
    ) -> _SegmentPath:
        """Merge a transformed demonstration and map its object events to path samples.

        Args:
            request: Prepared source skill, entry state and attachment geometry.

        Returns:
            Skill waypoints, IK joint samples and resolved object events.
        """

        assert request.source_skill is not None
        assert request.source_boundary is not None
        assert request.source_demo_index is not None
        has_execution_predecessor = bool(request.previous_waypoints)
        source_length = len(request.source_skill)
        merged_skill = request.data_generator.merge_eef_subtask_trajectory(
            env_id=request.env_id,
            eef_name=request.eef_name,
            subtask_index=request.subtask_index,
            prev_executed_traj=list(request.previous_waypoints) if has_execution_predecessor else None,
            subtask_trajectory=request.source_skill,
            force_use_prev_traj=has_execution_predecessor,
        )
        if not has_execution_predecessor:
            initial_joints = request.current_joints.clone()
            initial_pose = request.planner.joint_positions_to_ee_poses(initial_joints.unsqueeze(0))[0]
            merged_skill.insert(
                0,
                Waypoint(
                    pose=initial_pose,
                    passthrough_action=deepcopy(request.passthrough),
                    noise=0.0,
                    joint_seed=initial_joints.clone(),
                ),
            )
        skill_events, object_pose_keyframes = self._skill_events_resolve(
            data_generator=request.data_generator,
            planner=request.planner,
            segment_id=request.segment_id,
            events=request.segment_spec.events,
            source_demo_index=request.source_demo_index,
            source_boundary=request.source_boundary,
            source_length=source_length,
            merged_skill=merged_skill,
        )
        current_joints = request.current_joints
        carried_object = request.carried_object
        joint_positions: list[torch.Tensor] = []
        for skill_index, waypoint in enumerate(merged_skill):
            for event in skill_events.get(skill_index, ()):
                carried_object = self._object_event_apply(
                    request.planner,
                    request.env_id,
                    current_joints,
                    event,
                    carried_object,
                    request.object_collision_spheres,
                )
            solution = current_joints
            if has_execution_predecessor or skill_index:
                solution = request.planner.solve_ik_kinematic(waypoint.pose, current_joints)
                if solution is None:
                    raise EpisodeGenerationError(
                        "inverse_kinematics",
                        f"IK failed for {request.segment_id} at waypoint {skill_index}/{len(merged_skill)}",
                    )
                solution = self._joint_positions_match_passthrough(
                    request.data_generator.datastream,
                    request.planner,
                    solution,
                    waypoint.passthrough_action,
                )
            current_joints = solution
            joint_positions.append(solution)
        for event in skill_events.get(len(merged_skill), ()):
            carried_object = self._object_event_apply(
                request.planner,
                request.env_id,
                current_joints,
                event,
                carried_object,
                request.object_collision_spheres,
            )
        return _SegmentPath(
            waypoints=tuple(merged_skill),
            joint_positions=tuple(joint_positions),
            carried_object=carried_object,
            object_events=tuple(event for slot in sorted(skill_events) for event in skill_events[slot]),
            object_pose_keyframes=object_pose_keyframes,
        )

    def _scripted_segment_build(self, request: _SegmentBuildRequest) -> _SegmentPath:
        """Build a hold or Cartesian retreat while preserving attachment state.

        Args:
            request: Scripted segment, entry state and current hand commands.

        Returns:
            Scripted waypoints and joint samples with the existing attachment state.
        """
        data_generator, planner = request.data_generator, request.planner
        eef_name, segment_spec = request.eef_name, request.segment_spec
        current_joints, passthrough = request.current_joints, request.passthrough

        assert segment_spec.provider == SegmentProvider.SCRIPTED
        assert segment_spec.kind in {SegmentKind.HOLD, SegmentKind.TRANSIT}
        assert segment_spec.duration_steps > 0
        topology_links = set(planner.config.hand_link_names)
        task_links = {link_name for contact in segment_spec.robot_contacts for link_name in contact.robot_link_names}
        missing_links = task_links - topology_links
        assert not missing_links, f"task contact links are outside the planner hand topology: {sorted(missing_links)}"

        assert eef_name in passthrough, f"passthrough actions are missing the {eef_name!r} channel"
        target_passthrough = deepcopy(passthrough)
        if segment_spec.passthrough_preset:
            preset_values = data_generator.datastream.get_eef_passthrough_preset(
                eef_name,
                segment_spec.passthrough_preset,
            )
            preset_joint_names = data_generator.datastream.get_robot_eef_joint_names(eef_name)
            assert len(preset_values) == len(preset_joint_names)
            configured_auxiliary = planner.config.initial_auxiliary_joint_positions
            assert set(configured_auxiliary) <= set(preset_joint_names)
            assert set(preset_joint_names) <= set(
                planner.joint_positions_to_named_positions(current_joints)
            ), "passthrough preset joints must be active planner joints or their mimic followers"
            target_passthrough[eef_name] = passthrough[eef_name].new_tensor(preset_values)
            assert (
                target_passthrough[eef_name].shape == passthrough[eef_name].shape
            ), "release preset must match the incoming passthrough shape"
        seed_passthrough = passthrough if segment_spec.kind == SegmentKind.TRANSIT else target_passthrough
        seed_joints = self._joint_positions_match_passthrough(
            data_generator.datastream,
            planner,
            current_joints,
            seed_passthrough,
        )
        start_pose = planner.joint_positions_to_ee_poses(seed_joints.unsqueeze(0))[0]
        offset = start_pose.new_tensor(segment_spec.cartesian_offset_m or (0.0, 0.0, 0.0))
        waypoints: list[Waypoint] = []
        joint_positions: list[torch.Tensor] = []
        for step in range(1, segment_spec.duration_steps + 1):
            progress = step / segment_spec.duration_steps
            step_passthrough = target_passthrough
            if segment_spec.kind == SegmentKind.TRANSIT and segment_spec.passthrough_preset:
                step_passthrough = deepcopy(target_passthrough)
                step_passthrough[eef_name] = torch.lerp(
                    passthrough[eef_name],
                    target_passthrough[eef_name],
                    progress,
                )
            target_pose = start_pose.clone()
            target_pose[:3, 3] += offset * progress
            solution = seed_joints
            if segment_spec.kind == SegmentKind.TRANSIT:
                ik_seed = self._joint_positions_match_passthrough(
                    data_generator.datastream,
                    planner,
                    seed_joints,
                    step_passthrough,
                )
                solution = planner.solve_ik_kinematic(target_pose, ik_seed)
                if solution is None:
                    raise EpisodeGenerationError(
                        "inverse_kinematics",
                        f"IK failed for scripted segment {segment_spec.name!r} at "
                        f"waypoint {step - 1}/{segment_spec.duration_steps}",
                    )
                solution = self._joint_positions_match_passthrough(
                    data_generator.datastream,
                    planner,
                    solution,
                    step_passthrough,
                )
                seed_joints = solution
            waypoints.append(
                Waypoint(
                    pose=target_pose,
                    passthrough_action=deepcopy(step_passthrough),
                    noise=0.0,
                    joint_seed=solution.clone(),
                )
            )
            joint_positions.append(solution.clone())
        return _SegmentPath(tuple(waypoints), tuple(joint_positions), request.carried_object)

    def _demonstration_source_window_apply(
        self,
        *,
        data_generator: DataGenerator,
        eef_name: str,
        subtask_index: int,
        events: tuple[SegmentEventSpec, ...],
        source_demo_index: int,
        source_boundary: tuple[int, int],
        source_skill: WaypointTrajectory,
        source_window: DemonstrationSourceWindowSpec | None,
    ) -> tuple[WaypointTrajectory, tuple[int, int]]:
        """Trim a demonstration after its annotated event and configured trailing samples.

        Args:
            data_generator: Generator providing source annotations.
            eef_name: End effector owning the skill.
            subtask_index: Subtask containing the skill.
            events: Object events declared for this segment.
            source_demo_index: Selected source episode index.
            source_boundary: Original half-open action bounds in the source episode.
            source_skill: Transformed source waypoints, including any observed-pose prefix.
            source_window: Optional event-anchored window to retain.

        Returns:
            Retained waypoints and their half-open source action bounds.
        """

        if source_window is None:
            return source_skill, source_boundary
        boundary_start, boundary_end = source_boundary
        source_prefix = len(source_skill) - (boundary_end - boundary_start)
        assert source_prefix in (0, 1), "source skill may only prepend its recorded EEF pose"
        anchors = tuple(
            event
            for event in events
            if event.point == SegmentEventPoint.SOURCE_ANNOTATION
            and event.annotation_key == source_window.anchor_annotation_key
        )
        assert len(anchors) == 1, "source-window anchor must match exactly one segment event"
        pool = data_generator.src_demo_datagen_info_pool
        demo_key = pool.demo_keys[source_demo_index]

        def sample_index_resolve(event: SegmentEventSpec) -> int:
            """Resolve a source sample and classify annotation failures as source-window errors."""

            try:
                return pool.get_object_event_index(source_demo_index, event.annotation_key)
            except Exception as error:
                raise EpisodeGenerationError(
                    "source_window",
                    f"source annotation {event.annotation_key!r} query failed for {demo_key!r}: "
                    f"{type(error).__name__}: {error}",
                ) from error

        anchor_sample = sample_index_resolve(anchors[0])
        effective_end = anchor_sample + 1 + source_window.post_event_sample_count
        if not boundary_start < effective_end <= boundary_end:
            raise EpisodeGenerationError(
                "source_window",
                f"{eef_name} subtask {subtask_index} source window [{boundary_start}, {effective_end}) "
                f"must be a non-empty subset of [{boundary_start}, {boundary_end})",
            )
        for event in events:
            if event.point != SegmentEventPoint.SOURCE_ANNOTATION:
                continue
            sample = sample_index_resolve(event)
            if not boundary_start <= sample < effective_end:
                raise EpisodeGenerationError(
                    "source_window",
                    f"source event {event.annotation_key!r} sample {sample} is outside retained "
                    f"boundaries [{boundary_start}, {effective_end})",
                )

        retained_length = source_prefix + effective_end - boundary_start
        assert 0 < retained_length <= len(source_skill), "source window cannot expand its demonstration"
        retained = WaypointTrajectory()
        retained.add_waypoint_sequence(
            WaypointSequence(sequence=[source_skill[index] for index in range(retained_length)])
        )
        return retained, (boundary_start, effective_end)

    def _object_event_apply(
        self,
        planner: CuroboPlanner,
        env_id: int,
        current_joints: torch.Tensor,
        event: PathObjectEvent,
        carried_object: _CarriedObject | None,
        object_collision_spheres: dict[str, ObjectCollisionSpheres],
    ) -> _CarriedObject | None:
        """Update attachment state and reuse the object's fitted collision spheres.

        Args:
            planner: Arm planner whose attachment model is updated.
            env_id: Environment containing the object geometry.
            current_joints: Arm-local joint positions at the event [rad, m].
            event: Attach or detach event at this path boundary.
            carried_object: Object carried before the event, or None.
            object_collision_spheres: Episode geometry cache, updated on attachment.

        Returns:
            Carried-object state after the event, or None after release.
        """

        updated_carried_object = _carried_object_apply(carried_object, event)
        if event.spec.event_type == SegmentEventType.ATTACH:
            assert event.tool_from_object is not None
            object_ref = event.spec.object_ref
            episode_geometry = object_collision_spheres.get(object_ref)
            runtime_geometry = self._object_geometry_cache.get(object_ref)
            if episode_geometry is not None and runtime_geometry is not None:
                assert episode_geometry is runtime_geometry, f"episode geometry changed for {object_ref!r}"
            cached_geometry = episode_geometry or runtime_geometry
            try:
                try:
                    geometry = planner.object_collision_spheres_fit(
                        object_ref,
                        env_id,
                        current_joints,
                        event.tool_from_object,
                        cached_geometry,
                    )
                finally:
                    planner.update_world()
            except Exception as error:
                raise EpisodeGenerationError(
                    "object_sphere_fit",
                    f"failed to bind attachment geometry for {object_ref!r}: {type(error).__name__}: {error}",
                ) from error
            if cached_geometry is not None and geometry is not cached_geometry:
                raise EpisodeGenerationError(
                    "object_sphere_fit",
                    f"attachment geometry changed for {object_ref!r}",
                )
            episode_geometry = object_collision_spheres.setdefault(object_ref, geometry)
            runtime_geometry = self._object_geometry_cache.setdefault(object_ref, geometry)
            if episode_geometry is not geometry or runtime_geometry is not geometry:
                raise EpisodeGenerationError("object_sphere_fit", f"attachment geometry changed for {object_ref!r}")
        return updated_carried_object

    def _skill_events_resolve(
        self,
        *,
        data_generator: DataGenerator,
        planner: CuroboPlanner,
        segment_id: SegmentId,
        events: tuple[SegmentEventSpec, ...],
        source_demo_index: int,
        source_boundary: tuple[int, int],
        source_length: int,
        merged_skill: Sequence[Waypoint],
    ) -> tuple[dict[int, list[PathObjectEvent]], tuple[PathObjectPose, ...]]:
        """Map source object events and poses to boundaries in the generated skill.

        Args:
            data_generator: Owner of the source demonstrations and object-relative transforms.
            planner: Arm model providing the attachment frame and target tool poses.
            segment_id: Segment receiving the resolved events and keyframes.
            events: Attach/detach events declared for this segment.
            source_demo_index: Selected source demonstration index.
            source_boundary: Half-open action bounds in the source demonstration.
            source_length: Skill length before merge, including any prepended recorded tool pose.
            merged_skill: Generated skill after entry interpolation and trajectory merging.

        Returns:
            Events grouped by local pre-command boundary and environment-relative object poses.
            Recorded post-command poses map to the following boundary; a segment-end event uses
            the terminal boundary after its last command.
        """

        if not events:
            return {}, ()
        pool = data_generator.src_demo_datagen_info_pool
        assert len(pool.demo_keys) == len(pool.datagen_infos), "source demo keys and episodes are misaligned"
        source_episode = pool.datagen_infos[source_demo_index]
        boundary_start, boundary_end = source_boundary
        source_prefix = source_length - (boundary_end - boundary_start)
        assert source_prefix in (0, 1), "source skill may only prepend its recorded EEF pose"
        merged_length = len(merged_skill)
        interpolation_prefix = merged_length - source_length
        assert interpolation_prefix >= 0, "merged skill cannot be shorter than its source"

        eef_name = segment_id.eef_name
        resolved: dict[int, list[PathObjectEvent]] = {}
        for event in events:
            if event.point == SegmentEventPoint.START:
                source_sample_index = boundary_start
                skill_slot = 0
            elif event.point == SegmentEventPoint.END:
                source_sample_index = boundary_end
                skill_slot = merged_length
            else:
                assert event.point == SegmentEventPoint.SOURCE_ANNOTATION
                source_sample_index = pool.get_object_event_index(source_demo_index, event.annotation_key)
                if not boundary_start <= source_sample_index <= boundary_end:
                    raise EpisodeGenerationError(
                        "source_event",
                        f"source event {event.annotation_key!r} sample {source_sample_index} is outside "
                        f"segment {segment_id} state boundaries [{boundary_start}, {boundary_end}] for command slice "
                        f"[{boundary_start}, {boundary_end})",
                    )
                skill_slot = interpolation_prefix + source_prefix + source_sample_index - boundary_start

            tool_from_object = None
            if event.event_type == SegmentEventType.ATTACH:
                tool_link_name = planner.config.ee_link_name
                assert tool_link_name is not None, "native SkillGen planners require a physical tool link"
                assert (
                    event.attachment_frame == tool_link_name
                ), f"attachment frame {event.attachment_frame!r} does not match planner tool {tool_link_name!r}"
                assert source_episode.target_eef_pose is not None and eef_name in source_episode.target_eef_pose
                assert source_episode.object_poses is not None and event.object_ref in source_episode.object_poses
                source_tool_poses = source_episode.target_eef_pose[eef_name]
                source_object_poses = source_episode.object_poses[event.object_ref]
                source_tool_sample_index = source_sample_index - 1
                if not (
                    0 <= source_tool_sample_index < len(source_tool_poses)
                    and 0 <= source_sample_index < len(source_object_poses)
                ):
                    raise EpisodeGenerationError(
                        "source_event",
                        f"attach event {event.annotation_key or event.point.value!r} boundary sample "
                        f"{source_sample_index} requires preceding tool command {source_tool_sample_index} and "
                        "a matching object state",
                    )
                source_tool_pose = source_tool_poses[source_tool_sample_index]
                source_object_pose = source_object_poses[source_sample_index]
                tool_from_object = torch.linalg.solve(source_tool_pose, source_object_pose).clone()

            assert 0 <= skill_slot <= merged_length, "event slot must lie on the skill path boundary"
            resolved.setdefault(skill_slot, []).append(
                PathObjectEvent(
                    segment_id=segment_id,
                    local_boundary_index=skill_slot,
                    spec=event,
                    tool_from_object=tool_from_object,
                )
            )

        assert source_episode.target_eef_pose is not None and eef_name in source_episode.target_eef_pose
        assert source_episode.object_poses is not None
        source_tool_poses = source_episode.target_eef_pose[eef_name]
        source_target_slot = interpolation_prefix + source_prefix
        assert 0 <= boundary_start < len(source_tool_poses)
        assert source_target_slot < merged_length
        generated_target = merged_skill[source_target_slot].pose
        source_target = source_tool_poses[boundary_start].to(
            device=generated_target.device, dtype=generated_target.dtype
        )
        source_to_generated = generated_target @ torch.linalg.inv(source_target)

        object_refs = tuple(dict.fromkeys(event.object_ref for event in events))
        source_object_poses = {}
        for object_ref in object_refs:
            assert object_ref in source_episode.object_poses, f"source episode has no object {object_ref!r}"
            source_object_poses[object_ref] = source_episode.object_poses[object_ref]
            assert boundary_start < len(source_object_poses[object_ref])
        last_sample = min(boundary_end, *(len(poses) - 1 for poses in source_object_poses.values()))
        object_poses = tuple(
            PathObjectPose(
                segment_id=segment_id,
                local_boundary_index=interpolation_prefix + source_prefix + sample_index - boundary_start,
                object_ref=object_ref,
                env_from_object=source_to_generated
                @ source_object_poses[object_ref][sample_index].to(
                    device=generated_target.device, dtype=generated_target.dtype
                ),
            )
            for sample_index in range(boundary_start, last_sample + 1)
            for object_ref in object_refs
        )
        return resolved, object_poses

    def _transit_plan(
        self,
        data_generator: DataGenerator,
        planner: CuroboPlanner,
        env_id: int,
        target_pose: torch.Tensor,
        passthrough: dict[str, torch.Tensor],
        start_joints: torch.Tensor,
        carried_object: str | None,
        carried_object_pose: torch.Tensor | None,
        object_pose_overrides: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[list[Waypoint], list[torch.Tensor]]:
        """Plan a transit with object-pose overrides without moving simulator objects.

        Args:
            data_generator: Owner of the embodiment's passthrough-action conversion.
            planner: Arm planner used to build and retrieve the transit.
            env_id: Environment supplying the static planning world.
            target_pose: Environment-relative tool transform with shape ``(4, 4)`` [m].
            passthrough: Auxiliary commands held along the transit.
            start_joints: Initial configuration in the planner's active joint order [rad, m].
            carried_object: Logical object attached to the tool, or ``None``.
            carried_object_pose: Virtual environment-relative pose of the carried object [m].
            object_pose_overrides: Additional virtual object transforms in the environment frame [m].

        Returns:
            Aligned controller waypoints and planner-ordered joint configurations.

        Raises:
            EpisodeGenerationError: The planner cannot find a transit.
        """

        virtual_poses = dict(object_pose_overrides or {})
        if carried_object is not None:
            assert carried_object_pose is not None, "a carried object requires its virtual pose"
            virtual_poses[carried_object] = carried_object_pose
        succeeded = planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=carried_object,
            env_id=env_id,
            step_size=planner.step_size,
            enable_retiming=planner.step_size is not None,
            start_joint_positions=start_joints,
            force_object_attachment=carried_object is not None,
            retreat_distance=planner.config.retreat_distance if carried_object is not None else 0.0,
            object_pose_overrides=virtual_poses,
        )
        if not succeeded:
            raise EpisodeGenerationError(
                "motion_planning", f"transit planning failed for {planner.config.planning_group_name}"
            )
        poses = planner.get_planned_poses()
        joint_trajectory = planner.get_planned_joint_trajectory()
        assert joint_trajectory is not None and poses, "planner must expose both pose and named joint paths"
        assert joint_trajectory.joint_names == planner.active_joint_names, "planner joint path order is inconsistent"
        positions = joint_trajectory.positions
        assert len(poses) == len(positions), "planner path outputs are inconsistent"
        transit: list[Waypoint] = []
        transit_positions: list[torch.Tensor] = []
        for pose, position in zip(poses, positions, strict=True):
            transit_passthrough = deepcopy(passthrough)
            transit_position = self._joint_positions_match_passthrough(
                data_generator.datastream,
                planner,
                position,
                transit_passthrough,
            )
            transit.append(
                Waypoint(
                    pose=pose,
                    passthrough_action=transit_passthrough,
                    noise=planner.config.motion_noise_scale,
                )
            )
            transit_positions.append(transit_position)
        return transit, transit_positions

    @staticmethod
    def _joint_positions_match_passthrough(
        datastream: Datastream,
        planner: CuroboPlanner,
        joint_positions: torch.Tensor,
        passthrough: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Make scheduler geometry match joints commanded outside the pose controller."""

        result = joint_positions.clone()
        active_index = {name: index for index, name in enumerate(planner.active_joint_names)}
        for name, value in datastream.passthrough_actions_to_joint_positions(passthrough).items():
            if name not in active_index:
                continue
            assert isinstance(value, torch.Tensor) and value.numel() == 1
            result[active_index[name]] = value.reshape(()).to(device=result.device, dtype=result.dtype)
        return result


def _carried_object_apply(
    carried_object: _CarriedObject | None,
    event: PathObjectEvent,
) -> _CarriedObject | None:
    """Update which object the arm carries after an attach or detach event."""

    if event.spec.event_type == SegmentEventType.ATTACH:
        assert (
            carried_object is None
        ), f"cannot attach {event.spec.object_ref!r} while carrying {carried_object.object_name!r}"
        assert event.tool_from_object is not None, "attach event requires an object-to-tool transform"
        return _CarriedObject(event.spec.object_ref, event.tool_from_object)
    assert event.spec.event_type == SegmentEventType.DETACH
    assert (
        carried_object is not None and carried_object.object_name == event.spec.object_ref
    ), f"cannot detach uncarried object {event.spec.object_ref!r}"
    return None


def _initial_auxiliary_joint_positions_apply(
    planner: CuroboPlanner,
    joint_positions: torch.Tensor,
) -> torch.Tensor:
    """Apply the group's declared first-transit state without changing serial-arm joints."""

    configured = planner.config.initial_auxiliary_joint_positions
    if not configured:
        return joint_positions
    active_index = {name: index for index, name in enumerate(planner.active_joint_names)}
    unknown = set(configured) - set(active_index)
    assert not unknown, f"initial auxiliary joints are absent from planner state: {sorted(unknown)}"
    result = joint_positions.clone()
    for name, value in configured.items():
        result[active_index[name]] = value
    return result


def _source_candidate_diagnostics(
    data_generator: DataGenerator,
    attempted: Sequence[int],
    rejections: Sequence[tuple[int, EpisodeGenerationError]],
) -> tuple[tuple[str, str], ...]:
    """Summarize source candidates and their path-building failures."""

    demo_keys = data_generator.src_demo_datagen_info_pool.demo_keys
    diagnostics = (("source_candidates_tried", ",".join(demo_keys[index] for index in attempted)),)
    if not rejections:
        return diagnostics
    return diagnostics + (
        (
            "source_candidate_rejections",
            ";".join(f"{demo_keys[index]}:{failure.stage}:{failure}" for index, failure in rejections),
        ),
    )


def _passthrough_transitions_smooth(
    waypoints: list[Waypoint],
    eef_name: str,
    *,
    delay_steps: int,
    interpolation_steps: int,
    change_threshold: float = 0.5,
) -> None:
    """Delay and interpolate abrupt passthrough-action changes in place.

    Args:
        waypoints: Path whose selected passthrough channel is updated without adding samples.
        eef_name: Passthrough channel to smooth.
        delay_steps: Samples to retain the previous value after a detected change.
        interpolation_steps: Samples over which to interpolate to the new value.
        change_threshold: Maximum component change allowed without smoothing, in channel units.
    """

    if len(waypoints) < 2 or (delay_steps == 0 and interpolation_steps == 0):
        return
    values = torch.stack([waypoint.passthrough_action[eef_name] for waypoint in waypoints])
    smoothed = values.clone()
    index = 1
    while index < len(values):
        if float((values[index] - values[index - 1]).abs().max()) <= change_threshold:
            index += 1
            continue
        previous, target = values[index - 1], values[index]
        delay_end = min(index + delay_steps, len(values))
        smoothed[index:delay_end] = previous
        interpolation_end = min(delay_end + interpolation_steps, len(values))
        count = interpolation_end - delay_end
        if count:
            alpha = torch.arange(1, count + 1, device=values.device, dtype=values.dtype) / count
            smoothed[delay_end:interpolation_end] = torch.lerp(previous, target, alpha[:, None])
        index = max(interpolation_end, index + 1)
    for waypoint, value in zip(waypoints, smoothed, strict=True):
        waypoint.passthrough_action[eef_name] = value


def _rigid_poses_interpolate(first: torch.Tensor, second: torch.Tensor, steps: int) -> tuple[torch.Tensor, ...]:
    """Interpolate poses strictly between the endpoints using quaternion slerp.

    Args:
        first: Initial rigid transform with shape ``(4, 4)`` [m].
        second: Final rigid transform in the same reference frame [m].
        steps: Number of equal intervals between endpoints; must exceed one.

    Returns:
        The ``steps - 1`` interior poses, excluding both endpoints, on the first pose's device.
    """

    assert steps > 1 and first.shape == second.shape == (4, 4)
    second = second.to(device=first.device, dtype=first.dtype)
    alpha = torch.arange(1, steps, device=first.device, dtype=first.dtype) / steps
    positions = first[:3, 3] + alpha[:, None] * (second[:3, 3] - first[:3, 3])
    first_quaternion = pose_math.quat_from_matrix(first[:3, :3])
    second_quaternion = pose_math.quat_from_matrix(second[:3, :3])
    dot = torch.dot(first_quaternion, second_quaternion)
    if dot < 0.0:
        second_quaternion = -second_quaternion
        dot = -dot
    dot = torch.clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        quaternions = first_quaternion + alpha[:, None] * (second_quaternion - first_quaternion)
    else:
        angle = torch.acos(dot)
        quaternions = (
            torch.sin((1.0 - alpha) * angle)[:, None] * first_quaternion
            + torch.sin(alpha * angle)[:, None] * second_quaternion
        ) / torch.sin(angle)
    quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    return tuple(pose_math.make_pose(positions, pose_math.matrix_from_quat(quaternions)).unbind())


def _arm_path_refine(
    path: ArmPath,
    *,
    planner: CuroboPlanner,
    datastream: Datastream,
    max_joint_step_rad: float,
) -> ArmPath:
    """Insert FK-derived waypoints to bound joint steps while preserving segment events.

    Args:
        path: Complete arm path in the planner's active joint order.
        planner: Model providing tool FK for inserted joint configurations.
        datastream: Embodiment interface mapping inserted joints to passthrough commands.
        max_joint_step_rad: Maximum component change between consecutive joint samples [rad].

    Returns:
        Refined path with remapped segment bounds, object events and pose keyframes. Existing
        waypoints are retained; only adjacent object keyframes gain interpolated interior poses.
    """

    assert math.isfinite(max_joint_step_rad) and max_joint_step_rad > 0.0
    if len(path.waypoints) == 1:
        return path

    ordered_segments = tuple(sorted(path.segment_bounds.items(), key=lambda item: item[1][0]))
    positions: list[torch.Tensor] = []
    source_waypoints: list[Waypoint | None] = []
    passthrough_templates: list[dict[str, torch.Tensor] | None] = []
    segment_bounds: dict[SegmentId, tuple[int, int]] = {}
    boundary_maps: dict[SegmentId, tuple[int, ...]] = {}

    for segment_id, (start, end) in ordered_segments:
        refined_start = len(positions)
        boundary_map = [0] * (end - start + 1)
        for waypoint_index in range(start, end):
            local_index = waypoint_index - start
            target_positions = path.joint_positions[waypoint_index]
            if waypoint_index == 0:
                subdivisions = 1
            else:
                previous_positions = path.joint_positions[waypoint_index - 1]
                maximum_delta = float(torch.max(torch.abs(target_positions - previous_positions)).item())
                subdivision_ratio = maximum_delta / max_joint_step_rad
                rounded_ratio = round(subdivision_ratio)
                subdivisions = max(
                    1,
                    (
                        rounded_ratio
                        if math.isclose(subdivision_ratio, rounded_ratio, rel_tol=1.0e-6, abs_tol=1.0e-6)
                        else math.ceil(subdivision_ratio)
                    ),
                )

            boundary_map[local_index] = len(positions) - refined_start
            for subdivision in range(1, subdivisions + 1):
                is_original = subdivision == subdivisions
                if is_original:
                    positions.append(target_positions.clone())
                    source_waypoints.append(path.waypoints[waypoint_index])
                    passthrough_templates.append(None)
                    continue
                alpha = subdivision / subdivisions
                positions.append(torch.lerp(previous_positions, target_positions, alpha))
                source_waypoints.append(None)
                passthrough_templates.append(path.waypoints[waypoint_index - 1].passthrough_action)
        segment_bounds[segment_id] = refined_start, len(positions)
        boundary_map[-1] = len(positions) - refined_start
        boundary_maps[segment_id] = tuple(boundary_map)

    inserted_indices = [index for index, waypoint in enumerate(source_waypoints) if waypoint is None]
    if not inserted_indices:
        return path
    joint_positions = torch.stack(positions)
    inserted_poses = planner.joint_positions_to_ee_poses(joint_positions[inserted_indices])
    pose_by_index = dict(zip(inserted_indices, inserted_poses, strict=True))

    waypoints: list[Waypoint] = []
    for index, (position, source_waypoint, passthrough_template) in enumerate(
        zip(positions, source_waypoints, passthrough_templates, strict=True)
    ):
        if source_waypoint is not None:
            waypoints.append(source_waypoint)
            continue
        assert passthrough_template is not None
        named_positions = planner.joint_positions_to_named_positions(position)
        passthrough = datastream.joint_positions_to_passthrough_actions(named_positions, passthrough_template)
        waypoints.append(
            Waypoint(
                pose=pose_by_index[index],
                passthrough_action=passthrough,
                noise=0.0,
                joint_seed=position.clone(),
            )
        )

    maximum_refined_delta = float(torch.max(torch.abs(joint_positions[1:] - joint_positions[:-1])).item())
    assert (
        maximum_refined_delta <= max_joint_step_rad + 1.0e-6
    ), f"refined joint step {maximum_refined_delta} exceeds {max_joint_step_rad}"
    object_events = tuple(
        replace(
            event,
            local_boundary_index=(
                0
                if event.spec.point == SegmentEventPoint.START
                else boundary_maps[event.segment_id][event.local_boundary_index]
            ),
        )
        for event in path.object_events
    )
    object_pose_keyframes = [
        replace(
            keyframe,
            local_boundary_index=boundary_maps[keyframe.segment_id][keyframe.local_boundary_index],
        )
        for keyframe in path.object_pose_keyframes
    ]
    grouped_keyframes: dict[tuple[SegmentId, str], list[PathObjectPose]] = {}
    for keyframe in path.object_pose_keyframes:
        grouped_keyframes.setdefault((keyframe.segment_id, keyframe.object_ref), []).append(keyframe)
    for keyframes in grouped_keyframes.values():
        for first, second in zip(keyframes, keyframes[1:], strict=False):
            if second.local_boundary_index != first.local_boundary_index + 1:
                continue
            first_boundary = boundary_maps[first.segment_id][first.local_boundary_index]
            second_boundary = boundary_maps[second.segment_id][second.local_boundary_index]
            if second_boundary - first_boundary <= 1:
                continue
            for offset, pose in enumerate(
                _rigid_poses_interpolate(
                    first.env_from_object, second.env_from_object, second_boundary - first_boundary
                ),
                start=1,
            ):
                object_pose_keyframes.append(
                    replace(second, local_boundary_index=first_boundary + offset, env_from_object=pose)
                )
    segment_order = {segment_id: index for index, segment_id in enumerate(segment_bounds)}
    object_pose_keyframes.sort(
        key=lambda keyframe: (
            segment_order[keyframe.segment_id],
            keyframe.local_boundary_index,
            keyframe.object_ref,
        )
    )
    return ArmPath(
        eef_name=path.eef_name,
        waypoints=tuple(waypoints),
        joint_positions=joint_positions,
        segment_bounds=segment_bounds,
        object_events=object_events,
        object_pose_keyframes=tuple(object_pose_keyframes),
    )
