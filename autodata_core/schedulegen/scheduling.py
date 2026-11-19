# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule arm paths and map their events and contact rules to execution ticks."""

from __future__ import annotations

import torch
from collections.abc import Mapping

from autodata_core.schedulegen.types import (
    ArmPath,
    ArmWaypointConflict,
    ObjectStateSample,
    ScheduledObjectEvent,
    ScheduledObjectPose,
    ScheduledRobotContact,
    ScheduledWorldContact,
)
from autodata_core.scheduling import (
    Schedule,
    ScheduleFailure,
    SchedulePath,
    SegmentWaypoint,
    WaypointConflict,
    schedule_paths,
)
from autodata_interfaces.tasks.schedule_constraint_spec import (
    NonOverlapConstraintSpec,
    NonPreemptiveConstraintSpec,
    PrecedenceConstraintSpec,
    SchedulerStrategy,
    ScheduleSpec,
    SegmentId,
)
from autodata_interfaces.tasks.segment_spec import SegmentEventType, SegmentRobotContactSpec, SegmentWorldContactSpec


def _contact_interval(path: ArmPath, schedule: Schedule, segment_id: SegmentId) -> tuple[int, int]:
    """Include held lane occupancy, including the terminal boundary, in contact permissions."""

    # Contact permission follows lane occupancy, including waits and the terminal hold.
    held_indices = (*get_scheduled_path_indices(path, schedule), len(path.waypoints) - 1)
    start, stop = path.segment_bounds[segment_id]
    active_ticks = tuple(tick for tick, index in enumerate(held_indices) if start <= index < stop)
    assert active_ticks and active_ticks == tuple(
        range(active_ticks[0], active_ticks[-1] + 1)
    ), f"contact segment {segment_id} must own one contiguous scheduled interval"
    return active_ticks[0], active_ticks[-1] + 1


def schedule_world_contacts(
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
    contacts_by_segment: Mapping[SegmentId, tuple[SegmentWorldContactSpec, ...]],
) -> tuple[ScheduledWorldContact, ...]:
    """Map object/fixture permissions to their segment's held execution interval.

    Args:
        arm_paths: Complete paths keyed by end effector.
        schedule: Execution ticks for every segment waypoint.
        contacts_by_segment: Task-defined object/fixture permissions.

    Returns:
        Permissions sorted by start tick, segment and object/fixture pair.
    """

    path_by_segment = {segment_id: path for path in arm_paths.values() for segment_id in path.segment_bounds}
    missing = set(contacts_by_segment) - set(path_by_segment)
    assert not missing, f"world contacts reference segments without paths: {sorted(map(str, missing))}"
    scheduled = []
    for segment_id, specs in contacts_by_segment.items():
        # Fixture contact stays permitted while this segment's last command is held.
        start_tick, stop_tick = _contact_interval(path_by_segment[segment_id], schedule, segment_id)
        scheduled.extend(ScheduledWorldContact(segment_id, spec, start_tick, stop_tick) for spec in specs)
    return tuple(
        sorted(
            scheduled,
            key=lambda contact: (
                contact.start_tick,
                str(contact.segment_id),
                contact.spec.object_ref,
                contact.spec.fixture_ref,
            ),
        )
    )


def schedule_robot_contacts(
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
    contacts_by_segment: Mapping[SegmentId, tuple[SegmentRobotContactSpec, ...]],
    scheduled_object_events: tuple[ScheduledObjectEvent, ...],
    *,
    attachment_link_names: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[ScheduledRobotContact, ...]:
    """Map robot/object permissions through lane timing and optional release annotations.

    Args:
        arm_paths: Complete paths keyed by end effector.
        schedule: Execution ticks for every segment waypoint.
        contacts_by_segment: Task-defined robot/object permissions.
        scheduled_object_events: Resolved attach/detach events used by release permissions.
        attachment_link_names: Hand/tool collision links per lane. When provided, these links
            may contact an object from its grasp segment through its detach event.

    Returns:
        Permissions sorted by start tick, segment, object and robot links.
    """

    path_by_segment = {segment_id: path for path in arm_paths.values() for segment_id in path.segment_bounds}
    missing = set(contacts_by_segment) - set(path_by_segment)
    assert not missing, f"robot contacts reference segments without paths: {sorted(map(str, missing))}"
    scheduled = []
    for segment_id, specs in contacts_by_segment.items():
        first_tick, last_tick = _contact_interval(path_by_segment[segment_id], schedule, segment_id)
        for spec in specs:
            start_tick = first_tick
            if spec.start_annotation_key:
                # A release permission begins at the annotated detach, not at segment entry.
                matching_events = tuple(
                    event
                    for event in scheduled_object_events
                    if event.path_event.segment_id == segment_id
                    and event.path_event.spec.annotation_key == spec.start_annotation_key
                    and event.path_event.spec.object_ref == spec.object_ref
                )
                assert len(matching_events) == 1, (
                    f"robot contact on {segment_id} must resolve exactly one event "
                    f"{spec.start_annotation_key!r} for {spec.object_ref!r}"
                )
                assert (
                    matching_events[0].path_event.spec.event_type == SegmentEventType.DETACH
                ), "release contact must begin at a detach annotation"
                start_tick = matching_events[0].tick
            stop_tick = schedule.ticks_for(segment_id)[-1] if spec.clear_at_end else last_tick
            assert (
                first_tick <= start_tick < stop_tick
            ), f"robot contact on {segment_id} begins outside segment ownership"
            scheduled.append(ScheduledRobotContact(segment_id, spec, start_tick, stop_tick))
    # Grasping/carrying permits the owning hand, never the other arm or the robot body.
    for index, event in enumerate(scheduled_object_events):
        if attachment_link_names is None or event.path_event.spec.event_type != SegmentEventType.ATTACH:
            continue
        segment_id = event.path_event.segment_id
        object_ref = event.path_event.spec.object_ref
        start_tick, _ = _contact_interval(path_by_segment[segment_id], schedule, segment_id)
        stop_tick = next(
            (
                later.tick
                for later in scheduled_object_events[index + 1 :]
                if later.path_event.spec.object_ref == object_ref
                and later.path_event.spec.event_type == SegmentEventType.DETACH
            ),
            schedule.makespan_ticks + 1,
        )
        links = tuple(sorted(attachment_link_names[segment_id.eef_name]))
        if not links:
            continue
        scheduled.append(
            ScheduledRobotContact(segment_id, SegmentRobotContactSpec(object_ref, links), start_tick, stop_tick)
        )
    return tuple(
        sorted(
            scheduled,
            key=lambda contact: (
                contact.start_tick,
                str(contact.segment_id),
                contact.spec.object_ref,
                contact.spec.robot_link_names,
            ),
        )
    )


def schedule_arm_paths(
    arm_paths: Mapping[str, ArmPath],
    arm_conflicts: tuple[ArmWaypointConflict, ...],
    schedule_spec: ScheduleSpec,
    *,
    segment_resource_ids: Mapping[SegmentId, tuple[str, ...]] | None = None,
    serialize_segments: bool = False,
) -> Schedule | ScheduleFailure:
    """Schedule complete arm paths after every segment has been built.

    Segment provider details do not cross this boundary. Each arm path is split by its
    declared bounds, path-global collision indices are localized to those segments, and
    task-defined constraints are passed to the native scheduler unchanged.

    Args:
        arm_paths: Complete unscheduled paths keyed by end-effector lane.
        arm_conflicts: Cross-arm conflicts in complete-path waypoint coordinates.
        schedule_spec: Native scheduling policy and task constraints.
        segment_resource_ids: Additional task-local resource claims keyed by exact segment.
        serialize_segments: Whether all cross-lane segment intervals must be disjoint.

    Returns:
        A validated schedule, or a scheduling failure with no executable result.
    """

    assert isinstance(arm_paths, Mapping) and arm_paths, "arm_paths must be a non-empty mapping"
    assert all(
        isinstance(eef_name, str) and isinstance(path, ArmPath) for eef_name, path in arm_paths.items()
    ), "arm_paths must map end-effector names to ArmPath values"
    assert all(
        eef_name == path.eef_name for eef_name, path in arm_paths.items()
    ), "each arm_paths key must match its ArmPath.eef_name"
    assert isinstance(arm_conflicts, tuple) and all(
        isinstance(item, ArmWaypointConflict) for item in arm_conflicts
    ), "arm_conflicts must contain ArmWaypointConflict values"
    assert isinstance(schedule_spec, ScheduleSpec), "schedule_spec must be a ScheduleSpec"
    assert segment_resource_ids is None or isinstance(
        segment_resource_ids, Mapping
    ), "segment_resource_ids must be a mapping"
    assert isinstance(serialize_segments, bool), "serialize_segments must be a bool"
    resource_ids_by_segment = {} if segment_resource_ids is None else dict(segment_resource_ids)
    assert all(
        isinstance(segment_id, SegmentId) and isinstance(resource_ids, tuple)
        for segment_id, resource_ids in resource_ids_by_segment.items()
    ), "segment_resource_ids must map SegmentId values to tuples"
    assert schedule_spec.strategy in {
        SchedulerStrategy.WAYPOINT_RETIMING,
        SchedulerStrategy.BLOCK_SCHEDULING,
    }, f"unsupported scheduling strategy: {schedule_spec.strategy}"
    ordered_arm_paths = tuple(path for _, path in sorted(arm_paths.items()))
    arm_path_by_eef = {path.eef_name: path for path in ordered_arm_paths}
    assert len(arm_path_by_eef) == len(ordered_arm_paths), "arm paths must use distinct end-effector lanes"

    # Convert each provider's complete path into scheduler segments without copying geometry.
    bounds_by_eef: dict[str, tuple[tuple[SegmentId, tuple[int, int]], ...]] = {}
    scheduler_paths: list[SchedulePath] = []
    built_segment_ids: set[SegmentId] = set()
    for arm_path in ordered_arm_paths:
        bounds = tuple(sorted(arm_path.segment_bounds.items(), key=lambda item: (item[1][0], item[0])))
        bounds_by_eef[arm_path.eef_name] = bounds
        built_segment_ids.update(segment_id for segment_id, _ in bounds)
        scheduler_paths.extend(
            SchedulePath(
                segment_id=segment_id,
                lane=arm_path.eef_name,
                waypoint_count=end - start,
                resource_ids=resource_ids_by_segment.get(segment_id, ()),
            )
            for segment_id, (start, end) in bounds
        )
    unknown_resource_segments = set(resource_ids_by_segment) - built_segment_ids
    assert (
        not unknown_resource_segments
    ), f"segment_resource_ids references segments without paths: {sorted(map(str, unknown_resource_segments))}"

    # Collision detection indexes complete arm paths; the scheduler indexes segment-local waypoints.
    scheduler_conflicts: list[WaypointConflict] = []
    for conflict in sorted(arm_conflicts):
        first = _waypoint_localize(
            arm_path_by_eef,
            bounds_by_eef,
            eef_name=conflict.first_eef_name,
            waypoint_index=conflict.first_waypoint_index,
        )
        second = _waypoint_localize(
            arm_path_by_eef,
            bounds_by_eef,
            eef_name=conflict.second_eef_name,
            waypoint_index=conflict.second_waypoint_index,
        )
        scheduler_conflicts.append(WaypointConflict(first, second))

    return schedule_paths(
        tuple(scheduler_paths),
        conflicts=tuple(scheduler_conflicts),
        precedence=tuple(
            constraint for constraint in schedule_spec.constraints if isinstance(constraint, PrecedenceConstraintSpec)
        ),
        non_overlap=tuple(
            constraint for constraint in schedule_spec.constraints if isinstance(constraint, NonOverlapConstraintSpec)
        ),
        non_preemptive=tuple(
            constraint
            for constraint in schedule_spec.constraints
            if isinstance(constraint, NonPreemptiveConstraintSpec)
        ),
        strategy=schedule_spec.strategy,
        max_block_size=schedule_spec.max_block_size,
        serialize_segments=serialize_segments,
        time_limit_s=schedule_spec.time_limit_s,
        minimize_max_wait=schedule_spec.minimize_max_wait,
    )


def schedule_object_events(
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
) -> tuple[ScheduledObjectEvent, ...]:
    """Map attach/detach events from segment boundaries to execution ticks.

    A boundary before local waypoint ``i`` activates at that waypoint's occupancy start. The first
    writer on a persistent lane is occupied from tick zero; later waypoints use their scheduled
    ticks. A terminal boundary activates one tick after its segment's final waypoint. Same-object
    transitions on different lanes at one tick are rejected because they have no defined
    within-tick ordering.

    Args:
        arm_paths: Complete paths keyed by end-effector lane.
        schedule: Accepted waypoint schedule for those paths.

    Returns:
        Events ordered by execution tick, with stable per-lane declaration order for ties.
    """

    scheduled: list[ScheduledObjectEvent] = []
    for eef_name, path in sorted(arm_paths.items()):
        assert eef_name == path.eef_name, "each arm_paths key must match its ArmPath.eef_name"
        for event in path.object_events:
            start, end = path.segment_bounds[event.segment_id]
            ticks = schedule.ticks_for(event.segment_id)
            assert len(ticks) == end - start, f"schedule length differs for segment {event.segment_id}"
            # A lane's first pose is already occupied during any initial wait.
            if event.local_boundary_index == 0 and start == 0:
                tick = 0
            elif event.local_boundary_index < len(ticks):
                tick = ticks[event.local_boundary_index]
            else:
                tick = ticks[-1] + 1
            assert tick <= schedule.makespan_ticks, "object event lies beyond the schedule makespan"
            scheduled.append(ScheduledObjectEvent(tick=tick, path_event=event))
    scheduled.sort(key=lambda event: event.tick)

    # Validate the event stream before projecting poses: ownership must be single-valued.
    owner_by_object: dict[str, str] = {}
    object_by_lane: dict[str, str] = {}
    lanes_by_tick_and_object: dict[tuple[int, str], set[str]] = {}
    for scheduled_event in scheduled:
        path_event = scheduled_event.path_event
        event = path_event.spec
        lane = path_event.segment_id.eef_name
        lanes = lanes_by_tick_and_object.setdefault((scheduled_event.tick, event.object_ref), set())
        lanes.add(lane)
        assert len(lanes) == 1, (
            f"simultaneous transitions for object {event.object_ref!r} at tick {scheduled_event.tick} "
            "span multiple lanes"
        )
        if event.event_type == SegmentEventType.ATTACH:
            assert event.object_ref not in owner_by_object, (
                f"object {event.object_ref!r} is already carried by "
                f"{owner_by_object[event.object_ref]!r} at tick {scheduled_event.tick}"
            )
            assert (
                lane not in object_by_lane
            ), f"lane {lane!r} already carries object {object_by_lane.get(lane)!r} at tick {scheduled_event.tick}"
            owner_by_object[event.object_ref] = lane
            object_by_lane[lane] = event.object_ref
        else:
            assert (
                owner_by_object.get(event.object_ref) == lane
            ), f"lane {lane!r} cannot detach object {event.object_ref!r} at tick {scheduled_event.tick}"
            del owner_by_object[event.object_ref]
            del object_by_lane[lane]
    return tuple(scheduled)


def get_scheduled_path_indices(path: ArmPath, schedule: Schedule) -> tuple[int, ...]:
    """Map each execution tick to the waypoint held on one arm path.

    Args:
        path: Complete unscheduled arm path.
        schedule: Execution ticks for the path's segments.

    Returns:
        Path waypoint index for every tick before the schedule makespan.
        Initial, inter-segment and terminal waits repeat the held waypoint index.
    """

    scheduled: list[tuple[int, int]] = []
    for segment_id, (start, end) in path.segment_bounds.items():
        ticks = schedule.ticks_for(segment_id)
        assert len(ticks) == end - start, f"schedule length differs for segment {segment_id}"
        scheduled.extend((tick, start + local_index) for local_index, tick in enumerate(ticks))
    scheduled.sort()
    assert len(scheduled) == len(path.waypoints), "schedule must cover every path waypoint"
    timeline: list[int] = []
    cursor = 0
    # The last issued waypoint remains active during waits; waypoint zero covers the initial hold.
    for tick in range(schedule.makespan_ticks):
        while cursor + 1 < len(scheduled) and scheduled[cursor + 1][0] <= tick:
            cursor += 1
        timeline.append(scheduled[cursor][1])
    return tuple(timeline)


def schedule_object_poses(
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
) -> tuple[ScheduledObjectPose, ...]:
    """Map path-boundary object poses to their post-command execution samples.

    Args:
        arm_paths: Complete paths keyed by end-effector name.
        schedule: Execution ticks for every segment waypoint.

    Returns:
        Object poses ordered by sample tick and end effector.
        The initial pose uses sample tick -1, before any command executes.
    """

    scheduled: list[ScheduledObjectPose] = []
    for eef_name, path in sorted(arm_paths.items()):
        assert eef_name == path.eef_name, "each arm_paths key must match its ArmPath.eef_name"
        waypoint_ticks: dict[int, int] = {}
        for segment_id, (start, end) in path.segment_bounds.items():
            ticks = schedule.ticks_for(segment_id)
            assert len(ticks) == end - start, f"schedule length differs for segment {segment_id}"
            waypoint_ticks.update((start + local_index, tick) for local_index, tick in enumerate(ticks))
        assert len(waypoint_ticks) == len(path.waypoints), "schedule must cover every path waypoint"

        for path_pose in path.object_pose_keyframes:
            start, _ = path.segment_bounds[path_pose.segment_id]
            global_boundary = start + path_pose.local_boundary_index
            # Provider poses describe completed commands; -1 denotes the untouched reset state.
            sample_tick = -1 if global_boundary == 0 else waypoint_ticks[global_boundary - 1]
            assert sample_tick < schedule.makespan_ticks, "object pose lies beyond the schedule makespan"
            scheduled.append(ScheduledObjectPose(sample_tick=sample_tick, path_pose=path_pose))

    scheduled.sort(key=lambda pose: (pose.sample_tick, pose.path_pose.segment_id.eef_name))
    identities = tuple((pose.sample_tick, pose.path_pose.object_ref) for pose in scheduled)
    assert len(identities) == len(set(identities)), "multiple object poses target one execution sample"
    return tuple(scheduled)


def project_object_states(
    initial_object_poses: Mapping[str, torch.Tensor],
    tool_poses_by_eef: Mapping[str, torch.Tensor],
    object_refs: tuple[str, ...],
    scheduled_events: tuple[ScheduledObjectEvent, ...],
    scheduled_object_poses: tuple[ScheduledObjectPose, ...] = (),
    *,
    makespan_ticks: int,
) -> tuple[ObjectStateSample, ...]:
    """Predict object ownership and poses during a scheduled execution.

    Tool-pose row ``t`` is the state before command ``t``; row ``t + 1`` is the state afterward.
    An attach applies the annotated object pose and makes it follow the tool. A detach keeps
    the pose from before that command. The terminal boundary is included even though no
    command follows it.

    Args:
        initial_object_poses: Reset-time environment-relative object poses [m].
        tool_poses_by_eef: Environment-relative tool poses at boundaries ``0..makespan_ticks`` [m].
        object_refs: Exact episode object-geometry registry keys.
        scheduled_events: Object transitions mapped to schedule ticks.
        scheduled_object_poses: Provider object poses aligned to post-command execution samples.
        makespan_ticks: Exclusive terminal schedule tick.

    Returns:
        Immutable samples ordered by ``(boundary_tick, object_ref)``.
    """

    assert isinstance(initial_object_poses, Mapping)
    assert isinstance(tool_poses_by_eef, Mapping)
    assert isinstance(object_refs, tuple) and len(object_refs) == len(set(object_refs))
    assert isinstance(scheduled_events, tuple) and all(
        isinstance(event, ScheduledObjectEvent) for event in scheduled_events
    )
    assert isinstance(scheduled_object_poses, tuple) and all(
        isinstance(pose, ScheduledObjectPose) for pose in scheduled_object_poses
    )
    assert type(makespan_ticks) is int and makespan_ticks > 0
    assert tuple(event.tick for event in scheduled_events) == tuple(sorted(event.tick for event in scheduled_events))
    assert tuple(pose.sample_tick for pose in scheduled_object_poses) == tuple(
        sorted(pose.sample_tick for pose in scheduled_object_poses)
    )

    tracked_objects = tuple(sorted(object_refs))
    attach_refs = {
        event.path_event.spec.object_ref
        for event in scheduled_events
        if event.path_event.spec.event_type == SegmentEventType.ATTACH
    }
    assert attach_refs == set(tracked_objects), "object registry must exactly match scheduled attachments"
    assert all(event.path_event.spec.object_ref in attach_refs for event in scheduled_events)
    if not tracked_objects:
        return ()

    event_lanes = {event.path_event.segment_id.eef_name for event in scheduled_events}
    assert event_lanes <= set(tool_poses_by_eef), "object event references an unknown tool lane"
    reference = next(iter(tool_poses_by_eef.values()))
    assert isinstance(reference, torch.Tensor)
    tool_poses: dict[str, torch.Tensor] = {}
    for eef_name, poses in tool_poses_by_eef.items():
        assert isinstance(poses, torch.Tensor) and poses.shape == (makespan_ticks + 1, 4, 4)
        assert bool(torch.isfinite(poses).all()), f"tool timeline {eef_name!r} contains a non-finite pose"
        tool_poses[eef_name] = poses.to(device=reference.device, dtype=reference.dtype)

    current_poses: dict[str, torch.Tensor] = {}
    pose_history: dict[str, list[torch.Tensor]] = {object_ref: [] for object_ref in tracked_objects}
    owner_history: dict[str, list[ScheduledObjectEvent | None]] = {object_ref: [] for object_ref in tracked_objects}
    jump_history: dict[str, list[tuple[float, float] | None]] = {object_ref: [] for object_ref in tracked_objects}
    active_attach: dict[str, ScheduledObjectEvent] = {}
    active_object_by_lane: dict[str, str] = {}
    for object_ref in tracked_objects:
        assert object_ref in initial_object_poses, f"reset snapshot has no pose for object {object_ref!r}"
        pose = initial_object_poses[object_ref]
        assert isinstance(pose, torch.Tensor) and pose.shape == (4, 4) and bool(torch.isfinite(pose).all())
        current_poses[object_ref] = pose.to(device=reference.device, dtype=reference.dtype).detach().clone()

    poses_by_sample: dict[int, list[ScheduledObjectPose]] = {}
    for scheduled_pose in scheduled_object_poses:
        assert -1 <= scheduled_pose.sample_tick < makespan_ticks
        assert scheduled_pose.path_pose.object_ref in tracked_objects
        poses_by_sample.setdefault(scheduled_pose.sample_tick, []).append(scheduled_pose)
    for scheduled_pose in poses_by_sample.pop(-1, ()):
        path_pose = scheduled_pose.path_pose
        expected_pose = path_pose.env_from_object.to(device=reference.device, dtype=reference.dtype)
        position_delta, rotation_delta = _object_pose_delta(current_poses[path_pose.object_ref], expected_pose)
        assert position_delta <= 1.0e-4 and rotation_delta <= 1.0e-3, (
            f"initial object pose {path_pose.object_ref!r} differs from provider evidence by "
            f"{position_delta:.6f} m / {rotation_delta:.6f} rad"
        )

    events_by_tick: dict[int, list[ScheduledObjectEvent]] = {}
    for event in scheduled_events:
        assert event.tick <= makespan_ticks, "object event lies beyond the schedule makespan"
        events_by_tick.setdefault(event.tick, []).append(event)

    for tick in range(makespan_ticks + 1):
        # Boundary t contains the result of command t-1, not the result of command t.
        for object_ref, attach in active_attach.items():
            lane = attach.path_event.segment_id.eef_name
            tool_from_object = attach.path_event.tool_from_object
            assert tool_from_object is not None
            current_poses[object_ref] = tool_poses[lane][tick] @ tool_from_object.to(
                device=reference.device, dtype=reference.dtype
            )
        for scheduled_pose in poses_by_sample.get(tick - 1, ()):
            path_pose = scheduled_pose.path_pose
            if path_pose.object_ref not in active_attach:
                current_poses[path_pose.object_ref] = path_pose.env_from_object.to(
                    device=reference.device, dtype=reference.dtype
                )

        # Events change ownership at this boundary, before the next command executes.
        attachment_jumps: dict[str, tuple[float, float]] = {}
        for event in events_by_tick.get(tick, ()):
            path_event = event.path_event
            spec = path_event.spec
            object_ref = spec.object_ref
            lane = path_event.segment_id.eef_name
            if spec.event_type == SegmentEventType.ATTACH:
                assert object_ref not in active_attach, f"object {object_ref!r} is already attached"
                assert (
                    lane not in active_object_by_lane
                ), f"lane {lane!r} already carries object {active_object_by_lane.get(lane)!r}"
                assert path_event.tool_from_object is not None
                boundary_pose = tool_poses[lane][tick] @ path_event.tool_from_object.to(
                    device=reference.device, dtype=reference.dtype
                )
                attachment_jumps[object_ref] = _object_pose_delta(current_poses[object_ref], boundary_pose)
                current_poses[object_ref] = boundary_pose
                active_attach[object_ref] = event
                active_object_by_lane[lane] = object_ref
            else:
                assert spec.event_type == SegmentEventType.DETACH
                attach = active_attach.get(object_ref)
                assert attach is not None, f"cannot detach unowned object {object_ref!r}"
                assert attach.path_event.segment_id.eef_name == lane, (
                    f"lane {lane!r} cannot detach object {object_ref!r} owned by "
                    f"{attach.path_event.segment_id.eef_name!r}"
                )
                tool_from_object = attach.path_event.tool_from_object
                assert tool_from_object is not None
                current_poses[object_ref] = tool_poses[lane][tick] @ tool_from_object.to(
                    device=reference.device, dtype=reference.dtype
                )
                del active_attach[object_ref]
                del active_object_by_lane[lane]

        for object_ref in tracked_objects:
            pose_history[object_ref].append(current_poses[object_ref].detach().clone())
            owner_history[object_ref].append(active_attach.get(object_ref))
            jump_history[object_ref].append(attachment_jumps.get(object_ref))

    # Copy only after the timeline is complete so saved states cannot alias mutable tensors.
    immutable_poses = {
        object_ref: tuple(
            tuple(tuple(float(value) for value in row) for row in pose) for pose in torch.stack(poses).cpu().tolist()
        )
        for object_ref, poses in pose_history.items()
    }
    return tuple(
        ObjectStateSample(
            boundary_tick=tick,
            object_ref=object_ref,
            env_from_object=immutable_poses[object_ref][tick],
            active_attach_event=owner_history[object_ref][tick],
            attachment_pose_jump=jump_history[object_ref][tick],
        )
        for tick in range(makespan_ticks + 1)
        for object_ref in tracked_objects
    )


def _object_pose_delta(first_pose: torch.Tensor, second_pose: torch.Tensor) -> tuple[float, float]:
    """Return translation/rotation differences [m, rad], using float64 rotation math."""

    position_delta = float(torch.linalg.vector_norm(first_pose[:3, 3] - second_pose[:3, 3]).item())
    first_rotation = first_pose[:3, :3].to(dtype=torch.float64)
    second_rotation = second_pose[:3, :3].to(dtype=torch.float64)
    relative_rotation = first_rotation.transpose(0, 1) @ second_rotation
    cosine = torch.clamp((torch.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return position_delta, float(torch.acos(cosine).item())


def _waypoint_localize(
    arm_path_by_eef: dict[str, ArmPath],
    bounds_by_eef: dict[str, tuple[tuple[SegmentId, tuple[int, int]], ...]],
    *,
    eef_name: str,
    waypoint_index: int,
) -> SegmentWaypoint:
    """Map an arm-global waypoint index into its containing half-open segment interval."""

    arm_path = arm_path_by_eef.get(eef_name)
    assert arm_path is not None, f"conflict references unknown arm path {eef_name!r}"
    assert waypoint_index < len(
        arm_path.waypoints
    ), f"conflict waypoint {waypoint_index} is outside arm path {eef_name!r}"
    for segment_id, (start, end) in bounds_by_eef[eef_name]:
        if start <= waypoint_index < end:
            return SegmentWaypoint(segment_id, waypoint_index - start)
    raise AssertionError(f"conflict waypoint {waypoint_index} is not covered by arm path {eef_name!r}")


__all__ = [
    "get_scheduled_path_indices",
    "schedule_object_events",
    "schedule_object_poses",
    "schedule_robot_contacts",
    "schedule_world_contacts",
    "project_object_states",
    "schedule_arm_paths",
]
