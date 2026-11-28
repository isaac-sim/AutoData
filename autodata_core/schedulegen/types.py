# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path, object-event and contact data shared by ScheduleGen components."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from autodata_core.waypoint import Waypoint
from autodata_interfaces.tasks.schedule_constraint_spec import SegmentId
from autodata_interfaces.tasks.segment_spec import (
    SegmentEventPoint,
    SegmentEventSpec,
    SegmentEventType,
    SegmentRobotContactSpec,
    SegmentWorldContactSpec,
)

if TYPE_CHECKING:
    from autodata_interfaces.motion_planners.curobo.collision_spheres import ObjectCollisionSpheres


class EpisodeGenerationError(RuntimeError):
    """Expected path-building, scheduling, validation or replay failure.

    Args:
        stage: Failed operation recorded in the attempt result.
        message: Description of the failed operation.
        diagnostics: Additional scalar details for the attempt result.
    """

    def __init__(
        self,
        stage: str,
        message: str,
        diagnostics: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Record the failed stage, message and optional attempt diagnostics."""

        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class EpisodePaths:
    """Complete unscheduled arm paths and their attached-object geometry.

    Args:
        arm_paths: Paths keyed by end-effector name.
        object_collision_spheres: One fitted sphere set per object attached along the paths.
    """

    arm_paths: Mapping[str, ArmPath]
    object_collision_spheres: Mapping[str, ObjectCollisionSpheres]

    def __post_init__(self) -> None:
        """Check attachment geometry coverage and freeze copies of the path and geometry maps."""

        arm_paths = dict(self.arm_paths)
        geometries = dict(self.object_collision_spheres)
        attach_refs = {
            event.spec.object_ref
            for path in arm_paths.values()
            for event in path.object_events
            if event.spec.event_type == SegmentEventType.ATTACH
        }
        assert attach_refs == set(geometries), "every attachment must have exactly one sphere geometry"
        assert all(name == geometry.object_name for name, geometry in geometries.items())
        object.__setattr__(self, "arm_paths", MappingProxyType(arm_paths))
        object.__setattr__(self, "object_collision_spheres", MappingProxyType(geometries))


def _rigid_transform_copy(transform: torch.Tensor, name: str) -> torch.Tensor:
    """Validate and copy one rigid homogeneous transform.

    Args:
        transform: Homogeneous transform with shape (4, 4); translation is in [m].
        name: Field name used in validation errors.

    Returns:
        A detached copy with a finite, proper rotation and homogeneous final row.
    """

    assert isinstance(transform, torch.Tensor), f"{name} must be a torch.Tensor"
    result = transform.detach().clone()
    assert result.shape == (4, 4), f"{name} must have shape (4, 4)"
    assert bool(torch.isfinite(result).all()), f"{name} must be finite"
    assert torch.allclose(
        result[3], result.new_tensor((0.0, 0.0, 0.0, 1.0)), atol=1.0e-5, rtol=0.0
    ), f"invalid {name} homogeneous transform"
    rotation = result[:3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    assert torch.allclose(
        rotation.transpose(0, 1) @ rotation, identity, atol=1.0e-4, rtol=0.0
    ), f"{name} rotation must be orthonormal"
    assert torch.allclose(
        torch.linalg.det(rotation), rotation.new_tensor(1.0), atol=1.0e-4, rtol=0.0
    ), f"{name} rotation must be proper"
    return result


@dataclass(frozen=True)
class PathObjectEvent:
    """An attach or detach event at a segment-local path boundary.

    Args:
        segment_id: Segment containing the event.
        local_boundary_index: Boundary before local waypoint ``i``, or the segment length for its
            terminal boundary.
        spec: Original validated descriptor event and its source annotation.
        tool_from_object: Object transform in the attachment frame; translation is in [m].
    """

    segment_id: SegmentId
    local_boundary_index: int
    spec: SegmentEventSpec
    tool_from_object: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the event boundary and copy the rigid transform required by attachment."""

        assert isinstance(self.segment_id, SegmentId), "segment_id must be a SegmentId"
        assert (
            type(self.local_boundary_index) is int and self.local_boundary_index >= 0
        ), "local_boundary_index must be a non-negative integer"
        assert isinstance(self.spec, SegmentEventSpec), "spec must be a SegmentEventSpec"
        if self.spec.event_type == SegmentEventType.DETACH:
            assert self.tool_from_object is None, "detach events do not use tool_from_object"
            return

        assert self.spec.event_type == SegmentEventType.ATTACH
        object.__setattr__(self, "tool_from_object", _rigid_transform_copy(self.tool_from_object, "tool_from_object"))


@dataclass(frozen=True)
class PathObjectPose:
    """An object-pose sample at a segment-local path boundary.

    Args:
        segment_id: Segment containing the object-pose sample.
        local_boundary_index: Boundary before local waypoint ``i``, or the terminal boundary.
        object_ref: Exact logical scene-object name.
        env_from_object: Environment-relative object transform [m].
    """

    segment_id: SegmentId
    local_boundary_index: int
    object_ref: str
    env_from_object: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the object boundary and copy its environment-relative rigid transform."""

        assert isinstance(self.segment_id, SegmentId), "segment_id must be a SegmentId"
        assert type(self.local_boundary_index) is int and self.local_boundary_index >= 0
        assert self.object_ref and self.object_ref == self.object_ref.strip(), "object_ref must be non-empty"
        object.__setattr__(self, "env_from_object", _rigid_transform_copy(self.env_from_object, "env_from_object"))


@dataclass(frozen=True)
class ScheduledObjectEvent:
    """An attach or detach event mapped to an execution boundary tick.

    Args:
        tick: Integer boundary tick at which the transition becomes active.
        path_event: Path event and its source annotation.
    """

    tick: int
    path_event: PathObjectEvent

    def __post_init__(self) -> None:
        """Validate the nonnegative execution boundary and its path event."""

        assert type(self.tick) is int and self.tick >= 0, "tick must be a non-negative integer"
        assert isinstance(self.path_event, PathObjectEvent), "path_event must be a PathObjectEvent"


@dataclass(frozen=True)
class ScheduledObjectPose:
    """An object pose mapped to a recorded execution sample.

    Args:
        sample_tick: Post-command sample tick, or -1 for the pose before execution.
        path_pose: Object pose and its segment-local boundary.
    """

    sample_tick: int
    path_pose: PathObjectPose

    def __post_init__(self) -> None:
        """Validate the path pose and allow tick minus one for the pre-execution sample."""

        assert type(self.sample_tick) is int and self.sample_tick >= -1, "sample_tick must be at least -1"
        assert isinstance(self.path_pose, PathObjectPose), "path_pose must be a PathObjectPose"


@dataclass(frozen=True, kw_only=True)
class ObjectStateSample:
    """Predicted object state at one scheduled boundary.

    Args:
        boundary_tick: Boundary that starts this execution sample, including the terminal boundary.
        object_ref: Exact logical scene-object name.
        env_from_object: Environment-relative object transform [m].
        active_attach_event: Exact active attachment, or ``None`` while released in the world.
        attachment_pose_jump: Translation and rotation applied by an attach transition [m, rad],
            or ``None`` when no attachment starts this sample.
    """

    boundary_tick: int
    object_ref: str
    env_from_object: tuple[tuple[float, float, float, float], ...]
    active_attach_event: ScheduledObjectEvent | None
    attachment_pose_jump: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate attachment timing, finite pose jumps and the rigid object transform."""

        assert type(self.boundary_tick) is int and self.boundary_tick >= 0
        assert self.object_ref and self.object_ref == self.object_ref.strip(), "object_ref must be non-empty"
        if self.active_attach_event is not None:
            assert isinstance(self.active_attach_event, ScheduledObjectEvent)
            assert self.active_attach_event.tick <= self.boundary_tick
            spec = self.active_attach_event.path_event.spec
            assert spec.event_type == SegmentEventType.ATTACH and spec.object_ref == self.object_ref
        if self.attachment_pose_jump is not None:
            assert self.active_attach_event is not None and self.active_attach_event.tick == self.boundary_tick
            assert len(self.attachment_pose_jump) == 2
            jump = tuple(float(value) for value in self.attachment_pose_jump)
            assert all(math.isfinite(value) and value >= 0.0 for value in jump)
            object.__setattr__(self, "attachment_pose_jump", jump)

        transform = tuple(tuple(float(value) for value in row) for row in self.env_from_object)
        matrix = torch.tensor(transform, dtype=torch.float64)
        assert matrix.shape == (4, 4) and bool(torch.isfinite(matrix).all())
        assert torch.allclose(matrix[3], matrix.new_tensor((0.0, 0.0, 0.0, 1.0)), atol=1.0e-5, rtol=0.0)
        rotation = matrix[:3, :3]
        assert torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            torch.eye(3, dtype=matrix.dtype),
            atol=1.0e-4,
            rtol=0.0,
        )
        assert torch.allclose(torch.linalg.det(rotation), matrix.new_tensor(1.0), atol=1.0e-4, rtol=0.0)
        object.__setattr__(self, "env_from_object", transform)

    @property
    def owner_eef_name(self) -> str | None:
        """Return the carrying end effector, or ``None`` while the object is released."""

        if self.active_attach_event is None:
            return None
        return self.active_attach_event.path_event.segment_id.eef_name


@dataclass(frozen=True)
class ScheduledWorldContact:
    """An object/fixture permission over a half-open interval of execution ticks.

    Args:
        segment_id: Segment owning the permission.
        spec: Task-defined object/fixture pair.
        start_tick: First permitted boundary tick.
        stop_tick: Exclusive end of the permitted interval.
    """

    segment_id: SegmentId
    spec: SegmentWorldContactSpec
    start_tick: int
    stop_tick: int

    def __post_init__(self) -> None:
        """Require an object/fixture permission over a nonempty half-open tick interval."""

        assert isinstance(self.segment_id, SegmentId)
        assert isinstance(self.spec, SegmentWorldContactSpec)
        assert 0 <= self.start_tick < self.stop_tick


@dataclass(frozen=True)
class ScheduledRobotContact:
    """A robot/object permission over a half-open interval of execution ticks.

    Args:
        segment_id: Segment owning the permission.
        spec: Task-defined object and allowed robot links.
        start_tick: First permitted boundary tick.
        stop_tick: Exclusive end of the permitted interval.
    """

    segment_id: SegmentId
    spec: SegmentRobotContactSpec
    start_tick: int
    stop_tick: int

    def __post_init__(self) -> None:
        """Require a robot/object permission over a nonempty half-open tick interval."""

        assert isinstance(self.segment_id, SegmentId)
        assert isinstance(self.spec, SegmentRobotContactSpec)
        assert 0 <= self.start_tick < self.stop_tick


@dataclass(frozen=True)
class ArmPath:
    """One arm's complete, unscheduled episode path.

    ``segment_bounds`` maps every segment to a half-open ``[start, end)`` interval in both
    ``waypoints`` and ``joint_positions``. The intervals must partition the complete path exactly.

    Args:
        eef_name: End-effector/lane name used by the embodiment adapter.
        waypoints: Controller waypoints in path order.
        joint_positions: Arm-local planner joints with shape ``(T, J)`` [rad, m]. Columns use the
            corresponding arm planner's cuRobo joint order.
        segment_bounds: Exact half-open waypoint bounds for every declared segment.
        object_events: Attach/detach events in lane and boundary order.
        object_pose_keyframes: Object poses in lane and boundary order.
    """

    eef_name: str
    waypoints: tuple[Waypoint, ...]
    joint_positions: torch.Tensor
    segment_bounds: Mapping[SegmentId, tuple[int, int]]
    object_events: tuple[PathObjectEvent, ...] = field(default_factory=tuple)
    object_pose_keyframes: tuple[PathObjectPose, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Check aligned finite paths, exact segment coverage and ordered object boundaries."""

        assert self.eef_name and self.eef_name == self.eef_name.strip(), "eef_name must be non-empty and trimmed"
        assert isinstance(self.waypoints, tuple) and self.waypoints, "waypoints must be a non-empty tuple"
        assert all(
            isinstance(waypoint, Waypoint) for waypoint in self.waypoints
        ), "waypoints must contain Waypoint values"
        assert isinstance(self.joint_positions, torch.Tensor), "joint_positions must be a torch.Tensor"
        assert self.joint_positions.ndim == 2, "joint_positions must have shape (T, J)"
        assert self.joint_positions.shape[0] == len(
            self.waypoints
        ), "joint_positions and waypoints must have the same length"
        assert self.joint_positions.shape[1] > 0, "joint_positions must contain at least one joint"
        assert bool(torch.isfinite(self.joint_positions).all()), "joint_positions must be finite"
        assert isinstance(self.segment_bounds, Mapping) and self.segment_bounds, "segment_bounds must not be empty"
        assert isinstance(self.object_events, tuple) and all(
            isinstance(event, PathObjectEvent) for event in self.object_events
        ), "object_events must contain PathObjectEvent values"
        assert isinstance(self.object_pose_keyframes, tuple) and all(
            isinstance(keyframe, PathObjectPose) for keyframe in self.object_pose_keyframes
        ), "object_pose_keyframes must contain PathObjectPose values"

        waypoint_coverage = [0] * len(self.waypoints)
        bounds = dict(self.segment_bounds)
        for segment_id, interval in bounds.items():
            assert isinstance(segment_id, SegmentId), "segment_bounds keys must be SegmentId values"
            assert (
                segment_id.eef_name == self.eef_name
            ), f"segment {segment_id} belongs to {segment_id.eef_name!r}, expected {self.eef_name!r}"
            assert isinstance(interval, tuple) and len(interval) == 2, "segment bounds must be (start, end) tuples"
            start, end = interval
            assert type(start) is int and type(end) is int, "segment bounds must contain integers"
            assert 0 <= start < end <= len(self.waypoints), f"invalid half-open bounds {interval} for {segment_id}"
            for waypoint_index in range(start, end):
                waypoint_coverage[waypoint_index] += 1
        assert all(
            count == 1 for count in waypoint_coverage
        ), "segment bounds must partition the complete arm path without gaps or overlap"
        segment_order = {
            segment_id: index
            for index, (segment_id, _) in enumerate(sorted(bounds.items(), key=lambda item: item[1][0]))
        }
        event_order: list[tuple[int, int]] = []
        for event in self.object_events:
            assert event.segment_id in bounds, f"object event references unknown segment {event.segment_id}"
            assert event.segment_id.eef_name == self.eef_name, "object event belongs to another arm path"
            start, end = bounds[event.segment_id]
            assert (
                event.local_boundary_index <= end - start
            ), f"object event boundary {event.local_boundary_index} is outside segment {event.segment_id}"
            if event.spec.point == SegmentEventPoint.START:
                assert event.local_boundary_index == 0, "segment-start events must use boundary zero"
            elif event.spec.point == SegmentEventPoint.END:
                assert event.local_boundary_index == end - start, "segment-end events must use the terminal boundary"
            event_order.append((segment_order[event.segment_id], event.local_boundary_index))
        assert event_order == sorted(event_order), "object_events must be in lane and boundary order"
        keyframe_order: list[tuple[int, int, str]] = []
        keyframe_ids: set[tuple[SegmentId, int, str]] = set()
        for keyframe in self.object_pose_keyframes:
            assert keyframe.segment_id in bounds, f"object pose references unknown segment {keyframe.segment_id}"
            assert keyframe.segment_id.eef_name == self.eef_name, "object pose belongs to another arm path"
            start, end = bounds[keyframe.segment_id]
            assert (
                keyframe.local_boundary_index <= end - start
            ), f"object pose boundary {keyframe.local_boundary_index} is outside segment {keyframe.segment_id}"
            keyframe_id = (keyframe.segment_id, keyframe.local_boundary_index, keyframe.object_ref)
            assert keyframe_id not in keyframe_ids, f"duplicate object pose keyframe {keyframe_id}"
            keyframe_ids.add(keyframe_id)
            keyframe_order.append(
                (segment_order[keyframe.segment_id], keyframe.local_boundary_index, keyframe.object_ref)
            )
        assert keyframe_order == sorted(keyframe_order), "object_pose_keyframes must be in lane and boundary order"
        object.__setattr__(self, "segment_bounds", MappingProxyType(bounds))


@dataclass(frozen=True, order=True)
class ArmWaypointConflict:
    """A forbidden simultaneous pair of original arm waypoints.

    Args:
        first_eef_name: First path's end-effector/lane name.
        first_waypoint_index: Original waypoint index on the first path.
        second_eef_name: Second path's end-effector/lane name.
        second_waypoint_index: Original waypoint index on the second path.
        clearance_m: Minimum sampled sphere-surface clearance [m]. Negative values penetrate.
    """

    first_eef_name: str
    first_waypoint_index: int
    second_eef_name: str
    second_waypoint_index: int
    clearance_m: float

    def __post_init__(self) -> None:
        """Require distinct arm lanes, nonnegative waypoint indices and finite clearance."""

        assert self.first_eef_name and self.second_eef_name, "conflict lane names must be non-empty"
        assert self.first_eef_name != self.second_eef_name, "a cross-arm conflict requires distinct lanes"
        assert self.first_waypoint_index >= 0 and self.second_waypoint_index >= 0
        assert math.isfinite(self.clearance_m), "clearance_m must be finite"


__all__ = [
    "ArmPath",
    "ArmWaypointConflict",
    "ObjectStateSample",
    "PathObjectEvent",
    "PathObjectPose",
    "ScheduledObjectEvent",
    "ScheduledObjectPose",
    "ScheduledRobotContact",
    "ScheduledWorldContact",
]
