# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory segments used by complete-episode generation algorithms."""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field

_LOCAL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_local_name(value: str, *, field_name: str) -> None:
    """Check a lowercase descriptor name.

    Args:
        value: Name to validate.
        field_name: Field name included in validation errors.
    """

    assert isinstance(value, str) and _LOCAL_NAME_PATTERN.fullmatch(
        value
    ), f"{field_name} must match {_LOCAL_NAME_PATTERN.pattern!r}, got {value!r}"


@dataclass(frozen=True, order=True)
class SegmentId:
    """Identify a segment by end effector, subtask and segment name.

    Args:
        eef_name: End-effector name.
        subtask_name: Subtask name within that end effector.
        segment_name: Segment name within the subtask.
    """

    eef_name: str
    subtask_name: str
    segment_name: str

    def __post_init__(self) -> None:
        """Validate each part of the descriptor-local segment name."""

        validate_local_name(self.eef_name, field_name="eef_name")
        validate_local_name(self.subtask_name, field_name="subtask_name")
        validate_local_name(self.segment_name, field_name="segment_name")

    @classmethod
    def from_string(cls, value: str) -> SegmentId:
        """Parse a segment name such as ``left.pick.transit``.

        Args:
            value: End-effector, subtask and segment names separated by dots.

        Returns:
            Segment identified by the string.
        """

        assert isinstance(value, str), "segment ID must be a string"
        parts = value.split(".")
        assert len(parts) == 3, f"invalid segment ID: {value!r}"
        return cls(*parts)

    def __str__(self) -> str:
        """Return the dotted end-effector, subtask and segment name."""

        return f"{self.eef_name}.{self.subtask_name}.{self.segment_name}"


class SegmentKind(enum.StrEnum):
    """Semantic role of a trajectory segment."""

    TRANSIT = "transit"
    SKILL = "skill"
    HOLD = "hold"


class SegmentProvider(enum.StrEnum):
    """Source used to generate a segment trajectory."""

    MOTION_PLANNER = "motion_planner"
    DEMONSTRATION = "demonstration"
    SCRIPTED = "scripted"


@dataclass(frozen=True)
class MotionGoalSpec:
    """Tool-pose target for a motion-planned segment.

    The pose is expressed in the environment frame when ``object_ref`` is empty, otherwise in the
    current frame of that logical scene object.

    Args:
        position_m: Tool position [m].
        rotation_xyzw: Unit tool quaternion in ``(x, y, z, w)`` order.
        object_ref: Optional logical scene object used as the reference frame.
        object_pose_after_segment_id: Optional segment whose terminal released-object pose supplies
            ``object_ref``. Empty uses the reset-time scene pose.
    """

    position_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    object_ref: str = ""
    object_pose_after_segment_id: SegmentId | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Require a finite pose, unit XYZW quaternion and consistent object-frame reference."""

        assert len(self.position_m) == 3, "motion goal position_m must contain three values"
        assert len(self.rotation_xyzw) == 4, "motion goal rotation_xyzw must contain four values"
        values = (*self.position_m, *self.rotation_xyzw)
        assert all(
            type(value) in {int, float} and math.isfinite(value) for value in values
        ), "motion goal pose must contain finite numeric values"
        quaternion_norm = math.sqrt(sum(value * value for value in self.rotation_xyzw))
        assert math.isclose(
            quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5
        ), f"motion goal rotation_xyzw must be a unit quaternion, got norm {quaternion_norm:.6f}"
        if self.object_ref:
            validate_local_name(self.object_ref, field_name="motion goal object_ref")
        assert self.object_pose_after_segment_id is None or isinstance(
            self.object_pose_after_segment_id, SegmentId
        ), "object_pose_after_segment_id must be a SegmentId"
        assert (
            self.object_pose_after_segment_id is None or self.object_ref
        ), "object_pose_after_segment_id requires object_ref"


class SegmentEventType(enum.StrEnum):
    """Object-state transition on a segment boundary."""

    ATTACH = "attach"
    DETACH = "detach"


class SegmentEventPoint(enum.StrEnum):
    """Location of an object event within its segment."""

    START = "start"
    END = "end"
    SOURCE_ANNOTATION = "source_annotation"


@dataclass(frozen=True)
class DemonstrationSourceWindowSpec:
    """Event-relative end of the prefix retained from a demonstration segment.

    Args:
        anchor_annotation_key: Source annotation that anchors the retained endpoint.
        post_event_sample_count: Number of source samples retained after the event sample.
    """

    anchor_annotation_key: str
    post_event_sample_count: int

    def __post_init__(self) -> None:
        """Validate the annotation name and nonnegative trailing sample count."""

        validate_local_name(self.anchor_annotation_key, field_name="source-window anchor_annotation_key")
        assert (
            type(self.post_event_sample_count) is int and self.post_event_sample_count >= 0
        ), "post_event_sample_count must be a non-negative integer"


@dataclass(frozen=True, order=True)
class SegmentWorldContactSpec:
    """One object/fixture pair allowed to contact while a segment owns its lane.

    Args:
        object_ref: Logical projected object name.
        fixture_ref: Logical static world-object name.
    """

    object_ref: str
    fixture_ref: str

    def __post_init__(self) -> None:
        """Require distinct, valid logical object and fixture names."""

        validate_local_name(self.object_ref, field_name="world-contact object_ref")
        validate_local_name(self.fixture_ref, field_name="world-contact fixture_ref")
        assert self.object_ref != self.fixture_ref, "world contact requires distinct object and fixture refs"


@dataclass(frozen=True)
class SegmentRobotContactSpec:
    """Exact robot links allowed to contact one task object during a segment.

    Args:
        object_ref: Logical task object name.
        robot_link_names: Sorted robot collision-link names.
        start_annotation_key: Optional same-segment source annotation that starts the permission.
        clear_at_end: Whether the segment's final waypoint must be strictly clear of the object.
    """

    object_ref: str
    robot_link_names: tuple[str, ...]
    start_annotation_key: str = ""
    clear_at_end: bool = False

    def __post_init__(self) -> None:
        """Validate sorted unique link names and optional source-annotation timing."""

        validate_local_name(self.object_ref, field_name="robot-contact object_ref")
        assert (
            isinstance(self.robot_link_names, tuple) and self.robot_link_names
        ), "robot_link_names must be a non-empty tuple"
        assert all(
            isinstance(name, str) and name and name.strip() == name for name in self.robot_link_names
        ), "robot_link_names must contain non-empty trimmed strings"
        assert len(self.robot_link_names) == len(set(self.robot_link_names)), "robot_link_names must be unique"
        assert self.robot_link_names == tuple(sorted(self.robot_link_names)), "robot_link_names must be in sorted order"
        if self.start_annotation_key:
            validate_local_name(self.start_annotation_key, field_name="robot-contact start_annotation_key")
        assert type(self.clear_at_end) is bool, "clear_at_end must be a bool"


@dataclass(frozen=True)
class SegmentEventSpec:
    """Attach or detach an object at a stable segment boundary.

    Args:
        event_type: Object-state transition.
        object_ref: Logical task object name.
        point: Boundary or source annotation carrying the event.
        annotation_key: Dataset annotation used by ``SOURCE_ANNOTATION``.
        attachment_frame: Robot link used by ``ATTACH``.
    """

    event_type: SegmentEventType
    object_ref: str
    point: SegmentEventPoint = SegmentEventPoint.END
    annotation_key: str = ""
    attachment_frame: str = ""

    def __post_init__(self) -> None:
        """Validate annotation and attachment-frame fields for the selected event type."""

        validate_local_name(self.object_ref, field_name="object_ref")
        if self.point == SegmentEventPoint.SOURCE_ANNOTATION:
            validate_local_name(self.annotation_key, field_name="annotation_key")
        else:
            assert not self.annotation_key, "annotation_key requires point='source_annotation'"
        if self.event_type == SegmentEventType.ATTACH:
            assert self.attachment_frame, "attach events require attachment_frame"
        else:
            assert not self.attachment_frame, "detach events do not use attachment_frame"


@dataclass(frozen=True)
class SegmentSpec:
    """Declarative unit of a subtask path.

    Scheduling is provider-independent once the algorithm has built a segment's waypoint path.
    Descriptor validation must reject providers the selected algorithm does not support.

    Args:
        name: Stable descriptor-local segment name.
        kind: Semantic role of the segment.
        provider: Source used to build the segment.
        goal_segment_id: Segment whose prepared entry target is consumed by motion planning.
        motion_goal: Independent task-defined target consumed by motion planning.
        resource_ids: Additional task-local resources held for the nominal segment interval.
        events: Ordered object-state transitions on the segment.
        world_contacts: Exact object/fixture contact permissions.
        demonstration_source_window: Optional event-relative end of the retained demonstration.
        duration_steps: Number of samples for a scripted segment.
        passthrough_preset: Embodiment-owned auxiliary-joint target.
        cartesian_offset_m: Scripted environment-frame Cartesian offset [m].
        robot_contacts: Exact robot/object contact permissions.
    """

    name: str
    kind: SegmentKind
    provider: SegmentProvider
    goal_segment_id: SegmentId | None = field(default=None, kw_only=True)
    motion_goal: MotionGoalSpec | None = field(default=None, kw_only=True)
    resource_ids: tuple[str, ...] = field(default_factory=tuple, kw_only=True)
    events: tuple[SegmentEventSpec, ...] = field(default_factory=tuple)
    world_contacts: tuple[SegmentWorldContactSpec, ...] = field(default_factory=tuple)
    demonstration_source_window: DemonstrationSourceWindowSpec | None = None
    duration_steps: int = 0
    passthrough_preset: str = ""
    cartesian_offset_m: tuple[float, ...] = field(default_factory=tuple)
    robot_contacts: tuple[SegmentRobotContactSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate provider-specific goals, scripted commands, resources and contact permissions."""

        validate_local_name(self.name, field_name="segment name")
        assert isinstance(self.kind, SegmentKind), "kind must be a SegmentKind"
        assert isinstance(self.provider, SegmentProvider), "provider must be a SegmentProvider"
        if self.provider == SegmentProvider.MOTION_PLANNER:
            has_segment_goal = isinstance(self.goal_segment_id, SegmentId)
            has_motion_goal = isinstance(self.motion_goal, MotionGoalSpec)
            assert (
                has_segment_goal ^ has_motion_goal
            ), "motion-planner segments require a goal_segment_id or motion_goal, exclusively"
        else:
            assert self.goal_segment_id is None, "goal_segment_id is supported only by motion-planner segments"
            assert self.motion_goal is None, "motion_goal is supported only by motion-planner segments"
        assert isinstance(self.resource_ids, tuple), "resource_ids must be a tuple"
        for resource_id in self.resource_ids:
            validate_local_name(resource_id, field_name="resource_id")
        assert len(self.resource_ids) == len(set(self.resource_ids)), "resource_ids must be unique"
        assert self.resource_ids == tuple(sorted(self.resource_ids)), "resource_ids must be in sorted order"
        assert all(isinstance(event, SegmentEventSpec) for event in self.events)
        assert all(isinstance(contact, SegmentWorldContactSpec) for contact in self.world_contacts)
        assert len(self.world_contacts) == len(set(self.world_contacts)), "world contacts must be unique per segment"
        assert self.demonstration_source_window is None or isinstance(
            self.demonstration_source_window, DemonstrationSourceWindowSpec
        ), "demonstration_source_window must be a DemonstrationSourceWindowSpec"
        assert (
            type(self.duration_steps) is int and self.duration_steps >= 0
        ), "duration_steps must be a non-negative integer"
        assert isinstance(self.passthrough_preset, str), "passthrough_preset must be a string"
        if self.passthrough_preset:
            validate_local_name(self.passthrough_preset, field_name="passthrough_preset")
        assert isinstance(self.cartesian_offset_m, tuple) and len(self.cartesian_offset_m) in {
            0,
            3,
        }, "cartesian_offset_m must be empty or an XYZ tuple"
        assert all(
            type(value) in {int, float} and math.isfinite(value) for value in self.cartesian_offset_m
        ), "cartesian_offset_m must contain finite numeric values"
        assert all(isinstance(contact, SegmentRobotContactSpec) for contact in self.robot_contacts)
        assert len(self.robot_contacts) == len(set(self.robot_contacts)), "robot contacts must be unique per segment"


__all__ = [
    "DemonstrationSourceWindowSpec",
    "MotionGoalSpec",
    "SegmentEventPoint",
    "SegmentEventSpec",
    "SegmentEventType",
    "SegmentId",
    "SegmentKind",
    "SegmentProvider",
    "SegmentRobotContactSpec",
    "SegmentSpec",
    "SegmentWorldContactSpec",
    "validate_local_name",
]
