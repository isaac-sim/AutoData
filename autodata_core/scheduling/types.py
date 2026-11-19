# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent paths, collision conflicts and schedule results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from autodata_interfaces.tasks.schedule_constraint_spec import SegmentId
from autodata_interfaces.tasks.segment_spec import validate_local_name


@dataclass(frozen=True)
class SchedulePath:
    """Segment path metadata passed to the scheduling solver.

    Planned, demonstrated and scripted trajectories use the same ordered-path interface.

    Args:
        segment_id: Stable task segment identity.
        lane: Persistent command lane, normally an end-effector name.
        waypoint_count: Number of configurations in the segment path.
        resource_ids: Additional task-local resources held for the nominal segment interval.
            Each resource must be claimed on at least two lanes in the same scheduling call.
    """

    segment_id: SegmentId
    lane: str
    waypoint_count: int
    resource_ids: tuple[str, ...] = field(default_factory=tuple, kw_only=True)

    def __post_init__(self) -> None:
        """Validate segment metadata, positive length and sorted unique resource names."""

        assert isinstance(self.segment_id, SegmentId), "segment_id must be a SegmentId"
        assert (
            isinstance(self.lane, str) and self.lane and self.lane == self.lane.strip()
        ), "lane must be a non-empty, trimmed string"
        assert type(self.waypoint_count) is int and self.waypoint_count > 0, "waypoint_count must be a positive integer"
        assert isinstance(self.resource_ids, tuple), "resource_ids must be a tuple"
        for resource_id in self.resource_ids:
            validate_local_name(resource_id, field_name="resource_id")
        assert len(self.resource_ids) == len(set(self.resource_ids)), "resource_ids must be unique"
        assert self.resource_ids == tuple(sorted(self.resource_ids)), "resource_ids must be in sorted order"


@dataclass(frozen=True, order=True)
class SegmentWaypoint:
    """A configuration within a segment path.

    Args:
        segment_id: Segment containing the configuration.
        waypoint_index: Zero-based index within the segment.
    """

    segment_id: SegmentId
    waypoint_index: int

    def __post_init__(self) -> None:
        """Validate the segment identity and nonnegative local waypoint index."""

        assert isinstance(self.segment_id, SegmentId), "segment_id must be a SegmentId"
        assert (
            type(self.waypoint_index) is int and self.waypoint_index >= 0
        ), "waypoint_index must be a non-negative integer"


@dataclass(frozen=True)
class WaypointConflict:
    """Two configurations that may not be occupied at the same time.

    Args:
        first: Waypoint on one command lane.
        second: Conflicting waypoint on another command lane.
    """

    first: SegmentWaypoint
    second: SegmentWaypoint

    def __post_init__(self) -> None:
        """Require two distinct, typed waypoint references."""

        assert isinstance(self.first, SegmentWaypoint) and isinstance(
            self.second, SegmentWaypoint
        ), "conflict endpoints must be SegmentWaypoint values"
        assert self.first != self.second, "a waypoint cannot conflict with itself"

    @property
    def canonical_key(self) -> tuple[SegmentWaypoint, SegmentWaypoint]:
        """Return an order-independent conflict identity."""

        return tuple(sorted((self.first, self.second)))


@dataclass(frozen=True)
class Schedule:
    """Integer execution ticks for every segment waypoint.

    Args:
        waypoint_ticks: Segment-local waypoint ticks, keyed by stable identity.
        makespan_ticks: Exclusive terminal episode tick.
    """

    waypoint_ticks: Mapping[SegmentId, tuple[int, ...]]
    makespan_ticks: int

    def __post_init__(self) -> None:
        """Validate nonnegative command ticks and freeze a copy of the segment mapping."""

        assert (
            isinstance(self.waypoint_ticks, Mapping) and self.waypoint_ticks
        ), "waypoint_ticks must be a non-empty mapping"
        canonical: dict[SegmentId, tuple[int, ...]] = {}
        for segment_id, ticks in self.waypoint_ticks.items():
            assert isinstance(segment_id, SegmentId), "schedule keys must be SegmentId values"
            assert isinstance(ticks, tuple) and ticks, "every scheduled path must contain waypoint ticks"
            assert all(
                type(tick) is int and tick >= 0 for tick in ticks
            ), "waypoint ticks must be non-negative integers"
            canonical[segment_id] = ticks
        assert type(self.makespan_ticks) is int and self.makespan_ticks > 0, "makespan_ticks must be a positive integer"
        object.__setattr__(self, "waypoint_ticks", MappingProxyType(canonical))

    def ticks_for(self, segment_id: SegmentId) -> tuple[int, ...]:
        """Return the execution ticks for one segment.

        Args:
            segment_id: Segment to query.

        Returns:
            Execution ticks in segment waypoint order.
        """

        return self.waypoint_ticks[segment_id]


@dataclass(frozen=True)
class ScheduleFailure:
    """Scheduling failure with no executable result.

    Args:
        reason: Short failure category.
        message: Description of the failure.
    """

    reason: str
    message: str

    def __post_init__(self) -> None:
        """Require nonempty, trimmed failure descriptions."""

        assert self.reason and self.reason == self.reason.strip(), "reason must be non-empty"
        assert self.message and self.message == self.message.strip(), "message must be non-empty"


__all__ = [
    "Schedule",
    "ScheduleFailure",
    "SchedulePath",
    "SegmentWaypoint",
    "WaypointConflict",
]
