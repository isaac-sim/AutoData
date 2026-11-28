# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segment precedence and scheduling policy for ScheduleGen."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from autodata_interfaces.tasks.segment_spec import SegmentId


class SegmentAnchorPoint(enum.StrEnum):
    """Endpoint of a trajectory segment."""

    START = "start"
    END = "end"


@dataclass(frozen=True, order=True)
class SegmentAnchor:
    """Start or end of a scheduled segment.

    Args:
        segment_id: Segment being referenced.
        point: Start tick or exclusive end tick, one tick after the last waypoint starts.
    """

    segment_id: SegmentId
    point: SegmentAnchorPoint

    @classmethod
    def from_string(cls, value: str) -> SegmentAnchor:
        """Parse an endpoint such as ``left.pick.transit.end``.

        Args:
            value: End-effector, subtask, segment and endpoint names separated by dots.

        Returns:
            Segment and endpoint referenced by the string.
        """

        parts = value.split(".")
        assert len(parts) == 4, f"invalid segment anchor: {value!r}"
        return cls(SegmentId.from_string(".".join(parts[:3])), SegmentAnchorPoint(parts[3]))


@dataclass(frozen=True)
class PrecedenceConstraintSpec:
    """Bound the execution-tick lag from one segment endpoint to another.

    Args:
        before: Earlier segment endpoint.
        after: Later segment endpoint.
        min_lag_steps: Minimum number of control steps between the endpoints.
        max_lag_steps: Optional maximum number of control steps between the endpoints.
    """

    before: SegmentAnchor
    after: SegmentAnchor
    min_lag_steps: int = 0
    max_lag_steps: int | None = None

    def __post_init__(self) -> None:
        """Require distinct anchors and ordered nonnegative lag bounds."""

        assert self.before != self.after, "precedence anchors must differ"
        assert type(self.min_lag_steps) is int and self.min_lag_steps >= 0
        if self.max_lag_steps is not None:
            assert (
                type(self.max_lag_steps) is int and self.max_lag_steps >= self.min_lag_steps
            ), "max_lag_steps must be an integer no smaller than min_lag_steps"

    @property
    def referenced_segment_ids(self) -> tuple[SegmentId, SegmentId]:
        """Return segment identities referenced by this constraint."""

        return self.before.segment_id, self.after.segment_id


@dataclass(frozen=True)
class NonOverlapConstraintSpec:
    """Prevent two scheduled segment intervals from overlapping.

    Collision checks also cover initial and terminal holds outside these intervals.

    Args:
        segments: Pair of segments whose scheduled intervals must not overlap.
    """

    segments: tuple[SegmentId, SegmentId]

    def __post_init__(self) -> None:
        """Require exactly two distinct segment identities."""

        assert isinstance(self.segments, tuple) and len(self.segments) == 2, "non-overlap requires two segments"
        assert all(isinstance(segment_id, SegmentId) for segment_id in self.segments)
        assert self.segments[0] != self.segments[1], "non-overlap segments must differ"

    @property
    def referenced_segment_ids(self) -> tuple[SegmentId, SegmentId]:
        """Return segment identities referenced by this constraint."""

        return self.segments


@dataclass(frozen=True)
class NonPreemptiveConstraintSpec:
    """Prevent inserted waits within a same-lane segment sequence.

    The sequence may start late. Once it starts, every consecutive waypoint advances
    by one control step, including the boundaries between its segments.

    Args:
        segments: Consecutive segments on one command lane, listed in execution order.
    """

    segments: tuple[SegmentId, ...]

    def __post_init__(self) -> None:
        """Require a nonempty sequence of unique segment identities."""

        assert isinstance(self.segments, tuple) and self.segments, "non-preemptive requires segments"
        assert all(isinstance(segment_id, SegmentId) for segment_id in self.segments)
        assert len(self.segments) == len(set(self.segments)), "non-preemptive segments must be unique"

    @property
    def referenced_segment_ids(self) -> tuple[SegmentId, ...]:
        """Return segment identities referenced by this constraint."""

        return self.segments


type ScheduleConstraintSpec = PrecedenceConstraintSpec | NonOverlapConstraintSpec | NonPreemptiveConstraintSpec


class SchedulerStrategy(enum.StrEnum):
    """Choice between per-waypoint retiming and fixed-duration block scheduling."""

    WAYPOINT_RETIMING = "waypoint_retiming"
    BLOCK_SCHEDULING = "block_scheduling"


@dataclass(frozen=True)
class ScheduleSpec:
    """Scheduling settings and task-authored constraints.

    Args:
        strategy: Waypoint retiming or block scheduling.
        constraints: Timing constraints, independent of how each segment was generated.
        collision_margin_m: Collision clearance margin [m].
        time_limit_s: Optional solver wall-clock limit [s].
        max_block_size: Maximum waypoints per block; zero keeps each segment atomic.
        minimize_max_wait: Whether equal-makespan schedules minimize their largest inserted wait.
    """

    strategy: SchedulerStrategy = SchedulerStrategy.WAYPOINT_RETIMING
    constraints: tuple[ScheduleConstraintSpec, ...] = field(default_factory=tuple)
    collision_margin_m: float = 0.0
    time_limit_s: float | None = None
    max_block_size: int = 0
    minimize_max_wait: bool = True

    def __post_init__(self) -> None:
        """Validate solver settings, constraint types and strategy-specific block size."""

        assert isinstance(self.strategy, SchedulerStrategy)
        assert isinstance(self.constraints, tuple) and all(
            isinstance(item, (PrecedenceConstraintSpec, NonOverlapConstraintSpec, NonPreemptiveConstraintSpec))
            for item in self.constraints
        )
        assert math.isfinite(self.collision_margin_m) and self.collision_margin_m >= 0.0
        if self.time_limit_s is not None:
            assert math.isfinite(self.time_limit_s) and self.time_limit_s > 0.0
        assert type(self.max_block_size) is int and self.max_block_size >= 0
        assert (
            self.strategy == SchedulerStrategy.BLOCK_SCHEDULING or self.max_block_size == 0
        ), "max_block_size is supported only by block_scheduling"
        assert isinstance(self.minimize_max_wait, bool), "minimize_max_wait must be a bool"


__all__ = [
    "NonOverlapConstraintSpec",
    "NonPreemptiveConstraintSpec",
    "PrecedenceConstraintSpec",
    "ScheduleConstraintSpec",
    "ScheduleSpec",
    "SchedulerStrategy",
    "SegmentAnchor",
    "SegmentAnchorPoint",
    "SegmentId",
]
