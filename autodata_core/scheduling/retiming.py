# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native waypoint retiming with collision and segment-order constraints."""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass
from itertools import combinations
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from autodata_core.scheduling.types import Schedule, ScheduleFailure, SchedulePath, SegmentWaypoint, WaypointConflict
from autodata_interfaces.tasks.schedule_constraint_spec import (
    NonOverlapConstraintSpec,
    NonPreemptiveConstraintSpec,
    PrecedenceConstraintSpec,
    SchedulerStrategy,
    SegmentAnchor,
    SegmentAnchorPoint,
    SegmentId,
)

type _Expression = tuple[dict[int, float], float]
type _ConstraintRow = tuple[dict[int, float], float, float]
type _ConflictGroup = tuple[int, int, tuple[SegmentWaypoint, ...], tuple[SegmentWaypoint, ...]]


@dataclass(frozen=True)
class _Layout:
    """Flattened persistent command lanes and solver-variable indices."""

    paths: tuple[SchedulePath, ...]
    paths_by_id: dict[SegmentId, SchedulePath]
    lane_by_segment: dict[SegmentId, str]
    waypoints_by_lane: dict[str, tuple[SegmentWaypoint, ...]]
    time_by_waypoint: dict[SegmentWaypoint, _Expression]
    group_by_waypoint: dict[SegmentWaypoint, int]
    waypoints_by_group: dict[int, tuple[SegmentWaypoint, ...]]
    successor_by_waypoint: dict[SegmentWaypoint, SegmentWaypoint | None]

    @property
    def waypoint_count(self) -> int:
        """Return the total number of commands across all lanes."""

        return len(self.time_by_waypoint)

    @property
    def time_variable_count(self) -> int:
        """Return the number of independently timed waypoint groups."""

        return len(self.waypoints_by_group)


def schedule_paths(
    paths: tuple[SchedulePath, ...],
    *,
    conflicts: tuple[WaypointConflict, ...] = (),
    precedence: tuple[PrecedenceConstraintSpec, ...] = (),
    non_overlap: tuple[NonOverlapConstraintSpec, ...] = (),
    non_preemptive: tuple[NonPreemptiveConstraintSpec, ...] = (),
    strategy: SchedulerStrategy = SchedulerStrategy.WAYPOINT_RETIMING,
    max_block_size: int = 0,
    serialize_segments: bool = False,
    time_limit_s: float | None = None,
    minimize_max_wait: bool = True,
) -> Schedule | ScheduleFailure:
    """Retime ordered segment paths without changing or dropping waypoints.

    Paths sharing a lane are concatenated in declaration order. Before its first
    segment starts, a lane holds that segment's initial waypoint from tick zero.
    Every waypoint is then held until the next waypoint on that lane begins, and
    the final waypoint is held through the episode makespan. Collision conflicts
    therefore constrain exact held-state intervals, including initial,
    inter-segment, and terminal holds.

    A solver failure never produces an executable fallback schedule.

    Args:
        paths: Segment path metadata in lane execution order.
        conflicts: Cross-lane colliding waypoint pairs.
        precedence: Segment start/end precedence constraints.
        non_overlap: Unordered whole-segment non-overlap constraints.
        non_preemptive: Contiguous same-lane segment sequences that may not acquire waits.
        strategy: Singleton-waypoint retiming or conservative block scheduling.
        max_block_size: Maximum waypoints per block; zero keeps each segment atomic.
        serialize_segments: Whether all cross-lane segment intervals must be disjoint.
        time_limit_s: Optional HiGHS wall-clock limit [s].
        minimize_max_wait: Whether equal-makespan schedules minimize their largest inserted wait.

    Returns:
        Independently validated schedule, or a scheduling failure with no executable result.
    """

    assert isinstance(paths, tuple) and paths, "paths must be a non-empty tuple"
    assert all(isinstance(path, SchedulePath) for path in paths), "paths must contain SchedulePath values"
    assert isinstance(conflicts, tuple) and all(
        isinstance(item, WaypointConflict) for item in conflicts
    ), "conflicts must contain WaypointConflict values"
    assert isinstance(precedence, tuple) and all(
        isinstance(item, PrecedenceConstraintSpec) for item in precedence
    ), "precedence must contain PrecedenceConstraintSpec values"
    assert isinstance(non_overlap, tuple) and all(
        isinstance(item, NonOverlapConstraintSpec) for item in non_overlap
    ), "non_overlap must contain NonOverlapConstraintSpec values"
    assert isinstance(non_preemptive, tuple) and all(
        isinstance(item, NonPreemptiveConstraintSpec) for item in non_preemptive
    ), "non_preemptive must contain NonPreemptiveConstraintSpec values"
    assert isinstance(strategy, SchedulerStrategy), "strategy must be a SchedulerStrategy"
    assert type(max_block_size) is int and max_block_size >= 0, "max_block_size must be non-negative"
    assert (
        strategy == SchedulerStrategy.BLOCK_SCHEDULING or max_block_size == 0
    ), "max_block_size is supported only by block_scheduling"
    assert isinstance(serialize_segments, bool), "serialize_segments must be a bool"
    assert time_limit_s is None or time_limit_s > 0.0, "time_limit_s must be positive"
    assert isinstance(minimize_max_wait, bool), "minimize_max_wait must be a bool"

    group_size = 1 if strategy == SchedulerStrategy.WAYPOINT_RETIMING else max_block_size
    layout = _layout_create(paths, max_group_size=group_size)
    conflicts = _conflicts_validate_and_deduplicate(layout, conflicts)
    conflict_groups = _conflict_groups(layout, conflicts)
    _precedence_validate(layout, precedence)
    non_preemptive_pairs = _non_preemptive_pairs(layout, non_preemptive)
    if serialize_segments:
        non_overlap = (
            *non_overlap,
            *(
                NonOverlapConstraintSpec((first.segment_id, second.segment_id))
                for first, second in combinations(paths, 2)
                if first.lane != second.lane
            ),
        )
    non_overlap = _non_overlap_resolve(layout, non_overlap)

    horizon_ticks = _horizon_ticks(layout, precedence)
    makespan_variable = layout.time_variable_count
    max_wait_variable = makespan_variable + 1 if minimize_max_wait else None
    binary_offset = makespan_variable + 1 + int(minimize_max_wait)
    non_overlap_binary_offset = binary_offset + len(conflict_groups)
    variable_count = non_overlap_binary_offset + len(non_overlap)
    time_lower_bounds, time_upper_bounds = _time_variable_bounds(layout, horizon_ticks)
    rows: list[_ConstraintRow] = []

    for lane_waypoints in layout.waypoints_by_lane.values():
        for first, second in zip(lane_waypoints, lane_waypoints[1:], strict=False):
            if layout.group_by_waypoint[first] == layout.group_by_waypoint[second]:
                continue
            progress = _expressions_combine(
                (1.0, layout.time_by_waypoint[second]),
                (-1.0, layout.time_by_waypoint[first]),
            )
            _constraint_add(
                rows,
                progress,
                lower=1.0,
                upper=1.0 if (first, second) in non_preemptive_pairs else None,
            )
            if max_wait_variable is not None:
                max_wait = _expressions_combine(
                    (1.0, _variable(max_wait_variable)),
                    (-1.0, progress),
                )
                _constraint_add(rows, max_wait, lower=-1.0)

        terminal_progress = _expressions_combine(
            (1.0, _variable(makespan_variable)),
            (-1.0, layout.time_by_waypoint[lane_waypoints[-1]]),
        )
        _constraint_add(rows, terminal_progress, lower=1.0)

    for constraint in precedence:
        lag = _expressions_combine(
            (1.0, _anchor_expression(layout, constraint.after)),
            (-1.0, _anchor_expression(layout, constraint.before)),
        )
        _constraint_add(
            rows,
            lag,
            lower=float(constraint.min_lag_steps),
            upper=None if constraint.max_lag_steps is None else float(constraint.max_lag_steps),
        )

    for conflict_index, (first_group, second_group, first_conflicts, second_conflicts) in enumerate(conflict_groups):
        order_variable = binary_offset + conflict_index
        first_start, first_end = _conflict_group_interval(
            layout,
            first_group,
            first_conflicts,
            makespan_variable,
        )
        second_start, second_end = _conflict_group_interval(
            layout,
            second_group,
            second_conflicts,
            makespan_variable,
        )

        _non_overlap_rows_add(
            rows,
            first_start=first_start,
            first_end=first_end,
            second_start=second_start,
            second_end=second_end,
            order_variable=order_variable,
            time_lower_bounds=time_lower_bounds,
            time_upper_bounds=time_upper_bounds,
        )

    for constraint_index, constraint in enumerate(non_overlap):
        first_id, second_id = constraint.segments
        order_variable = non_overlap_binary_offset + constraint_index
        _non_overlap_rows_add(
            rows,
            first_start=_anchor_expression(layout, SegmentAnchor(first_id, SegmentAnchorPoint.START)),
            first_end=_anchor_expression(layout, SegmentAnchor(first_id, SegmentAnchorPoint.END)),
            second_start=_anchor_expression(layout, SegmentAnchor(second_id, SegmentAnchorPoint.START)),
            second_end=_anchor_expression(layout, SegmentAnchor(second_id, SegmentAnchorPoint.END)),
            order_variable=order_variable,
            time_lower_bounds=time_lower_bounds,
            time_upper_bounds=time_upper_bounds,
        )

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[makespan_variable] = 1.0
    if max_wait_variable is not None:
        objective[max_wait_variable] = 1.0 / (horizon_ticks + 1)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, float(horizon_ticks), dtype=np.float64)
    upper_bounds[: layout.time_variable_count] = float(horizon_ticks - 1)
    upper_bounds[binary_offset:] = 1.0
    integrality = np.ones(variable_count, dtype=np.uint8)
    linear_constraints = _scipy_constraints(rows, variable_count)
    options: dict[str, float | bool] = {}
    deadline = None if time_limit_s is None else time.monotonic() + time_limit_s

    try:
        # The pinned HiGHS presolver can falsely report valid disjunctive models as infeasible.
        for presolve in (True, False):
            options["presolve"] = presolve
            if deadline is not None:
                remaining_time_s = deadline - time.monotonic()
                if remaining_time_s <= 0.0:
                    return ScheduleFailure(
                        reason="limit_reached",
                        message="MILP time limit was exhausted before the solver attempt",
                    )
                options["time_limit"] = remaining_time_s
            result = milp(
                c=objective,
                integrality=integrality,
                bounds=Bounds(lower_bounds, upper_bounds),
                constraints=linear_constraints,
                options=options,
            )
            if int(result.status) != 2:
                break
    except (RuntimeError, ValueError) as exc:
        return ScheduleFailure(reason="solver_error", message=f"{type(exc).__name__}: {exc}")

    if int(result.status) != 0 or result.x is None:
        reasons = {1: "limit_reached", 2: "infeasible", 3: "unbounded"}
        reason = reasons.get(int(result.status), "solver_error")
        message = str(result.message).strip() or f"MILP returned status {result.status}"
        return ScheduleFailure(reason=reason, message=message)

    values = np.asarray(result.x, dtype=np.float64)
    rounded = np.rint(values)
    if not bool(np.allclose(values, rounded, atol=1.0e-6, rtol=0.0)):
        return ScheduleFailure(reason="invalid_candidate", message="MILP returned a fractional integer candidate")

    ticks_by_segment = {
        path.segment_id: tuple(
            int(_expression_value(layout.time_by_waypoint[SegmentWaypoint(path.segment_id, waypoint_index)], rounded))
            for waypoint_index in range(path.waypoint_count)
        )
        for path in paths
    }
    schedule = Schedule(
        waypoint_ticks=ticks_by_segment,
        makespan_ticks=int(rounded[makespan_variable]),
    )
    validation_error = _schedule_validation_error(
        layout,
        conflicts,
        precedence,
        non_overlap,
        non_preemptive,
        schedule,
    )
    if validation_error is not None:
        return ScheduleFailure(reason="invalid_candidate", message=validation_error)
    return schedule


def _layout_create(paths: tuple[SchedulePath, ...], *, max_group_size: int) -> _Layout:
    """Assign time variables to fixed-rate groups and link consecutive lane commands.

    Args:
        paths: Segments in execution order within each lane.
        max_group_size: Maximum commands per fixed-rate group; zero groups each whole segment.

    Returns:
        Waypoint expressions, group membership and successors across segment boundaries.
    """

    paths_by_id = {path.segment_id: path for path in paths}
    assert len(paths_by_id) == len(paths), "segment IDs must be unique"
    assert type(max_group_size) is int and max_group_size >= 0
    lane_by_segment = {path.segment_id: path.lane for path in paths}
    lane_waypoint_lists: dict[str, list[SegmentWaypoint]] = {}
    time_by_waypoint: dict[SegmentWaypoint, _Expression] = {}
    group_by_waypoint: dict[SegmentWaypoint, int] = {}
    waypoints_by_group: dict[int, tuple[SegmentWaypoint, ...]] = {}
    for path in paths:
        path_waypoints = tuple(
            SegmentWaypoint(path.segment_id, waypoint_index) for waypoint_index in range(path.waypoint_count)
        )
        lane_waypoint_lists.setdefault(path.lane, []).extend(path_waypoints)
        group_size = path.waypoint_count if max_group_size == 0 else max_group_size
        for start in range(0, path.waypoint_count, group_size):
            group_waypoints = path_waypoints[start : start + group_size]
            group_index = len(waypoints_by_group)
            waypoints_by_group[group_index] = group_waypoints
            for offset, waypoint in enumerate(group_waypoints):
                time_by_waypoint[waypoint] = _variable(group_index, constant=float(offset))
                group_by_waypoint[waypoint] = group_index
    waypoints_by_lane = {lane: tuple(waypoints) for lane, waypoints in lane_waypoint_lists.items()}
    successor_by_waypoint: dict[SegmentWaypoint, SegmentWaypoint | None] = {}
    for lane_waypoints in waypoints_by_lane.values():
        for waypoint, successor in zip(lane_waypoints, (*lane_waypoints[1:], None), strict=True):
            successor_by_waypoint[waypoint] = successor
    return _Layout(
        paths=paths,
        paths_by_id=paths_by_id,
        lane_by_segment=lane_by_segment,
        waypoints_by_lane=waypoints_by_lane,
        time_by_waypoint=time_by_waypoint,
        group_by_waypoint=group_by_waypoint,
        waypoints_by_group=waypoints_by_group,
        successor_by_waypoint=successor_by_waypoint,
    )


def _conflicts_validate_and_deduplicate(
    layout: _Layout,
    conflicts: tuple[WaypointConflict, ...],
) -> tuple[WaypointConflict, ...]:
    """Validate cross-lane conflict indices and retain each unordered pair once."""

    canonical: dict[tuple[SegmentWaypoint, SegmentWaypoint], WaypointConflict] = {}
    for conflict in conflicts:
        for waypoint in (conflict.first, conflict.second):
            path = layout.paths_by_id.get(waypoint.segment_id)
            assert path is not None, f"conflict references unknown segment {waypoint.segment_id}"
            assert (
                waypoint.waypoint_index < path.waypoint_count
            ), f"conflict waypoint {waypoint.waypoint_index} is outside segment {waypoint.segment_id}"
        first_lane = layout.lane_by_segment[conflict.first.segment_id]
        second_lane = layout.lane_by_segment[conflict.second.segment_id]
        assert first_lane != second_lane, "waypoint conflicts must be cross-lane"
        first, second = conflict.canonical_key
        canonical.setdefault((first, second), WaypointConflict(first, second))
    return tuple(canonical.values())


def _conflict_groups(
    layout: _Layout,
    conflicts: tuple[WaypointConflict, ...],
) -> tuple[_ConflictGroup, ...]:
    """Return unique time-group pairs requiring conservative non-overlap."""

    waypoints_by_pair: dict[tuple[int, int], tuple[set[SegmentWaypoint], set[SegmentWaypoint]]] = {}
    for conflict in conflicts:
        first_group = layout.group_by_waypoint[conflict.first]
        second_group = layout.group_by_waypoint[conflict.second]
        if first_group <= second_group:
            pair = first_group, second_group
            first, second = conflict.first, conflict.second
        else:
            pair = second_group, first_group
            first, second = conflict.second, conflict.first
        first_waypoints, second_waypoints = waypoints_by_pair.setdefault(pair, (set(), set()))
        first_waypoints.add(first)
        second_waypoints.add(second)
    return tuple(
        (
            first_group,
            second_group,
            tuple(sorted(first_waypoints)),
            tuple(sorted(second_waypoints)),
        )
        for (first_group, second_group), (first_waypoints, second_waypoints) in sorted(waypoints_by_pair.items())
    )


def _precedence_validate(layout: _Layout, precedence: tuple[PrecedenceConstraintSpec, ...]) -> None:
    """Require every precedence anchor to reference a segment in the problem."""

    for constraint in precedence:
        for anchor in (constraint.before, constraint.after):
            assert (
                anchor.segment_id in layout.paths_by_id
            ), f"precedence anchor references unknown segment {anchor.segment_id}"


def _non_preemptive_pairs(
    layout: _Layout,
    constraints: tuple[NonPreemptiveConstraintSpec, ...],
) -> set[tuple[SegmentWaypoint, SegmentWaypoint]]:
    """Return adjacent command pairs that must execute exactly one tick apart.

    Each configured sequence must be contiguous and ordered within a single lane. Initial
    waiting remains allowed; no wait can be inserted between the returned pairs.
    """

    pairs: set[tuple[SegmentWaypoint, SegmentWaypoint]] = set()
    for constraint in constraints:
        for segment_id in constraint.segments:
            assert segment_id in layout.paths_by_id, f"non-preemptive references unknown segment {segment_id}"
        lanes = {layout.lane_by_segment[segment_id] for segment_id in constraint.segments}
        assert len(lanes) == 1, "non-preemptive segments must use one lane"
        lane_waypoints = layout.waypoints_by_lane[lanes.pop()]
        constrained = tuple(
            SegmentWaypoint(segment_id, waypoint_index)
            for segment_id in constraint.segments
            for waypoint_index in range(layout.paths_by_id[segment_id].waypoint_count)
        )
        start = lane_waypoints.index(constrained[0])
        assert (
            lane_waypoints[start : start + len(constrained)] == constrained
        ), "non-preemptive segments must be contiguous and in lane order"
        pairs.update(zip(constrained, constrained[1:], strict=False))
    return pairs


def _non_overlap_resolve(
    layout: _Layout,
    authored: tuple[NonOverlapConstraintSpec, ...],
) -> tuple[NonOverlapConstraintSpec, ...]:
    """Combine authored and shared-resource mutexes into one deduplicated constraint tuple."""

    constraints: list[NonOverlapConstraintSpec] = []
    seen_pairs: set[tuple[SegmentId, SegmentId]] = set()

    def append_pair(segments: tuple[SegmentId, SegmentId]) -> None:
        """Append a validated cross-lane segment pair unless it is already present."""

        for segment_id in segments:
            assert segment_id in layout.paths_by_id, f"non-overlap references unknown segment {segment_id}"
        pair = tuple(sorted(segments))
        if layout.lane_by_segment[pair[0]] == layout.lane_by_segment[pair[1]]:
            return
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            constraints.append(NonOverlapConstraintSpec(pair))

    for constraint in authored:
        append_pair(constraint.segments)

    paths_by_resource: dict[str, list[SchedulePath]] = {}
    for path in layout.paths:
        for resource_id in path.resource_ids:
            paths_by_resource.setdefault(resource_id, []).append(path)
    for resource_id, resource_paths in sorted(paths_by_resource.items()):
        lanes = {path.lane for path in resource_paths}
        assert len(lanes) >= 2, f"resource {resource_id!r} must be claimed by segments on at least two lanes"
        for first, second in combinations(sorted(resource_paths, key=lambda path: path.segment_id), 2):
            if first.lane != second.lane:
                append_pair((first.segment_id, second.segment_id))
    return tuple(constraints)


def _horizon_ticks(layout: _Layout, precedence: tuple[PrecedenceConstraintSpec, ...]) -> int:
    """Bound schedule length by serialized commands plus all minimum precedence lags."""

    serialized_path_ticks = sum(path.waypoint_count for path in layout.paths)
    explicit_wait_ticks = sum(constraint.min_lag_steps for constraint in precedence)
    return serialized_path_ticks + explicit_wait_ticks + 1


def _time_variable_bounds(layout: _Layout, horizon_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Return lane-order-implied time bounds for numerically tight disjunctions."""

    lower_bounds = np.zeros(layout.time_variable_count + 1, dtype=np.float64)
    upper_bounds = np.full(layout.time_variable_count + 1, float(horizon_ticks), dtype=np.float64)
    bounded_groups: set[int] = set()
    for lane_waypoints in layout.waypoints_by_lane.values():
        lane_length = len(lane_waypoints)
        for rank, waypoint in enumerate(lane_waypoints):
            group = layout.group_by_waypoint[waypoint]
            if group in bounded_groups:
                continue
            bounded_groups.add(group)
            lower_bounds[group] = float(rank)
            upper_bounds[group] = float(horizon_ticks - (lane_length - rank))
    return lower_bounds, upper_bounds


def _variable(index: int, *, constant: float = 0.0) -> _Expression:
    """Represent one solver variable plus a constant as an affine expression."""

    return {index: 1.0}, constant


def _expressions_combine(*terms: tuple[float, _Expression]) -> _Expression:
    """Sum scaled affine expressions, removing zero coefficients."""

    coefficients: dict[int, float] = {}
    constant = 0.0
    for scale, (expression_coefficients, expression_constant) in terms:
        constant += scale * expression_constant
        for variable, coefficient in expression_coefficients.items():
            coefficients[variable] = coefficients.get(variable, 0.0) + scale * coefficient
    return {variable: value for variable, value in coefficients.items() if value}, constant


def _expression_value(expression: _Expression, values: np.ndarray) -> float:
    """Evaluate an affine expression at the supplied solver values."""

    coefficients, constant = expression
    return constant + sum(coefficient * values[variable] for variable, coefficient in coefficients.items())


def _anchor_expression(layout: _Layout, anchor: SegmentAnchor) -> _Expression:
    """Express a segment's first command tick or exclusive terminal tick."""

    path = layout.paths_by_id[anchor.segment_id]
    waypoint_index = 0 if anchor.point == SegmentAnchorPoint.START else path.waypoint_count - 1
    waypoint = SegmentWaypoint(path.segment_id, waypoint_index)
    expression = layout.time_by_waypoint[waypoint]
    if anchor.point == SegmentAnchorPoint.START:
        return expression
    return expression[0], expression[1] + 1.0


def _waypoint_end_expression(
    layout: _Layout,
    waypoint: SegmentWaypoint,
    makespan_variable: int,
) -> _Expression:
    """Express occupancy end at the next lane command, or the episode makespan."""

    successor = layout.successor_by_waypoint[waypoint]
    if successor is None:
        return _variable(makespan_variable)
    return layout.time_by_waypoint[successor]


def _conflict_group_interval(
    layout: _Layout,
    group: int,
    conflict_waypoints: tuple[SegmentWaypoint, ...],
    makespan_variable: int,
) -> tuple[_Expression, _Expression]:
    """Return a block interval, extending only holds covered by the collision conflict.

    Args:
        layout: Solver variables and waypoint groups for the ordered paths.
        group: Waypoint group whose occupied interval is needed.
        conflict_waypoints: Group waypoints participating in this collision conflict.
        makespan_variable: Solver variable used when the lane holds its final waypoint.

    Returns:
        Affine expressions for the interval's start and exclusive end ticks.
    """

    waypoints = layout.waypoints_by_group[group]
    first = waypoints[0]
    last = waypoints[-1]
    lane = layout.lane_by_segment[first.segment_id]
    start = layout.time_by_waypoint[first]
    if first in conflict_waypoints and first == layout.waypoints_by_lane[lane][0]:
        start = {}, 0.0
    end = layout.time_by_waypoint[last][0], layout.time_by_waypoint[last][1] + 1.0
    if last in conflict_waypoints:
        end = _waypoint_end_expression(layout, last, makespan_variable)
    return start, end


def _non_overlap_rows_add(
    rows: list[_ConstraintRow],
    *,
    first_start: _Expression,
    first_end: _Expression,
    second_start: _Expression,
    second_end: _Expression,
    order_variable: int,
    time_lower_bounds: np.ndarray,
    time_upper_bounds: np.ndarray,
) -> None:
    """Append bounded big-M rows selecting one of two non-overlapping interval orders.

    The binary order variable selects first-before-second when one, and second-before-first
    when zero. Each row uses its own expression bound rather than a shared arbitrary big-M.
    """

    first_before_second_difference = _expressions_combine(
        (1.0, first_end),
        (-1.0, second_start),
    )
    first_before_second_m = max(
        0.0,
        _expression_upper_bound(first_before_second_difference, time_lower_bounds, time_upper_bounds),
    )
    first_before_second = _expressions_combine(
        (1.0, first_before_second_difference),
        (first_before_second_m, _variable(order_variable)),
    )
    _constraint_add(rows, first_before_second, upper=first_before_second_m)

    second_before_first_difference = _expressions_combine(
        (1.0, second_end),
        (-1.0, first_start),
    )
    second_before_first_m = max(
        0.0,
        _expression_upper_bound(second_before_first_difference, time_lower_bounds, time_upper_bounds),
    )
    second_before_first = _expressions_combine(
        (1.0, second_before_first_difference),
        (-second_before_first_m, _variable(order_variable)),
    )
    _constraint_add(rows, second_before_first, upper=0.0)


def _expression_upper_bound(
    expression: _Expression,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    """Maximize an affine expression over independent variable bounds."""

    coefficients, constant = expression
    return constant + sum(
        coefficient * (upper_bounds[variable] if coefficient > 0.0 else lower_bounds[variable])
        for variable, coefficient in coefficients.items()
    )


def _constraint_add(
    rows: list[_ConstraintRow],
    expression: _Expression,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Append a bounded linear row after moving the expression constant into its bounds."""

    assert lower is not None or upper is not None, "a constraint requires at least one bound"
    coefficients, constant = expression
    rows.append((
        coefficients,
        -np.inf if lower is None else lower - constant,
        np.inf if upper is None else upper - constant,
    ))


def _scipy_constraints(rows: list[_ConstraintRow], variable_count: int) -> LinearConstraint:
    """Assemble affine constraint rows into the sparse matrix accepted by SciPy MILP."""

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for row_index, (coefficients, lower, upper) in enumerate(rows):
        for variable, coefficient in coefficients.items():
            matrix_rows.append(row_index)
            matrix_columns.append(variable)
            matrix_values.append(coefficient)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    matrix = coo_array(
        (
            np.asarray(matrix_values, dtype=np.float64),
            (np.asarray(matrix_rows, dtype=np.int32), np.asarray(matrix_columns, dtype=np.int32)),
        ),
        shape=(len(rows), variable_count),
    ).tocsc()
    return LinearConstraint(matrix, np.asarray(lower_bounds), np.asarray(upper_bounds))


def _schedule_validation_error(
    layout: _Layout,
    conflicts: tuple[WaypointConflict, ...],
    precedence: tuple[PrecedenceConstraintSpec, ...],
    non_overlap: tuple[NonOverlapConstraintSpec, ...],
    non_preemptive: tuple[NonPreemptiveConstraintSpec, ...],
    schedule: Schedule,
) -> str | None:
    """Check the integer schedule independently of solver rows and tolerances.

    Args:
        layout: Segment coverage, command order and persistent lane successors.
        conflicts: Forbidden simultaneous waypoint occupancies, including initial/final holds.
        precedence: Required lags between segment anchors.
        non_overlap: Segment intervals that must be disjoint.
        non_preemptive: Contiguous command sequences that cannot contain waits.
        schedule: Rounded candidate schedule returned by the solver.

    Returns:
        The first violated constraint, or ``None`` when all checks pass.
    """

    if set(schedule.waypoint_ticks) != set(layout.paths_by_id):
        return "candidate segment coverage differs from the scheduling problem"

    for path in layout.paths:
        ticks = schedule.ticks_for(path.segment_id)
        if len(ticks) != path.waypoint_count:
            return f"candidate waypoint count differs for segment {path.segment_id}"
    for lane, lane_waypoints in layout.waypoints_by_lane.items():
        ticks = tuple(schedule.ticks_for(waypoint.segment_id)[waypoint.waypoint_index] for waypoint in lane_waypoints)
        if any(second <= first for first, second in zip(ticks, ticks[1:], strict=False)):
            return f"lane {lane!r} does not preserve strict waypoint order"

    expected_makespan = max(schedule.ticks_for(path.segment_id)[-1] + 1 for path in layout.paths)
    if schedule.makespan_ticks != expected_makespan:
        return "candidate makespan is not the latest segment end"

    for constraint in precedence:
        before = _anchor_tick(schedule, constraint.before)
        after = _anchor_tick(schedule, constraint.after)
        lag = after - before
        if lag < constraint.min_lag_steps:
            return f"candidate violates precedence minimum lag for {constraint.before} -> {constraint.after}"
        if constraint.max_lag_steps is not None and lag > constraint.max_lag_steps:
            return f"candidate violates precedence maximum lag for {constraint.before} -> {constraint.after}"

    for constraint in non_overlap:
        first_id, second_id = constraint.segments
        first_interval = schedule.ticks_for(first_id)[0], schedule.ticks_for(first_id)[-1] + 1
        second_interval = schedule.ticks_for(second_id)[0], schedule.ticks_for(second_id)[-1] + 1
        if first_interval[0] < second_interval[1] and second_interval[0] < first_interval[1]:
            return f"candidate overlaps constrained segments {first_id} and {second_id}"

    for constraint in non_preemptive:
        ticks = tuple(tick for segment_id in constraint.segments for tick in schedule.ticks_for(segment_id))
        if any(second != first + 1 for first, second in zip(ticks, ticks[1:], strict=False)):
            return f"candidate preempts constrained segment sequence {constraint.segments}"

    for conflict in conflicts:
        first_interval = _occupancy_interval(layout, schedule, conflict.first)
        second_interval = _occupancy_interval(layout, schedule, conflict.second)
        if first_interval[0] < second_interval[1] and second_interval[0] < first_interval[1]:
            return f"candidate retains collision between {conflict.first} and {conflict.second}"
    return None


def _anchor_tick(schedule: Schedule, anchor: SegmentAnchor) -> int:
    """Return the first command tick or exclusive terminal tick of a segment."""

    ticks = schedule.ticks_for(anchor.segment_id)
    return ticks[0] if anchor.point == SegmentAnchorPoint.START else ticks[-1] + 1


def _occupancy_interval(
    layout: _Layout,
    schedule: Schedule,
    waypoint: SegmentWaypoint,
) -> tuple[int, int]:
    """Return a waypoint's half-open occupancy interval, including boundary holds.

    The first lane configuration occupies tick zero onward, even before its command tick.
    Every configuration persists until its successor; the final one persists to makespan.
    """

    lane = layout.lane_by_segment[waypoint.segment_id]
    start = (
        0
        if waypoint == layout.waypoints_by_lane[lane][0]
        else schedule.ticks_for(waypoint.segment_id)[waypoint.waypoint_index]
    )
    successor = layout.successor_by_waypoint[waypoint]
    if successor is None:
        return start, schedule.makespan_ticks
    end = schedule.ticks_for(successor.segment_id)[successor.waypoint_index]
    return start, end


__all__ = ["schedule_paths"]
