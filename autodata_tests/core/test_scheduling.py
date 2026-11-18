# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduling contracts independent of simulator and trajectory provider."""

from concurrent.futures import ThreadPoolExecutor

import pytest


def test_secondary_objective_is_not_hidden_by_a_relative_mip_gap(monkeypatch):
    from autodata_core.scheduling import retiming

    original = retiming.milp
    options = []

    def solve(*args, **kwargs):
        options.append(kwargs["options"].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(retiming, "milp", solve)
    segment = SegmentId.from_string("left.move.transit")
    assert isinstance(schedule_paths((SchedulePath(segment, "left", 2),)), Schedule)
    assert options and all(value["mip_rel_gap"] == 0.0 for value in options)


from autodata_core.scheduling import (
    Schedule,
    ScheduleFailure,
    SchedulePath,
    SegmentWaypoint,
    WaypointConflict,
    schedule_paths,
)
from autodata_interfaces.tasks.schedule_constraint_spec import (
    NonPreemptiveConstraintSpec,
    PrecedenceConstraintSpec,
    SchedulerStrategy,
    SegmentAnchor,
    SegmentAnchorPoint,
    SegmentId,
)


@pytest.mark.parametrize("strategy", tuple(SchedulerStrategy))
@pytest.mark.parametrize("minimize_max_wait", (False, True))
def test_collision_conflicts_preserve_precedence_and_non_preemption(strategy, minimize_max_wait):
    left = SegmentId.from_string("left.pour.transit")
    right = SegmentId.from_string("right.pick.skill")
    paths = (SchedulePath(left, "left", 4), SchedulePath(right, "right", 4))
    constraint = PrecedenceConstraintSpec(
        before=SegmentAnchor.from_string("left.pour.transit.end"),
        after=SegmentAnchor.from_string("right.pick.skill.end"),
    )

    result = schedule_paths(
        paths,
        conflicts=(WaypointConflict(SegmentWaypoint(left, 1), SegmentWaypoint(right, 1)),),
        precedence=(constraint,),
        non_preemptive=(NonPreemptiveConstraintSpec((right,)),),
        strategy=strategy,
        minimize_max_wait=minimize_max_wait,
    )

    assert isinstance(result, Schedule)
    left_ticks, right_ticks = result.ticks_for(left), result.ticks_for(right)
    assert left_ticks[-1] <= right_ticks[-1]
    assert all(second - first == 1 for first, second in zip(right_ticks, right_ticks[1:]))
    assert left_ticks[2] <= right_ticks[1] or right_ticks[2] <= left_ticks[1]


def test_independent_schedules_solve_in_parallel_without_sharing_results():
    left = SegmentId.from_string("left.pick.transit")
    right = SegmentId.from_string("right.pick.transit")

    def solve(length):
        return schedule_paths((SchedulePath(left, "left", length), SchedulePath(right, "right", length + 1)))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(solve, (3, 5, 4, 7)))
    assert all(isinstance(result, Schedule) for result in results)
    assert [result.makespan_ticks for result in results] == [4, 6, 5, 8]


def test_colliding_initial_states_fail_instead_of_delaying_occupied_endpoints():
    left = SegmentId.from_string("left.pick.transit")
    right = SegmentId.from_string("right.pick.transit")

    result = schedule_paths(
        (SchedulePath(left, "left", 3), SchedulePath(right, "right", 3)),
        conflicts=(WaypointConflict(SegmentWaypoint(left, 0), SegmentWaypoint(right, 0)),),
    )

    assert isinstance(result, ScheduleFailure)


def test_equal_start_and_non_preemption_synchronize_equal_length_skills():
    """Existing endpoint and continuity constraints compose into fixed-rate synchronization."""
    left = SegmentId.from_string("left.cooperate.skill")
    right = SegmentId.from_string("right.cooperate.skill")
    paths = (
        SchedulePath(SegmentId.from_string("left.cooperate.transit"), "left", 3),
        SchedulePath(left, "left", 4),
        SchedulePath(SegmentId.from_string("right.cooperate.transit"), "right", 5),
        SchedulePath(right, "right", 4),
    )

    result = schedule_paths(
        paths,
        precedence=(
            PrecedenceConstraintSpec(
                SegmentAnchor(left, SegmentAnchorPoint.START),
                SegmentAnchor(right, SegmentAnchorPoint.START),
                max_lag_steps=0,
            ),
        ),
        non_preemptive=(NonPreemptiveConstraintSpec((left,)), NonPreemptiveConstraintSpec((right,))),
    )

    assert isinstance(result, Schedule)
    assert result.ticks_for(left) == result.ticks_for(right)
    assert result.makespan_ticks == 9


def test_precedence_keeps_provider_at_hold_until_consumer_finishes():
    """Constrain departure, not the nominal hold end, when an arm sustains a resource."""
    hold = SegmentId.from_string("left.support.hold")
    leave = SegmentId.from_string("left.support.departure")
    work = SegmentId.from_string("right.insert.skill")
    paths = (
        SchedulePath(SegmentId.from_string("left.support.transit"), "left", 2),
        SchedulePath(hold, "left", 1),
        SchedulePath(leave, "left", 2),
        SchedulePath(SegmentId.from_string("right.insert.transit"), "right", 1),
        SchedulePath(work, "right", 4),
    )

    result = schedule_paths(
        paths,
        precedence=(
            PrecedenceConstraintSpec(
                SegmentAnchor(hold, SegmentAnchorPoint.START), SegmentAnchor(work, SegmentAnchorPoint.START)
            ),
            PrecedenceConstraintSpec(
                SegmentAnchor(work, SegmentAnchorPoint.END), SegmentAnchor(leave, SegmentAnchorPoint.START)
            ),
        ),
        non_preemptive=(NonPreemptiveConstraintSpec((work,)),),
    )

    assert isinstance(result, Schedule)
    assert result.ticks_for(hold)[0] <= result.ticks_for(work)[0]
    assert result.ticks_for(leave)[0] >= result.ticks_for(work)[-1] + 1
