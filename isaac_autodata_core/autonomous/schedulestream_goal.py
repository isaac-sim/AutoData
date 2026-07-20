# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed Arena/AutoData goal translation into ScheduleStream formulas."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from isaac_autodata_core.autonomous.task_motion import GoalPredicate


class GoalCompilationError(ValueError):
    """Raised when a planner-neutral goal cannot be represented by the selected domain."""


@dataclass(frozen=True)
class ScheduleStreamGoalSymbols:
    """Minimal injected ScheduleStream language surface used by the pure goal compiler."""

    attached_equals: Callable[[str, str], Any]
    holding_equals: Callable[[str, str], Any]


def compile_schedulestream_goal(
    predicates: Iterable[GoalPredicate],
    symbols: ScheduleStreamGoalSymbols,
    *,
    arm: str,
    supported_relations: frozenset[str] = frozenset({"on", "in", "at", "holding"}),
) -> Any:
    """Compile planner-neutral predicates into one ScheduleStream conjunction.

    Args:
        predicates: Goal clauses resolved to live scene IDs.
        symbols: Injected ScheduleStream language constructors.
        arm: ScheduleStream arm identifier used by ``holding``.
        supported_relations: Relations admitted by the selected domain and runtime-support profile.

    Returns:
        A ScheduleStream formula object. Its concrete type remains behind the adapter boundary.
    """

    clauses: list[Any] = []
    for index, predicate in enumerate(predicates):
        relation = predicate.relation.lower()
        if relation not in supported_relations:
            raise GoalCompilationError(
                f"goal[{index}] relation {predicate.relation!r} is not supported by this "
                f"ScheduleStream capability; supported: {sorted(supported_relations)}"
            )
        if relation == "holding":
            if predicate.target is not None:
                raise GoalCompilationError(f"goal[{index}] holding is unary and must not define target")
            clauses.append(symbols.holding_equals(arm, predicate.subject))
            continue
        if predicate.target is None:
            raise GoalCompilationError(f"goal[{index}] relation {relation!r} requires target")
        # Both ScheduleStream domains represent final support/containment/at-placement facts through
        # Attached(object) == destination. Domain placement streams remain responsible for testing
        # whether that particular relation is geometrically feasible.
        clauses.append(symbols.attached_equals(predicate.subject, predicate.target))

    if not clauses:
        raise GoalCompilationError("ScheduleStream goal must contain at least one predicate")
    goal = clauses[0]
    for clause in clauses[1:]:
        goal = goal & clause
    return goal


def load_schedulestream_goal_symbols(application: str) -> ScheduleStreamGoalSymbols:
    """Lazily load goal symbols for ``custream`` (v1) or ``custream2`` (v2).

    Args:
        application: ScheduleStream manipulation application selected by the runtime capability
            probe.
    """

    if application == "custream":
        from schedulestream.applications.custream.example import Attached, Holding
    elif application == "custream2":
        from schedulestream.applications.custream2.tamp import Attached, Holding
    else:
        raise GoalCompilationError(
            f"Unknown ScheduleStream application {application!r}; expected 'custream' or 'custream2'"
        )
    return ScheduleStreamGoalSymbols(
        attached_equals=lambda subject, target: Attached(subject) == target,
        holding_equals=lambda arm, subject: Holding(arm) == subject,
    )
