# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for planner-neutral goal compilation into ScheduleStream formulas."""

from __future__ import annotations

import pytest

from isaac_autodata_interfaces.task_planners.schedulestream.goal import (
    GoalCompilationError,
    ScheduleStreamGoalSymbols,
    compile_schedulestream_goal,
)
from isaac_autodata_interfaces.tasks.task_goal import GoalPredicate


class _Formula:
    def __init__(self, text: str) -> None:
        self.text = text

    def __and__(self, other: _Formula) -> _Formula:
        return _Formula(f"({self.text} & {other.text})")


SYMBOLS = ScheduleStreamGoalSymbols(
    attached_equals=lambda subject, target: _Formula(f"Attached({subject}) == {target}"),
    holding_equals=lambda arm, subject: _Formula(f"Holding({arm}) == {subject}"),
)


def test_compiles_ordered_goal_conjunction():
    result = compile_schedulestream_goal(
        (
            GoalPredicate("on", "cube_2", "cube_1"),
            GoalPredicate("on", "cube_3", "cube_2"),
        ),
        SYMBOLS,
        arm="panda_arm",
    )

    assert result.text == "(Attached(cube_2) == cube_1 & Attached(cube_3) == cube_2)"


def test_compiles_holding_goal_with_selected_arm():
    result = compile_schedulestream_goal(
        (GoalPredicate("holding", "cube", None),),
        SYMBOLS,
        arm="panda_arm",
    )

    assert result.text == "Holding(panda_arm) == cube"


def test_rejects_capability_unsupported_relation():
    with pytest.raises(GoalCompilationError, match="not supported"):
        compile_schedulestream_goal(
            (GoalPredicate("in", "cube", "drawer"),),
            SYMBOLS,
            arm="panda_arm",
            supported_relations=frozenset({"on"}),
        )


def test_rejects_invalid_predicate_arity_and_empty_goals():
    with pytest.raises(GoalCompilationError, match="requires target"):
        compile_schedulestream_goal((GoalPredicate("on", "cube"),), SYMBOLS, arm="panda_arm")
    with pytest.raises(GoalCompilationError, match="at least one"):
        compile_schedulestream_goal((), SYMBOLS, arm="panda_arm")
