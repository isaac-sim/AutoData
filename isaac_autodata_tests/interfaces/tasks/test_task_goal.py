# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaac_autodata_core.autonomous.task_motion import GoalPredicate as LegacyGoalPredicate
from isaac_autodata_interfaces.tasks import GoalPredicate


def test_goal_predicate_is_shared_with_autonomous_plan_contracts() -> None:
    assert LegacyGoalPredicate is GoalPredicate


def test_goal_predicate_round_trip() -> None:
    predicate = GoalPredicate("on", "cube", "bowl")

    assert GoalPredicate.from_dict(predicate.to_dict()) == predicate


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", "cube", "bowl"), "relation"),
        (("on", "", "bowl"), "subject"),
        (("on", "cube", ""), "target"),
    ],
)
def test_goal_predicate_rejects_invalid_text(args, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GoalPredicate(*args)
