# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import pytest

from isaac_autodata_core.waypoint import Waypoint, WaypointSequence, WaypointTrajectory
from isaac_autodata_interfaces.task_planners import TaskPlannerBase, TaskPlanningError, TaskPlanningNoSolutionError
from isaac_autodata_interfaces.tasks import GoalPredicate


class _Planner(TaskPlannerBase):
    def _plan(self, goal, *, seed: int):
        del goal, seed
        return WaypointTrajectory()


class _SuccessfulPlanner(TaskPlannerBase):
    def _plan(self, goal, *, seed: int):
        del goal, seed
        trajectory = WaypointTrajectory()
        trajectory.add_waypoint_sequence(WaypointSequence([Waypoint(pose=torch.eye(4), gripper_action=torch.ones(1))]))
        return trajectory


def test_task_planner_binds_datastream_and_environment() -> None:
    datastream = object()
    planner = _Planner(datastream, env_id=2)

    assert planner.datastream is datastream
    assert planner.env_id == 2
    assert planner.close() is None


@pytest.mark.parametrize("env_id", [-1, True, 1.5])
def test_task_planner_rejects_invalid_environment_id(env_id) -> None:
    with pytest.raises(ValueError, match="env_id"):
        _Planner(object(), env_id=env_id)


def test_no_solution_error_is_a_task_planning_error() -> None:
    assert isinstance(TaskPlanningNoSolutionError("no feasible plan"), TaskPlanningError)


@pytest.mark.parametrize(
    ("goal", "seed", "message"),
    [
        ((), 0, "goal"),
        (("not-a-predicate",), 0, "goal"),
        ((GoalPredicate("on", "cube", "bowl"),), -1, "seed"),
        ((GoalPredicate("on", "cube", "bowl"),), True, "seed"),
    ],
)
def test_task_planner_validates_common_inputs(goal, seed: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _Planner(object()).plan(goal, seed=seed)


def test_task_planner_translates_empty_backend_result_to_no_solution() -> None:
    goal = (GoalPredicate("on", "cube", "bowl"),)

    with pytest.raises(TaskPlanningNoSolutionError, match="empty waypoint trajectory"):
        _Planner(object()).plan(goal, seed=0)


def test_task_planner_returns_existing_autodata_trajectory() -> None:
    goal = (GoalPredicate("on", "cube", "bowl"),)

    trajectory = _SuccessfulPlanner(object()).plan(goal, seed=0)

    assert isinstance(trajectory, WaypointTrajectory)
    assert len(trajectory) == 1
