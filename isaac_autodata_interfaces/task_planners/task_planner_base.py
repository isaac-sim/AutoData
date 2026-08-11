# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for planners that solve an entire semantic task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaac_autodata_interfaces.tasks.task_goal import GoalPredicate

if TYPE_CHECKING:
    from isaac_autodata_core.waypoint import WaypointTrajectory
    from isaac_autodata_interfaces.datastream.datastream import Datastream


class TaskPlanningError(RuntimeError):
    """Base error for task-planner configuration or execution failures."""


class TaskPlanningNoSolutionError(TaskPlanningError):
    """The bounded search completed without finding a feasible task plan."""


class TaskPlannerBase(ABC):
    """Plan whole-task behavior from state exposed by a :class:`Datastream`.

    This interface is intentionally separate from point-to-point motion planners. A task planner
    may invoke motion planning internally while deciding which objects to manipulate and in what
    order.
    """

    def __init__(self, datastream: Datastream, env_id: int = 0) -> None:
        """Bind the planner to one environment in a Datastream.

        Args:
            datastream: Shared source of live robot, object, and scene state.
            env_id: Vectorized environment index.
        """

        if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
            raise ValueError("env_id must be a non-negative integer")
        self.datastream = datastream
        self.env_id = env_id

    def plan(self, goal: tuple[GoalPredicate, ...], *, seed: int) -> WaypointTrajectory:
        """Plan and lower one semantic goal into an AutoData waypoint trajectory.

        Args:
            goal: Planner-neutral terminal predicates resolved to live scene IDs.
            seed: Non-negative random seed for this planning attempt.

        Returns:
            Existing AutoData trajectory representation for the complete task.

        Raises:
            TaskPlanningNoSolutionError: If bounded search finds no feasible plan.
            TaskPlanningError: If planning cannot run or its output cannot be lowered safely.
        """

        if not goal or any(not isinstance(predicate, GoalPredicate) for predicate in goal):
            raise ValueError("goal must contain GoalPredicate instances")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        trajectory = self._plan(goal, seed=seed)
        if not trajectory:
            raise TaskPlanningNoSolutionError("task planner returned an empty waypoint trajectory")
        return trajectory

    @abstractmethod
    def _plan(self, goal: tuple[GoalPredicate, ...], *, seed: int) -> WaypointTrajectory:
        """Implement backend-specific task planning after common request validation."""

        raise NotImplementedError

    def close(self) -> None:
        """Release planner-owned resources idempotently."""
