# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ScheduleStream generation algorithm (cuStream2 / cuRoboV2 backend).

Unlike MimicGen/SkillGen — which transform *recorded source demonstrations* per subtask —
ScheduleStream plans the **whole task from scratch** from a symbolic goal via cuStream2's
``solve_tamp`` and replays the resulting command sequence. The planning itself lives in
:class:`~isaac_autodata_core.schedulestream_planner.ScheduleStreamPlanner`, a cuStream2
:class:`Planner` subclass grounded in the Isaac AutoData env (see that module).

To fit the per-subtask plug-in contract, the matching task descriptor declares a **single
subtask** per EEF (see ``tasks/franka_cube_stack_schedulestream.yaml``). The single
:meth:`ScheduleStream.plan_subtask_trajectory` call solves TAMP once and returns the entire
trajectory as a flat ``list[Waypoint]``; :class:`DataGenerator` then steps and records it like any
other generated demo.

The planner module (and its cuRobo / cuStream2 / isaaclab dependency chain) is imported lazily
inside :meth:`ScheduleStream._get_planner`, so importing this module to register the algorithm
stays cheap and dependency-free until a run actually selects ``--alg schedulestream``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaac_autodata_core.algorithms import GenerationAlgorithm

if TYPE_CHECKING:
    from isaac_autodata_core.data_generator import DataGenerator, _EEFGenerationState
    from isaac_autodata_core.schedulestream_planner import ScheduleStreamPlanner
    from isaac_autodata_core.waypoint import Waypoint
    from isaac_autodata_interfaces.datastream.datastream import Datastream


class ScheduleStream(GenerationAlgorithm):
    """Whole-task TAMP via cuStream2 (cuRoboV2), exposed as a single-subtask generation algorithm.

    The algorithm owns one :class:`ScheduleStreamPlanner` per ``env_id`` (lazily built, reused
    across ``generate()`` calls). Each invocation re-syncs the planner world from the live env,
    solves TAMP for the task goal, and converts the command rollout into executable waypoints.
    """

    name = "schedulestream"
    expected_eef_count = 1
    requires_motion_planner = False
    uses_subtask_start_signals = False
    supports_coordination = False

    def __init__(self, success_term: Any) -> None:
        """
        Args:
            success_term: The task's success :class:`TerminationTermCfg`; the symbolic goal is
                derived from it (e.g. the cube-stacking termination yields stacking ``Attached``
                relations). It is passed in because ``setup_env_config`` strips it from the env, so
                it cannot be recovered at plan time. All other tuning (collisions, max_time,
                profile, hold, animate) is read from the task descriptor's single-subtask
                ``algo_params`` (:class:`ScheduleStreamSubtaskAlgoParams`) via the datastream —
                see :meth:`_get_planner`.
        """
        self.success_term = success_term
        self._planners: dict[int, ScheduleStreamPlanner] = {}

    def validate_setup(self, datastream: Datastream) -> None:
        """Require exactly one subtask per EEF — ScheduleStream plans the whole task at once."""
        for eef_name in datastream.get_eef_names():
            num_subtasks = datastream.num_subtasks(eef_name)
            if num_subtasks != 1:
                raise ValueError(
                    "schedulestream plans the whole task in one shot and expects exactly one "
                    f"subtask per EEF, but EEF {eef_name!r} declares {num_subtasks}. Use a "
                    "single-subtask descriptor (e.g. tasks/franka_cube_stack_schedulestream.yaml)."
                )

    def plan_subtask_trajectory(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        eef_name: str,
        eef_state: _EEFGenerationState,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> tuple[list[Waypoint], bool] | None:
        """Solve TAMP once for the (single) subtask and return the full trajectory.

        Returns ``(waypoints, False)`` to execute the planned trajectory as the subtask.

        On planning failure the planner RAISES rather than returning ``None``. Returning ``None``
        makes ``generate()`` report failure without ever enqueuing an action, but ``env_loop``
        blocks waiting for an action and so never reaches its attempt-count stop check — the failed
        attempt is retried forever (a livelock of repeated tracebacks). Raising propagates out
        through ``generate()`` and the data-gen task, where ``env_loop`` re-raises it and the
        process terminates with the traceback. For multi-trial production with retry-on-failure,
        this would need the upstream env_loop to count no-action attempts; until then, fail loudly.
        """
        planner = self._get_planner(data_generator.datastream, env_id)
        waypoints = planner.get_waypoints()
        if not waypoints:
            raise RuntimeError(
                f"schedulestream planning produced no trajectory for env {env_id} "
                f"(solve_tamp found no plan or returned empty commands)."
            )
        return waypoints, False

    def _get_planner(self, datastream: Datastream, env_id: int) -> ScheduleStreamPlanner:
        if env_id not in self._planners:
            # Deferred import: the planner module pulls isaaclab + cuStream2/cuRobo, which require
            # the running Isaac app; importing this module (to register the algorithm) stays cheap.
            from isaac_autodata_core.schedulestream_planner import ScheduleStreamPlanner

            # Tuning lives on the single subtask's algo_params (ScheduleStreamSubtaskAlgoParams),
            # read here via the datastream so it stays in the task descriptor, not the CLI.
            params = datastream.get_subtask_algo_params(datastream.get_eef_names()[0])[0]
            self._planners[env_id] = ScheduleStreamPlanner(
                datastream,
                self.success_term,
                env_id=env_id,
                metric="ee",
                motion="tool",
                collisions=params.collisions,
                max_time=params.max_time,
                profile=params.profile,
                hold=params.hold,
                animate=params.animate,
            )
        return self._planners[env_id]
