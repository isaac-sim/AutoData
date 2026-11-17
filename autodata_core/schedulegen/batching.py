# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffered preparation and scheduling with the ordinary asynchronous episode replay."""

from __future__ import annotations

import asyncio
import contextlib
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from autodata_core.data_generator import GenerationResult
from autodata_core.schedulegen.types import EpisodeGenerationError
from autodata_core.waypoint import MultiWaypoint
from autodata_interfaces.env.isaaclab_env_interface import env_loop
from autodata_utils.generation_result import record_generation_attempt

if TYPE_CHECKING:
    from isaaclab.managers import TerminationTermCfg

    from autodata_core.data_generator import DataGenerator
    from autodata_core.schedulegen.algorithm import ScheduleGen, _PreparedEpisode
    from autodata_utils.generation_result import AttemptLog


def generate_buffered(
    env: Any,
    data_generator: DataGenerator,
    algorithm: ScheduleGen,
    success_term: TerminationTermCfg,
    schedule_workers: int,
    *,
    attempt_log: AttemptLog | None = None,
) -> dict:
    """Prepare waves on environment zero, schedule independently and replay in vectorized environments.

    Only CPU scheduling uses worker threads. Scene resets, GPU planning, validation and
    simulation stay on the calling thread. Geometry, dynamics and robot base poses must match
    across replay environments; each episode supplies its own environment-relative reset state.
    Resets finish outside the event loop before replay coroutines start.

    Args:
        env: Environment to step, including an optional recording wrapper.
        data_generator: Owner of source selection, recording and episode export.
        algorithm: ScheduleGen with one shared planning context bound to environment zero.
        success_term: Task success predicate evaluated by the ordinary replay coroutine.
        schedule_workers: Maximum number of simultaneous CPU schedule solves.
        attempt_log: Optional native metrics log receiving each completed attempt.

    Returns:
        Counters, completion-ordered attempts, timing totals [s] and target-reached status.
    """

    assert set(algorithm.planning_contexts) == {0}, "buffered generation requires one environment-zero context"
    assert type(schedule_workers) is int and schedule_workers > 0
    datastream = data_generator.datastream
    base_env = datastream.get_env()
    policy = datastream.get_generation_policy()
    schedule_spec = datastream.get_schedule_spec()
    assert schedule_spec is not None
    loop = asyncio.get_event_loop()
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}
    attempts: list[dict[str, object]] = []
    # Keep the saved timing keys stable; certify_s measures validation.
    timings = {name: 0.0 for name in ("prepare_s", "conflicts_s", "schedule_wall_s", "certify_s", "replay_s")}
    workers = min(schedule_workers, base_env.num_envs)
    planning_ids = torch.tensor([0], dtype=torch.int64, device=datastream.device)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            count = min(base_env.num_envs, policy.remaining_attempts(stats["num_attempts"], stats["num_success"]))
            if count <= 0:
                break
            print(f"Preparing buffered batch of {count} candidates.", flush=True)
            candidates = []
            for _ in range(count):
                attempt_started = time.monotonic()
                started = time.perf_counter()
                base_env.recorder_manager.reset(env_ids=planning_ids)
                base_env.reset(env_ids=planning_ids)
                prepared = algorithm._episode_prepare(data_generator, 0, success_term)
                prepared.start_time_s = attempt_started
                timings["prepare_s"] += time.perf_counter() - started
                candidates.append(prepared)
                if prepared.result is None:
                    assert prepared.paths is not None
                    started = time.perf_counter()
                    try:
                        prepared.conflicts = algorithm._conflicts_find(
                            dict(prepared.paths.arm_paths), schedule_spec.collision_margin_m, 0
                        )
                    except EpisodeGenerationError as failure:
                        _failed(data_generator, algorithm, prepared, success_term, failure)
                    timings["conflicts_s"] += time.perf_counter() - started

            started = time.perf_counter()
            futures = [
                (
                    prepared,
                    executor.submit(
                        algorithm._episode_schedule_solve,
                        data_generator=data_generator,
                        episode_paths=prepared.paths,
                        conflicts=prepared.conflicts,
                        schedule_spec=schedule_spec,
                    ),
                )
                for prepared in candidates
                if prepared.result is None
            ]
            solved = []
            for prepared, future in futures:
                try:
                    solved.append((prepared, future.result()))
                except EpisodeGenerationError as failure:
                    _failed(data_generator, algorithm, prepared, success_term, failure)
            timings["schedule_wall_s"] += time.perf_counter() - started
            started = time.perf_counter()
            for prepared, schedule in solved:
                algorithm._state_restore(datastream, prepared.initial_state, 0)
                try:
                    prepared.validated = algorithm._episode_validate_scheduled(
                        data_generator=data_generator,
                        episode_paths=prepared.paths,
                        initial_object_poses=prepared.initial_object_poses,
                        initial_joint_positions=prepared.initial_joint_positions,
                        schedule_spec=schedule_spec,
                        conflicts=prepared.conflicts,
                        schedule=schedule,
                        env_id=0,
                    )
                except EpisodeGenerationError as failure:
                    _failed(data_generator, algorithm, prepared, success_term, failure)
            timings["certify_s"] += time.perf_counter() - started
            for prepared in candidates:
                if prepared.result is not None:
                    record_generation_attempt(
                        GenerationResult(prepared.initial_state, False, prepared.result.diagnostics),
                        0,
                        stats,
                        attempts,
                        attempt_log=attempt_log,
                        start_time_s=prepared.start_time_s,
                    )
            ready = [prepared for prepared in candidates if prepared.validated is not None]
            started = time.perf_counter()
            if ready:
                _replay_wave(
                    env, data_generator, algorithm, success_term, ready, loop, stats, attempts, attempt_log=attempt_log
                )
            timings["replay_s"] += time.perf_counter() - started
    env_ids = torch.arange(base_env.num_envs, device=datastream.device)
    base_env.recorder_manager.reset(env_ids=env_ids)
    completed = stats["num_success"] if policy.guarantee_success else stats["num_attempts"]
    return {
        "stats": stats,
        "attempts": attempts,
        "completed": completed >= policy.num_trials,
        "execution": {"schedule_workers": workers, "timings_s": timings},
    }


def _failed(
    data_generator: DataGenerator,
    algorithm: ScheduleGen,
    prepared: _PreparedEpisode,
    success_term: TerminationTermCfg,
    failure: EpisodeGenerationError,
) -> None:
    """Retain a pre-replay failure on the candidate's original state.

    Args:
        data_generator: Generator owning the live datastream.
        algorithm: Algorithm providing state restoration and failure diagnostics.
        prepared: Candidate whose preparation or scheduling failed.
        success_term: Task success predicate used by failure diagnostics.
        failure: Expected generation failure reported on the calling thread.
    """

    algorithm._state_restore(data_generator.datastream, prepared.initial_state, 0)
    prepared.result = algorithm._failure_result(
        data_generator=data_generator,
        env_id=0,
        success_term=success_term,
        selected_source=prepared.selected_source,
        source_candidate_diagnostics=prepared.source_candidate_diagnostics,
        failure=failure,
    )


def _states_stack(states: list[dict]) -> dict:
    """Stack matching single-environment scene states without modifying the copies.

    Args:
        states: Environment-relative states, each containing one tensor row per scene field.

    Returns:
        A scene state with one row per replay slot.
    """

    assert states and all(set(state) == set(states[0]) for state in states)
    stacked = {}
    for name, value in states[0].items():
        values = [state[name] for state in states]
        if isinstance(value, dict):
            stacked[name] = _states_stack(values)
        else:
            assert all(isinstance(item, torch.Tensor) and item.shape == value.shape for item in values)
            assert value.shape[0] == 1, "buffered states must contain exactly one environment"
            stacked[name] = torch.cat(values, dim=0)
    return stacked


def _replay_wave(
    env: Any,
    data_generator: DataGenerator,
    algorithm: ScheduleGen,
    success_term: TerminationTermCfg,
    ready: list[_PreparedEpisode],
    loop: asyncio.AbstractEventLoop,
    stats: dict[str, int],
    attempts: list[dict[str, object]],
    *,
    attempt_log: AttemptLog | None = None,
) -> None:
    """Replay a wave and hold finished slots without exporting their tails.

    Unassigned slots hold their initial validated target without executing another episode.

    Args:
        env: Environment to step, including an optional recording wrapper.
        data_generator: Owner of recording and episode export.
        algorithm: Algorithm providing the shared episode replay coroutine.
        success_term: Task success predicate evaluated during replay.
        ready: Validated candidates assigned to the first available replay slots.
        attempt_log: Optional native metrics log receiving each completed attempt.
        loop: Event loop driven by the synchronous simulator step loop.
        stats: Mutable counters for the complete generation run.
        attempts: Mutable completion-ordered attempt records.
    """

    datastream = data_generator.datastream
    base_env = datastream.get_env()
    assert 0 < len(ready) <= base_env.num_envs
    slots = [*ready, *([ready[0]] * (base_env.num_envs - len(ready)))]
    env_ids = torch.arange(base_env.num_envs, device=datastream.device)
    base_env.recorder_manager.reset(env_ids=env_ids)
    base_env.reset_to(_states_stack([prepared.initial_state for prepared in slots]), env_ids=env_ids, is_relative=True)
    action_queue = asyncio.Queue()
    attempts_before = stats["num_attempts"]

    async def replay(env_id: int, prepared: _PreparedEpisode) -> None:
        """Export a real slot at its exact endpoint, then emit valid terminal holds until the wave ends."""

        if env_id < len(ready):
            result = await algorithm._episode_execute(data_generator, prepared, env_id, success_term, action_queue)
            data_generator._finalize_episode_recording(
                env_id_tensor=env_ids[env_id : env_id + 1], success=result.success, export_demo=True
            )
            record_generation_attempt(
                GenerationResult(prepared.initial_state, result.success, result.diagnostics),
                env_id,
                stats,
                attempts,
                attempt_log=attempt_log,
                start_time_s=prepared.start_time_s,
            )
        hold_index = -1 if env_id < len(ready) else 0
        waypoints = {name: path.waypoints[hold_index] for name, path in prepared.validated.arm_paths.items()}
        while True:
            trajectory = prepared.validated.replay_joint_trajectory
            if trajectory is not None:
                datastream.set_robot_joint_posture_target(
                    trajectory.joint_names, trajectory.positions[hold_index], env_id
                )
            await MultiWaypoint(waypoints).execute(datastream, success_term, env_id, action_queue)

    tasks = asyncio.gather(*(replay(env_id, prepared) for env_id, prepared in enumerate(slots)))
    try:
        env_loop(
            env,
            asyncio.Queue(),
            action_queue,
            loop,
            generation_policy_params=datastream.get_generation_policy(),
            stats=stats,
            data_gen_tasks=tasks,
            stop_after_attempts=len(ready),
        )
        assert stats["num_attempts"] - attempts_before == len(
            ready
        ), "buffered replay stopped before all episodes completed"
    finally:
        tasks.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(tasks)
        for env_id in range(base_env.num_envs):
            datastream.reset_robot_joint_posture_target(env_id)
