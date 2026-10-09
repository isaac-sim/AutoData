# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel (multi-env) retargeting replay.

Replays many source demos at once on a single vectorized env, reusing the proven async-worker +
``env_loop`` machinery that generation runs. One async worker per env pulls episodes from a shared
feed, resets *its* env to the source scene (``prepare_episode`` with ``reset_sim=False``), and drives
the trajectory by putting one action per env-step on the shared action queue; ``env_loop`` batches all
envs into a single ``env.step``. When the episodes run out, a worker holds (zero action) so the loop
keeps stepping until every episode is done.

Not supported in parallel mode (use ``--num_envs 1``): ``init_robot_from_ik`` (env-0 PinkIK) and the
per-step tracking-error report -- both single-env only.
"""

import asyncio
import contextlib
import sys
import torch
import traceback
from copy import deepcopy

from autodata_interfaces.env import env_loop

from .provider import PlanProvider, ReplayResult
from .replay import EpisodeOutcome, ReplayContext, episode_steps, prepare_episode


async def _async_step(env, env_id, action_queue, target_adapter, target_eef_pose_dict, passthrough_action_dict) -> None:
    """Encode one action for ``env_id`` and hand it to ``env_loop`` (which batches all envs and steps)."""
    action = target_adapter.target_eef_pose_to_action(
        target_eef_pose_dict=target_eef_pose_dict,
        passthrough_action_dict=passthrough_action_dict,
        env_id=env_id,
    )
    if action.dim() > 1:
        action = action[0]
    await action_queue.put((env_id, action.to(device=env.device)))
    await action_queue.join()


async def _replay_one_episode(
    ctx: ReplayContext, env, env_id: int, plan, action_queue: asyncio.Queue
) -> EpisodeOutcome:
    """Reset ``env_id`` to the source scene and drive one plan's trajectory async.

    Runs the same step loop as the sequential path (:func:`~.replay.episode_steps`); each step is handed
    to ``env_loop``, which batches every env into one ``env.step``. The per-EEF scheduler state is local to
    this env, so a sync barrier joins the arms of one env -- bimanual + hand-off run in parallel.
    """
    prep = prepare_episode(ctx, env, env_id, plan.episode, reset_sim=False)
    steps = episode_steps(ctx, env, env_id, prep)
    try:
        command = next(steps)
        while True:
            await _async_step(env, env_id, action_queue, ctx.target_adapter, *command)
            command = steps.send(None)
    except StopIteration as stop:
        return stop.value


def _zeros_passthrough(env, target_adapter) -> dict[str, torch.Tensor]:
    """A zero-filled passthrough dict matching the adapter's channel layout (for never-replayed workers)."""
    zeros = torch.zeros(1, int(env.action_space.shape[-1]), device=env.device)
    return {name: value[0] for name, value in target_adapter.actions_to_passthrough_actions(zeros).items()}


def _hold_action(env, env_id, target_adapter, passthrough) -> torch.Tensor:
    """A valid 'stay here' action for any controller: command the EEF's *current* pose.

    For a delta-pose env this is a zero delta; for an absolute-pose PinkIK env it is the current pose --
    both hold the robot in place, unlike a zero action, whose zero/identity target pose is unreachable
    and makes the IK solver spam "could not find a solution / NaN" every step for each finished env.
    """
    current = {eef: pose[0] for eef, pose in target_adapter.get_eef_poses(env_ids=[env_id]).items()}
    action = target_adapter.target_eef_pose_to_action(current, passthrough, env_id=env_id)
    return action[0] if action.dim() > 1 else action


async def _replay_worker(
    ctx: ReplayContext,
    env,
    env_id: int,
    provider: PlanProvider,
    action_queue: asyncio.Queue,
    results: list[tuple[str, bool]],
    stats: dict[str, int],
) -> None:
    """One env's worker: pull plans from ``provider`` and replay them.

    When the provider hands out nothing: if it is fully drained (nothing in flight on any env) the worker
    *returns* so ``env_loop`` terminates via its task-completion check (this covers an unreachable success
    target on a finite source); otherwise it *holds* at the current pose so the batched step keeps
    advancing the still-running workers.
    """
    target_adapter = ctx.target_adapter
    env_ids = torch.tensor([env_id], device=env.device)
    hold_passthrough = None  # last episode's passthrough, reused to hold the gripper at a valid command
    while True:
        # ``next_for_env`` is env-agnostic for the copy provider (its plans do not depend on which env runs
        # them). Synchronous, so each worker claims a distinct source episode + index.
        plan = provider.next_for_env(env_id)
        if plan is None:
            if provider.in_flight() == 0:
                return
            if hold_passthrough is None:
                hold_passthrough = _zeros_passthrough(env, target_adapter)
            await action_queue.put((env_id, _hold_action(env, env_id, target_adapter, hold_passthrough)))
            await action_queue.join()
            continue
        try:
            outcome = await _replay_one_episode(ctx, env, env_id, plan, action_queue)
            success, hold_passthrough = outcome.task_succeeded, outcome.passthrough_action_dict
        except Exception:
            sys.stderr.write(f"[parallel-replay] env {env_id} failed on {plan.name}:\n{traceback.format_exc()}")
            sys.stderr.flush()
            success = False
        env.recorder_manager.set_success_to_episodes(
            env_ids, torch.tensor([[success]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes(env_ids)
        provider.observe(plan, ReplayResult(success=bool(success)))
        results.append((plan.name, bool(success)))
        stats["num_attempts"] += 1
        stats["num_success"] += int(success)


def run_parallel_replay(
    ctx: ReplayContext,
    env,
    provider: PlanProvider,
    num_envs: int,
    generation_policy,
) -> list[tuple[str, bool]]:
    """Replay plans from ``provider`` across ``num_envs`` parallel workers; return ``[(name, success), ...]``.

    Reuses generation's ``env_loop``: it batches one action per env each step and terminates on the
    provider's stop condition -- ``provider.loop_policy()`` gives ``(guarantee_success, num_trials)`` so
    the loop exits on enough successes or attempts, and the workers exit once the provider is drained (so
    an unreachable target on a finite source still terminates). Workers reset their env with
    ``env.reset_to`` directly (not the reset queue) since each needs its plan's scene.
    """
    event_loop = asyncio.get_event_loop()
    action_queue: asyncio.Queue = asyncio.Queue()
    reset_queue: asyncio.Queue = asyncio.Queue()  # unused here; env_loop expects the handle
    results: list[tuple[str, bool]] = []
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}

    tasks = [
        event_loop.create_task(_replay_worker(ctx, env, env_id, provider, action_queue, results, stats))
        for env_id in range(num_envs)
    ]
    data_gen_tasks = asyncio.ensure_future(asyncio.gather(*tasks))

    # Terminate the step loop on the provider's stop condition (successes or attempts).
    guarantee_success, num_trials = provider.loop_policy()
    loop_policy = deepcopy(generation_policy)
    loop_policy.num_trials = num_trials
    loop_policy.guarantee_success = guarantee_success
    try:
        env_loop(
            env,
            reset_queue,
            action_queue,
            event_loop,
            generation_policy_params=loop_policy,
            stats=stats,
            data_gen_tasks=data_gen_tasks,
        )
    finally:
        # Cancelling the workers makes the gathered future raise CancelledError (a BaseException, so it
        # is not caught by ``suppress(Exception)``) — list it explicitly so results are returned cleanly.
        data_gen_tasks.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            event_loop.run_until_complete(data_gen_tasks)
    return results
