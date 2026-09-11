# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
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
from dataclasses import dataclass
from typing import Any

from autodata_interfaces.env import env_loop
from autodata_utils.tensor_utils import as_torch

from .config import DefaultObjectTracking
from .provider import PlanProvider, ReplayResult
from .replay import (
    _apply_object_centric_override,
    _build_controlled_link_pose_reader,
    _monitored_miss,
    _record_datagen_poses,
    _record_signal_frame,
    prepare_episode,
    read_achieved_eef_poses,
)
from .util import pose_tracking_error


@dataclass
class ReplayParams:
    """Bundle of per-run retargeting knobs shared by every parallel worker (see ``RetargetConfig``)."""

    robot_asset_name: str
    reference_pose: str
    eef_name_map: dict[str, str] | None
    replay_speed: float
    retarget_frame: str
    scene_translation: tuple[float, float, float]
    eef_offsets: dict[str, torch.Tensor] | None
    num_interpolation_steps: int
    init_robot_from_ik: bool
    subtasks: dict
    default_object_tracking: DefaultObjectTracking
    write_datagen_info: bool
    eef_reference_link: dict[str, str] | str | None
    source_hand_postures: dict[str, dict[str, list[float]]] | None
    success_settle_steps: int
    segment_settle_steps: int
    settle_pos_tol_m: float
    settle_rot_tol_deg: float
    settle_joint_tol: float
    config: Any = None  # RetargetConfig (early-abort thresholds, synchronization, speed caps)


async def _async_step(
    env, env_id, action_queue, target_adapter, target_eef_pose_dict, passthrough_action_dict
) -> None:
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
    env, env_id, plan, action_queue, target_default_state, adapters, params, success_term
) -> bool:
    """Reset ``env_id`` to the source scene and drive one plan's trajectory async; return whether it
    succeeded."""
    target_adapter, source_adapter, remap_passthrough = adapters
    prep = prepare_episode(
        env,
        env_id,
        plan.episode,
        target_adapter,
        source_adapter,
        remap_passthrough,
        target_default_state,
        params.robot_asset_name,
        params.reference_pose,
        params.eef_name_map,
        params.replay_speed,
        params.retarget_frame,
        params.scene_translation,
        params.eef_offsets,
        params.num_interpolation_steps,
        init_robot_from_ik=params.init_robot_from_ik,
        subtasks=params.subtasks,
        default_object_tracking=params.default_object_tracking,
        source_hand_postures=params.source_hand_postures,
        need_source_objects=any(
            st.object_tracking is not None for entries in params.subtasks.values() for st in entries
        ),
        reset_sim=False,
        eef_reference_link=params.eef_reference_link,
        write_datagen_info=params.write_datagen_info,
        segment_settle_steps=params.segment_settle_steps,
        max_eef_linear_velocity=params.config.max_eef_linear_velocity,
        max_eef_rotation_speed=params.config.max_eef_rotation_speed,
        synchronization=params.config.synchronization,
    )
    eef_names = prep["eef_names"]
    commanded_poses, commanded_passthrough = prep["commanded_poses"], prep["commanded_passthrough"]
    num_interpolation_steps, num_steps = prep["num_interpolation_steps"], prep["num_steps"]
    carry_segments, source_objects = prep["carry_segments"], prep["source_objects"]
    source_signals = prep["source_signals"]  # {name: (num_traj_steps, 1)} to forward, or {}
    segment_ends = prep["segment_ends"]  # {eef: {trajectory step: settle-hold cap}} for the motion-aware settle
    sync_of = prep.get("sync_of", {})  # {eef: {end step: group id}} -- a cross-EEF rendezvous barrier
    group_members = prep.get("group_members", {})  # {group id: [(eef, end step)]}
    # Measure the object-centric grasp transform (and any debug tracking) at the same IK-controlled link
    # the command drives when the reference is reconstructed there, else the observed EEF frame.
    controlled_reader = (
        _build_controlled_link_pose_reader(env, target_adapter) if params.eef_reference_link is not None else None
    )

    # Per-EEF single-step scheduler (mirrors replay_episode_on_target, one ``_async_step`` per tick for THIS
    # env): each EEF advances its own pointer through its (possibly unequal-length) commanded sequence; at a
    # subtask boundary it HOLDS until its motion-aware settle AND its sync group's rendezvous are satisfied.
    # This is per-env local state, so the barrier joins the two arms of one env -- bimanual + hand-off run in
    # parallel. Pose / passthrough / override are (re)computed only for EEFs that just advanced (a holding
    # carrier keeps its pose so the override does not remeasure the grasp mid-release).
    task_succeeded = success_term is None
    lengths = {eef_name: commanded_poses[eef_name].shape[0] for eef_name in eef_names}
    ptr = {eef_name: 0 for eef_name in eef_names}
    hold = {eef_name: 0 for eef_name in eef_names}
    prev_pose: dict[str, torch.Tensor | None] = {eef_name: None for eef_name in eef_names}
    prev_ptr = {eef_name: -1 for eef_name in eef_names}
    prev_qpos = None
    channel_eef = {name: name if name in eef_names else eef_names[0] for name in commanded_passthrough}
    target_eef_pose_dict: dict[str, torch.Tensor] = {}
    passthrough_action_dict: dict[str, torch.Tensor] = {}
    record_signals = None
    # Full datagen_info (poses) per step when write_datagen_info (copy) -> the output is generate-ready.
    object_names = list(env.scene.rigid_objects.keys()) if params.write_datagen_info else []
    tick = 0
    early_failure = False
    monitor_early = params.config.stop_early_on_failure and (
        params.config.max_translation_error is not None or params.config.max_rotation_error is not None
    )
    while any(ptr[eef_name] < lengths[eef_name] for eef_name in eef_names):
        idx = {eef_name: min(ptr[eef_name], lengths[eef_name] - 1) for eef_name in eef_names}
        traj_step = {eef_name: idx[eef_name] - num_interpolation_steps for eef_name in eef_names}
        advanced = {eef_name for eef_name in eef_names if ptr[eef_name] != prev_ptr[eef_name]}
        for eef_name in advanced:
            target_eef_pose_dict[eef_name] = commanded_poses[eef_name][idx[eef_name]]
            prev_ptr[eef_name] = ptr[eef_name]
        for name, tensor in commanded_passthrough.items():
            if channel_eef[name] in advanced or name not in passthrough_action_dict:
                passthrough_action_dict[name] = tensor[idx[channel_eef[name]]]
        if carry_segments and advanced:
            _apply_object_centric_override(
                carry_segments,
                traj_step,
                target_eef_pose_dict,
                source_objects,
                target_adapter,
                env,
                env_id,
                controlled_reader=controlled_reader,
                commanded_poses=commanded_poses,
                num_interpolation_steps=num_interpolation_steps,
                eefs=advanced,
            )
        ref_ts = max(traj_step.values())  # shared clock for signals only
        signal_frame = {name: sig[min(max(ref_ts, 0), sig.shape[0] - 1)] for name, sig in source_signals.items()}
        record_signals = (lambda sf=signal_frame: _record_signal_frame(env, env_id, sf)) if signal_frame else None

        await _async_step(
            env,
            env_id,
            action_queue,
            target_adapter,
            target_eef_pose_dict,
            passthrough_action_dict,
        )
        if record_signals is not None:
            record_signals()
        if params.write_datagen_info:
            _record_datagen_poses(env, env_id, target_adapter, target_eef_pose_dict, object_names)
        if success_term is not None and bool(success_term.func(env, **success_term.params)[env_id]):
            task_succeeded = True

        if monitor_early:
            achieved_poses = read_achieved_eef_poses(target_adapter, controlled_reader, env_id)
            if _monitored_miss(
                eef_names,
                traj_step,
                ptr,
                lengths,
                carry_segments,
                source_objects,
                target_eef_pose_dict,
                achieved_poses,
                env,
                env_id,
                params.config.max_translation_error,
                params.config.max_rotation_error,
                tick,
            ):
                early_failure = True
                break

        curr_poses = target_adapter.get_eef_poses(env_ids=[env_id])
        curr_qpos = (
            as_torch(env.scene[params.robot_asset_name].data.joint_pos)[env_id] if params.robot_asset_name else None
        )
        joint_moved = (
            prev_qpos is not None
            and curr_qpos is not None
            and float(torch.max(torch.abs(curr_qpos - prev_qpos))) > params.settle_joint_tol
        )
        for eef_name in eef_names:
            if ptr[eef_name] >= lengths[eef_name]:
                continue
            cap = segment_ends.get(eef_name, {}).get(traj_step[eef_name], 0)
            gid = sync_of.get(eef_name, {}).get(traj_step[eef_name])
            if cap > 0 or gid is not None:
                hold[eef_name] += 1
                settle_ok = True
                if cap > 0:
                    moved = joint_moved
                    if prev_pose[eef_name] is not None:
                        dpos, drot = pose_tracking_error(prev_pose[eef_name], curr_poses[eef_name][0])
                        moved = moved or dpos > params.settle_pos_tol_m or drot > params.settle_rot_tol_deg
                    settle_ok = (not moved) or hold[eef_name] >= cap
                sync_ok = gid is None or all(
                    ptr[m_eef] - num_interpolation_steps >= m_step for m_eef, m_step in group_members[gid]
                )
                if settle_ok and sync_ok:
                    ptr[eef_name] += 1
                    hold[eef_name] = 0
            else:
                ptr[eef_name] += 1
            prev_pose[eef_name] = curr_poses[eef_name][0]
        prev_qpos = curr_qpos.clone() if curr_qpos is not None else None
        tick += 1

    # Final success settle: hold the last pose (final gripper release) until success or the cap.
    if (
        not task_succeeded
        and not early_failure
        and params.success_settle_steps > 0
        and num_steps > 0
        and success_term is not None
    ):
        for _ in range(params.success_settle_steps):
            await _async_step(env, env_id, action_queue, target_adapter, target_eef_pose_dict, passthrough_action_dict)
            if record_signals is not None:  # hold the last waypoint's signal across the success settle
                record_signals()
            if params.write_datagen_info:
                _record_datagen_poses(env, env_id, target_adapter, target_eef_pose_dict, object_names)
            if bool(success_term.func(env, **success_term.params)[env_id]):
                task_succeeded = True
                break
    return task_succeeded, passthrough_action_dict


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
    env,
    env_id,
    provider: PlanProvider,
    action_queue,
    target_default_state,
    adapters,
    params,
    success_term,
    results,
    stats,
) -> None:
    """One env's worker: pull plans from ``provider`` and replay them.

    When the provider hands out nothing: if it is fully drained (nothing in flight on any env) the worker
    *returns* so ``env_loop`` terminates via its task-completion check (this covers an unreachable success
    target on a finite source); otherwise it *holds* at the current pose so the batched step keeps
    advancing the still-running workers.
    """
    target_adapter = adapters[0]
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
            success, hold_passthrough = await _replay_one_episode(
                env, env_id, plan, action_queue, target_default_state, adapters, params, success_term
            )
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
    env,
    provider: PlanProvider,
    num_envs: int,
    adapters: tuple,
    target_default_state: dict,
    success_term,
    params: ReplayParams,
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
        event_loop.create_task(
            _replay_worker(
                env,
                env_id,
                provider,
                action_queue,
                target_default_state,
                adapters,
                params,
                success_term,
                results,
                stats,
            )
        )
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
