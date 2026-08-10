# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reset settling in the synchronous data-generation environment loop."""

from __future__ import annotations

import asyncio
import contextlib
import torch
from types import SimpleNamespace

from isaac_autodata_interfaces.env.isaaclab_env_interface import env_loop
from isaac_autodata_interfaces.env.reset_request import EnvResetRequest
from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy


class _MockRecorderManager:
    def __init__(self) -> None:
        self.reset_env_ids: list[tuple[int, ...]] = []
        self.post_reset_env_ids: list[tuple[int, ...]] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        self.reset_env_ids.append(tuple(env_ids.tolist()))

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        self.post_reset_env_ids.append(tuple(env_ids.tolist()))


class _MockSimulation:
    def is_stopped(self) -> bool:
        return False


class _MockEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.action_space = SimpleNamespace(shape=(num_envs, 1))
        self.recorder_manager = _MockRecorderManager()
        self.sim = _MockSimulation()
        self.reset_env_ids: list[tuple[int, ...]] = []
        self.actions: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        self.reset_env_ids.append(tuple(env_ids.tolist()))

    def step(self, actions: torch.Tensor) -> None:
        self.actions.append(actions.clone())


def _close_event_loop(loop: asyncio.AbstractEventLoop, tasks: asyncio.Future) -> None:
    if not tasks.done():
        tasks.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(tasks)
    loop.close()
    asyncio.set_event_loop(None)


def test_reset_without_settling_completes_without_environment_steps():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    env = _MockEnv(num_envs=1)
    reset_queue: asyncio.Queue = asyncio.Queue()
    action_queue: asyncio.Queue = asyncio.Queue()
    completion = loop.create_future()
    reset_queue.put_nowait(EnvResetRequest(env_id=0, completion=completion))

    async def wait_for_reset() -> None:
        await completion

    tasks = asyncio.gather(wait_for_reset())
    try:
        completed = env_loop(
            env=env,
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
            asyncio_event_loop=loop,
            generation_policy_params=GenerationPolicy(reset_settling_steps=0),
            stats={"num_success": 0, "num_failures": 0, "num_attempts": 0},
            data_gen_tasks=tasks,
        )
        assert completed is False
        assert env.reset_env_ids == [(0,)]
        assert env.actions == []
        assert env.recorder_manager.reset_env_ids == []
        assert env.recorder_manager.post_reset_env_ids == []
        loop.run_until_complete(asyncio.wait_for(reset_queue.join(), timeout=0.1))
    finally:
        _close_event_loop(loop, tasks)


def test_settling_env_uses_zero_actions_while_other_env_continues():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    env = _MockEnv(num_envs=2)
    reset_queue: asyncio.Queue = asyncio.Queue()
    action_queue: asyncio.Queue = asyncio.Queue()
    completion = loop.create_future()
    reset_queue.put_nowait(EnvResetRequest(env_id=0, completion=completion))

    async def provide_running_env_actions() -> None:
        for value in (3.0, 4.0):
            await action_queue.put((1, torch.tensor([value])))
            await action_queue.join()

    async def wait_for_reset() -> None:
        await completion

    tasks = asyncio.gather(provide_running_env_actions(), wait_for_reset())
    try:
        completed = env_loop(
            env=env,
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
            asyncio_event_loop=loop,
            generation_policy_params=GenerationPolicy(reset_settling_steps=2),
            stats={"num_success": 0, "num_failures": 0, "num_attempts": 0},
            data_gen_tasks=tasks,
        )
        assert completed is False
        assert env.reset_env_ids == [(0,)]
        assert len(env.actions) == 2
        assert env.actions[0][:, 0].tolist() == [0.0, 3.0]
        assert env.actions[1][:, 0].tolist() == [0.0, 4.0]
        assert env.recorder_manager.reset_env_ids == [(0,)]
        assert env.recorder_manager.post_reset_env_ids == [(0,)]
        loop.run_until_complete(asyncio.wait_for(reset_queue.join(), timeout=0.1))
    finally:
        _close_event_loop(loop, tasks)
