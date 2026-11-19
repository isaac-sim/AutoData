# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`autodata_core.data_generator.DataGenerator`.

Construction-time validation and navigation playback decisions are covered here.
Simulation integration is covered by the e2e suite.
"""

import asyncio
import dataclasses
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from autodata_core.algorithms import DexMimicGen, MimicGen, SkillGen
from autodata_core.data_generator import DataGenerator, GenerationResult, _EEFGenerationState
from autodata_core.datagen_info import DatagenInfo
from autodata_core.waypoint import MultiWaypoint, Waypoint
from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from autodata_interfaces.tasks.subtask_spec import MimicGenSubtaskAlgoParams, Subtask


def test_generation_result_preserves_positional_extras():
    extras = {"example": 1}
    result = GenerationResult({}, True, extras)
    assert result.extras is extras
    assert result.diagnostics == ()


def _subtask(term_signal: str = "s", term_offset: tuple[int, int] = (0, 0)) -> Subtask:
    return Subtask(
        subtask_term_signal=term_signal,
        subtask_term_offset_range=term_offset,
        algo_params=MimicGenSubtaskAlgoParams(),
    )


class _MockDatastream:
    """Minimal datastream for construction and navigation playback tests."""

    def __init__(self, subtasks_by_eef: dict, constraints: list | None = None) -> None:
        self.source_pool = SimpleNamespace(
            asyncio_lock=None,
            datagen_infos=[object()],
            subtask_boundaries={
                name: [[(2 * i, 2 * i + 2) for i in range(len(subtasks))]] for name, subtasks in subtasks_by_eef.items()
            },
        )
        self._subtasks = subtasks_by_eef
        self._constraints = constraints or []
        self.navigation_state = None
        self.generation_policy = GenerationPolicy()
        self.device = "cpu"
        self.env = None

    def get_eef_names(self):
        return list(self._subtasks)

    def get_subtasks(self, eef_name):
        return self._subtasks[eef_name]

    def get_task_constraints(self):
        return self._constraints

    def get_navigation_state(self, env_id):
        return self.navigation_state

    def get_generation_policy(self):
        return self.generation_policy

    def get_env(self):
        return self.env


def _assert_constructed(generator: DataGenerator, datastream: _MockDatastream, algorithm) -> None:
    assert generator.algorithm is algorithm
    assert generator.datastream is datastream
    assert generator.src_demo_datagen_info_pool is datastream.source_pool


def test_valid_mimicgen_construction():
    datastream = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]})
    algorithm = MimicGen()
    generator = DataGenerator(datastream=datastream, algorithm=algorithm)
    _assert_constructed(generator, datastream, algorithm)


def test_valid_dexmimicgen_construction():
    datastream = _MockDatastream({"left": [_subtask("l")], "right": [_subtask("r")]})
    algorithm = DexMimicGen()
    generator = DataGenerator(datastream=datastream, algorithm=algorithm)
    _assert_constructed(generator, datastream, algorithm)


def test_valid_skillgen_construction():
    datastream = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]})
    algorithm = SkillGen(motion_planners={0: object()})
    generator = DataGenerator(datastream=datastream, algorithm=algorithm)
    _assert_constructed(generator, datastream, algorithm)
    assert generator.algorithm.requires_motion_planner is True


def test_eef_count_mismatch_error():
    datastream = _MockDatastream({"left": [_subtask("l")], "right": [_subtask("r")]})  # 2 EEFs
    with pytest.raises(ValueError, match="expects 1 EEF"):
        DataGenerator(datastream=datastream, algorithm=MimicGen())


def test_coordination_unsupported_error():
    datastream = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]}, constraints=[object()])
    with pytest.raises(ValueError, match="does not support coordination"):
        DataGenerator(datastream=datastream, algorithm=MimicGen())


def test_terminal_subtask_offset_must_be_zero():
    datastream = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("", term_offset=(0, 2))]})
    with pytest.raises(AssertionError):
        DataGenerator(datastream=datastream, algorithm=MimicGen())


def test_dexmimicgen_with_two_eefs_and_constraints_ok():
    # DexMimicGen supports coordination, so a datastream carrying constraints still constructs.
    datastream = _MockDatastream({"left": [_subtask("l")], "right": [_subtask("r")]}, constraints=[object()])
    algorithm = DexMimicGen()
    generator = DataGenerator(datastream=datastream, algorithm=algorithm)
    _assert_constructed(generator, datastream, algorithm)


def test_skillgen_requires_motion_planners():
    # SkillGen's only non-trivial construction invariant: at least one planner must be supplied.
    with pytest.raises(AssertionError, match="at least one motion planner"):
        SkillGen(motion_planners={})


def test_skillgen_coordination_unsupported_error():
    datastream = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]}, constraints=[object()])
    with pytest.raises(ValueError, match="does not support coordination"):
        DataGenerator(datastream=datastream, algorithm=SkillGen(motion_planners={0: object()}))


def test_generation_result_fields_and_frozen():
    result = GenerationResult(initial_state={"foo": 1}, success=True)
    assert result.initial_state == {"foo": 1}
    assert result.success is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False  # type: ignore[misc]


def _navigation_generator(active: bool, reached: bool) -> DataGenerator:
    datastream = _MockDatastream({name: [_subtask("")] for name in ("left", "right", "base")})
    datastream.navigation_state = ("base", active, reached)
    datastream.generation_policy.use_navigation_controller = True
    return DataGenerator(datastream=datastream, algorithm=DexMimicGen())


def _playback_state(length: int, index: int, *, paused: bool = False, done: bool = False) -> _EEFGenerationState:
    waypoint = Waypoint(pose=torch.eye(4), passthrough_action={})
    return _EEFGenerationState(
        current_trajectory=[waypoint] * length,
        subtask_step_index=index,
        is_currently_paused=paused,
        subtasks_done=done,
    )


def test_navigation_holds_playback_until_goal_reached():
    states = {"base": _playback_state(10, 8), "left": _playback_state(20, 8), "right": _playback_state(20, 8)}
    generator = _navigation_generator(active=True, reached=False)
    assert generator._synchronize_navigation(states, 0, True) == (True, True)
    assert all(state.subtask_step_index == 8 for state in states.values())

    generator.datastream.navigation_state = ("base", False, True)
    assert generator._synchronize_navigation(states, 0, True) == (False, False)


@pytest.mark.parametrize(
    "length, paused, done",
    [(20, True, False), (20, False, True), (5, False, False)],
    ids=["paused", "completed", "last_waypoint"],
)
def test_navigation_completion_skips_idle_steps_without_rewinding_other_channels(length, paused, done):
    states = {
        "base": _playback_state(10, 4),
        "left": _playback_state(20, 4),
        "right": _playback_state(length, 4, paused=paused, done=done),
    }
    generator = _navigation_generator(active=False, reached=True)
    assert generator._synchronize_navigation(states, 0, True) == (False, False)
    assert {name: state.subtask_step_index for name, state in states.items()} == {"base": 8, "left": 8, "right": 4}


@pytest.mark.parametrize("paused, reached", [(False, False), (True, True)], ids=["cancelled", "paused"])
def test_navigation_does_not_skip_playback_after_cancellation_or_while_paused(paused, reached):
    states = {
        "base": _playback_state(10, 4, paused=paused),
        "left": _playback_state(20, 4),
        "right": _playback_state(20, 4),
    }
    generator = _navigation_generator(active=False, reached=reached)
    assert generator._synchronize_navigation(states, 0, True) == (False, False)
    assert all(state.subtask_step_index == 4 for state in states.values())


def test_unresponsive_navigation_fails_and_finalizes_the_attempt(monkeypatch):
    generator = _navigation_generator(active=True, reached=False)
    generator.datastream.generation_policy.max_navigation_hold_steps = 3
    recorder = Mock()
    recorder.get_episode.return_value = SimpleNamespace(data={"actions": torch.zeros(1, 1)})
    generator.datastream.env = SimpleNamespace(recorder_manager=recorder)
    waypoint = Waypoint(pose=torch.eye(4), passthrough_action={})
    monkeypatch.setattr(generator.algorithm, "plan_subtask_trajectory", lambda **kwargs: ([waypoint, waypoint], False))
    monkeypatch.setattr(generator, "_reset_environment_for_generation", AsyncMock(return_value=(torch.tensor([0]), {})))
    execute = AsyncMock(return_value=False)
    monkeypatch.setattr(MultiWaypoint, "execute", execute)

    result = asyncio.run(
        asyncio.wait_for(
            generator.generate(
                env_id=0,
                success_term=None,
                env_reset_queue=asyncio.Queue(),
                env_action_queue=asyncio.Queue(),
            ),
            timeout=1.0,
        )
    )

    assert result.success is False
    assert execute.await_count == 3
    assert recorder.set_success_to_episodes.call_args.args[1].item() is False
    recorder.export_episodes.assert_called_once()


@pytest.mark.parametrize("object_poses", [None, {"other": torch.eye(4).unsqueeze(0)}])
def test_rigid_reference_requires_source_object_pose_at_setup(object_poses):
    subtask = _subtask("")
    subtask.object_ref = "cube"
    datastream = _MockDatastream({"franka": [subtask]})
    datastream.source_pool.datagen_infos = [DatagenInfo(object_poses=object_poses)]

    with pytest.raises(AssertionError, match="Source demo 0 lacks object_pose for 'cube'"):
        DataGenerator(datastream, MimicGen())


def test_source_selection_preserves_positional_arguments():
    subtask = _subtask("")
    subtask.object_ref = "cube"
    datastream = _MockDatastream({"franka": [subtask]})
    poses = torch.eye(4).repeat(2, 1, 1)
    datastream.source_pool.datagen_infos = [
        DatagenInfo(
            eef_pose={"franka": poses},
            object_poses={"cube": poses},
            target_eef_pose={"franka": poses},
            passthrough_action={"franka": torch.zeros(2, 1)},
        )
    ]
    generator = DataGenerator(datastream, MimicGen())

    selected = generator.select_source_demo(
        "franka", torch.eye(4), torch.eye(4), np.array([[0, 2]]), "cube", "nearest_neighbor_object", {"nn_k": 1}
    )

    assert selected == 0


@pytest.mark.parametrize("recorded_steps", [False, True])
def test_finalization_does_not_export_a_reset_snapshot_without_steps(recorded_steps):
    recorder = Mock()
    data = {"initial_state": {}}
    if recorded_steps:
        data["obs"] = torch.zeros(1, 2)
    recorder.get_episode.return_value = SimpleNamespace(data=data)
    generator = object.__new__(DataGenerator)
    generator.datastream = SimpleNamespace(device="cpu", get_env=lambda: SimpleNamespace(recorder_manager=recorder))
    generator._finalize_episode_recording(env_id_tensor=torch.tensor([0]), success=False, export_demo=True)
    recorder.set_success_to_episodes.assert_called_once()
    assert recorder.export_episodes.call_count == int(recorded_steps)


@pytest.mark.parametrize(
    "num_envs,num_trials,guarantee_success,fail_first,step_count,expected_attempts,expected_success",
    [(5, 6, False, 0, 2, 6, 6), (5, 2, False, 0, 2, 2, 2), (3, 4, True, 2, 2, 6, 4), (3, 4, True, 100, 0, 6, 0)],
)
def test_generation_loop_reserves_exact_budget_and_holds_idle_relative_targets(
    num_envs, num_trials, guarantee_success, fail_first, step_count, expected_attempts, expected_success
):
    from autodata_interfaces.env.isaaclab_env_interface import env_loop

    policy = GenerationPolicy(num_trials=num_trials, guarantee_success=guarantee_success, max_num_failures=2)
    env = SimpleNamespace(
        num_envs=num_envs,
        cfg=SimpleNamespace(),
        device="cpu",
        action_space=SimpleNamespace(shape=(num_envs, 1)),
        action_manager=SimpleNamespace(action=torch.zeros(num_envs, 1)),
        sim=SimpleNamespace(is_stopped=lambda: False),
        positions=torch.zeros(num_envs),
    )

    def step(actions):
        env.positions += actions[:, 0]
        env.action_manager.action = actions.clone()

    def pose(env_ids, name):
        result = torch.eye(4).repeat(len(env_ids), 1, 1)
        result[:, 0, 3] = env.positions[env_ids]
        return result

    def encode(target_eef_pose_dict, env_id, **kwargs):
        return (target_eef_pose_dict["eef"][0, 3] - env.positions[env_id]).reshape(1)

    env.step = step
    generator = object.__new__(DataGenerator)
    generator.datastream = SimpleNamespace(
        get_env=lambda: env,
        get_generation_policy=lambda: policy,
        get_eef_names=lambda: ["eef"],
        get_robot_eef_pose=pose,
        get_passthrough_actions=lambda ids: None if num_trials < num_envs else {},
        actions_to_passthrough_actions=lambda actions: {},
        target_eef_pose_to_action=encode,
    )
    calls = []
    pending = set()
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}
    attempts = []

    async def generate(env_id, env_action_queue, **kwargs):
        index = len(calls)
        calls.append(env_id)
        assert len(pending) <= min(num_envs, num_trials)
        for _ in range(step_count):
            await env_action_queue.put((env_id, torch.ones(1)))
            await env_action_queue.join()
        return GenerationResult({}, success=index >= fail_first)

    generator.generate = generate
    success_term = SimpleNamespace(func=lambda env: torch.zeros(num_envs, dtype=torch.bool), params={})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    action_queue = asyncio.Queue()
    reset_queue = asyncio.Queue()
    tasks = asyncio.gather(*(
        generator.generate_loop(index, success_term, reset_queue, action_queue, False, stats, attempts, pending)
        for index in range(num_envs)
    ))
    guarded = loop.create_task(asyncio.wait_for(tasks, timeout=2.0))
    try:
        completed = env_loop(env, reset_queue, action_queue, loop, policy, stats, data_gen_tasks=guarded)
    finally:
        guarded.cancel()
        loop.run_until_complete(asyncio.gather(guarded, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)
    assert completed is (expected_success >= num_trials if guarantee_success else True)
    assert len(calls) == len(attempts) == stats["num_attempts"] == expected_attempts
    assert stats["num_success"] == expected_success
    assert stats["num_failures"] == expected_attempts - expected_success
    assert not pending
    assert env.positions.tolist() == [calls.count(index) * step_count for index in range(num_envs)]
