# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`autodata_core.data_generator.DataGenerator`.

Construction-time validation and navigation playback decisions are covered here.
Simulation integration is covered by the e2e suite.
"""

import asyncio
import dataclasses
import torch
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from autodata_core.algorithms import DexMimicGen, MimicGen, SkillGen
from autodata_core.data_generator import DataGenerator, GenerationResult, _EEFGenerationState
from autodata_core.waypoint import MultiWaypoint, Waypoint
from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from autodata_interfaces.tasks.subtask_spec import MimicGenSubtaskAlgoParams, Subtask


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


@pytest.mark.parametrize("command", ["pose", "gripper"])
def test_navigation_completion_preserves_manipulation_commands(command):
    states = {name: _playback_state(10, 4) for name in ("base", "left", "right")}
    endpoint = Waypoint(pose=torch.eye(4), passthrough_action={"base": torch.zeros(3)})
    states["base"].current_trajectory[-1] = endpoint
    pose = torch.eye(4)
    gripper = torch.zeros(1)
    if command == "pose":
        pose[0, 3] = 0.25
    else:
        gripper[0] = 1.0
    next_arm_waypoint = Waypoint(pose=pose, passthrough_action={"left": gripper})
    states["left"].current_trajectory[5] = next_arm_waypoint

    generator = _navigation_generator(active=False, reached=True)
    assert generator._synchronize_navigation(states, 0, True) == (False, False)
    assert all(state.subtask_step_index == 4 for state in states.values())
    assert all(waypoint is endpoint for waypoint in states["base"].current_trajectory[5:])

    generator._advance_subtask_progress(states, {}, pause_subtask=False)
    assert states["left"].subtask_step_index == 5
    assert states["left"].current_trajectory[5] is next_arm_waypoint


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
