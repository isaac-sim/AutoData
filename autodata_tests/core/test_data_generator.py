# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`autodata_core.data_generator.DataGenerator`.

Construction-time validation and navigation playback decisions are covered here.
The async generation loop is exercised by the e2e suite.
"""

import dataclasses
import torch

import pytest

from autodata_core.algorithms import DexMimicGen, MimicGen, SkillGen
from autodata_core.data_generator import DataGenerator, GenerationResult, _EEFGenerationState
from autodata_core.waypoint import Waypoint
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
        self.source_pool = object()
        self._subtasks = subtasks_by_eef
        self._constraints = constraints or []
        self.navigation_state = None

    def get_eef_names(self):
        return list(self._subtasks)

    def get_subtasks(self, eef_name):
        return self._subtasks[eef_name]

    def get_task_constraints(self):
        return self._constraints

    def get_navigation_state(self, env_id):
        return self.navigation_state


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
