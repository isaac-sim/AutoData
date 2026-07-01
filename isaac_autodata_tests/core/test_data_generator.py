# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`isaac_autodata_core.data_generator.DataGenerator` construction guards.

Only the (synchronous) construction-time validation is covered here -- the async generation loop is
exercised by the e2e suite. A minimal mock datastream supplies just the read methods the
constructor calls (``source_pool``, ``get_eef_names``, ``get_subtasks``, ``get_task_constraints``).
"""

import dataclasses

import pytest

from isaac_autodata_core.algorithms import DexMimicGen, MimicGen
from isaac_autodata_core.data_generator import DataGenerator, GenerationResult
from isaac_autodata_interfaces.tasks.subtask_spec import MimicGenSubtaskAlgoParams, Subtask


def _subtask(term_signal: str = "s", term_offset: tuple[int, int] = (0, 0)) -> Subtask:
    return Subtask(
        subtask_term_signal=term_signal,
        subtask_term_offset_range=term_offset,
        algo_params=MimicGenSubtaskAlgoParams(),
    )


class _MockDatastream:
    """Minimal stand-in exposing only what ``DataGenerator.__init__`` reads."""

    def __init__(self, subtasks_by_eef: dict, constraints: list | None = None) -> None:
        self.source_pool = object()
        self._subtasks = subtasks_by_eef
        self._constraints = constraints or []

    def get_eef_names(self):
        return list(self._subtasks)

    def get_subtasks(self, eef_name):
        return self._subtasks[eef_name]

    def get_task_constraints(self):
        return self._constraints


def test_valid_mimicgen_construction():
    ds = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]})
    gen = DataGenerator(datastream=ds, algorithm=MimicGen())
    assert gen.algorithm.name == "mimicgen"
    assert gen.src_demo_datagen_info_pool is ds.source_pool


def test_eef_count_mismatch_raises():
    ds = _MockDatastream({"left": [_subtask("l")], "right": [_subtask("r")]})  # 2 EEFs
    with pytest.raises(ValueError, match="expects 1 EEF"):
        DataGenerator(datastream=ds, algorithm=MimicGen())


def test_coordination_unsupported_raises():
    ds = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("")]}, constraints=[object()])
    with pytest.raises(ValueError, match="does not support coordination"):
        DataGenerator(datastream=ds, algorithm=MimicGen())


def test_terminal_subtask_offset_must_be_zero():
    ds = _MockDatastream({"franka": [_subtask("grasp_1"), _subtask("", term_offset=(0, 2))]})
    with pytest.raises(AssertionError):
        DataGenerator(datastream=ds, algorithm=MimicGen())


def test_dexmimicgen_with_two_eefs_and_constraints_ok():
    ds = _MockDatastream({"left": [_subtask("l")], "right": [_subtask("r")]}, constraints=[object()])
    gen = DataGenerator(datastream=ds, algorithm=DexMimicGen())
    assert gen.algorithm.expected_eef_count == 2


def test_generation_result_fields_and_frozen():
    result = GenerationResult(initial_state={"foo": 1}, success=True)
    assert result.initial_state == {"foo": 1}
    assert result.success is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False  # type: ignore[misc]
