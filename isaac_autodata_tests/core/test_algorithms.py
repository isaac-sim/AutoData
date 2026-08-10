# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.algorithms`."""

import pytest

from isaac_autodata_core.algorithms import (
    REGISTERED_ALGORITHMS,
    DexMimicGen,
    GenerationAlgorithm,
    MimicGen,
    SkillGen,
    get_algorithm,
    iter_algorithms,
)


def test_registry_contents():
    assert REGISTERED_ALGORITHMS == {
        "mimicgen": MimicGen,
        "dexmimicgen": DexMimicGen,
        "skillgen": SkillGen,
    }


def test_base_class_not_registered():
    # GenerationAlgorithm has an empty name, so the metaclass must not register it.
    assert GenerationAlgorithm not in REGISTERED_ALGORITHMS.values()
    assert "" not in REGISTERED_ALGORITHMS


def test_iter_algorithms():
    assert set(iter_algorithms()) == {MimicGen, DexMimicGen, SkillGen}


@pytest.mark.parametrize("name, cls", [("mimicgen", MimicGen), ("dexmimicgen", DexMimicGen)])
def test_get_algorithm_no_kwargs(name, cls):
    assert isinstance(get_algorithm(name), cls)


def test_get_algorithm_unknown_error():
    with pytest.raises(KeyError, match="Unknown algorithm"):
        get_algorithm("nope")


def test_mimicgen_attributes():
    algo = MimicGen()
    assert algo.name == "mimicgen"
    assert algo.expected_eef_count == 1
    assert algo.requires_motion_planner is False
    assert algo.uses_subtask_start_signals is False
    assert algo.supports_coordination is False


def test_dexmimicgen_attributes():
    algo = DexMimicGen()
    assert algo.name == "dexmimicgen"
    assert algo.expected_eef_count == 2
    assert algo.supports_coordination is True
    assert algo.requires_motion_planner is False


def test_skillgen_attributes():
    algo = SkillGen(motion_planners={0: object()})
    assert algo.name == "skillgen"
    assert algo.expected_eef_count == 1
    assert algo.requires_motion_planner is True
    assert algo.uses_subtask_start_signals is True
    assert algo.supports_coordination is False


def test_skillgen_requires_at_least_one_planner():
    with pytest.raises(AssertionError, match="requires at least one motion planner"):
        SkillGen(motion_planners={})


def test_get_algorithm_skillgen_needs_planners():
    with pytest.raises(TypeError):
        get_algorithm("skillgen")  # motion_planners is required
    assert isinstance(
        get_algorithm("skillgen", motion_planners={0: object()}), SkillGen
    )  # base implementation must not raise
