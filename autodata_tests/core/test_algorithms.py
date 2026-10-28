# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`autodata_core.algorithms`."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from autodata_core.algorithms import (
    REGISTERED_ALGORITHMS,
    DexMimicGen,
    GenerationAlgorithm,
    MimicGen,
    SkillGen,
    get_algorithm,
    iter_algorithms,
)
from autodata_core.data_generator import _EEFGenerationState


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
    assert algo.expected_eef_count == (2, 3)
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


def test_skillgen_reports_a_failed_motion_plan():
    planner = Mock(update_world_and_plan_motion=Mock(return_value=False))
    algorithm = SkillGen(motion_planners={0: planner})
    data_generator = Mock()
    data_generator.generate_eef_subtask_trajectory.return_value = [SimpleNamespace(pose=None, passthrough_action=None)]
    eef_state = _EEFGenerationState()

    result = algorithm.plan_subtask_trajectory(
        data_generator=data_generator,
        env_id=0,
        eef_name="franka",
        eef_state=eef_state,
        all_randomized_subtask_boundaries={},
        runtime_subtask_constraints_dict={},
        selected_src_demo_inds={},
    )

    assert result is None and algorithm.attempt_extras([eef_state])["planning_failed"] is True
