# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`autodata_core.algorithms`."""

import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from autodata_core.algorithms import (
    REGISTERED_ALGORITHMS,
    DexMimicGen,
    GenerationAlgorithm,
    MimicGen,
    SkillGen,
    SoftMimicGen,
    get_algorithm,
    iter_algorithms,
)
from autodata_core.data_generator import _EEFGenerationState
from autodata_core.datagen_info import DatagenInfo
from autodata_interfaces.tasks.subtask_spec import SoftMimicGenSubtaskAlgoParams, Subtask


def test_registry_contents():
    assert REGISTERED_ALGORITHMS == {
        "mimicgen": MimicGen,
        "dexmimicgen": DexMimicGen,
        "softmimicgen": SoftMimicGen,
        "skillgen": SkillGen,
    }


def test_base_class_not_registered():
    # GenerationAlgorithm has an empty name, so the metaclass must not register it.
    assert GenerationAlgorithm not in REGISTERED_ALGORITHMS.values()
    assert "" not in REGISTERED_ALGORITHMS


def test_iter_algorithms():
    assert set(iter_algorithms()) == {MimicGen, DexMimicGen, SoftMimicGen, SkillGen}


@pytest.mark.parametrize(
    "name, cls",
    [("mimicgen", MimicGen), ("dexmimicgen", DexMimicGen), ("softmimicgen", SoftMimicGen)],
)
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


def test_softmimicgen_attributes():
    algo = SoftMimicGen()
    assert algo.name == "softmimicgen"
    assert algo.expected_eef_count == (1, 2)
    assert algo.requires_motion_planner is False
    assert algo.uses_subtask_start_signals is False
    assert algo.supports_coordination is False


@pytest.mark.parametrize(
    "object_soft, strategy, compatible",
    [
        (False, "random", True),
        (False, "nearest_neighbor_object", True),
        (False, "nearest_neighbor_robot_distance", True),
        (False, "registration_cost", False),
        (True, "random", True),
        (True, "nearest_neighbor_object", False),
        (True, "nearest_neighbor_robot_distance", False),
        (True, "registration_cost", True),
    ],
)
def test_softmimicgen_validates_selection_strategy_at_setup(object_soft, strategy, compatible):
    subtask = Subtask(
        object_ref="object",
        selection_strategy=strategy,
        algo_params=SoftMimicGenSubtaskAlgoParams(object_soft=object_soft),
    )
    nodes = torch.zeros(1, 6, 3)
    datastream = SimpleNamespace(
        get_eef_names=lambda: ["franka"],
        get_subtasks=lambda _eef: [subtask],
        get_object_nodal_positions=lambda: {"object": nodes},
        source_pool=SimpleNamespace(datagen_infos=[DatagenInfo(object_nodal_positions={"object": nodes})]),
    )
    algorithm = SoftMimicGen()
    if compatible:
        algorithm.validate_setup(datastream)
    else:
        object_type = "deformable" if object_soft else "rigid"
        with pytest.raises(AssertionError, match=f"{strategy!r} does not support {object_type} reference objects"):
            algorithm.validate_setup(datastream)


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
