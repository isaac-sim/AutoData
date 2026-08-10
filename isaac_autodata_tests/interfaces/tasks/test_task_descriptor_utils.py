# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.tasks.task_descriptor_utils`.

The private ``_coerce_enum`` / ``_coerce_tuples`` helpers are exercised through the public
``build_subtask`` / ``build_constraint`` entry points rather than tested directly.
"""

import copy

import pytest

from isaac_autodata_interfaces.tasks.subtask_constraint_spec import (
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaac_autodata_interfaces.tasks.subtask_spec import MimicGenSubtaskAlgoParams, SkillGenSubtaskAlgoParams, Subtask
from isaac_autodata_interfaces.tasks.task_descriptor_utils import build_constraint, build_subtask, validate_task_dict


def _minimal_valid() -> dict:
    return {
        "name": "t",
        "algo": "mimicgen",
        "subtasks": {"arm": [{"object_ref": "cube", "subtask_term_signal": "grasp_1"}]},
    }


# ---------------------------------------------------------------------------------------------------
# validate_task_dict
# ---------------------------------------------------------------------------------------------------
def test_validate_minimal_valid_passes():
    validate_task_dict(_minimal_valid())  # must not raise


def test_validate_full_valid_passes():
    data = _minimal_valid()
    data["description"] = "a task"
    data["constraints"] = [{"constraint_type": "sequential", "eef_subtask_constraint_tuple": [["arm", 0], ["arm", 0]]}]
    data["generation_policy"] = {"num_trials": 3}
    validate_task_dict(data)


def test_validate_non_dict():
    with pytest.raises(AssertionError, match="Expected top-level dict"):
        validate_task_dict(["not", "a", "dict"])  # type: ignore[arg-type]


@pytest.mark.parametrize("missing_key", ["name", "algo", "subtasks"])
def test_validate_missing_required_key(missing_key):
    data = _minimal_valid()
    del data[missing_key]
    with pytest.raises(AssertionError, match="Missing required top-level keys"):
        validate_task_dict(data)


def test_validate_unknown_top_level_key():
    data = _minimal_valid()
    data["bogus"] = 1
    with pytest.raises(AssertionError, match="Unknown top-level keys"):
        validate_task_dict(data)


def test_validate_name_not_str():
    data = _minimal_valid()
    data["name"] = 5
    with pytest.raises(AssertionError, match="'name' must be a string"):
        validate_task_dict(data)


def test_validate_description_not_str():
    data = _minimal_valid()
    data["description"] = 5
    with pytest.raises(AssertionError, match="'description' must be a string"):
        validate_task_dict(data)


def test_validate_algo_not_str():
    data = _minimal_valid()
    data["algo"] = 5
    with pytest.raises(AssertionError, match="'algo' must be a string"):
        validate_task_dict(data)


def test_validate_unknown_algo():
    data = _minimal_valid()
    data["algo"] = "nope"
    with pytest.raises(AssertionError, match="Unknown algo"):
        validate_task_dict(data)


def test_validate_subtasks_not_dict():
    data = _minimal_valid()
    data["subtasks"] = []
    with pytest.raises(AssertionError, match="'subtasks' must be a dict"):
        validate_task_dict(data)


def test_validate_eef_subtasks_not_list():
    data = _minimal_valid()
    data["subtasks"] = {"arm": {"object_ref": "cube"}}
    with pytest.raises(AssertionError, match="must be a list"):
        validate_task_dict(data)


def test_validate_subtask_not_dict():
    data = _minimal_valid()
    data["subtasks"] = {"arm": [5]}
    with pytest.raises(AssertionError, match=r"subtasks\['arm'\]\[0\] must be a dict"):
        validate_task_dict(data)


def test_validate_subtask_unknown_key():
    data = _minimal_valid()
    data["subtasks"] = {"arm": [{"bogus_field": 1}]}
    with pytest.raises(AssertionError, match="has unknown keys"):
        validate_task_dict(data)


def test_validate_algo_params_not_dict():
    data = _minimal_valid()
    data["subtasks"] = {"arm": [{"algo_params": 5}]}
    with pytest.raises(AssertionError, match="algo_params must be a dict"):
        validate_task_dict(data)


def test_validate_algo_params_unknown_key_for_algo():
    # subtask_start_offset_range is a SkillGen param; invalid under mimicgen.
    data = _minimal_valid()
    data["subtasks"] = {"arm": [{"algo_params": {"subtask_start_offset_range": [0, 0]}}]}
    with pytest.raises(AssertionError, match="algo_params has unknown keys"):
        validate_task_dict(data)


def test_validate_algo_params_valid_for_skillgen():
    data = _minimal_valid()
    data["algo"] = "skillgen"
    data["subtasks"] = {"arm": [{"algo_params": {"subtask_start_offset_range": [0, 2]}}]}
    validate_task_dict(data)  # must not raise


def test_validate_constraints_not_list():
    data = _minimal_valid()
    data["constraints"] = {}
    with pytest.raises(AssertionError, match="'constraints' must be a list"):
        validate_task_dict(data)


def test_validate_constraint_not_dict():
    data = _minimal_valid()
    data["constraints"] = [5]
    with pytest.raises(AssertionError, match=r"constraints\[0\] must be a dict"):
        validate_task_dict(data)


def test_validate_constraint_missing_type():
    data = _minimal_valid()
    data["constraints"] = [{"sequential_min_time_diff": 1}]
    with pytest.raises(AssertionError, match="missing required key 'constraint_type'"):
        validate_task_dict(data)


def test_validate_constraint_unknown_key():
    data = _minimal_valid()
    data["constraints"] = [{"constraint_type": "sequential", "bogus": 1}]
    with pytest.raises(AssertionError, match=r"constraints\[0\] has unknown keys"):
        validate_task_dict(data)


def test_validate_generation_policy_not_dict():
    data = _minimal_valid()
    data["generation_policy"] = []
    with pytest.raises(AssertionError, match="'generation_policy' must be a dict"):
        validate_task_dict(data)


def test_validate_generation_policy_unknown_key():
    data = _minimal_valid()
    data["generation_policy"] = {"bogus": 1}
    with pytest.raises(AssertionError, match="'generation_policy' has unknown keys"):
        validate_task_dict(data)


def test_validate_does_not_mutate_input():
    data = _minimal_valid()
    snapshot = copy.deepcopy(data)
    validate_task_dict(data)
    assert data == snapshot


# ---------------------------------------------------------------------------------------------------
# build_subtask (covers list->tuple coercion of tuple-typed fields)
# ---------------------------------------------------------------------------------------------------
def test_build_subtask_basic_with_tuple_coercion():
    st = build_subtask(
        {"object_ref": "cube", "subtask_term_signal": "grasp_1", "subtask_term_offset_range": [1, 2]},
        MimicGenSubtaskAlgoParams,
    )
    assert isinstance(st, Subtask)
    assert st.object_ref == "cube"
    assert st.subtask_term_signal == "grasp_1"
    # YAML scalar sequences load as lists; tuple-typed fields are coerced to tuples.
    assert st.subtask_term_offset_range == (1, 2)
    assert isinstance(st.subtask_term_offset_range, tuple)
    assert isinstance(st.algo_params, MimicGenSubtaskAlgoParams)


def test_build_subtask_leaves_non_tuple_fields_untouched():
    st = build_subtask(
        {"object_ref": "cube", "selection_strategy_kwargs": {"nn_k": 3}},
        MimicGenSubtaskAlgoParams,
    )
    assert st.object_ref == "cube"
    assert st.selection_strategy_kwargs == {"nn_k": 3}


def test_build_subtask_skillgen_algo_params_tuple_coercion():
    st = build_subtask({"algo_params": {"subtask_start_offset_range": [2, 4]}}, SkillGenSubtaskAlgoParams)
    algo = st.algo_params
    assert isinstance(algo, SkillGenSubtaskAlgoParams)
    assert algo.subtask_start_offset_range == (2, 4)


def test_build_subtask_does_not_mutate_input():
    data = {"object_ref": "cube", "algo_params": {"subtask_start_offset_range": [1, 2]}}
    snapshot = copy.deepcopy(data)
    build_subtask(data, SkillGenSubtaskAlgoParams)
    assert data == snapshot


# ---------------------------------------------------------------------------------------------------
# build_constraint (covers enum coercion + nested tuple coercion)
# ---------------------------------------------------------------------------------------------------
def test_build_constraint_coerces_enums_and_tuples():
    c = build_constraint({
        "constraint_type": "coordination",
        "eef_subtask_constraint_tuple": [["left", 0], ["right", 1]],
        "coordination_scheme": "transform",
    })
    assert c.constraint_type == SubTaskConstraintType.COORDINATION
    assert c.coordination_scheme == SubTaskConstraintCoordinationScheme.TRANSFORM
    assert c.eef_subtask_constraint_tuple == (("left", 0), ("right", 1))
    assert isinstance(c.eef_subtask_constraint_tuple, tuple)
    assert all(isinstance(pair, tuple) for pair in c.eef_subtask_constraint_tuple)


@pytest.mark.parametrize("type_value", ["sequential", "SEQUENTIAL", "Sequential", 0])
def test_build_constraint_type_name_case_insensitive_and_int(type_value):
    c = build_constraint({"constraint_type": type_value})
    assert c.constraint_type == SubTaskConstraintType.SEQUENTIAL


def test_build_constraint_invalid_type_name_raises():
    with pytest.raises(AssertionError, match="not a valid SubTaskConstraintType"):
        build_constraint({"constraint_type": "bogus"})


def test_build_constraint_does_not_mutate_input():
    data = {
        "constraint_type": "sequential",
        "eef_subtask_constraint_tuple": [["left", 0], ["right", 1]],
    }
    snapshot = copy.deepcopy(data)
    build_constraint(data)
    assert data == snapshot
