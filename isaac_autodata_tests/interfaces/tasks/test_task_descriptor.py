# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.tasks.task_descriptor`."""

import os

import pytest

from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from isaac_autodata_interfaces.tasks.subtask_constraint_spec import SubtaskConstraint, SubTaskConstraintType
from isaac_autodata_interfaces.tasks.subtask_spec import (
    MimicGenSubtaskAlgoParams,
    SkillGenSubtaskAlgoParams,
    SoftMimicGenSubtaskAlgoParams,
)
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from isaac_autodata_tests.utils.constants import TestPaths


def _stack_task_dict() -> dict:
    """A four-subtask single-arm stack task (each subtask has a term signal, like SkillGen)."""
    return {
        "name": "stack",
        "description": "stack cubes",
        "algo": "mimicgen",
        "subtasks": {
            "franka": [
                {
                    "object_ref": "cube_2",
                    "description": "grasp red",
                    "subtask_term_signal": "grasp_1",
                    "subtask_term_offset_range": [0, 0],
                },
                {"object_ref": "cube_1", "description": "stack red", "subtask_term_signal": "stack_1"},
                {"object_ref": "cube_3", "description": "grasp green", "subtask_term_signal": "grasp_2"},
                {"object_ref": "cube_2", "description": "stack green", "subtask_term_signal": "stack_2"},
            ]
        },
        "generation_policy": {"name": "stack", "num_trials": 7},
    }


def _bimanual_task_dict() -> dict:
    """A two-arm task with a coordination constraint."""
    return {
        "name": "bimanual",
        "algo": "dexmimicgen",
        "subtasks": {
            "left": [{"object_ref": "obj", "subtask_term_signal": "l1"}, {"subtask_term_signal": ""}],
            "right": [{"object_ref": "obj", "subtask_term_signal": "r1"}, {"subtask_term_signal": ""}],
        },
        "constraints": [{
            "constraint_type": "coordination",
            "eef_subtask_constraint_tuple": [["left", 0], ["right", 0]],
            "coordination_scheme": "replay",
        }],
    }


# ---------------------------------------------------------------------------------------------------
# from_dict construction
# ---------------------------------------------------------------------------------------------------
def test_from_dict_basic_fields():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    assert td.name == "stack"
    assert td.description == "stack cubes"
    assert td.get_eef_names() == ["franka"]
    assert len(td.get_subtasks("franka")) == 4
    assert td.env is None


def test_from_dict_invokes_validation():
    with pytest.raises(AssertionError, match="Missing required top-level keys"):
        TaskDescriptor.from_dict({"name": "x"})


def test_from_dict_defaults_for_optional_fields():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    # constraints default to empty; generation_policy is built from the provided block.
    assert td.get_task_constraints() == []
    assert isinstance(td.get_generation_policy(), GenerationPolicy)
    assert td.get_generation_policy().num_trials == 7


def test_from_dict_description_defaults_empty():
    data = _stack_task_dict()
    del data["description"]
    td = TaskDescriptor.from_dict(data)
    assert td.description == ""


def test_franka_rope_softmimicgen_yaml():
    path = os.path.join(TestPaths.tasks_dir, "franka_rope_softmimicgen.yaml")
    task = TaskDescriptor.from_yaml(path)
    assert task.name == "franka_rope"
    assert task.get_eef_names() == ["robot0"]
    assert len(task.get_subtasks("robot0")) == 2
    first = task.get_subtasks("robot0")[0]
    assert first.object_ref == "object"
    assert first.selection_strategy == "registration_cost"
    assert first.subtask_term_offset_range == (10, 15)
    assert isinstance(first.algo_params, SoftMimicGenSubtaskAlgoParams)
    assert first.algo_params.object_soft is True


# ---------------------------------------------------------------------------------------------------
# per-EEF, subtask-ordered getters
# ---------------------------------------------------------------------------------------------------
def test_getters_return_subtask_ordered_lists():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    assert td.get_object_refs("franka") == ["cube_2", "cube_1", "cube_3", "cube_2"]
    assert td.get_term_signal_names("franka") == ["grasp_1", "stack_1", "grasp_2", "stack_2"]
    assert td.get_start_signal_names("franka") == ["", "", "", ""]
    assert td.get_subtask_descriptions("franka") == ["grasp red", "stack red", "grasp green", "stack green"]
    algo_params = td.get_subtask_algo_params("franka")
    assert len(algo_params) == 4
    assert all(isinstance(p, MimicGenSubtaskAlgoParams) for p in algo_params)


def test_subtasks_are_ordered():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    subtasks = td.get_subtasks("franka")
    assert subtasks[0].subtask_term_signal == "grasp_1"
    assert subtasks[3].object_ref == "cube_2"


def test_offset_range_coerced_to_tuple():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    first = td.get_subtasks("franka")[0]
    assert first.subtask_term_offset_range == (0, 0)
    assert isinstance(first.subtask_term_offset_range, tuple)


@pytest.mark.parametrize(
    "getter",
    [
        "get_subtasks",
        "get_object_refs",
        "get_start_signal_names",
        "get_term_signal_names",
        "get_subtask_descriptions",
        "get_subtask_algo_params",
    ],
)
def test_getters_reject_unknown_eef(getter):
    td = TaskDescriptor.from_dict(_stack_task_dict())
    with pytest.raises(AssertionError, match="Unknown eef name"):
        getattr(td, getter)("nope")


# ---------------------------------------------------------------------------------------------------
# get_expected_attached_object (grasp -> stack carry inference)
# ---------------------------------------------------------------------------------------------------
def test_expected_attached_object_stack_after_grasp():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    # subtask 1 (stack_1) follows subtask 0 (grasp_1, object_ref cube_2) -> carries cube_2.
    assert td.get_expected_attached_object("franka", 1) == "cube_2"
    # subtask 3 (stack_2) follows subtask 2 (grasp_2, object_ref cube_3) -> carries cube_3.
    assert td.get_expected_attached_object("franka", 3) == "cube_3"


def test_expected_attached_object_none_for_grasp_subtasks():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    assert td.get_expected_attached_object("franka", 0) is None  # grasp_1, not a stack
    assert td.get_expected_attached_object("franka", 2) is None  # grasp_2, not a stack


def test_expected_attached_object_out_of_range_and_unknown_eef():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    assert td.get_expected_attached_object("franka", 99) is None
    assert td.get_expected_attached_object("franka", -1) is None
    assert td.get_expected_attached_object("nope", 0) is None


def test_expected_attached_object_first_subtask_stack_is_none():
    # A stack at index 0 has no preceding grasp, so nothing is carried.
    data = {
        "name": "x",
        "algo": "mimicgen",
        "subtasks": {"arm": [{"object_ref": "c", "subtask_term_signal": "stack_1"}]},
    }
    td = TaskDescriptor.from_dict(data)
    assert td.get_expected_attached_object("arm", 0) is None


def test_expected_attached_object_grasp_without_object_ref_is_none():
    # stack preceded by a grasp that references no object -> None (prev.object_ref or None).
    data = {
        "name": "x",
        "algo": "mimicgen",
        "subtasks": {"arm": [{"subtask_term_signal": "grasp_1"}, {"subtask_term_signal": "stack_1"}]},
    }
    td = TaskDescriptor.from_dict(data)
    assert td.get_expected_attached_object("arm", 1) is None


def test_expected_attached_object_stack_not_after_grasp_is_none():
    # stack preceded by a non-grasp subtask -> None.
    data = {
        "name": "x",
        "algo": "mimicgen",
        "subtasks": {
            "arm": [{"object_ref": "c", "subtask_term_signal": "approach_1"}, {"subtask_term_signal": "stack_1"}]
        },
    }
    td = TaskDescriptor.from_dict(data)
    assert td.get_expected_attached_object("arm", 1) is None


# ---------------------------------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------------------------------
def test_constraints_parsed_and_exposed():
    td = TaskDescriptor.from_dict(_bimanual_task_dict())
    assert set(td.get_eef_names()) == {"left", "right"}
    constraints = td.get_task_constraints()
    assert len(constraints) == 1
    c = constraints[0]
    assert isinstance(c, SubtaskConstraint)
    assert c.constraint_type == SubTaskConstraintType.COORDINATION
    assert c.eef_subtask_constraint_tuple == (("left", 0), ("right", 0))
    # The exposed constraint expands to the runtime form the generator consumes.
    runtime = c.generate_runtime_subtask_constraints()
    assert set(runtime) == {("left", 0), ("right", 0)}


# ---------------------------------------------------------------------------------------------------
# bind_env
# ---------------------------------------------------------------------------------------------------
def test_bind_env_sets_env_once():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    sentinel = object()
    td.bind_env(sentinel)
    assert td.env is sentinel


def test_bind_env_rejects_double_bind():
    td = TaskDescriptor.from_dict(_stack_task_dict())
    td.bind_env(object())
    with pytest.raises(AssertionError, match="env already bound"):
        td.bind_env(object())


# ---------------------------------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------------------------------
def test_from_yaml_round_trip(tmp_path):
    import yaml

    data = _stack_task_dict()
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data))
    td = TaskDescriptor.from_yaml(str(path))
    assert td.name == "stack"
    assert td.get_term_signal_names("franka") == ["grasp_1", "stack_1", "grasp_2", "stack_2"]
    assert td.get_generation_policy().num_trials == 7


@pytest.mark.parametrize(
    "yaml_name",
    ["franka_cube_stack.yaml", "franka_cube_stack_skillgen.yaml", "g1_pick_place.yaml", "gr1_pick_place.yaml"],
)
def test_shipped_example_task_yamls_parse(yaml_name):
    """The example task descriptors shipped in the repo must parse into valid TaskDescriptors."""
    path = os.path.join(TestPaths.tasks_dir, yaml_name)
    td = TaskDescriptor.from_yaml(path)
    assert td.name
    eef_names = td.get_eef_names()
    assert len(eef_names) >= 1
    for eef in eef_names:
        subtasks = td.get_subtasks(eef)
        assert len(subtasks) >= 1
        # Each accessor is subtask-ordered and the same length as the subtask list.
        assert len(td.get_term_signal_names(eef)) == len(subtasks)
        assert len(td.get_object_refs(eef)) == len(subtasks)
    assert isinstance(td.get_generation_policy(), GenerationPolicy)


def test_shipped_skillgen_yaml_uses_skillgen_algo_params():
    path = os.path.join(TestPaths.tasks_dir, "franka_cube_stack_skillgen.yaml")
    td = TaskDescriptor.from_yaml(path)
    for eef in td.get_eef_names():
        for params in td.get_subtask_algo_params(eef):
            assert isinstance(params, SkillGenSubtaskAlgoParams)
