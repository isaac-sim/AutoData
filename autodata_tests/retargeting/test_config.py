# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for retarget-descriptor parsing into core subtasks."""

import os

import pytest

from autodata_interfaces.tasks.subtask_spec import Subtask
from autodata_retargeting.config import RetargetConfig, RetargetSubtaskAlgoParams, parse_subtask

_RETARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "autodata_examples", "retarget")


def test_parse_subtask_splits_core_and_retarget_fields():
    st = parse_subtask({
        "name": "grasp",
        "object_ref": "cube",
        "description": "Grasp the cube.",
        "subtask_end": {"method": "gripper_close", "offset": 5},
        "object_tracking": "cube",
        "offset": {"translation": [0, 0, 0.01]},
    })
    assert isinstance(st, Subtask)
    assert st.object_ref == "cube"
    assert st.description == "Grasp the cube."
    params = st.algo_params
    assert isinstance(params, RetargetSubtaskAlgoParams)
    assert params.name == "grasp"
    assert params.subtask_end.method == "gripper_close" and params.subtask_end.offset == 5
    assert params.object_tracking.object == "cube"
    assert params.offset.translation == [0.0, 0.0, 0.01]


def test_parse_subtask_defaults_to_core_defaults():
    st = parse_subtask({"frame_ref": "world"})
    assert st.object_ref == ""
    assert st.algo_params.frame_ref == "world"
    assert st.algo_params.subtask_end is None
    assert st.algo_params.object_tracking is None
    assert st.algo_params.offset is None


@pytest.mark.parametrize("key", ["selection_strategy", "action_noise", "not_a_key"])
def test_parse_subtask_rejects_unsupported_keys(key):
    with pytest.raises(AssertionError, match="unknown subtask keys"):
        parse_subtask({key: 0})


def test_synchronization_resolves_names_from_algo_params():
    config = RetargetConfig(
        source_embodiment="src.yaml",
        target_embodiment="tgt.yaml",
        target_env_name="env",
        subtasks={
            "left": [{"name": "l_release", "subtask_end": "gripper_open"}, {}],
            "right": [{"name": "r_grasp", "subtask_end": "gripper_close"}, {}],
        },
        synchronization=[["l_release", "r_grasp"]],
    )
    assert config.subtasks["left"][0].algo_params.name == "l_release"


def test_duplicate_subtask_names_rejected():
    with pytest.raises(AssertionError, match="unique"):
        RetargetConfig(
            source_embodiment="src.yaml",
            target_embodiment="tgt.yaml",
            target_env_name="env",
            subtasks={"left": [{"name": "a", "subtask_end": "gripper_open"}, {"name": "a"}]},
        )


@pytest.mark.parametrize("yaml_name", ["g1_to_gr1_pick_place.yaml", "gr1_to_g1_pick_place.yaml"])
def test_shipped_retarget_descriptors_parse(yaml_name):
    config = RetargetConfig.from_yaml(os.path.join(_RETARGET_DIR, yaml_name))
    assert config.subtasks
    for entries in config.subtasks.values():
        assert all(isinstance(st.algo_params, RetargetSubtaskAlgoParams) for st in entries)


def test_from_yaml_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("source_embodiment: a\nsource_embodiment: b\ntarget_embodiment: c\ntarget_env_name: d\n")
    with pytest.raises(AssertionError, match="duplicate key"):
        RetargetConfig.from_yaml(str(path))
