# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.datagen_info`."""

import torch

from isaac_autodata_core.datagen_info import DatagenInfo


def test_defaults_all_none():
    di = DatagenInfo()
    assert di.eef_pose is None
    assert di.object_poses is None
    assert di.object_nodal_positions is None
    assert di.subtask_term_signals is None
    assert di.subtask_start_signals is None
    assert di.target_eef_pose is None
    assert di.passthrough_action is None
    assert di.to_dict() == {}


def test_to_dict_only_includes_populated_fields():
    di = DatagenInfo(
        eef_pose={"franka": torch.zeros(2, 4, 4)},
        subtask_term_signals={"grasp_1": torch.zeros(2)},
    )
    assert set(di.to_dict()) == {"eef_pose", "subtask_term_signals"}


def test_to_dict_key_names():
    # Every field populated (empty dicts are not None): note object_poses is plural on disk/out,
    # while eef_pose / target_eef_pose are singular.
    di = DatagenInfo(
        eef_pose={},
        object_poses={},
        object_nodal_positions={},
        subtask_term_signals={},
        subtask_start_signals={},
        target_eef_pose={},
        passthrough_action={},
    )
    assert set(di.to_dict()) == {
        "eef_pose",
        "object_poses",
        "object_nodal_positions",
        "subtask_term_signals",
        "subtask_start_signals",
        "target_eef_pose",
        "passthrough_action",
    }


def test_object_poses_and_signals_are_copied_at_construction():
    obj = {"cube": torch.zeros(2, 4, 4)}
    term = {"grasp_1": torch.zeros(2)}
    di = DatagenInfo(object_poses=obj, subtask_term_signals=term)
    # Constructor takes a shallow dict() copy of these, so adding keys later must not leak in.
    obj["extra"] = torch.zeros(1)
    term["extra"] = torch.zeros(1)
    assert "extra" not in di.object_poses
    assert "extra" not in di.subtask_term_signals


def test_eef_pose_is_stored_by_reference():
    # eef_pose / target_eef_pose / passthrough_action are stored as-is (documented behavior).
    eef = {"franka": torch.zeros(2, 4, 4)}
    di = DatagenInfo(eef_pose=eef)
    assert di.eef_pose is eef


def test_to_dict_deepcopies_object_poses():
    obj = {"cube": torch.zeros(2, 4, 4)}
    di = DatagenInfo(object_poses=obj)
    out = di.to_dict()
    out["object_poses"]["cube"] += 1.0  # mutate the returned copy
    assert torch.count_nonzero(di.object_poses["cube"]) == 0  # original untouched


def test_to_dict_deepcopies_object_nodal_positions():
    nodes = {"rope": torch.zeros(2, 8, 3)}
    di = DatagenInfo(object_nodal_positions=nodes)
    out = di.to_dict()
    out["object_nodal_positions"]["rope"] += 1.0
    assert torch.count_nonzero(di.object_nodal_positions["rope"]) == 0
