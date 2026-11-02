# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

from autodata_interfaces.env.datagen_info_recorder import (
    DatagenInfoRecorderManagerCfg,
    PreStepDatagenInfoRecorder,
    add_pose_recorder,
    add_step_success_recorder,
)


def test_metric_recorders_extend_a_custom_configuration():
    custom = ActionStateRecorderManagerCfg()

    add_pose_recorder(custom)
    add_step_success_recorder(custom, success_term=None)

    assert custom.record_pre_step_datagen_info.record_object_poses is False
    assert not hasattr(custom, "record_post_step_success")


def test_pose_recorder_keeps_an_existing_datagen_info_term():
    annotation = DatagenInfoRecorderManagerCfg()

    add_pose_recorder(annotation)

    assert annotation.record_pre_step_datagen_info.record_object_poses is True


@pytest.mark.parametrize("annotation", [True, False], ids=["annotation", "metrics"])
def test_recorder_includes_deformable_state_only_for_annotation(annotation):
    recorder_cfg = DatagenInfoRecorderManagerCfg() if annotation else ActionStateRecorderManagerCfg()
    if not annotation:
        add_pose_recorder(recorder_cfg)

    nodes = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    datastream = Mock()
    datastream.embodiment_adapter.get_eef_poses.return_value = {}
    datastream.action_to_target_eef_pose.return_value = {}
    datastream.get_object_poses.return_value = {}
    datastream.get_object_nodal_positions.return_value = {"rope": nodes}
    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(1, 7)))
    recorder = PreStepDatagenInfoRecorder(recorder_cfg.record_pre_step_datagen_info, env)
    recorder.datastream = datastream

    key, info = recorder.record_pre_step()

    assert key == "obs/datagen_info"
    if annotation:
        assert set(info) == {"eef_pose", "target_eef_pose", "object_pose", "object_nodal_position"}
        torch.testing.assert_close(info["object_nodal_position"]["rope"], nodes)
        datastream.get_object_nodal_positions.assert_called_once_with()
    else:
        assert set(info) == {"eef_pose", "target_eef_pose"}
        datastream.get_object_poses.assert_not_called()
        datastream.get_object_nodal_positions.assert_not_called()
