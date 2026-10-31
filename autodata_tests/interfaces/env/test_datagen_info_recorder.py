# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

from autodata_interfaces.env.datagen_info_recorder import (
    DatagenInfoRecorderManagerCfg,
    PreStepDatagenInfoRecorderCfg,
    add_pose_recorder,
    add_step_success_recorder,
)


def test_metric_recorders_extend_a_custom_configuration():
    custom = ActionStateRecorderManagerCfg()

    add_pose_recorder(custom)
    add_step_success_recorder(custom, success_term=None)

    assert isinstance(custom.record_pre_step_datagen_info, PreStepDatagenInfoRecorderCfg)
    assert not hasattr(custom, "record_post_step_success")


def test_pose_recorder_keeps_an_existing_datagen_info_term():
    annotation = DatagenInfoRecorderManagerCfg()
    term = annotation.record_pre_step_datagen_info

    add_pose_recorder(annotation)

    assert annotation.record_pre_step_datagen_info is term
