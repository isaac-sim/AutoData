# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorder that stores ``obs/datagen_info`` on every step of annotation and generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from autodata_interfaces.datastream import Datastream


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Records object, end-effector and commanded target poses from the Datastream before each step.

    Attributes:
        datastream: Datastream to read. Set it on the class once the Datastream exists, since Isaac Lab
            copies recorder configurations when building the env.
    """

    datastream: Datastream | None = None

    def record_pre_step(self):
        assert self.datastream is not None, "Set PreStepDatagenInfoRecorder.datastream before stepping the env."
        datagen_info = {
            "object_pose": self.datastream.get_object_poses(),
            "eef_pose": self.datastream.embodiment_adapter.get_eef_poses(env_ids=None),
            "target_eef_pose": self.datastream.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`PreStepDatagenInfoRecorder`."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


@configclass
class DatagenInfoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus the per-step ``obs/datagen_info`` term."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
