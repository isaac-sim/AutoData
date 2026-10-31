# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorder terms that store the per-step data annotation, generation and the dataset metrics read."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from autodata_interfaces.datastream import Datastream


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Records end-effector, commanded target and object poses before each step, in one frame.

    Attributes:
        datastream: Datastream to read. Set it on the class once the Datastream exists, since Isaac Lab
            copies recorder configurations when building the env.
    """

    datastream: Datastream | None = None

    def record_pre_step(self):
        assert self.datastream is not None, "Set PreStepDatagenInfoRecorder.datastream before stepping the env."
        return "obs/datagen_info", {
            "eef_pose": self.datastream.embodiment_adapter.get_eef_poses(env_ids=None),
            "target_eef_pose": self.datastream.action_to_target_eef_pose(self._env.action_manager.action),
            "object_pose": self.datastream.get_object_poses(),
        }


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`PreStepDatagenInfoRecorder`."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PostStepSuccessRecorder(RecorderTerm):
    """Records whether the task's success term holds after each step, as ``step_success``."""

    def record_post_step(self):
        success_term = self.cfg.success_term
        return "step_success", success_term.func(self._env, **success_term.params)


@configclass
class PostStepSuccessRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`PostStepSuccessRecorder`."""

    class_type: type[RecorderTerm] = PostStepSuccessRecorder
    success_term: TerminationTermCfg | None = None
    """The task's success term; the env itself runs without termination terms."""


@configclass
class DatagenInfoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus the per-step ``obs/datagen_info`` term."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()


def add_step_success_recorder(recorder_cfg: object, success_term: TerminationTermCfg | None) -> None:
    """Add the ``step_success`` term to any recorder manager configuration.

    Nothing is added without a success term, since there is then no success to record.

    Args:
        recorder_cfg: Recorder manager configuration the env will be built with.
        success_term: The task's success term, or None when the task has none.
    """
    if success_term is not None:
        recorder_cfg.record_post_step_success = PostStepSuccessRecorderCfg(success_term=success_term)


def add_pose_recorder(recorder_cfg: object) -> None:
    """Add the per-step end-effector, commanded target and object pose term to any recorder manager configuration.

    A configuration that already has the ``record_pre_step_datagen_info`` term, such as
    :class:`DatagenInfoRecorderManagerCfg`, keeps its own.

    Args:
        recorder_cfg: Recorder manager configuration the env will be built with.
    """
    if not hasattr(recorder_cfg, "record_pre_step_datagen_info"):
        recorder_cfg.record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
