# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Record image observations exposed in a separate camera group."""

    def record_pre_step(self):
        """Return camera observations alongside the recorded policy observations."""
        return "obs", self._env.obs_buf.get("camera_obs", {})


def make_action_state_recorder_manager_cfg(record_images: bool = False) -> ActionStateRecorderManagerCfg:
    """Return Isaac Lab's action-state recorder with optional camera observations.

    Args:
        record_images: Include the separate camera observation group when present.
    """

    cfg = ActionStateRecorderManagerCfg()
    if record_images:
        cfg.record_pre_step_camera_observations = RecorderTermCfg(class_type=PreStepCameraObservationsRecorder)
    return cfg
