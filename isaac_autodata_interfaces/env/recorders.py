# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoData recorder terms for parallel scene-state capture."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm

from isaac_autodata_interfaces.env.scene_state import get_scene_state


class InitialStateRecorder(RecorderTerm):
    """Record the correctly env-relative initial scene state after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        """Return initial state for the reset environments."""

        def select_envs(value):
            if isinstance(value, dict):
                return {key: select_envs(item) for key, item in value.items()}
            return value if env_ids is None else value[env_ids]

        return "initial_state", select_envs(get_scene_state(self._env.scene, is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Record correctly env-relative scene state after each environment step."""

    def record_post_step(self):
        """Return state for every environment."""

        return "states", get_scene_state(self._env.scene, is_relative=True)


def make_action_state_recorder_manager_cfg() -> ActionStateRecorderManagerCfg:
    """Return Isaac Lab's action-state recorder with corrected state terms."""

    cfg = ActionStateRecorderManagerCfg()
    cfg.record_initial_state.class_type = InitialStateRecorder
    cfg.record_post_step_states.class_type = PostStepStatesRecorder
    return cfg
