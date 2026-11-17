# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary Newton frame-sensor refresh fix for Arena rope episode restores."""

import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer


# TODO: Remove this file and the override in FrankaRopeIKEmbodiment when Arena's
# Lab pin refreshes native frame transforms on sensor reads after reset/reset_to.
class FrameTransformerResetPatch(FrameTransformer):
    """Read restored poses before the first physics step of an episode."""

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        # Lab otherwise only updates the native sensor while stepping physics.
        # Refresh from the current state before copying its output into Lab buffers.
        sensor = NewtonManager._newton_frame_transform_sensors[self._sensor_index]
        sensor.update(NewtonManager.get_state())
        super()._update_buffers_impl(env_mask)
