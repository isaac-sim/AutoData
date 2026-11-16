# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary Isaac Lab body-offset Jacobian backport for Arena rope SoftMimicGen."""

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils


# TODO: Remove this file and the override in FrankaRopeIKEmbodiment after Arena's Lab pin
# includes https://github.com/isaac-sim/IsaacLab/pull/7989 (2f1983ca7247).
class DifferentialInverseKinematicsActionOffset(DifferentialInverseKinematicsAction):
    """Apply the upstream tool-offset correction without changing the Lab installation."""

    def _compute_frame_jacobian(self) -> torch.Tensor:
        self._jacobian_b[:] = self.jacobian_b
        if self.cfg.body_offset is not None:
            body_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(self._asset.data.root_quat_w.torch),
                self._asset.data.body_quat_w.torch[:, self._body_idx],
            )
            offset_b = math_utils.quat_apply(body_quat_b, self._offset_pos)
            self._jacobian_b[:, :3] += torch.linalg.cross(self._jacobian_b[:, 3:], offset_b[:, :, None], dim=1)
            # A rigidly attached tool has the same angular velocity in base axes.
        return self._jacobian_b
