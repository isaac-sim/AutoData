# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dataclasses used by single-arm and bimanual embodiment adapters."""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionTargetFrame:
    """Exact robot frame commanded by one ordered pose-action block.

    Args:
        eef_name: End-effector name exposed by the embodiment adapter.
        frame_name: Exact controlled frame name in the live robot model.
    """

    eef_name: str
    frame_name: str

    def __post_init__(self) -> None:
        """Require named end-effector and controller frames."""
        assert self.eef_name, "eef_name must be a non-empty string"
        assert self.frame_name, "frame_name must be a non-empty string"


@dataclass(frozen=True)
class NamedJointTrajectory:
    """Joint positions with an explicit, stable name order.

    Args:
        joint_names: Unique articulation joint names.
        positions: Joint positions with shape ``(T, J)`` [rad, m].
    """

    joint_names: tuple[str, ...]
    positions: torch.Tensor

    def __post_init__(self) -> None:
        """Require unique joint names and finite, nonempty position samples."""
        assert self.joint_names and len(self.joint_names) == len(set(self.joint_names))
        assert all(isinstance(name, str) and name for name in self.joint_names)
        assert isinstance(self.positions, torch.Tensor) and self.positions.ndim == 2
        assert self.positions.shape[1] == len(self.joint_names)
        assert len(self.positions) > 0 and bool(torch.isfinite(self.positions).all())


@dataclass(frozen=True)
class PoseObsKeys:
    """Observation-buffer keys for a single end-effector's pose.

    Both keys index into ``env.obs_buf[obs_group]``. ``pos`` returns a position
    tensor; ``quat`` returns a (x, y, z, w) orientation quaternion.

    Args:
        pos: Observation key for end-effector position.
        quat: Observation key for end-effector orientation quaternion.
    """

    pos: str
    quat: str
