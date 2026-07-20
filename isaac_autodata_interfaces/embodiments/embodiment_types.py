# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared dataclasses used by single-arm and bimanual embodiment adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseObsKeys:
    """Observation-buffer keys for a single end-effector's pose.

    Both keys index into ``env.obs_buf[obs_group]``. ``pos`` returns a position
    tensor; ``quat`` returns an (x, y, z, w) orientation quaternion. In Isaac
    Lab observation names such as ``target_quat_w``, the ``_w`` suffix denotes
    the world frame; it does not denote scalar-first quaternion storage.

    Args:
        pos: Observation key for end-effector position.
        quat: Observation key for end-effector orientation quaternion.
    """

    pos: str
    quat: str
