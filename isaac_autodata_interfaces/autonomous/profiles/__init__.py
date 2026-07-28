# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete autonomous task profiles supported by the live runtime."""

from isaac_autodata_interfaces.autonomous.profiles.franka_pick_cube_into_bowl import (
    FRANKA_PICK_CUBE_INTO_BOWL,
    AutonomousTaskProfile,
    PickPlaceSuccessThresholds,
)

__all__ = [
    "FRANKA_PICK_CUBE_INTO_BOWL",
    "AutonomousTaskProfile",
    "PickPlaceSuccessThresholds",
]
