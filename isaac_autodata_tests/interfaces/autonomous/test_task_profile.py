# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from isaac_autodata_interfaces.autonomous.pick_place_success import PickPlaceSuccessThresholds as VerifierThresholds
from isaac_autodata_interfaces.autonomous.profiles.franka_pick_cube_into_bowl import (
    FRANKA_PICK_CUBE_INTO_BOWL,
    PickPlaceSuccessThresholds,
)
from isaac_autodata_interfaces.autonomous.runtime_support import CURRENT_RUNTIME_SUPPORT
from isaac_autodata_interfaces.autonomous.schedulestream.custream_v1 import (
    V1_DESTINATION_PLACEMENT_PROFILE,
    V1_FRANKA_USD_BASENAMES,
    V1_GRASP_GEOMETRY_PROFILE,
    V1_REVIEWED_DESTINATION_ASSET,
    V1_REVIEWED_GRASPABLE_ASSET,
)


def test_live_task_profile_is_the_single_owner_of_shared_runtime_facts() -> None:
    profile = FRANKA_PICK_CUBE_INTO_BOWL

    assert CURRENT_RUNTIME_SUPPORT == profile.name
    assert V1_FRANKA_USD_BASENAMES is profile.franka_usd_basenames
    assert V1_REVIEWED_GRASPABLE_ASSET == profile.graspable_asset
    assert V1_REVIEWED_DESTINATION_ASSET == profile.destination_asset
    assert V1_GRASP_GEOMETRY_PROFILE == profile.grasp_geometry_profile
    assert V1_DESTINATION_PLACEMENT_PROFILE == profile.destination_placement_profile
    assert VerifierThresholds is PickPlaceSuccessThresholds


def test_live_task_profile_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        FRANKA_PICK_CUBE_INTO_BOWL.motion_backend = "other"
