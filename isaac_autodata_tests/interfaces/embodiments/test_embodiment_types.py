# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.embodiments.embodiment_types`."""

import dataclasses

import pytest

from isaac_autodata_interfaces.embodiments.embodiment_types import PoseObsKeys


def test_pose_obs_keys_fields():
    keys = PoseObsKeys(pos="eef_pos", quat="eef_quat")
    assert keys.pos == "eef_pos"
    assert keys.quat == "eef_quat"


def test_pose_obs_keys_is_frozen():
    keys = PoseObsKeys(pos="p", quat="q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        keys.pos = "x"  # type: ignore[misc]


def test_pose_obs_keys_equality_and_hash():
    assert PoseObsKeys("p", "q") == PoseObsKeys("p", "q")
    assert PoseObsKeys("p", "q") != PoseObsKeys("p", "r")
    # frozen dataclass is hashable, so it can key a dict / live in a set.
    assert len({PoseObsKeys("p", "q"), PoseObsKeys("p", "q")}) == 1
