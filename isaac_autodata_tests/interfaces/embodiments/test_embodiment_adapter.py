# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :class:`EmbodimentAdapter` ABC's shared behavior.

The ABC cannot be instantiated directly, so its env-binding and joint-state logic is exercised
through the concrete :class:`DeltaPoseIKSingleArmAdapter`.
"""

import torch

import pytest

from isaac_autodata_interfaces.embodiments import DeltaPoseIKSingleArmAdapter, PoseObsKeys
from isaac_autodata_tests.interfaces.mocks import MockArticulationData, MockAsset, MockEnv, MockScene


def _adapter() -> DeltaPoseIKSingleArmAdapter:
    return DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
    )


def _robot_env(joint_pos: torch.Tensor, joint_names: list[str]) -> MockEnv:
    data = MockArticulationData(joint_pos=joint_pos, joint_names=joint_names)
    return MockEnv(scene=MockScene(assets={"robot": MockAsset(data)}))


def test_default_robot_asset_name():
    assert _adapter().robot_asset_name == "robot"


def test_bind_env_sets_env_once():
    adapter = _adapter()
    assert adapter.env is None
    env = MockEnv()
    adapter.bind_env(env)
    assert adapter.env is env


def test_bind_env_rejects_double_bind():
    adapter = _adapter()
    adapter.bind_env(MockEnv())
    with pytest.raises(AssertionError, match="env already bound"):
        adapter.bind_env(MockEnv())


def test_get_joint_positions_all_and_indexed():
    adapter = _adapter()
    joint_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    adapter.bind_env(_robot_env(joint_pos, ["j0", "j1", "j2"]))
    assert torch.equal(adapter.get_joint_positions(), joint_pos)
    assert torch.equal(adapter.get_joint_positions(env_ids=[1]), joint_pos[[1]])


def test_get_joint_names():
    adapter = _adapter()
    adapter.bind_env(_robot_env(torch.zeros(1, 3), ["j0", "j1", "j2"]))
    assert adapter.get_joint_names() == ["j0", "j1", "j2"]


def test_joint_queries_require_bound_env():
    with pytest.raises(AssertionError, match="bind_env"):
        _adapter().get_joint_positions()
    with pytest.raises(AssertionError, match="bind_env"):
        _adapter().get_joint_names()
