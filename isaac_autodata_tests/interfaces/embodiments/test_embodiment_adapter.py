# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :class:`EmbodimentAdapter`."""

import torch

import pytest

from isaac_autodata_interfaces.embodiments import DeltaPoseIKSingleArmAdapter, PoseObsKeys
from isaac_autodata_tests.interfaces.mocks import MockArticulationData, MockAsset, MockEnv, MockScene


def _embodiment_adapter() -> DeltaPoseIKSingleArmAdapter:
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
    assert _embodiment_adapter().robot_asset_name == "robot"


def test_bind_env_sets_env_once():
    embodiment_adapter = _embodiment_adapter()
    assert embodiment_adapter.env is None
    env = MockEnv()
    embodiment_adapter.bind_env(env)
    assert embodiment_adapter.env is env


def test_bind_env_rejects_double_bind():
    embodiment_adapter = _embodiment_adapter()
    embodiment_adapter.bind_env(MockEnv())
    with pytest.raises(AssertionError, match="env already bound"):
        embodiment_adapter.bind_env(MockEnv())


def test_get_joint_positions_all_and_indexed():
    embodiment_adapter = _embodiment_adapter()
    joint_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    embodiment_adapter.bind_env(_robot_env(joint_pos, ["j0", "j1", "j2"]))
    assert torch.equal(embodiment_adapter.get_joint_positions(), joint_pos)
    assert torch.equal(embodiment_adapter.get_joint_positions(env_ids=[1]), joint_pos[[1]])


def test_get_joint_names():
    embodiment_adapter = _embodiment_adapter()
    embodiment_adapter.bind_env(_robot_env(torch.zeros(1, 3), ["j0", "j1", "j2"]))
    assert embodiment_adapter.get_joint_names() == ["j0", "j1", "j2"]


def test_joint_queries_require_bound_env():
    with pytest.raises(AssertionError, match="bind_env"):
        _embodiment_adapter().get_joint_positions()
    with pytest.raises(AssertionError, match="bind_env"):
        _embodiment_adapter().get_joint_names()
