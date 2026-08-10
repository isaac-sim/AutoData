# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.embodiments.single_arm_embodiment_adapter`."""

import torch

import pytest

from isaac_autodata_interfaces.embodiments import DeltaPoseIKSingleArmAdapter, PoseObsKeys
from isaac_autodata_tests.interfaces.mocks import MockEnv
from isaac_autodata_utils import pose_math

_IDENTITY_QUAT_XYZW = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


def _embodiment_adapter(
    gripper_action_dim: int = 1, clip: bool = True, eef_offset=(0.0, 0.0, 0.0)
) -> DeltaPoseIKSingleArmAdapter:
    return DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=gripper_action_dim,
        clip_pose_action_to_unit=clip,
        eef_offset=eef_offset,
    )


def _bound(embodiment_adapter: DeltaPoseIKSingleArmAdapter, pos: torch.Tensor, quat: torch.Tensor | None = None):
    quat = _IDENTITY_QUAT_XYZW if quat is None else quat
    embodiment_adapter.bind_env(MockEnv(obs_buf={"policy": {"eef_pos": pos, "eef_quat": quat}}))
    return embodiment_adapter


# ---------------------------------------------------------------------------------------------------
# action_dim
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gripper, expected", [(0, 6), (1, 7), (2, 8)])
def test_action_dim(gripper, expected):
    assert _embodiment_adapter(gripper_action_dim=gripper).action_dim == expected


# ---------------------------------------------------------------------------------------------------
# __post_init__
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": ""}, "name must be"),
        ({"eef_name": ""}, "eef_name must be"),
        ({"gripper_action_dim": -1}, "gripper_action_dim must be non-negative"),
        ({"obs_group": ""}, "obs_group must be"),
        ({"eef_offset": (0.0, 0.0)}, "eef_offset must have 3 elements"),
    ],
)
def test_post_init_assertions(kwargs, match):
    base = {
        "name": "franka",
        "eef_name": "franka",
        "pose_obs_keys": PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        "gripper_action_dim": 1,
    }
    base.update(kwargs)
    with pytest.raises(AssertionError, match=match):
        DeltaPoseIKSingleArmAdapter(**base)


def test_post_init_rejects_non_pose_obs_keys():
    with pytest.raises(AssertionError, match="pose_obs_keys must be a PoseObsKeys"):
        DeltaPoseIKSingleArmAdapter(
            name="franka",
            eef_name="franka",
            pose_obs_keys={"pos": "eef_pos", "quat": "eef_quat"},  # type: ignore[arg-type]
            gripper_action_dim=1,
        )


# ---------------------------------------------------------------------------------------------------
# get_eef_names
# ---------------------------------------------------------------------------------------------------
def test_get_eef_names():
    assert _embodiment_adapter().get_eef_names() == ("franka",)


# ---------------------------------------------------------------------------------------------------
# get_eef_poses / _observed_to_control_link
# ---------------------------------------------------------------------------------------------------
def test_get_eef_poses_identity_quat_is_translation():
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[1.0, 2.0, 3.0]]))
    pose = embodiment_adapter.get_eef_poses()["franka"]
    assert pose.shape == (1, 4, 4)
    assert torch.allclose(pose[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(pose[0, :3, :3], torch.eye(3))


def test_get_eef_poses_applies_eef_offset():
    # control-link pose = observed @ translate(-offset); with identity rotation this is pos - offset.
    embodiment_adapter = _bound(_embodiment_adapter(eef_offset=(0.0, 0.0, 0.1)), pos=torch.tensor([[0.0, 0.0, 1.0]]))
    pose = embodiment_adapter.get_eef_poses()["franka"]
    assert torch.allclose(pose[0, :3, 3], torch.tensor([0.0, 0.0, 0.9]))


def test_get_eef_poses_zero_offset_is_noop():
    embodiment_adapter = _embodiment_adapter(eef_offset=(0.0, 0.0, 0.0))
    raw = pose_math.make_pose(torch.tensor([[1.0, 2.0, 3.0]]), torch.eye(3).unsqueeze(0))
    assert torch.equal(embodiment_adapter._observed_to_control_link(raw), raw)


def test_get_eef_poses_requires_bound_env():
    with pytest.raises(AssertionError, match="bind_env"):
        _embodiment_adapter().get_eef_poses()


# ---------------------------------------------------------------------------------------------------
# action_to_target_eef_pose
# ---------------------------------------------------------------------------------------------------
def test_action_to_target_eef_pose_adds_delta_to_current():
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    action = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]])  # delta_pos, zero rot delta, gripper
    target = embodiment_adapter.action_to_target_eef_pose(action)["franka"]
    assert target.shape == (1, 4, 4)
    assert torch.allclose(target[0, :3, 3], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(target[0, :3, :3], torch.eye(3), atol=1e-6)


def test_action_to_target_eef_pose_rejects_wrong_shape():
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    with pytest.raises(AssertionError, match="action shape must be"):
        embodiment_adapter.action_to_target_eef_pose(torch.zeros(7))  # 1-D, not (num_envs, action_dim)


# ---------------------------------------------------------------------------------------------------
# target_eef_pose_to_action
# ---------------------------------------------------------------------------------------------------
def test_target_eef_pose_to_action_recovers_delta():
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    target = pose_math.make_pose(torch.tensor([0.1, 0.2, 0.3]), torch.eye(3))
    action = embodiment_adapter.target_eef_pose_to_action({"franka": target}, {"franka": torch.tensor([0.5])}, env_id=0)
    assert action.shape == (7,)
    assert torch.allclose(action[:3], torch.tensor([0.1, 0.2, 0.3]), atol=1e-6)
    assert torch.allclose(action[3:6], torch.zeros(3), atol=1e-6)
    assert action[6] == 0.5


def test_target_eef_pose_to_action_clips_when_configured():
    embodiment_adapter = _bound(_embodiment_adapter(clip=True), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    target = pose_math.make_pose(torch.tensor([2.0, 0.0, 0.0]), torch.eye(3))  # delta of 2.0 -> clipped to 1.0
    action = embodiment_adapter.target_eef_pose_to_action({"franka": target}, {"franka": torch.tensor([0.0])}, env_id=0)
    assert action[0] == 1.0


def test_target_eef_pose_to_action_no_clip_preserves_large_delta():
    embodiment_adapter = _bound(_embodiment_adapter(clip=False), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    target = pose_math.make_pose(torch.tensor([2.0, 0.0, 0.0]), torch.eye(3))
    action = embodiment_adapter.target_eef_pose_to_action({"franka": target}, {"franka": torch.tensor([0.0])}, env_id=0)
    assert torch.allclose(action[0], torch.tensor(2.0))


@pytest.mark.parametrize(
    "target_keys, passthrough_keys, match",
    [
        ({"wrong"}, {"franka"}, "target_eef_pose_dict must have exactly one key"),
        ({"franka"}, {"wrong"}, "passthrough_action_dict must have exactly one key"),
    ],
)
def test_target_eef_pose_to_action_rejects_wrong_keys(target_keys, passthrough_keys, match):
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    target = pose_math.make_pose(torch.zeros(3), torch.eye(3))
    target_dict = {k: target for k in target_keys}
    passthrough = {k: torch.zeros(1) for k in passthrough_keys}
    with pytest.raises(AssertionError, match=match):
        embodiment_adapter.target_eef_pose_to_action(target_dict, passthrough, env_id=0)


def test_target_eef_pose_to_action_rejects_wrong_target_shape():
    embodiment_adapter = _bound(_embodiment_adapter(), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    with pytest.raises(AssertionError, match=r"target pose must be \(4, 4\)"):
        embodiment_adapter.target_eef_pose_to_action(
            {"franka": torch.zeros(3, 3)}, {"franka": torch.zeros(1)}, env_id=0
        )


def test_target_eef_pose_to_action_rejects_wrong_gripper_shape():
    embodiment_adapter = _bound(_embodiment_adapter(gripper_action_dim=1), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    target = pose_math.make_pose(torch.zeros(3), torch.eye(3))
    with pytest.raises(AssertionError, match="gripper action must be"):
        embodiment_adapter.target_eef_pose_to_action({"franka": target}, {"franka": torch.zeros(2)}, env_id=0)


def test_action_round_trip():
    # action -> target pose -> action recovers the pose + gripper parts (no clip needed for small deltas).
    embodiment_adapter = _bound(_embodiment_adapter(clip=False), pos=torch.tensor([[0.0, 0.0, 0.0]]))
    action_in = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.7]])
    target = embodiment_adapter.action_to_target_eef_pose(action_in)["franka"][0]
    action_out = embodiment_adapter.target_eef_pose_to_action(
        {"franka": target}, {"franka": action_in[0, 6:]}, env_id=0
    )
    assert torch.allclose(action_out[:6], action_in[0, :6], atol=1e-5)
    assert torch.allclose(action_out[6:], action_in[0, 6:])


# ---------------------------------------------------------------------------------------------------
# actions_to_passthrough_actions
# ---------------------------------------------------------------------------------------------------
def test_actions_to_passthrough_actions_slices_gripper_and_preserves_batch():
    embodiment_adapter = _embodiment_adapter(gripper_action_dim=1)
    actions = torch.randn(2, 5, 7)  # (num_envs, num_steps, action_dim)
    out = embodiment_adapter.actions_to_passthrough_actions(actions)
    assert set(out) == {"franka"}
    assert out["franka"].shape == (2, 5, 1)
    assert torch.equal(out["franka"], actions[..., -1:])


def test_actions_to_passthrough_actions_rejects_wrong_last_dim():
    embodiment_adapter = _embodiment_adapter(gripper_action_dim=1)
    with pytest.raises(AssertionError, match="actions last dim must be"):
        embodiment_adapter.actions_to_passthrough_actions(torch.zeros(2, 5))


# ---------------------------------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------------------------------
def test_from_dict_full():
    data = {
        "name": "franka_panda",
        "description": "franka arm",
        "eef_name": "franka",
        "obs_group": "policy",
        "pose_obs_keys": {"pos": "eef_pos", "quat": "eef_quat"},
        "action_layout": {"gripper_dim": 1, "clip_pose_action_to_unit": False},
        "eef_offset": [0.0, 0.0, 0.1034],
    }
    embodiment_adapter = DeltaPoseIKSingleArmAdapter.from_dict(data)
    assert embodiment_adapter.name == "franka_panda"
    assert embodiment_adapter.description == "franka arm"
    assert embodiment_adapter.eef_name == "franka"
    assert embodiment_adapter.gripper_action_dim == 1
    assert embodiment_adapter.clip_pose_action_to_unit is False
    assert embodiment_adapter.eef_offset == (0.0, 0.0, 0.1034)
    assert embodiment_adapter.pose_obs_keys == PoseObsKeys("eef_pos", "eef_quat")


def test_from_dict_defaults():
    data = {
        "name": "franka",
        "eef_name": "franka",
        "pose_obs_keys": {"pos": "eef_pos", "quat": "eef_quat"},
        "action_layout": {"gripper_dim": 1},
    }
    embodiment_adapter = DeltaPoseIKSingleArmAdapter.from_dict(data)
    assert embodiment_adapter.description == ""
    assert embodiment_adapter.obs_group == "policy"
    assert embodiment_adapter.clip_pose_action_to_unit is True
    assert embodiment_adapter.eef_offset == (0.0, 0.0, 0.0)
