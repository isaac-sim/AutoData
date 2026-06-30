# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.embodiments.bimanual_embodiment_adapter`."""

import dataclasses
import torch

import pytest

from isaac_autodata_interfaces.embodiments import AbsolutePoseWholeBodyBimanualAdapter, BimanualEefConfig, PoseObsKeys
from isaac_autodata_tests.interfaces.mocks import MockEnv
from isaac_autodata_utils import pose_math

_IDENTITY_QUAT_XYZW = [0.0, 0.0, 0.0, 1.0]


def _eef(name: str, indices: tuple[int, ...]) -> BimanualEefConfig:
    return BimanualEefConfig(
        name=name,
        pose_obs_keys=PoseObsKeys(pos=f"{name}_eef_pos", quat=f"{name}_eef_quat"),
        gripper_action_indices=indices,
    )


def _adapter(
    passthrough_channels: dict | None = None, canonicalize: bool = True
) -> AbsolutePoseWholeBodyBimanualAdapter:
    """A small bimanual adapter: pose blocks [0:7] / [7:14], hand-joints [14:18] (width 4)."""
    return AbsolutePoseWholeBodyBimanualAdapter(
        name="g1",
        left=_eef("left", (0, 1)),
        right=_eef("right", (2, 3)),
        left_pose_slice=(0, 7),
        right_pose_slice=(7, 14),
        hand_joints_slice=(14, 18),
        canonicalize_quat=canonicalize,
        passthrough_channels=passthrough_channels or {},
    )


def _bound(adapter: AbsolutePoseWholeBodyBimanualAdapter, left_pos, right_pos):
    obs = {
        "policy": {
            "left_eef_pos": left_pos,
            "left_eef_quat": torch.tensor([_IDENTITY_QUAT_XYZW]),
            "right_eef_pos": right_pos,
            "right_eef_quat": torch.tensor([_IDENTITY_QUAT_XYZW]),
        }
    }
    adapter.bind_env(MockEnv(obs_buf=obs))
    return adapter


# ---------------------------------------------------------------------------------------------------
# action_dim -- includes the regression test for empty passthrough_channels
# ---------------------------------------------------------------------------------------------------
def test_action_dim_no_passthrough_channels():
    # Regression: with no passthrough channels, action_dim is the hand-joints end and must not
    # raise (the previous `max(int, *empty)` form raised "int object is not iterable").
    assert _adapter().action_dim == 18


def test_action_dim_with_passthrough_channels():
    assert _adapter({"body": (18, 22)}).action_dim == 22


def test_action_dim_multiple_passthrough_channels_takes_max_end():
    adapter = _adapter({"body": (18, 22), "head": (22, 25)})
    assert adapter.action_dim == 25


def test_get_eef_names():
    assert _adapter().get_eef_names() == ("left", "right")


# ---------------------------------------------------------------------------------------------------
# BimanualEefConfig
# ---------------------------------------------------------------------------------------------------
def test_eef_config_is_frozen():
    cfg = _eef("left", (0, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------------
# __post_init__ assertions
# ---------------------------------------------------------------------------------------------------
def test_post_init_rejects_duplicate_eef_names():
    with pytest.raises(AssertionError, match="distinct names"):
        AbsolutePoseWholeBodyBimanualAdapter(
            name="g1",
            left=_eef("hand", (0, 1)),
            right=_eef("hand", (2, 3)),
            left_pose_slice=(0, 7),
            right_pose_slice=(7, 14),
            hand_joints_slice=(14, 18),
        )


def test_post_init_rejects_overlapping_gripper_indices():
    with pytest.raises(AssertionError, match="gripper_action_indices overlap"):
        AbsolutePoseWholeBodyBimanualAdapter(
            name="g1",
            left=_eef("left", (0, 1)),
            right=_eef("right", (1, 2)),  # index 1 overlaps left
            left_pose_slice=(0, 7),
            right_pose_slice=(7, 14),
            hand_joints_slice=(14, 18),
        )


def test_post_init_rejects_wrong_pose_span():
    with pytest.raises(AssertionError, match="must span 7 dims"):
        AbsolutePoseWholeBodyBimanualAdapter(
            name="g1",
            left=_eef("left", (0, 1)),
            right=_eef("right", (2, 3)),
            left_pose_slice=(0, 6),  # span 6, not 7
            right_pose_slice=(6, 13),
            hand_joints_slice=(13, 17),
        )


def test_post_init_rejects_non_contiguous_pose_blocks():
    with pytest.raises(AssertionError, match="right_pose_slice must immediately follow"):
        AbsolutePoseWholeBodyBimanualAdapter(
            name="g1",
            left=_eef("left", (0, 1)),
            right=_eef("right", (2, 3)),
            left_pose_slice=(0, 7),
            right_pose_slice=(8, 15),  # gap after left
            hand_joints_slice=(15, 19),
        )


def test_post_init_rejects_gripper_index_out_of_hand_width():
    with pytest.raises(AssertionError, match="gripper_action_indices reference position"):
        AbsolutePoseWholeBodyBimanualAdapter(
            name="g1",
            left=_eef("left", (0, 1)),
            right=_eef("right", (2, 9)),  # 9 >= hand width 4
            left_pose_slice=(0, 7),
            right_pose_slice=(7, 14),
            hand_joints_slice=(14, 18),
        )


def test_post_init_rejects_passthrough_channel_named_like_eef():
    with pytest.raises(AssertionError, match="collides with an eef name"):
        _adapter({"left": (18, 20)})


def test_post_init_rejects_passthrough_channel_overlapping_pose_region():
    with pytest.raises(AssertionError, match="overlaps action region"):
        _adapter({"body": (10, 16)})  # overlaps the [0, 18) pose+hand region


# ---------------------------------------------------------------------------------------------------
# get_eef_poses
# ---------------------------------------------------------------------------------------------------
def test_get_eef_poses_returns_both_arms():
    adapter = _bound(_adapter(), left_pos=torch.tensor([[1.0, 2.0, 3.0]]), right_pos=torch.tensor([[4.0, 5.0, 6.0]]))
    poses = adapter.get_eef_poses()
    assert set(poses) == {"left", "right"}
    assert torch.allclose(poses["left"][0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(poses["right"][0, :3, 3], torch.tensor([4.0, 5.0, 6.0]))


# ---------------------------------------------------------------------------------------------------
# action_to_target_eef_pose
# ---------------------------------------------------------------------------------------------------
def test_action_to_target_eef_pose_decodes_both_pose_blocks():
    adapter = _adapter()
    action = torch.zeros(1, 18)
    action[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    action[0, 3:7] = torch.tensor(_IDENTITY_QUAT_XYZW)
    action[0, 7:10] = torch.tensor([4.0, 5.0, 6.0])
    action[0, 10:14] = torch.tensor(_IDENTITY_QUAT_XYZW)
    out = adapter.action_to_target_eef_pose(action)
    assert torch.allclose(out["left"][0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out["right"][0, :3, 3], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(out["left"][0, :3, :3], torch.eye(3), atol=1e-6)


def test_action_to_target_eef_pose_rejects_wrong_shape():
    with pytest.raises(AssertionError, match="action shape must be"):
        _adapter().action_to_target_eef_pose(torch.zeros(18))  # 1-D


# ---------------------------------------------------------------------------------------------------
# target_eef_pose_to_action + interleaving
# ---------------------------------------------------------------------------------------------------
def test_target_eef_pose_to_action_packs_poses_and_interleaves_hands():
    adapter = _adapter()
    left_target = pose_math.make_pose(torch.tensor([1.0, 2.0, 3.0]), torch.eye(3))
    right_target = pose_math.make_pose(torch.tensor([4.0, 5.0, 6.0]), torch.eye(3))
    passthrough = {"left": torch.tensor([0.1, 0.2]), "right": torch.tensor([0.3, 0.4])}
    action = adapter.target_eef_pose_to_action({"left": left_target, "right": right_target}, passthrough)
    assert action.shape == (18,)
    assert torch.allclose(action[0:3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(action[3:7], torch.tensor(_IDENTITY_QUAT_XYZW), atol=1e-6)
    assert torch.allclose(action[7:10], torch.tensor([4.0, 5.0, 6.0]))
    # hand-joints block: left indices (0,1) -> 0.1,0.2 ; right indices (2,3) -> 0.3,0.4
    assert torch.allclose(action[14:18], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_target_eef_pose_to_action_writes_extra_channels():
    adapter = _adapter({"body": (18, 22)})
    target = pose_math.make_pose(torch.zeros(3), torch.eye(3))
    passthrough = {
        "left": torch.tensor([0.0, 0.0]),
        "right": torch.tensor([0.0, 0.0]),
        "body": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    action = adapter.target_eef_pose_to_action({"left": target, "right": target}, passthrough)
    assert action.shape == (22,)
    assert torch.allclose(action[18:22], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_target_eef_pose_to_action_rejects_missing_keys():
    adapter = _adapter()
    target = pose_math.make_pose(torch.zeros(3), torch.eye(3))
    with pytest.raises(AssertionError, match="target_eef_pose_dict must have exactly"):
        adapter.target_eef_pose_to_action({"left": target}, {"left": torch.zeros(2), "right": torch.zeros(2)})
    with pytest.raises(AssertionError, match="passthrough_action_dict must have exactly"):
        adapter.target_eef_pose_to_action({"left": target, "right": target}, {"left": torch.zeros(2)})


# ---------------------------------------------------------------------------------------------------
# actions_to_passthrough_actions (de-interleave)
# ---------------------------------------------------------------------------------------------------
def test_actions_to_passthrough_actions_deinterleaves_and_slices():
    adapter = _adapter({"body": (18, 22)})
    actions = torch.zeros(2, 22)
    actions[..., 14:18] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    actions[..., 18:22] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = adapter.actions_to_passthrough_actions(actions)
    assert set(out) == {"left", "right", "body"}
    assert torch.allclose(out["left"], torch.tensor([0.1, 0.2]).expand(2, 2))
    assert torch.allclose(out["right"], torch.tensor([0.3, 0.4]).expand(2, 2))
    assert torch.allclose(out["body"], torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4))


def test_actions_to_passthrough_actions_rejects_wrong_last_dim():
    with pytest.raises(AssertionError, match="actions last dim must be"):
        _adapter().actions_to_passthrough_actions(torch.zeros(2, 17))


def test_passthrough_round_trip():
    # target -> action -> passthrough recovers the per-arm hand actions and extra channels.
    adapter = _adapter({"body": (18, 22)})
    target = pose_math.make_pose(torch.zeros(3), torch.eye(3))
    passthrough = {
        "left": torch.tensor([0.1, 0.2]),
        "right": torch.tensor([0.3, 0.4]),
        "body": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    action = adapter.target_eef_pose_to_action({"left": target, "right": target}, passthrough)
    recovered = adapter.actions_to_passthrough_actions(action.unsqueeze(0))
    assert torch.allclose(recovered["left"][0], passthrough["left"])
    assert torch.allclose(recovered["right"][0], passthrough["right"])
    assert torch.allclose(recovered["body"][0], passthrough["body"])


# ---------------------------------------------------------------------------------------------------
# quaternion canonicalization + action noise (exercised through the public action encoder)
# ---------------------------------------------------------------------------------------------------
def test_canonicalize_quat_flips_negative_w():
    # A z-rotation whose recovered quaternion has w < 0 (quat_from_matrix's z-branch preserves the
    # sign here; a 90-degree axis rotation would be a degenerate tie and is avoided).
    quat = torch.tensor([0.0, 0.0, 0.9, -0.4359])
    quat = quat / quat.norm()
    target = pose_math.make_pose(torch.zeros(3), pose_math.matrix_from_quat(quat))
    passthrough = {"left": torch.tensor([0.0, 0.0]), "right": torch.tensor([0.0, 0.0])}

    # The left quaternion occupies action[3:7] (xyzw), so its w component is action[6].
    left_quat_canon = _adapter(canonicalize=True).target_eef_pose_to_action(
        {"left": target, "right": target}, passthrough
    )[3:7]
    left_quat_raw = _adapter(canonicalize=False).target_eef_pose_to_action(
        {"left": target, "right": target}, passthrough
    )[3:7]
    assert left_quat_canon[3] >= 0.0  # canonicalized to w >= 0
    assert left_quat_raw[3] < 0.0  # un-canonicalized keeps the negative-w sign
    # Both represent the same rotation: canonicalization only flips the overall sign.
    assert torch.allclose(left_quat_canon, -left_quat_raw, atol=1e-5)


def test_action_noise_zero_is_noop():
    adapter = _adapter()
    target = pose_math.make_pose(torch.tensor([0.1, 0.2, 0.3]), torch.eye(3))
    passthrough = {"left": torch.tensor([0.0, 0.0]), "right": torch.tensor([0.0, 0.0])}
    no_noise = adapter.target_eef_pose_to_action({"left": target, "right": target}, passthrough, action_noise_dict=None)
    zero_noise = adapter.target_eef_pose_to_action(
        {"left": target, "right": target}, passthrough, action_noise_dict={"left": 0.0, "right": 0.0}
    )
    assert torch.equal(no_noise, zero_noise)


# ---------------------------------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------------------------------
def test_from_dict_full():
    data = {
        "name": "g1",
        "description": "g1 humanoid",
        "eefs": {
            "left": {
                "pose_obs_keys": {"pos": "left_eef_pos", "quat": "left_eef_quat"},
                "gripper_action_indices": [0, 1],
            },
            "right": {
                "pose_obs_keys": {"pos": "right_eef_pos", "quat": "right_eef_quat"},
                "gripper_action_indices": [2, 3],
            },
        },
        "action_layout": {
            "left_pose_slice": [0, 7],
            "right_pose_slice": [7, 14],
            "hand_joints_slice": [14, 18],
            "canonicalize_quat": False,
            "passthrough_channels": {"body": [18, 22]},
        },
    }
    adapter = AbsolutePoseWholeBodyBimanualAdapter.from_dict(data)
    assert adapter.name == "g1"
    assert adapter.left.gripper_action_indices == (0, 1)
    assert adapter.right.gripper_action_indices == (2, 3)
    assert adapter.left_pose_slice == (0, 7)
    assert adapter.hand_joints_slice == (14, 18)
    assert adapter.canonicalize_quat is False
    assert adapter.passthrough_channels == {"body": (18, 22)}
    assert adapter.action_dim == 22


def test_from_dict_defaults():
    data = {
        "name": "g1",
        "eefs": {
            "left": {
                "pose_obs_keys": {"pos": "left_eef_pos", "quat": "left_eef_quat"},
                "gripper_action_indices": [0, 1],
            },
            "right": {
                "pose_obs_keys": {"pos": "right_eef_pos", "quat": "right_eef_quat"},
                "gripper_action_indices": [2, 3],
            },
        },
        "action_layout": {
            "left_pose_slice": [0, 7],
            "right_pose_slice": [7, 14],
            "hand_joints_slice": [14, 18],
        },
    }
    adapter = AbsolutePoseWholeBodyBimanualAdapter.from_dict(data)
    assert adapter.description == ""
    assert adapter.obs_group == "policy"
    assert adapter.canonicalize_quat is True
    assert adapter.passthrough_channels == {}
