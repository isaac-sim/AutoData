# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual embodiment adapters.

Defines the morphology ABC :class:`BimanualEmbodimentAdapter` and one
concrete subclass :class:`AbsolutePoseWholeBodyBimanualAdapter` whose
action vector is
``[left_pos(3), left_quat(4), right_pos(3), right_quat(4), interleaved_hand_joints]``.

Covers both one-base bimanual (humanoid with whole-body IK) and two-base
bimanual (two arms with independent IK).
"""

from __future__ import annotations

import torch
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaac_autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from isaac_autodata_interfaces.embodiments.embodiment_types import PoseObsKeys
from isaac_autodata_utils import pose_math


@dataclass(frozen=True)
class BimanualEefConfig:
    """Per-EEF configuration for a bimanual embodiment.

    Args:
        name: Name of this EEF.
        pose_obs_keys: Observation-buffer keys for this EEF's pose state.
        gripper_action_indices: Indices into the interleaved hand-joints
            slice of the env action vector. Length equals this EEF's gripper
            action dim.
    """

    name: str
    pose_obs_keys: PoseObsKeys
    gripper_action_indices: tuple[int, ...]


@dataclass(kw_only=True)
class BimanualEmbodimentAdapter(EmbodimentAdapter):
    """Morphology ABC for two-EEF embodiments.

    Subclasses inherit the two-EEF invariant and pose-reading logic and
    implement the three action-encoding methods.

    Args:
        name: Identifier used in YAML and registry lookups.
        description: Human-readable description.
        left: Left-EEF configuration.
        right: Right-EEF configuration.
        obs_group: Observation-buffer group name under which the per-EEF
            pose obs keys live. Defaults to ``"policy"``.
    """

    name: str
    description: str = ""
    left: BimanualEefConfig
    right: BimanualEefConfig
    obs_group: str = "policy"

    def __post_init__(self) -> None:
        assert self.name, "name must be a non-empty string"
        assert isinstance(
            self.left, BimanualEefConfig
        ), f"left must be a BimanualEefConfig, got {type(self.left).__name__}"
        assert isinstance(
            self.right, BimanualEefConfig
        ), f"right must be a BimanualEefConfig, got {type(self.right).__name__}"
        assert (
            self.left.name != self.right.name
        ), f"left and right EEFs must have distinct names; both are {self.left.name!r}"
        overlap = set(self.left.gripper_action_indices) & set(self.right.gripper_action_indices)
        assert not overlap, f"left and right gripper_action_indices overlap on {sorted(overlap)}"
        assert self.obs_group, "obs_group must be a non-empty string"

    def get_eef_names(self) -> tuple[str, ...]:
        return (self.left.name, self.right.name)

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        obs = self.env.obs_buf[self.obs_group]
        return {eef.name: self._read_eef_pose(eef, obs, index) for eef in (self.left, self.right)}

    @staticmethod
    def _read_eef_pose(eef: BimanualEefConfig, obs: dict, index) -> torch.Tensor:
        pos = obs[eef.pose_obs_keys.pos][index]
        rot = pose_math.matrix_from_quat(obs[eef.pose_obs_keys.quat][index])
        return pose_math.make_pose(pos, rot)

    @abstractmethod
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert env action to per-EEF target pose. Implemented by concrete subclasses."""

    @abstractmethod
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert per-EEF target pose and gripper to env action. Implemented by concrete subclasses."""

    @abstractmethod
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper-actuation slices from a sequence of env actions. Implemented by concrete subclasses."""


@dataclass(kw_only=True)
class AbsolutePoseWholeBodyBimanualAdapter(BimanualEmbodimentAdapter):
    """Bimanual adapter for absolute-pose whole-body IK control.

    Action layout (concatenated):
    ``[left_pos(3), left_quat(4), right_pos(3), right_quat(4), interleaved_hand_joints]``.
    Pose is absolute and tracked by a whole-body IK controller. Hand joints
    are interleaved by URDF order; which indices belong to which arm is
    captured in :attr:`BimanualEefConfig.gripper_action_indices`.

    Args:
        left_pose_slice: Half-open ``(start, end)`` range in the action vector
            for the left ``[pos(3), quat(4)]`` block. Span must equal 7.
        right_pose_slice: ``(start, end)`` for the right ``[pos(3), quat(4)]``
            block. Must immediately follow ``left_pose_slice``.
        hand_joints_slice: ``(start, end)`` for the interleaved hand-joints
            block. Must immediately follow ``right_pose_slice``.
        canonicalize_quat: If True, flip quaternions to ``w >= 0`` before
            packing into the action. Prevents sign drift in recorded actions.
    """

    left_pose_slice: tuple[int, int]
    right_pose_slice: tuple[int, int]
    hand_joints_slice: tuple[int, int]
    canonicalize_quat: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for label, sl in (("left", self.left_pose_slice), ("right", self.right_pose_slice)):
            assert sl[1] - sl[0] == 7, f"{label}_pose_slice must span 7 dims, got {sl[1] - sl[0]}"
        assert self.right_pose_slice[0] == self.left_pose_slice[1], (
            "right_pose_slice must immediately follow left_pose_slice; "
            f"got left=({self.left_pose_slice[0]}, {self.left_pose_slice[1]}), "
            f"right=({self.right_pose_slice[0]}, {self.right_pose_slice[1]})"
        )
        assert (
            self.hand_joints_slice[0] == self.right_pose_slice[1]
        ), "hand_joints_slice must immediately follow right_pose_slice"
        hand_joints_width = self.hand_joints_slice[1] - self.hand_joints_slice[0]
        all_indices = set(self.left.gripper_action_indices) | set(self.right.gripper_action_indices)
        if all_indices:
            assert max(all_indices) < hand_joints_width, (
                f"gripper_action_indices reference position {max(all_indices)} but "
                f"hand-joints slice only has width {hand_joints_width}"
            )

    @property
    def action_dim(self) -> int:
        """Total width of the env action vector."""
        return self.hand_joints_slice[1]

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert env action to target EEF poses for both arms.

        Args:
            action: Env action of shape (num_envs, action_dim).

        Returns:
            Dictionary with one key per EEF, each value of shape
            (num_envs, 4, 4).
        """
        assert (
            action.dim() == 2 and action.shape[-1] == self.action_dim
        ), f"action shape must be (num_envs, {self.action_dim}), got {tuple(action.shape)}"
        result: dict[str, torch.Tensor] = {}
        for eef, sl in ((self.left, self.left_pose_slice), (self.right, self.right_pose_slice)):
            pos = action[:, sl[0] : sl[0] + 3]
            quat = action[:, sl[0] + 3 : sl[1]]
            rot = pose_math.matrix_from_quat(quat)
            result[eef.name] = pose_math.make_pose(pos, rot)
        return result

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,  # noqa: ARG002
    ) -> torch.Tensor:
        """Convert per-arm target pose and gripper to env action.

        Args:
            target_eef_pose_dict: ``{eef_name: target_pose(4, 4)}`` for both arms.
            gripper_action_dict: ``{eef_name: gripper_tensor}``; each tensor is
                shape ``(len(gripper_action_indices),)`` for that arm.
            action_noise_dict: Optional per-arm noise scales. Applied to pose
                and quat slices independently per arm; gripper joints are not
                noisified.
            env_id: Present for ABC parity. Unused (absolute-pose encoding
                does not read env state).

        Returns:
            Env action tensor of shape (action_dim,).
        """
        expected_keys = {self.left.name, self.right.name}
        assert (
            set(target_eef_pose_dict) == expected_keys
        ), f"target_eef_pose_dict must have exactly {expected_keys}, got {set(target_eef_pose_dict)}"
        assert (
            set(gripper_action_dict) == expected_keys
        ), f"gripper_action_dict must have exactly {expected_keys}, got {set(gripper_action_dict)}"
        left_pos, left_quat = self._encode_pose(target_eef_pose_dict[self.left.name])
        right_pos, right_quat = self._encode_pose(target_eef_pose_dict[self.right.name])
        if action_noise_dict is not None:
            left_pos, left_quat = self._add_noise(left_pos, left_quat, action_noise_dict.get(self.left.name, 0.0))
            right_pos, right_quat = self._add_noise(right_pos, right_quat, action_noise_dict.get(self.right.name, 0.0))
        hand_joints = self._build_interleaved_hand_joints(gripper_action_dict)
        return torch.cat([left_pos, left_quat, right_pos, right_quat, hand_joints], dim=0)

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """De-interleave the hand-joints slice into per-arm gripper actions.

        Args:
            actions: Action tensor whose last dim is ``action_dim``. Leading
                batch dims are preserved.

        Returns:
            Dictionary ``{left_name: left_gripper, right_name: right_gripper}``
            where each tensor's last dim equals that arm's
            ``len(gripper_action_indices)``.
        """
        assert (
            actions.shape[-1] == self.action_dim
        ), f"actions last dim must be {self.action_dim}, got {actions.shape[-1]}"
        hand_joints = actions[..., self.hand_joints_slice[0] : self.hand_joints_slice[1]]
        return {
            self.left.name: self._gather_indices(hand_joints, self.left.gripper_action_indices),
            self.right.name: self._gather_indices(hand_joints, self.right.gripper_action_indices),
        }

    def _encode_pose(self, target_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose (4, 4) target pose into (pos[3], quat[4]); canonicalize quat sign if configured."""
        assert target_pose.shape == (4, 4), f"target pose must be (4, 4), got {tuple(target_pose.shape)}"
        pos, rot = pose_math.unmake_pose(target_pose)
        quat = pose_math.quat_from_matrix(rot)
        if self.canonicalize_quat:
            quat = pose_math.quat_unique(quat)
        return pos, quat

    @staticmethod
    def _add_noise(pos: torch.Tensor, quat: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise per scalar; pos and quat are noisified independently."""
        if scale <= 0.0:
            return pos, quat
        return pos + scale * torch.randn_like(pos), quat + scale * torch.randn_like(quat)

    def _build_interleaved_hand_joints(self, gripper_action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Place per-arm gripper actions at their configured indices in the hand-joints block."""
        width = self.hand_joints_slice[1] - self.hand_joints_slice[0]
        left = gripper_action_dict[self.left.name]
        right = gripper_action_dict[self.right.name]
        assert left.shape == (
            len(self.left.gripper_action_indices),
        ), f"left gripper action must be shape ({len(self.left.gripper_action_indices)},), got {tuple(left.shape)}"
        assert right.shape == (
            len(self.right.gripper_action_indices),
        ), f"right gripper action must be shape ({len(self.right.gripper_action_indices)},), got {tuple(right.shape)}"
        out = torch.zeros(width, dtype=left.dtype, device=left.device)
        out[list(self.left.gripper_action_indices)] = left
        out[list(self.right.gripper_action_indices)] = right
        return out

    @staticmethod
    def _gather_indices(hand_joints: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        """Select named indices from the trailing dim of ``hand_joints``."""
        idx_tensor = torch.tensor(list(indices), dtype=torch.long, device=hand_joints.device)
        return torch.index_select(hand_joints, dim=-1, index=idx_tensor)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbsolutePoseWholeBodyBimanualAdapter:
        """Build an adapter instance from a parsed config dict.

        Expected schema::

            name: <str>
            description: <str>          # optional
            obs_group: <str>            # optional, default "policy"
            eefs:
              left:
                pose_obs_keys: {pos: <str>, quat: <str>}
                gripper_action_indices: [<int>, ...]
              right:
                pose_obs_keys: {pos: <str>, quat: <str>}
                gripper_action_indices: [<int>, ...]
            action_layout:
              left_pose_slice: [<start>, <end>]
              right_pose_slice: [<start>, <end>]
              hand_joints_slice: [<start>, <end>]
              canonicalize_quat: <bool>   # optional, default true
        """
        eefs = data["eefs"]
        layout = data["action_layout"]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            left=cls._eef_from_dict("left", eefs["left"]),
            right=cls._eef_from_dict("right", eefs["right"]),
            obs_group=str(data.get("obs_group", "policy")),
            left_pose_slice=tuple(layout["left_pose_slice"]),
            right_pose_slice=tuple(layout["right_pose_slice"]),
            hand_joints_slice=tuple(layout["hand_joints_slice"]),
            canonicalize_quat=bool(layout.get("canonicalize_quat", True)),
        )

    @staticmethod
    def _eef_from_dict(name: str, data: dict[str, Any]) -> BimanualEefConfig:
        pose_keys = data["pose_obs_keys"]
        return BimanualEefConfig(
            name=name,
            pose_obs_keys=PoseObsKeys(pos=pose_keys["pos"], quat=pose_keys["quat"]),
            gripper_action_indices=tuple(int(i) for i in data["gripper_action_indices"]),
        )
