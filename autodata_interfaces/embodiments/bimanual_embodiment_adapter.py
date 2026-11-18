# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import torch
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_interfaces.embodiments.embodiment_types import ActionTargetFrame, NamedJointTrajectory, PoseObsKeys
from autodata_utils import pose_math


@dataclass(frozen=True)
class BimanualEefConfig:
    """Per-EEF configuration for a bimanual embodiment.

    Args:
        name: Name of this EEF.
        pose_obs_keys: Observation-buffer keys for this EEF's pose state.
        gripper_action_indices: Indices into the interleaved hand-joints
            slice of the env action vector. Length equals this EEF's gripper
            action dim.
        joint_names: Positional passthrough joint names in local channel order.
        passthrough_presets: Named positional passthrough values in the same local joint order.
    """

    name: str
    pose_obs_keys: PoseObsKeys
    gripper_action_indices: tuple[int, ...]
    joint_names: tuple[str, ...] = ()
    passthrough_presets: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check local hand-joint ordering and freeze the named passthrough presets."""
        assert not self.joint_names or len(self.joint_names) == len(
            self.gripper_action_indices
        ), "joint_names must align with gripper_action_indices"
        assert len(set(self.joint_names)) == len(self.joint_names), "joint_names must be unique"
        if self.passthrough_presets:
            assert len(self.joint_names) == len(
                self.gripper_action_indices
            ), "passthrough presets require joint_names aligned with gripper_action_indices"

        presets: dict[str, tuple[float, ...]] = {}
        for preset_name, preset_values in self.passthrough_presets.items():
            assert isinstance(preset_name, str) and preset_name, "passthrough preset names must be non-empty strings"
            values = tuple(float(value) for value in preset_values)
            assert len(values) == len(self.gripper_action_indices), (
                f"passthrough preset {preset_name!r} must contain {len(self.gripper_action_indices)} values "
                f"in joint_names order, got {len(values)}"
            )
            assert all(
                math.isfinite(value) for value in values
            ), f"passthrough preset {preset_name!r} values must be finite"
            presets[preset_name] = values
        object.__setattr__(self, "passthrough_presets", MappingProxyType(presets))


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

    def get_eef_joint_names(self, eef_name: str) -> tuple[str, ...]:
        """Return the named hand joints commanded by one EEF passthrough channel."""

        by_name = {eef.name: eef.joint_names for eef in (self.left, self.right)}
        assert eef_name in by_name, f"unknown end-effector {eef_name!r}"
        return by_name[eef_name]

    def get_eef_passthrough_preset(self, eef_name: str, preset_name: str) -> tuple[float, ...]:
        """Return a named hand preset in the EEF's configured joint order."""

        by_name = {eef.name: eef for eef in (self.left, self.right)}
        assert eef_name in by_name, f"unknown end-effector {eef_name!r}"
        assert isinstance(preset_name, str) and preset_name, "preset_name must be a non-empty string"
        presets = by_name[eef_name].passthrough_presets
        if preset_name not in presets:
            raise KeyError(
                f"unknown passthrough preset {preset_name!r} for end-effector {eef_name!r}; "
                f"available presets: {sorted(presets)}"
            )
        return presets[preset_name]

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
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert per-EEF target pose and passthrough actions to env action. Implemented by subclasses."""

    @abstractmethod
    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract passthrough-action slices from a sequence of env actions. Implemented by subclasses."""


@dataclass(kw_only=True)
class AbsolutePoseWholeBodyBimanualAdapter(BimanualEmbodimentAdapter):
    """Bimanual adapter for absolute-pose whole-body IK control.

    Action layout (concatenated):
    ``[left_pos(3), left_quat(4), right_pos(3), right_quat(4), interleaved_hand_joints, *extra]``.
    Pose is absolute and tracked by a whole-body IK controller. Hand joints
    are interleaved by URDF order; which indices belong to which arm is
    captured in :attr:`BimanualEefConfig.gripper_action_indices`.

    Beyond the two eef (hand) passthrough channels, the embodiment may declare any number of
    additional non-eef passthrough channels via :attr:`passthrough_channels` (e.g. a mobile
    base / locomotion command). Each is a named, contiguous slice of the action vector that is copied verbatim from the source demo, exactly
    like the gripper.

    Args:
        left_pose_slice: Half-open ``(start, end)`` range in the action vector
            for the left ``[pos(3), quat(4)]`` block. Span must equal 7.
        right_pose_slice: ``(start, end)`` for the right ``[pos(3), quat(4)]``
            block. Must immediately follow ``left_pose_slice``.
        hand_joints_slice: ``(start, end)`` for the interleaved hand-joints
            block. Must immediately follow ``right_pose_slice``.
        canonicalize_quat: If True, flip quaternions to ``w >= 0`` before
            packing into the action. Prevents sign drift in recorded actions.
        passthrough_channels: Map ``channel_name → (start, end)`` of additional non-eef
            passthrough channels, each a contiguous slice of the action vector.
            Channel names must be distinct from the eef names. Empty by default.
    """

    left_pose_slice: tuple[int, int]
    right_pose_slice: tuple[int, int]
    hand_joints_slice: tuple[int, int]
    canonicalize_quat: bool = True
    passthrough_channels: dict[str, tuple[int, int]] = field(default_factory=dict)
    planning_model_action_term: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for label, pose_slice in (("left", self.left_pose_slice), ("right", self.right_pose_slice)):
            span = pose_slice[1] - pose_slice[0]
            assert span == 7, f"{label}_pose_slice must span 7 dims, got {span}"
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

        # Validate the extra (non-eef) passthrough channels: distinct names, valid non-overlapping
        # slices outside the contiguous pose+hand region.
        eef_names = {self.left.name, self.right.name}
        occupied_ranges: list[tuple[int, int]] = [(self.left_pose_slice[0], self.hand_joints_slice[1])]
        for channel_name, channel_slice in self.passthrough_channels.items():
            assert channel_name not in eef_names, f"passthrough channel {channel_name!r} collides with an eef name"
            assert (
                len(channel_slice) == 2 and channel_slice[1] > channel_slice[0] >= 0
            ), f"passthrough channel {channel_name!r} has invalid slice {channel_slice}"
            for occupied_range in occupied_ranges:
                assert channel_slice[1] <= occupied_range[0] or channel_slice[0] >= occupied_range[1], (
                    f"passthrough channel {channel_name!r} slice {tuple(channel_slice)} "
                    f"overlaps action region {occupied_range}"
                )
            occupied_ranges.append(tuple(channel_slice))

    def get_passthrough_actions(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Recover positional hand commands from the live articulation state.

        Args:
            env_ids: Environment indices, or None for all environments.

        Returns:
            Hand-action tensors per end effector, or None if a required channel is unavailable.
        """

        eefs = (self.left, self.right)
        if self.passthrough_channels or any(not eef.joint_names for eef in eefs):
            return None
        joint_names = tuple(self.get_joint_names())
        missing = {name for eef in eefs for name in eef.joint_names} - set(joint_names)
        assert not missing, f"passthrough joints are absent from the live articulation: {sorted(missing)}"
        joint_index = {name: index for index, name in enumerate(joint_names)}
        positions = self.get_joint_positions(env_ids)
        return {eef.name: positions[:, [joint_index[name] for name in eef.joint_names]].clone() for eef in eefs}

    def _planning_action_term(self) -> Any:
        """Resolve the configured planning action from the bound environment."""
        assert self.env is not None, "Call bind_env(env) before resolving the planning model."
        terms = getattr(self.env.action_manager, "_terms", {})
        assert (
            self.planning_model_action_term in terms
        ), f"unknown planning action term {self.planning_model_action_term!r}"
        return terms[self.planning_model_action_term]

    def get_action_target_frames(self) -> tuple[ActionTargetFrame, ...]:
        """Match end effectors to the action's controlled frames.

        Returns:
            Ordered end-effector/frame pairs, or an empty tuple without a planning action.
        """
        if self.planning_model_action_term is None:
            return ()
        eef_names = self.get_eef_names()
        frame_names = self._planning_action_term().get_action_target_frame_names()
        assert len(frame_names) == len(eef_names), "planning action frame count must match the embodiment EEF count"
        return tuple(ActionTargetFrame(eef, frame) for eef, frame in zip(eef_names, frame_names, strict=True))

    def get_planning_model_path(self) -> Path | None:
        """Locate the URDF exported for the live Pink controller.

        Returns:
            Existing URDF path, or None without a planning action.
        """

        if self.planning_model_action_term is None:
            return None
        controller = self._planning_action_term().cfg.controller
        if controller.urdf_path:
            path = Path(controller.urdf_path).expanduser()
        else:
            assert controller.usd_path and controller.urdf_output_dir
            path = Path(controller.urdf_output_dir).expanduser() / "urdf" / f"{Path(controller.usd_path).stem}.urdf"
        path = path.resolve()
        assert path.is_file(), f"controller URDF file does not exist: {path}"
        return path

    def solve_joint_trajectory(
        self,
        target_eef_poses: Mapping[str, torch.Tensor],
        joint_seed_trajectories: Mapping[str, NamedJointTrajectory],
        initial_joint_positions: torch.Tensor,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        joint_state_is_valid: Callable[[tuple[str, ...], torch.Tensor, int], bool] | None = None,
        env_id: int = 0,
    ) -> NamedJointTrajectory | None:
        """Solve synchronized targets through the configured planning action.

        Args:
            target_eef_poses: Target poses per end effector with shape (T, 4, 4) [m].
            joint_seed_trajectories: Named planner joint proposals per end effector.
            initial_joint_positions: Initial state in live articulation joint order [rad, m].
            position_tolerance_m: Required tool-position accuracy [m].
            rotation_tolerance_rad: Required tool-rotation accuracy [rad].
            joint_state_is_valid: Optional feasibility check for each joint sample.
            env_id: Environment controller to use.

        Returns:
            Named joint trajectory, or None without a planning action.
        """
        if self.planning_model_action_term is None:
            return None
        frames = self.get_action_target_frames()
        assert set(target_eef_poses) == set(joint_seed_trajectories) == {frame.eef_name for frame in frames}
        return self._planning_action_term().solve_joint_trajectory(
            {frame.frame_name: target_eef_poses[frame.eef_name] for frame in frames},
            {frame.frame_name: joint_seed_trajectories[frame.eef_name] for frame in frames},
            NamedJointTrajectory(tuple(self.get_joint_names()), initial_joint_positions.unsqueeze(0)),
            position_tolerance_m,
            rotation_tolerance_rad,
            joint_state_is_valid,
            env_id,
        )

    def set_joint_posture_target(
        self,
        joint_names: tuple[str, ...],
        joint_positions: torch.Tensor,
        env_id: int = 0,
    ) -> None:
        """Set the planning action's posture reference.

        Args:
            joint_names: Names aligned with the position values.
            joint_positions: Posture target with shape (J,) [rad, m].
            env_id: Environment controller to update.
        """
        if self.planning_model_action_term is not None:
            self._planning_action_term().set_joint_posture_target(joint_names, joint_positions, env_id)

    def reset_joint_posture_target(self, env_id: int = 0) -> None:
        """Restore the planning action's default posture reference.

        Args:
            env_id: Environment controller to update.
        """
        if self.planning_model_action_term is not None:
            self._planning_action_term().reset_joint_posture_target(env_id)

    def get_planning_root_pose(
        self,
        root_pose_source_frame_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return the planning-root pose, using the live controller frame when available.

        Args:
            root_pose_source_frame_name: Robot frame used as the planning root.
            env_ids: Environment indices, or None for all environments.

        Returns:
            Environment-relative root transforms with shape (N, 4, 4) [m].
        """

        if self.planning_model_action_term is None:
            return super().get_planning_root_pose(root_pose_source_frame_name, env_ids)
        action_term = self._planning_action_term()
        assert action_term.cfg.controller.base_link_name == root_pose_source_frame_name
        return action_term.get_base_link_pose(env_ids)

    def passthrough_actions_to_joint_positions(
        self,
        passthrough_action_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map hand-action channels to named articulation joint positions.

        Args:
            passthrough_action_dict: Unbatched hand-action tensors keyed by end effector.

        Returns:
            Scalar joint-position tensors keyed by articulation joint name [rad, m].
        """

        result: dict[str, torch.Tensor] = {}
        for eef in (self.left, self.right):
            if not eef.joint_names or eef.name not in passthrough_action_dict:
                continue
            values = passthrough_action_dict[eef.name]
            assert values.shape == (len(eef.joint_names),)
            result.update(zip(eef.joint_names, values, strict=True))
        return result

    def joint_positions_to_passthrough_actions(
        self,
        joint_position_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Update hand-action channels from named joint positions.

        Args:
            joint_position_dict: Scalar joint positions keyed by name [rad, m].
            passthrough_action_dict: Action tensors providing the channel shapes and defaults.

        Returns:
            Copied action channels with the supplied hand-joint positions applied.
        """

        result = {name: value.clone() for name, value in passthrough_action_dict.items()}
        for eef in (self.left, self.right):
            selected = tuple(name in joint_position_dict for name in eef.joint_names)
            if not any(selected):
                continue
            assert all(selected), f"planned joint state only partially covers the {eef.name!r} hand"
            assert eef.name in result, f"passthrough actions are missing the {eef.name!r} hand channel"
            template = result[eef.name]
            assert template.shape == (len(eef.joint_names),)
            values = tuple(joint_position_dict[name] for name in eef.joint_names)
            assert all(
                isinstance(value, torch.Tensor) and value.numel() == 1 for value in values
            ), "planned hand joint positions must be scalar tensors"
            result[eef.name] = torch.stack(
                [value.reshape(()).to(device=template.device, dtype=template.dtype) for value in values]
            )
        return result

    @property
    def action_dim(self) -> int:
        """Total width of the env action vector (pose + hands + any extra passthrough channels)."""
        channel_ends = [channel_slice[1] for channel_slice in self.passthrough_channels.values()]
        return max([self.hand_joints_slice[1], *channel_ends])

    def metric_context(self) -> dict[str, Any]:
        """``left_pose_slice`` and ``right_pose_slice`` of the action vector hold the hands' target poses.

        Each holds a position, then a quaternion. These entries are not clipped.
        """
        pose_action = {
            self.left.name: {"action_slice": list(self.left_pose_slice), "bound": None, "absolute": True},
            self.right.name: {"action_slice": list(self.right_pose_slice), "bound": None, "absolute": True},
        }
        return {**super().metric_context(), "pose_action": pose_action}

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
        for eef, pose_slice in ((self.left, self.left_pose_slice), (self.right, self.right_pose_slice)):
            pos = action[:, pose_slice[0] : pose_slice[0] + 3]
            quat = action[:, pose_slice[0] + 3 : pose_slice[1]]
            rot = pose_math.matrix_from_quat(quat)
            result[eef.name] = pose_math.make_pose(pos, rot)
        return result

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,  # noqa: ARG002
    ) -> torch.Tensor:
        """Convert per-arm target pose and passthrough actions to env action.

        Args:
            target_eef_pose_dict: ``{eef_name: target_pose(4, 4)}`` for both arms.
            passthrough_action_dict: ``{channel_name: tensor}`` for both arms and any declared extra channels.
            action_noise_dict: Optional per-arm noise scales. Applied to pose
                and quat slices independently per arm; passthrough actions are not
                noisified.
            env_id: Present for ABC parity. Unused (absolute-pose encoding
                does not read env state).

        Returns:
            Env action tensor of shape (action_dim,).
        """
        eef_keys = {self.left.name, self.right.name}
        expected_keys = eef_keys | set(self.passthrough_channels)
        assert (
            set(target_eef_pose_dict) == eef_keys
        ), f"target_eef_pose_dict must have exactly {eef_keys}, got {set(target_eef_pose_dict)}"
        assert (
            set(passthrough_action_dict) == expected_keys
        ), f"passthrough_action_dict must have exactly {expected_keys}, got {set(passthrough_action_dict)}"
        left_pos, left_quat = self._encode_pose(target_eef_pose_dict[self.left.name])
        right_pos, right_quat = self._encode_pose(target_eef_pose_dict[self.right.name])
        if action_noise_dict is not None:
            left_pos, left_quat = self._add_noise(left_pos, left_quat, action_noise_dict.get(self.left.name, 0.0))
            right_pos, right_quat = self._add_noise(right_pos, right_quat, action_noise_dict.get(self.right.name, 0.0))
        hand_joints = self._build_interleaved_hand_joints(passthrough_action_dict)

        action = torch.zeros(self.action_dim, dtype=left_pos.dtype, device=left_pos.device)
        action[self.left_pose_slice[0] : self.left_pose_slice[1]] = torch.cat([left_pos, left_quat], dim=0)
        action[self.right_pose_slice[0] : self.right_pose_slice[1]] = torch.cat([right_pos, right_quat], dim=0)
        action[self.hand_joints_slice[0] : self.hand_joints_slice[1]] = hand_joints
        for channel_name, channel_slice in self.passthrough_channels.items():
            action[channel_slice[0] : channel_slice[1]] = passthrough_action_dict[channel_name]
        return action

    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract all passthrough channels from a sequence of env actions.

        De-interleaves the hand-joints slice into per-arm gripper actions and slices out every
        declared extra (non-eef) channel. Leading batch dims are preserved.

        Args:
            actions: Action tensor whose last dim is ``action_dim``.

        Returns:
            Dictionary keyed by channel name: the two eef hand channels (each tensor's last dim
            equals that arm's ``len(gripper_action_indices)``) plus every entry in
            :attr:`passthrough_channels` (each the width of its slice).
        """
        assert (
            actions.shape[-1] == self.action_dim
        ), f"actions last dim must be {self.action_dim}, got {actions.shape[-1]}"
        hand_joints = actions[..., self.hand_joints_slice[0] : self.hand_joints_slice[1]]
        result = {
            self.left.name: self._gather_indices(hand_joints, self.left.gripper_action_indices),
            self.right.name: self._gather_indices(hand_joints, self.right.gripper_action_indices),
        }
        for channel_name, channel_slice in self.passthrough_channels.items():
            result[channel_name] = actions[..., channel_slice[0] : channel_slice[1]]
        return result

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

    def _build_interleaved_hand_joints(self, passthrough_action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Place per-arm gripper actions at their configured indices in the hand-joints block.

        Reads only the two eef channels from ``passthrough_action_dict``; extra non-eef channels
        are placed separately by :meth:`target_eef_pose_to_action`.
        """
        width = self.hand_joints_slice[1] - self.hand_joints_slice[0]
        left = passthrough_action_dict[self.left.name]
        right = passthrough_action_dict[self.right.name]
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
                joint_names: [<str>, ...]       # optional
                passthrough_presets:             # optional
                  <name>: [<float>, ...]
              right:
                pose_obs_keys: {pos: <str>, quat: <str>}
                gripper_action_indices: [<int>, ...]
                joint_names: [<str>, ...]       # optional
                passthrough_presets:             # optional
                  <name>: [<float>, ...]
            action_layout:
              left_pose_slice: [<start>, <end>]
              right_pose_slice: [<start>, <end>]
              hand_joints_slice: [<start>, <end>]
              canonicalize_quat: <bool>   # optional, default true
              passthrough_channels:       # optional, non-eef channels copied from the action
                <name>: [<start>, <end>]
        """
        eefs = data["eefs"]
        layout = data["action_layout"]
        passthrough_channels = {
            channel_name: tuple(channel_slice)
            for channel_name, channel_slice in layout.get("passthrough_channels", {}).items()
        }
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
            passthrough_channels=passthrough_channels,
            planning_model_action_term=data.get("planning_model_action_term"),
        )

    @staticmethod
    def _eef_from_dict(name: str, data: dict[str, Any]) -> BimanualEefConfig:
        """Build one end-effector configuration from its YAML fields."""
        pose_keys = data["pose_obs_keys"]
        return BimanualEefConfig(
            name=name,
            pose_obs_keys=PoseObsKeys(pos=pose_keys["pos"], quat=pose_keys["quat"]),
            gripper_action_indices=tuple(int(i) for i in data["gripper_action_indices"]),
            joint_names=tuple(data.get("joint_names", ())),
            passthrough_presets=data.get("passthrough_presets", {}),
        )
