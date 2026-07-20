# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-arm embodiment adapters.

Defines the morphology ABC :class:`SingleArmEmbodimentAdapter` and one
concrete subclass :class:`DeltaPoseIKSingleArmAdapter` whose action vector
is ``[delta_pos(3), delta_rot_axis_angle(3), gripper]``.
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

# The delta-pose IK action layout is always 3D Cartesian + 3D compact
# axis-angle. No real upstream embodiment varies these widths.
_DELTA_POSE_ACTION_DIM = 6


@dataclass(kw_only=True)
class SingleArmEmbodimentAdapter(EmbodimentAdapter):
    """Morphology ABC for single-arm fixed-base manipulators.

    Subclasses inherit the one-EEF invariant and pose-reading logic and
    implement the three action-encoding methods.

    Args:
        name: Identifier used in YAML and registry lookups.
        description: Human-readable description.
        eef_name: Name of the single end-effector.
        pose_obs_keys: Observation-buffer keys for the EEF pose state.
        gripper_action_dim: Number of trailing action vector dimensions
            occupied by the gripper actuation.
        obs_group: Observation-buffer group name under which the pose obs
            keys live. Defaults to ``"policy"`` to match Isaac Lab's
            standard observation manager.
        eef_offset: Translation [m] from the robot's kinematic control link
            (the frame a motion planner plans, e.g. the wrist/hand) to the EEF
            frame the pose observation reports, expressed in the control link's
            frame. The adapter reports poses in the control-link frame so the
            whole pipeline (planner targets, recorded skill targets, delta-pose
            actions) shares one frame. Defaults to no offset.
    """

    name: str
    description: str = ""
    eef_name: str
    pose_obs_keys: PoseObsKeys
    gripper_action_dim: int
    obs_group: str = "policy"
    eef_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        assert self.name, "name must be a non-empty string"
        assert self.eef_name, "eef_name must be a non-empty string"
        assert self.gripper_action_dim >= 0, f"gripper_action_dim must be non-negative, got {self.gripper_action_dim}"
        assert isinstance(
            self.pose_obs_keys, PoseObsKeys
        ), f"pose_obs_keys must be a PoseObsKeys instance, got {type(self.pose_obs_keys).__name__}"
        assert self.obs_group, "obs_group must be a non-empty string"
        assert len(self.eef_offset) == 3, f"eef_offset must have 3 elements, got {len(self.eef_offset)}"

    def get_eef_names(self) -> tuple[str, ...]:
        return (self.eef_name,)

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        obs = self.env.obs_buf[self.obs_group]
        # Isaac Lab FrameTransformer observations such as ``target_quat_w`` are already XYZW;
        # ``_w`` identifies the world frame, not the quaternion component order.
        quaternion_xyzw = obs[self.pose_obs_keys.quat][index]
        rot = pose_math.matrix_from_quat(quaternion_xyzw)
        pose = pose_math.make_pose(obs[self.pose_obs_keys.pos][index], rot)
        return {self.eef_name: self._observed_to_control_link(pose)}

    def _observed_to_control_link(self, pose: torch.Tensor) -> torch.Tensor:
        """Shift an observed EEF pose back to the kinematic control-link frame.

        The observed EEF sits :attr:`eef_offset` ahead of the control link, so the control-link
        pose is ``pose @ translate(-eef_offset)``. Identity when no offset is configured.
        """

        if not any(self.eef_offset):
            return pose
        transform = torch.eye(4, dtype=pose.dtype, device=pose.device)
        transform[:3, 3] = -torch.tensor(self.eef_offset, dtype=pose.dtype, device=pose.device)
        return pose @ transform

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
class DeltaPoseIKSingleArmAdapter(SingleArmEmbodimentAdapter):
    """Single-arm adapter for delta-pose IK control.

    Action layout (concatenated):
    ``[delta_pos(3), delta_rot_axis_angle(3), gripper(gripper_action_dim)]``.
    The pose part is the delta from the current EEF pose to the target;
    axis-angle is the compact ``axis * angle`` form. The pose part is
    optionally clamped to ``[-1, 1]`` after noise.

    Args:
        clip_pose_action_to_unit: If True, clamp the pose part of the action
            to ``[-1, 1]`` after adding noise.
    """

    clip_pose_action_to_unit: bool = True

    @property
    def action_dim(self) -> int:
        """Total width of the env action vector."""
        return _DELTA_POSE_ACTION_DIM + self.gripper_action_dim

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert env action to target EEF pose.

        Args:
            action: Env action of shape (num_envs, action_dim).

        Returns:
            Dictionary ``{eef_name: target_pose}`` with target pose of shape
            (num_envs, 4, 4).
        """
        assert (
            action.dim() == 2 and action.shape[-1] == self.action_dim
        ), f"action shape must be (num_envs, {self.action_dim}), got {tuple(action.shape)}"
        # Isaac Lab applies the DifferentialInverseKinematicsAction term's scale *after* receiving
        # raw environment actions. Decode the physical delta rather than treating raw policy units
        # as metres/radians. Environments without a discoverable DIK term retain the legacy unit
        # scale for compatibility.
        pose_action = action[:, :_DELTA_POSE_ACTION_DIM] * self._get_pose_action_scale(
            num_envs=action.shape[0], dtype=action.dtype, device=action.device
        )
        delta_pos = pose_action[:, :3]
        delta_aa = pose_action[:, 3:6]
        curr_pos, curr_rot = pose_math.unmake_pose(self.get_eef_poses(env_ids=None)[self.eef_name])
        target_pos = curr_pos + delta_pos
        delta_rot = pose_math.matrix_from_quat(pose_math.quat_from_axis_angle_vec(delta_aa))
        target_rot = torch.matmul(delta_rot, curr_rot)
        return {self.eef_name: pose_math.make_pose(target_pos, target_rot)}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose and gripper to env action for a single env.

        Args:
            target_eef_pose_dict: ``{eef_name: target_pose}`` with target pose
                of shape (4, 4).
            gripper_action_dict: ``{eef_name: gripper_action}`` of shape
                (gripper_action_dim,).
            action_noise_dict: Optional ``{eef_name: noise_scale}``. Noise is
                applied to the pose part only; the result is clipped if
                :attr:`clip_pose_action_to_unit` is set.
            env_id: Env index used when reading the current pose.

        Returns:
            Env action tensor of shape (action_dim,).
        """
        assert set(target_eef_pose_dict) == {
            self.eef_name
        }, f"target_eef_pose_dict must have exactly one key '{self.eef_name}', got {list(target_eef_pose_dict)}"
        assert set(gripper_action_dict) == {
            self.eef_name
        }, f"gripper_action_dict must have exactly one key '{self.eef_name}', got {list(gripper_action_dict)}"
        target_pose = target_eef_pose_dict[self.eef_name]
        assert target_pose.shape == (4, 4), f"target pose must be (4, 4), got {tuple(target_pose.shape)}"
        target_pos, target_rot = pose_math.unmake_pose(target_pose)
        curr_pos, curr_rot = pose_math.unmake_pose(self.get_eef_poses(env_ids=[env_id])[self.eef_name][0])
        delta_pos = target_pos - curr_pos
        delta_rot = torch.matmul(target_rot, curr_rot.transpose(-1, -2))
        delta_aa = pose_math.axis_angle_from_quat(pose_math.quat_from_matrix(delta_rot))
        pose_action = torch.cat([delta_pos, delta_aa], dim=0)
        # Encode physical deltas back into raw environment-action units. This is required for
        # Franka IK-Rel, whose action term commonly uses scale=0.5.
        pose_action = (
            pose_action
            / self._get_pose_action_scale(
                num_envs=1, dtype=pose_action.dtype, device=pose_action.device, env_id=env_id
            )[0]
        )
        if action_noise_dict is not None:
            scale = action_noise_dict.get(self.eef_name, 0.0)
            if scale > 0.0:
                pose_action = pose_action + scale * torch.randn_like(pose_action)
        if self.clip_pose_action_to_unit:
            pose_action = torch.clamp(pose_action, -1.0, 1.0)
        gripper_action = gripper_action_dict[self.eef_name]
        assert gripper_action.shape == (
            self.gripper_action_dim,
        ), f"gripper action must be ({self.gripper_action_dim},), got {tuple(gripper_action.shape)}"
        return torch.cat([pose_action, gripper_action], dim=0)

    def _get_pose_action_scale(
        self,
        *,
        num_envs: int,
        dtype: torch.dtype,
        device: torch.device,
        env_id: int | None = None,
    ) -> torch.Tensor:
        """Return the live DIK action scale in pose-action order.

        Args:
            num_envs: Number of rows the caller needs.
            dtype: Result dtype.
            device: Result device.
            env_id: Optional single environment row to select.
        """

        unit = torch.ones((num_envs, _DELTA_POSE_ACTION_DIM), dtype=dtype, device=device)
        if self.env is None:
            return unit
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is None:
            return unit
        for term_name in getattr(action_manager, "active_terms", ()):
            term = action_manager.get_term(term_name)
            # Use a name/shape capability check instead of importing Isaac's action class at module
            # load time. This keeps embodiment parsing usable before SimulationApp starts.
            if type(term).__name__ != "DifferentialInverseKinematicsAction":
                continue
            scale = getattr(term, "_scale", None)
            if scale is None:
                scale = getattr(getattr(term, "cfg", None), "scale", None)
            if scale is None:
                return unit
            scale_tensor = torch.as_tensor(scale, dtype=dtype, device=device)
            if scale_tensor.ndim == 1:
                scale_tensor = scale_tensor.unsqueeze(0)
            if scale_tensor.shape[-1] != _DELTA_POSE_ACTION_DIM:
                raise ValueError(
                    "DifferentialInverseKinematicsAction scale must have 6 pose dimensions, "
                    f"got shape {tuple(scale_tensor.shape)}"
                )
            if torch.any(scale_tensor == 0):
                raise ValueError("DifferentialInverseKinematicsAction scale must be non-zero")
            if env_id is not None:
                if not 0 <= env_id < scale_tensor.shape[0]:
                    raise ValueError(f"env_id {env_id} is outside action-scale rows {scale_tensor.shape[0]}")
                scale_tensor = scale_tensor[env_id : env_id + 1]
            if scale_tensor.shape[0] == 1 and num_envs > 1:
                scale_tensor = scale_tensor.expand(num_envs, -1)
            if scale_tensor.shape[0] != num_envs:
                raise ValueError(
                    f"DIK action-scale rows {scale_tensor.shape[0]} do not match requested num_envs {num_envs}"
                )
            return scale_tensor
        return unit

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice the trailing gripper dims off a sequence of env actions.

        Leading batch dims are preserved (e.g. ``(num_envs, num_steps, action_dim)``).

        Args:
            actions: Action tensor whose final dim is ``action_dim``.

        Returns:
            Dictionary ``{eef_name: gripper_actions}``.
        """
        assert (
            actions.shape[-1] == self.action_dim
        ), f"actions last dim must be {self.action_dim}, got {actions.shape[-1]}"
        gripper_start = self.action_dim - self.gripper_action_dim
        return {self.eef_name: actions[..., gripper_start:]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeltaPoseIKSingleArmAdapter:
        """Build an adapter instance from a parsed config dict.

        Expected schema::

            name: <str>
            description: <str>          # optional
            eef_name: <str>
            obs_group: <str>            # optional, default "policy"
            pose_obs_keys:
              pos: <str>
              quat: <str>
            action_layout:
              gripper_dim: <int>              # required
              clip_pose_action_to_unit: <bool># optional, default true
            eef_offset: [<float>, <float>, <float>]  # optional, default [0, 0, 0]
        """
        pose_obs_keys_data = data["pose_obs_keys"]
        layout = data.get("action_layout", {})
        eef_offset = data.get("eef_offset", (0.0, 0.0, 0.0))
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            eef_name=data["eef_name"],
            pose_obs_keys=PoseObsKeys(pos=pose_obs_keys_data["pos"], quat=pose_obs_keys_data["quat"]),
            gripper_action_dim=int(layout["gripper_dim"]),
            obs_group=str(data.get("obs_group", "policy")),
            clip_pose_action_to_unit=bool(layout.get("clip_pose_action_to_unit", True)),
            eef_offset=tuple(float(v) for v in eef_offset),
        )
