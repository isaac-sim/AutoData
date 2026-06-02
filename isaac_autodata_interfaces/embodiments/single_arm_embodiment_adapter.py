# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
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
    """

    name: str
    description: str = ""
    eef_name: str
    pose_obs_keys: PoseObsKeys
    gripper_action_dim: int
    obs_group: str = "policy"

    def __post_init__(self) -> None:
        assert self.name, "name must be a non-empty string"
        assert self.eef_name, "eef_name must be a non-empty string"
        assert self.gripper_action_dim >= 0, f"gripper_action_dim must be non-negative, got {self.gripper_action_dim}"
        assert isinstance(
            self.pose_obs_keys, PoseObsKeys
        ), f"pose_obs_keys must be a PoseObsKeys instance, got {type(self.pose_obs_keys).__name__}"
        assert self.obs_group, "obs_group must be a non-empty string"

    def get_eef_names(self) -> tuple[str, ...]:
        return (self.eef_name,)

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        obs = self.env.obs_buf[self.obs_group]
        rot = pose_math.matrix_from_quat(obs[self.pose_obs_keys.quat][index])
        return {self.eef_name: pose_math.make_pose(obs[self.pose_obs_keys.pos][index], rot)}

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
        delta_pos = action[:, :3]
        delta_aa = action[:, 3:6]
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

        # --- DEBUG: every 10th step, dump curr / target / delta to spot scaling, NaN, frame, or sign-flip issues ---
        self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        if self._dbg_step % 10 == 1:
            cp = curr_pos.detach().cpu().tolist()
            tp = target_pos.detach().cpu().tolist()
            dp = delta_pos.detach().cpu().tolist()
            da = delta_aa.detach().cpu().tolist()
            g = gripper_action.detach().cpu().tolist()
            print(
                f"[ADAPTER] step={self._dbg_step:4d}  "
                f"curr=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f})  "
                f"target=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f})  "
                f"dpos=({dp[0]:+.4f},{dp[1]:+.4f},{dp[2]:+.4f})  "
                f"delta_aa=({da[0]:+.3f},{da[1]:+.3f},{da[2]:+.3f})  "
                f"grip={g}",
                flush=True,
            )

        return torch.cat([pose_action, gripper_action], dim=0)

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
        return {self.eef_name: actions[..., -self.gripper_action_dim :]}

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
        """
        pose_obs_keys_data = data["pose_obs_keys"]
        layout = data.get("action_layout", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            eef_name=data["eef_name"],
            pose_obs_keys=PoseObsKeys(pos=pose_obs_keys_data["pos"], quat=pose_obs_keys_data["quat"]),
            gripper_action_dim=int(layout["gripper_dim"]),
            obs_group=str(data.get("obs_group", "policy")),
            clip_pose_action_to_unit=bool(layout.get("clip_pose_action_to_unit", True)),
        )
