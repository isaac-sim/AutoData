# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configurable absolute-pose control with independently laid out action channels."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_utils import pose_math
from autodata_utils.tensor_utils import as_torch


@dataclass(kw_only=True)
class AbsolutePoseEmbodimentAdapter(EmbodimentAdapter):
    """Encode arbitrary absolute-pose and passthrough channels from a descriptor.

    Each entry in ``eefs`` declares ``pose_obs_keys``, an optional seven-component
    ``pose_action_slice`` (position [m] and quaternion xyzw), and a ``passthrough_action_slice``.
    Tracking-only channels omit the pose slice; their actions contain only passthrough commands.

    Args:
        name: Embodiment name.
        eefs: Per-channel observation keys and action slices.
        action_dim: Width of the complete action vector.
        description: Human-readable description.
        obs_group: Observation group containing the tracked poses.
        reference_frame: Optional controller frame, specified by asset name and optional body name.
        navigation: Optional navigation channel and observation keys used to synchronize playback.
        canonicalize_quat: Whether encoded quaternions have a nonnegative scalar component.
    """

    name: str
    eefs: dict[str, dict[str, Any]]
    action_dim: int
    description: str = ""
    obs_group: str = "policy"
    reference_frame: dict[str, str] | None = None
    navigation: dict[str, str] | None = None
    canonicalize_quat: bool = True

    def __post_init__(self) -> None:
        assert self.name and self.eefs and self.action_dim > 0
        occupied: set[int] = set()
        for name, spec in self.eefs.items():
            assert set(spec) <= {
                "pose_obs_keys",
                "pose_action_slice",
                "passthrough_action_slice",
            }, f"Unknown channel parameters for {name!r}: {set(spec)}."
            assert set(spec["pose_obs_keys"]) == {"pos", "quat"}
            for key in ("pose_action_slice", "passthrough_action_slice"):
                indices = spec.get(key)
                if indices is None:
                    continue
                assert len(indices) == 2 and 0 <= indices[0] < indices[1] <= self.action_dim
                if key == "pose_action_slice":
                    assert indices[1] - indices[0] == 7
                positions = set(range(*indices))
                assert not positions & occupied, f"Overlapping action slices for {name!r}."
                occupied.update(positions)
            assert "passthrough_action_slice" in spec, f"Missing passthrough action slice for {name!r}."
        assert occupied == set(range(self.action_dim)), "Every action component must belong to exactly one channel."
        if self.reference_frame is not None:
            assert "asset_name" in self.reference_frame
            assert set(self.reference_frame) <= {"asset_name", "body_name"}
        if self.navigation is not None:
            assert set(self.navigation) == {"eef_name", "observation_group", "active_key", "goal_reached_key"}
            assert self.navigation["eef_name"] in self.eefs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbsolutePoseEmbodimentAdapter:
        """Build the adapter from a validated descriptor."""
        return cls(**data)

    def get_eef_names(self) -> tuple[str, ...]:
        return tuple(self.eefs)

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        assert self.env is not None, "Call bind_env(env) before reading state."
        index = slice(None) if env_ids is None else env_ids
        obs = self.env.obs_buf[self.obs_group]
        return {
            name: pose_math.make_pose(
                as_torch(obs[spec["pose_obs_keys"]["pos"]])[index],
                pose_math.matrix_from_quat(as_torch(obs[spec["pose_obs_keys"]["quat"]])[index]),
            )
            for name, spec in self.eefs.items()
        }

    def get_reference_frame_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor | None:
        if self.reference_frame is None:
            return None
        assert self.env is not None, "Call bind_env(env) before reading state."
        index = slice(None) if env_ids is None else env_ids
        asset = self.env.scene[self.reference_frame["asset_name"]]
        body_name = self.reference_frame.get("body_name")
        if body_name is None:
            position = as_torch(asset.data.root_pos_w)[index]
            quat = as_torch(asset.data.root_quat_w)[index]
        else:
            body_index = asset.data.body_names.index(body_name)
            state = as_torch(asset.data.body_link_state_w)[index, body_index]
            position, quat = state[..., :3], state[..., 3:7]
        return pose_math.make_pose(position - self.env.scene.env_origins[index], pose_math.matrix_from_quat(quat))

    def get_navigation_state(self, env_id: int) -> tuple[str, bool, bool] | None:
        """Return the navigation channel, active flag, and goal flag from standard observations."""
        if self.navigation is None:
            return None
        assert self.env is not None, "Call bind_env(env) before reading state."
        obs = self.env.obs_buf[self.navigation["observation_group"]]
        return (
            self.navigation["eef_name"],
            bool(obs[self.navigation["active_key"]][env_id]),
            bool(obs[self.navigation["goal_reached_key"]][env_id]),
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        assert action.ndim == 2 and action.shape[-1] == self.action_dim
        result = {}
        for name, spec in self.eefs.items():
            indices = spec.get("pose_action_slice")
            if indices is None:
                # A tracking-only channel has no commanded target pose. Its pose does not
                # participate in encoding; use a valid neutral transform for the source pool.
                result[name] = torch.eye(4, dtype=action.dtype, device=action.device).repeat(len(action), 1, 1)
            else:
                values = action[:, slice(*indices)]
                result[name] = pose_math.make_pose(values[:, :3], pose_math.matrix_from_quat(values[:, 3:7]))
        return result

    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        assert actions.shape[-1] == self.action_dim
        return {name: actions[..., slice(*spec["passthrough_action_slice"])] for name, spec in self.eefs.items()}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        assert set(target_eef_pose_dict) == set(self.eefs)
        assert set(passthrough_action_dict) == set(self.eefs)
        first_action = next(iter(passthrough_action_dict.values()))
        action = first_action.new_zeros(self.action_dim)
        for name, spec in self.eefs.items():
            action[slice(*spec["passthrough_action_slice"])] = passthrough_action_dict[name].reshape(-1)
            indices = spec.get("pose_action_slice")
            if indices is not None:
                position, rotation = pose_math.unmake_pose(target_eef_pose_dict[name])
                quat = pose_math.quat_from_matrix(rotation)
                if self.canonicalize_quat:
                    quat = pose_math.quat_unique(quat)
                values = torch.cat((position, quat))
                noise = (action_noise_dict or {}).get(name, 0.0)
                action[slice(*indices)] = values + noise * torch.randn_like(values) if noise else values
        return action
