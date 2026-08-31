# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embodiment adapter that delegates transforms to a live mimic environment."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaac_autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter


@dataclass(kw_only=True)
class EnvironmentEmbodimentAdapter(EmbodimentAdapter):
    """Delegate pose/action/frame operations to an environment's mimic API.

    Args:
        name: Stable embodiment identifier.
        eef_names: End-effectors transformed by AutoData. Extra passthrough channels returned by
            the environment, such as a humanoid body command, need not be listed here.
        robot_asset_name: Scene key of the controlled robot articulation.
        description: Human-readable description.
    """

    name: str
    eef_names: tuple[str, ...]
    robot_asset_name: str = "robot"
    description: str = ""

    def __post_init__(self) -> None:
        assert self.name, "name must be non-empty"
        assert self.eef_names and all(self.eef_names), "eef_names must contain non-empty names"
        assert len(set(self.eef_names)) == len(self.eef_names), "eef_names must be unique"
        assert self.robot_asset_name, "robot_asset_name must be non-empty"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentEmbodimentAdapter:
        """Construct from the YAML payload consumed by the embodiment factory."""

        payload = dict(data)
        payload["eef_names"] = tuple(payload["eef_names"])
        return cls(**payload)

    def _require_method(self, name: str):
        assert self.env is not None, "Call bind_env(env) before using the environment adapter."
        method = getattr(self.env, name, None)
        assert callable(method), f"Environment does not implement required mimic method {name!r}."
        return method

    def get_eef_names(self) -> tuple[str, ...]:
        return self.eef_names

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        get_pose = self._require_method("get_robot_eef_pose")
        return {name: get_pose(name, env_ids=env_ids) for name in self.eef_names}

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        poses = self._require_method("action_to_target_eef_pose")(action)
        assert all(
            name in poses for name in self.eef_names
        ), f"Environment pose result is missing an EEF; expected {self.eef_names}, got {tuple(poses)}."
        return {name: poses[name] for name in self.eef_names}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        return self._require_method("target_eef_pose_to_action")(
            target_eef_pose_dict,
            passthrough_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._require_method("actions_to_gripper_actions")(actions)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return object poses in the environment's controller frame."""

        return self._require_method("get_object_poses")(env_ids=env_ids)


__all__ = ["EnvironmentEmbodimentAdapter"]
