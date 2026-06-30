# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight mock env/scene used by the interface unit tests.

These mocks expose just the attributes the embodiment adapters and the Datastream read
(``env.device``, ``env.obs_buf``, ``env.scene[...]``, ``scene.rigid_objects``,
``scene.env_origins``, ``scene.get_state``, and articulation ``.data`` handles) so the
pure-Python pose/action logic can be exercised without launching Isaac Sim.
"""

from __future__ import annotations

import torch
from typing import Any


class MockArticulationData:
    """Stand-in for an Isaac Lab articulation/rigid-object ``.data`` handle."""

    def __init__(
        self,
        joint_pos: torch.Tensor | None = None,
        joint_names: list[str] | None = None,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
    ) -> None:
        self.joint_pos = joint_pos
        self.joint_names = joint_names
        self.root_pos_w = root_pos_w
        self.root_quat_w = root_quat_w


class MockAsset:
    """Stand-in for a scene asset (articulation or rigid object) wrapping a ``.data`` handle."""

    def __init__(self, data: MockArticulationData) -> None:
        self.data = data


class MockScene:
    """Stand-in for ``env.scene`` supporting key lookup, rigid-object iteration, and state reads."""

    def __init__(
        self,
        assets: dict[str, MockAsset] | None = None,
        rigid_objects: dict[str, MockAsset] | None = None,
        env_origins: torch.Tensor | None = None,
        state: Any = None,
    ) -> None:
        self._assets = assets or {}
        self.rigid_objects = rigid_objects or {}
        self.env_origins = env_origins
        self._state = state

    def __getitem__(self, key: str) -> MockAsset:
        return self._assets[key]

    def get_state(self, is_relative: bool = True) -> Any:
        return self._state


class MockEnv:
    """Stand-in for the live env exposing only what the adapters / Datastream read."""

    def __init__(self, device: str = "cpu", obs_buf: dict | None = None, scene: MockScene | None = None) -> None:
        self.device = device
        self.obs_buf = obs_buf
        self.scene = scene
