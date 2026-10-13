# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-state reads with optional environment selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def get_scene_state(scene: Any, is_relative: bool = True, env_ids: Sequence[int] | None = None) -> dict:
    """Return scene state in world or per-environment coordinates.

    Isaac Lab owns the coordinate conversion for every supported asset type.
    Selection is applied afterwards to retain the requested environment order.

    Args:
        scene: Isaac Lab interactive scene.
        is_relative: Whether positions are relative to each environment origin.
        env_ids: Environment rows to include, in the requested order; None selects all.

    Returns:
        Nested scene-state mapping containing cloned tensors.
    """

    state = scene.get_state(is_relative=is_relative)
    if env_ids is None:
        return state

    def select_envs(value):
        if isinstance(value, dict):
            return {key: select_envs(item) for key, item in value.items()}
        return value[env_ids]

    return select_envs(state)
