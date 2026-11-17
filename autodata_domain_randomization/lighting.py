# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab event functions for episodic lighting randomization."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def color_temperature_to_rgb(color_temperature: float) -> tuple[float, float, float]:
    """Convert a color temperature to an approximate sRGB light color.

    Uses the Tanner Helland approximation over its supported 1000-40000 K interval.

    Args:
        color_temperature: Color temperature [K].

    Returns:
        RGB components in the range ``[0, 1]``.
    """
    assert 1000.0 <= color_temperature <= 40000.0, "Color temperature must be in [1000, 40000] K."
    temperature = color_temperature / 100.0

    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = 0.0 if temperature <= 19.0 else 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * (temperature - 60.0) ** -0.1332047592
        green = 288.1221695283 * (temperature - 60.0) ** -0.0755148492
        blue = 255.0

    return tuple(max(0.0, min(1.0, component / 255.0)) for component in (red, green, blue))


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_temperature_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize existing non-dome lights for the selected environments.

    Dome lighting is unsupported because dome lights illuminate the whole scene. The target
    asset must provide one non-dome light beneath each environment's prim path. This event
    edits intensity and color only; light placement and linking remain task configuration.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose lights should be randomized.
        intensity_range: Uniform USD emitter intensity range.
        color_temperature_range: Uniform color-temperature range [K].
        asset_cfg: Scene entity identifying the per-environment light.
    """
    if len(env_ids) == 0:
        return
    prims = _find_light_prims(env, env_ids, asset_cfg)
    for prim in prims:
        intensity = _sample_uniform(intensity_range, env.device)
        color_temperature = _sample_uniform(color_temperature_range, env.device)
        prim.GetAttribute("inputs:intensity").Set(intensity)
        prim.GetAttribute("inputs:color").Set(color_temperature_to_rgb(color_temperature))


def _find_light_prims(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> list[Any]:
    from isaaclab.sim.utils import find_matching_prims
    from pxr import UsdLux

    stage = env.scene.stage
    assert not any(prim.IsA(UsdLux.DomeLight) for prim in stage.Traverse()), (
        "Lighting DR is not supported in environments with dome lighting. "
        "Omit domain_randomization.lighting or configure per-environment non-dome lights."
    )
    asset = env.scene[asset_cfg.name]
    candidates = find_matching_prims(asset.cfg.prim_path, stage=stage)
    selected = []
    for index in env_ids.tolist():
        env_path = env.scene.env_prim_paths[index]
        matches = [
            prim
            for prim in candidates
            if str(prim.GetPath()).startswith(f"{env_path}/") and prim.HasAPI(UsdLux.LightAPI)
        ]
        assert len(matches) == 1, (
            f"Lighting DR requires exactly one non-dome light for {asset_cfg.name!r} under {env_path!r}; "
            f"found {len(matches)}. Global lights cannot be randomized independently per environment."
        )
        selected.append(matches[0])
    return selected


def _sample_uniform(value_range: tuple[float, float], device: str) -> float:
    low, high = value_range
    if low == high:
        return float(low)
    return float(low + (high - low) * torch.rand((), device=device).item())
