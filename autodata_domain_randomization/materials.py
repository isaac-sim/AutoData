# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab event functions for episodic material color and scalar randomization."""

from __future__ import annotations

import hashlib
import re
import torch
from typing import TYPE_CHECKING, Any

from autodata_utils.pose_math import sample_uniform_scalar

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def randomize_material_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prim_name_markers: tuple[str, ...],
    shader_input: str,
    color_palette: tuple[tuple[float, float, float], ...],
) -> None:
    """Apply an independently sampled color multiplier to each resetting environment.

    The sampler interpolates between adjacent palette anchors, which produces a continuous
    color gradient instead of selecting from a small discrete set. The draw uses Torch's
    process RNG so Isaac Lab's environment seed controls reproducibility.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose assets should be recolored.
        asset_cfg: Scene entity containing the target visual prims.
        prim_name_markers: Substrings used to select target prim names.
        shader_input: Material shader input that receives the sampled RGB multiplier.
        color_palette: RGB multiplier anchors defining a piecewise-linear gradient.
    """
    if len(env_ids) == 0:
        return
    inputs_by_env = _find_shader_inputs(env, env_ids, asset_cfg, prim_name_markers, shader_input)

    from pxr import Gf

    for shader_inputs in inputs_by_env.values():
        value = Gf.Vec3f(*sample_palette_color(color_palette, env.device))
        for material_input in shader_inputs:
            material_input.Set(value)


def randomize_material_scalar(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prim_name_markers: tuple[str, ...],
    shader_input: str,
    value_range: tuple[float, float],
) -> None:
    """Apply an independently sampled scalar to each resetting environment.

    Matching shaders within one environment share the sampled value. Equal bounds pin the
    input without consuming random numbers.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose assets should be updated.
        asset_cfg: Scene entity containing the target visual prims.
        prim_name_markers: Substrings used to select target prim names.
        shader_input: Material shader input that receives the sampled value.
        value_range: Uniform sampling range.
    """
    if len(env_ids) == 0:
        return
    inputs_by_env = _find_shader_inputs(env, env_ids, asset_cfg, prim_name_markers, shader_input)
    for shader_inputs in inputs_by_env.values():
        value = sample_uniform_scalar(value_range, env.device)
        for material_input in shader_inputs:
            material_input.Set(value)


def sample_palette_color(
    color_palette: tuple[tuple[float, float, float], ...],
    device: str,
) -> tuple[float, float, float]:
    """Sample uniformly along one segment of a piecewise-linear RGB palette."""
    assert color_palette, "A material color palette must contain at least one color."
    if len(color_palette) == 1:
        return tuple(float(component) for component in color_palette[0])

    segment = int(torch.randint(len(color_palette) - 1, (), device=device).item())
    alpha = float(torch.rand((), device=device).item())
    start = color_palette[segment]
    end = color_palette[segment + 1]
    return tuple(float(start[index] + alpha * (end[index] - start[index])) for index in range(3))


def _find_shader_inputs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prim_name_markers: tuple[str, ...],
    shader_input: str,
) -> dict[int, list[Any]]:
    """Find editable shader inputs grouped by environment."""
    from pxr import Usd, UsdShade

    asset = env.scene[asset_cfg.name]
    root_pattern = re.compile(f"^{asset.cfg.prim_path}$")
    selected_env_paths = {int(env_id): env.scene.env_prim_paths[int(env_id)] for env_id in env_ids.tolist()}
    roots = []
    for prim in env.scene.stage.Traverse():
        path = str(prim.GetPath())
        if not root_pattern.fullmatch(path):
            continue
        for env_id, env_path in selected_env_paths.items():
            if path.startswith(f"{env_path}/"):
                roots.append((env_id, prim))
                break

    inputs_by_env = {env_id: {} for env_id in selected_env_paths}
    for env_id, root in roots:
        for prim in Usd.PrimRange(root):
            if not any(marker in prim.GetName() for marker in prim_name_markers):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            # Isolate each prim's binding from other targets sharing its source material,
            # including clones in other environments.
            overrides_path = root.GetPath().AppendChild("DomainRandomizationMaterials")
            if not material.GetPath().HasPrefix(overrides_path):
                assert not prim.IsInstanceProxy(), "Material DR requires non-instanceable target visual prims."
                identifier = hashlib.sha256(f"{prim.GetPath()}|{material.GetPath()}".encode()).hexdigest()[:16]
                local_path = overrides_path.AppendChild(f"Material_{identifier}")
                local_material = UsdShade.Material.Get(env.scene.stage, local_path)
                if not local_material:
                    local_material = UsdShade.Material.Define(env.scene.stage, local_path)
                    local_material.GetPrim().GetSpecializes().AddSpecialize(material.GetPath())
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(local_material)
                material = local_material
            for material_prim in Usd.PrimRange(material.GetPrim()):
                if material_prim.GetTypeName() != "Shader":
                    continue
                material_input = UsdShade.Shader(material_prim).GetInput(shader_input)
                if material_input:
                    inputs_by_env[env_id][str(material_input.GetAttr().GetPath())] = material_input

    for env_id, inputs in inputs_by_env.items():
        if not inputs:
            raise RuntimeError(
                f"No material shader input {shader_input!r} found on prims matching {prim_name_markers!r} "
                f"for scene asset {asset_cfg.name!r} in environment {env_id}."
            )
    return {env_id: list(inputs.values()) for env_id, inputs in inputs_by_env.items()}
