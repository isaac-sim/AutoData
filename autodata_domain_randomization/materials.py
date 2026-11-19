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

_OVERRIDE_MARKER = "autodata:domain_randomization_override"
_OVERRIDE_TARGET = "autodata:domain_randomization_target"
_INHERITS_PARENT = "autodata:domain_randomization_inherits_parent"
_BINDING_EXCEPTIONS = "autodata:domain_randomization_exceptions"


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
    Connections on the selected input are replaced on the target's private material.

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
            material_input.DisconnectSource()
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
    Connections on the selected input are replaced on the target's private material.

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
            material_input.DisconnectSource()
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
    """Find editable shader inputs only in the resetting environments."""
    from pxr import Usd, UsdShade

    scene_asset_cfg = getattr(env.scene.cfg, asset_cfg.name)
    root_pattern = re.compile(f"^{scene_asset_cfg.prim_path}$")
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
        overrides_path = root.GetPath().AppendChild("DomainRandomizationMaterials")
        prim_iterator = iter(Usd.PrimRange(root))
        for prim in prim_iterator:
            if prim.GetPath() == overrides_path:
                prim_iterator.PruneChildren()
                continue
            if not any(marker in prim.GetName() for marker in prim_name_markers):
                continue
            material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            # Use the original source for a stable key, while preserving the effective
            # material as the base of a new override.
            source_material = material
            while source_material.GetPath().HasPrefix(overrides_path) and source_material.GetPrim().GetCustomDataByKey(
                _OVERRIDE_MARKER
            ):
                source_material = source_material.GetBaseMaterial()
            identifier = hashlib.sha256(f"{prim.GetPath()}|{source_material.GetPath()}".encode()).hexdigest()[:16]
            local_path = overrides_path.AppendChild(f"Material_{identifier}")
            target = str(prim.GetPath().MakeRelativePath(root.GetPath()))
            # Asset-relative ownership also identifies overrides copied from another environment.
            if (
                material.GetPath().HasPrefix(overrides_path)
                and material.GetPrim().GetCustomDataByKey(_OVERRIDE_TARGET) == target
            ):
                local_path = material.GetPath()
            local_material = UsdShade.Material.Get(env.scene.stage, local_path)
            input_material = local_material or material
            # Skip unsupported targets before creating an override or changing their binding.
            if not any(
                material_prim.GetTypeName() == "Shader" and UsdShade.Shader(material_prim).GetInput(shader_input)
                for material_prim in Usd.PrimRange(input_material.GetPrim())
            ):
                continue
            if material.GetPath() != local_path:
                assert not prim.IsInstanceProxy(), "Material DR requires non-instanceable target visual prims."
                if not local_material:
                    local_material = UsdShade.Material.Define(env.scene.stage, local_path)
                    local_material.SetBaseMaterial(material)
                    local_material.GetPrim().SetCustomDataByKey(_OVERRIDE_MARKER, True)
                    local_material.GetPrim().SetCustomDataByKey(_OVERRIDE_TARGET, target)
                    _, parent_binding = UsdShade.MaterialBindingAPI(prim.GetParent()).ComputeBoundMaterial()
                    # Collections can assign a prim a different material from its parent.
                    local_material.GetPrim().SetCustomDataByKey(
                        _INHERITS_PARENT, binding.GetPath() == parent_binding.GetPath()
                    )
                _bind_material_override(prim, local_material)
                material = local_material
            for material_prim in Usd.PrimRange(material.GetPrim()):
                if material_prim.GetTypeName() != "Shader":
                    continue
                material_input = UsdShade.Shader(material_prim).GetInput(shader_input)
                if material_input:
                    inputs_by_env[env_id][str(material_input.GetAttr().GetPath())] = material_input
        _update_material_inheritance(root, overrides_path)

    for env_id, inputs in inputs_by_env.items():
        if not inputs:
            raise RuntimeError(
                f"No material shader input {shader_input!r} found on prims matching {prim_name_markers!r} "
                f"for scene asset {asset_cfg.name!r} in environment {env_id}."
            )
    return {env_id: list(inputs.values()) for env_id, inputs in inputs_by_env.items()}


def _bind_material_override(prim: Any, material: Any) -> None:
    """Bypass blocking bindings only inside the target subtree."""
    from pxr import Usd, UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)
    api.GetDirectBindingRel().SetCustomDataByKey(_BINDING_EXCEPTIONS, "")
    # An ancestor collection can cover a child without covering the target itself.
    # Keep bindings authored within the subtree so children retain their own materials.
    for descendant in Usd.PrimRange(prim):
        api = UsdShade.MaterialBindingAPI(descendant)
        bound, binding = api.ComputeBoundMaterial()
        while binding and (
            (descendant == prim and bound.GetPath() != material.GetPath())
            or not binding.GetPrim().GetPath().HasPrefix(prim.GetPath())
        ):
            previous = binding.GetPath()
            _exclude_binding_subtree(binding, bound, prim.GetPath())
            bound, binding = api.ComputeBoundMaterial()
            assert (
                binding.GetPath() != previous
            ), f"Could not override material binding {previous} for {descendant.GetPath()}."


def _exclude_binding_subtree(binding: Any, material: Any, target_path: Any) -> None:
    """Retain a binding's original membership except for explicitly randomized subtrees."""
    from pxr import Sdf, Usd, UsdShade

    owner = binding.GetPrim()
    stage = owner.GetStage()
    exception_path = binding.GetCustomDataByKey(_BINDING_EXCEPTIONS)
    target_expression = Sdf.PathExpression(str(target_path) + "//")
    if exception_path:
        exceptions = Usd.CollectionAPI.GetCollection(stage, Sdf.Path(exception_path))
        attribute = exceptions.GetMembershipExpressionAttr()
        attribute.Set(Sdf.PathExpression.MakeOp(Sdf.PathExpression.Union, attribute.Get(), target_expression))
        return

    identifier = hashlib.sha256(str(binding.GetPath()).encode()).hexdigest()[:16]
    # Global exceptions keep absolute target paths when environment prims inherit one another.
    scope = stage.DefinePrim("/DomainRandomizationCollections")
    exceptions = Usd.CollectionAPI.Apply(scope, f"binding_{identifier}")
    exceptions.CreateMembershipExpressionAttr().Set(target_expression)
    exception_reference = Sdf.PathExpression(f"%{scope.GetPath()}:{exceptions.GetName()}")

    api = UsdShade.MaterialBindingAPI.Apply(owner)
    collection_binding = UsdShade.MaterialBindingAPI.CollectionBinding(binding)
    if collection_binding.IsValid():
        source = collection_binding.GetCollection()
        if not source.IsInExpressionMode():
            expression_source = Usd.CollectionAPI.Apply(source.GetPrim(), f"autodata_source_{identifier}")
            expression_source.CreateMembershipExpressionAttr().Set(_collection_as_expression(source))
            source = expression_source
        base = Sdf.PathExpression(f"%{source.GetPrim().GetPath()}:{source.GetName()}")
    else:
        base = Sdf.PathExpression.Everything()

    guard_name = f"autodata_guard_{identifier}"
    guard = Usd.CollectionAPI.Apply(owner, guard_name)
    guard.CreateMembershipExpressionAttr().Set(
        Sdf.PathExpression.MakeOp(Sdf.PathExpression.Difference, base, exception_reference)
    )
    strength = api.GetMaterialBindingStrength(binding)
    if collection_binding.IsValid():
        name = binding.GetName().rsplit(":", 1)[-1]
        api.Bind(guard, material, bindingName=name, bindingStrength=strength)
        guard_binding = binding
    else:
        # A strong direct binding affects all descendants. Move that precedence to
        # a fallback collection, after the ancestor's existing collection bindings.
        collections = api.GetCollectionBindingRels()
        order = [rel.GetName() for rel in collections]
        # Those collections previously shared the direct binding's strong precedence.
        for relation in collections:
            api.SetMaterialBindingStrength(relation, UsdShade.Tokens.strongerThanDescendants)
        api.Bind(guard, material, bindingName=guard_name, bindingStrength=strength)
        guard_binding = api.GetCollectionBindingRel(guard_name)
        order.append(guard_binding.GetName())
        owner.SetPropertyOrder(order + [name for name in owner.GetPropertyOrder() if name not in order])
        api.SetMaterialBindingStrength(binding, UsdShade.Tokens.weakerThanDescendants)
    guard_binding.SetCustomDataByKey(_BINDING_EXCEPTIONS, str(exceptions.GetCollectionPath()))


def _collection_as_expression(collection: Any) -> Any:
    """Express relationship-based membership without changing the authored collection."""
    from pxr import Sdf, Usd

    expression = Sdf.PathExpression.Nothing()
    rules = collection.ComputeMembershipQuery().GetAsPathExpansionRuleMap()
    for path, rule in sorted(rules.items(), key=lambda item: item[0].pathElementCount):
        pattern = str(path)
        if not path.IsPropertyPath() and rule != Usd.Tokens.explicitOnly:
            pattern = pattern.rstrip("/") + "//"
        operation = Sdf.PathExpression.Difference if rule == Usd.Tokens.exclude else Sdf.PathExpression.Union
        expression = Sdf.PathExpression.MakeOp(operation, expression, Sdf.PathExpression(pattern))
    return expression


def _update_material_inheritance(root: Any, overrides_path: Any) -> None:
    """Keep inherited inputs linked when an ancestor gets an override after its children."""
    from pxr import UsdShade

    stage = root.GetStage()
    scope = stage.GetPrimAtPath(overrides_path)
    if not scope:
        return
    for material_prim in scope.GetChildren():
        if not material_prim.GetCustomDataByKey(_INHERITS_PARENT):
            continue
        target = stage.GetPrimAtPath(root.GetPath().AppendPath(material_prim.GetCustomDataByKey(_OVERRIDE_TARGET)))
        material = UsdShade.Material(material_prim)
        bound_material, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
        if not bound_material or bound_material.GetPath() != material.GetPath():
            continue
        parent_material, _ = UsdShade.MaterialBindingAPI(target.GetParent()).ComputeBoundMaterial()
        if (
            parent_material
            and parent_material.GetPath() != material.GetPath()
            and material.GetBaseMaterialPath() != parent_material.GetPath()
        ):
            material.SetBaseMaterial(parent_material)
