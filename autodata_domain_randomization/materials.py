# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab event functions for episodic material color and scalar randomization."""

from __future__ import annotations

import hashlib
import math
import re
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

_OVERRIDE_MARKER = "autodata:domain_randomization_override"
_OVERRIDE_TARGET = "autodata:domain_randomization_target"
_EXPLICIT_INPUTS = "autodata:domain_randomization_inputs"
_BINDING_EXCEPTIONS = "autodata:domain_randomization_exceptions"
_PRIVATE_SCOPE = "/DomainRandomizationMaterials"
_INTEGER_LIMITS = {
    "bool": (0, 1),
    "uchar": (0, 2**8 - 1),
    "int": (-(2**31), 2**31 - 1),
    "uint": (0, 2**32 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint64": (0, 2**64 - 1),
}


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
    color gradient instead of selecting from a small discrete set. Each environment uses a
    separate Torch RNG stream initialized from the environment seed, so other environments do not
    affect its material draws.
    Connections on the selected input are replaced on the target's private material.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose assets should be recolored.
        asset_cfg: Scene entity containing the target visual prims.
        prim_name_markers: Substrings selecting prims whose DR propagates to descendants.
            An explicit descendant setting takes precedence for the same shader input.
        shader_input: Material shader input that receives the sampled RGB multiplier.
        color_palette: RGB multiplier anchors defining a piecewise-linear gradient.

    Raises:
        TypeError: A selected shader input cannot accept an RGB color.
        RuntimeError: No matching input exists or a replacement cannot be written.
    """
    if len(env_ids) == 0:
        return
    inputs_by_env = _find_shader_inputs(env, env_ids, asset_cfg, prim_name_markers, shader_input, "color")

    from pxr import Gf

    for env_id, shader_inputs in inputs_by_env.items():
        generator = _material_generator(env, env_id) if len(color_palette) > 1 else None
        value = Gf.Vec3f(*sample_palette_color(color_palette, env.device, generator))
        for material_input in shader_inputs:
            if not material_input.Set(value):
                raise RuntimeError(f"Failed to write material shader input {material_input.GetAttr().GetPath()}.")
            material_input.DisconnectSource()


def randomize_material_scalar(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prim_name_markers: tuple[str, ...],
    shader_input: str,
    value_range: tuple[float, float],
) -> None:
    """Apply an independently sampled scalar to each resetting environment.

    Matching shaders within one environment share the sampled value. If any input is integral,
    sample uniformly among the integers in range that fit all integral inputs. A single possible
    value pins the input without consuming random numbers.
    Connections on the selected input are replaced on the target's private material.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose assets should be updated.
        asset_cfg: Scene entity containing the target visual prims.
        prim_name_markers: Substrings selecting prims whose DR propagates to descendants.
            An explicit descendant setting takes precedence for the same shader input.
        shader_input: Material shader input that receives the sampled value.
        value_range: Uniform sampling range.

    Raises:
        TypeError: A selected shader input cannot accept a scalar value.
        ValueError: The range contains no representable integer for a selected input.
        RuntimeError: No matching input exists or a replacement cannot be written.
    """
    if len(env_ids) == 0:
        return
    integer_ranges: dict[int, tuple[int, int]] = {}

    def check_integer_range(env_id: int, material_input: Any) -> None:
        usd_type = str(material_input.GetTypeName())
        limits = _INTEGER_LIMITS.get(usd_type)
        if limits is None:
            return
        low, high = integer_ranges.get(env_id, limits)
        low = max(low, limits[0], math.ceil(value_range[0]))
        high = min(high, limits[1], math.floor(value_range[1]))
        if low > high:
            raise ValueError(
                f"value_range {value_range} contains no representable integer for "
                f"{material_input.GetAttr().GetPath()} ({usd_type})."
            )
        integer_ranges[env_id] = (low, high)

    inputs_by_env = _find_shader_inputs(
        env, env_ids, asset_cfg, prim_name_markers, shader_input, "scalar", check_integer_range
    )
    for env_id, shader_inputs in inputs_by_env.items():
        if env_id in integer_ranges:
            low, high = integer_ranges[env_id]
            value = low if low == high else _sample_integer(low, high, _material_generator(env, env_id))
        else:
            low, high = value_range
            value = (
                float(low)
                if low == high
                else torch.empty((), device=env.device)
                .uniform_(low, high, generator=_material_generator(env, env_id))
                .item()
            )
        for material_input in shader_inputs:
            if not material_input.Set(value):
                raise RuntimeError(f"Failed to write material shader input {material_input.GetAttr().GetPath()}.")
            material_input.DisconnectSource()


def _sample_integer(low: int, high: int, generator: torch.Generator) -> int:
    """Sample an inclusive USD integer range, including unsigned 64-bit endpoints."""
    count = high - low + 1
    if count <= torch.iinfo(torch.int64).max:
        return low + torch.randint(count, (), device=generator.device, generator=generator).item()

    # randint's upper bound must fit a signed int64. For wider intervals, draw two
    # 32-bit words and reject out-of-range offsets to keep every value equally likely.
    while True:
        upper, lower = torch.randint(2**32, (2,), device=generator.device, generator=generator).tolist()
        offset = (upper << 32) | lower
        if offset < count:
            return low + offset


def sample_palette_color(
    color_palette: tuple[tuple[float, float, float], ...],
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    """Sample uniformly along one segment of a piecewise-linear RGB palette.

    Args:
        color_palette: RGB anchors defining the gradient.
        device: Device on which to sample.
        generator: Random stream; None uses Torch's process stream.
    """
    assert color_palette, "A material color palette must contain at least one color."
    if len(color_palette) == 1:
        return tuple(float(component) for component in color_palette[0])

    segment = int(torch.randint(len(color_palette) - 1, (), device=device, generator=generator).item())
    alpha = float(torch.rand((), device=device, generator=generator).item())
    start = color_palette[segment]
    end = color_palette[segment + 1]
    return tuple(float(start[index] + alpha * (end[index] - start[index])) for index in range(3))


def _material_generator(env: ManagerBasedEnv, env_id: int) -> torch.Generator:
    """Keep each environment's material draws independent of reset batching and order."""
    state = getattr(env, "_autodata_material_generators", None)
    if state is None:
        seed = getattr(getattr(env, "cfg", None), "seed", None)
        if seed is None:
            seed = torch.initial_seed()
        state = (seed, {})
        env._autodata_material_generators = state
    generators = state[1]
    if env_id not in generators:
        digest = hashlib.sha256(f"material:{state[0]}:{env_id}".encode()).digest()
        generators[env_id] = torch.Generator(device=env.device).manual_seed(int.from_bytes(digest[:8], "big"))
    return generators[env_id]


def _find_shader_inputs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prim_name_markers: tuple[str, ...],
    shader_input: str,
    input_kind: Literal["scalar", "color"],
    validate_scalar_range: Callable[[int, Any], None] | None = None,
) -> dict[int, list[Any]]:
    """Find inherited inputs, preserving separately authored materials and child DR."""
    from pxr import Sdf, Usd, UsdShade

    type_names = (
        ("Half3", "Float3", "Double3")
        if input_kind == "color"
        else ("Bool", "UChar", "Int", "UInt", "Int64", "UInt64", "Half", "Float", "Double", "TimeCode")
    )
    compatible_types = {getattr(Sdf.ValueTypeNames, name).type for name in type_names}

    scene_asset_cfg = getattr(env.scene.cfg, asset_cfg.name)
    root_pattern = re.compile(f"^{scene_asset_cfg.prim_path}$")
    selected_env_paths = {int(env_id): env.scene.env_prim_paths[int(env_id)] for env_id in env_ids.tolist()}
    env_ids_by_path = {Sdf.Path(path): env_id for env_id, path in enumerate(env.scene.env_prim_paths)}
    parallel = len(env_ids_by_path) > 1
    roots = []
    iterator = iter(env.scene.stage.Traverse())
    for prim in iterator:
        if prim.GetPath() == _PRIVATE_SCOPE:
            iterator.PruneChildren()
            continue
        if not root_pattern.fullmatch(str(prim.GetPath())):
            continue
        env_id = next((env_ids_by_path[path] for path in prim.GetPath().GetPrefixes() if path in env_ids_by_path), None)
        if env_id is None:
            assert (
                len(env.scene.env_prim_paths) == 1
            ), "Material DR requires per-environment assets for parallel resets."
            env_id = 0
        if env_id in selected_env_paths:
            roots.append((env_id, prim))

    # Snapshot effective materials before changing any bindings. A strong ancestor may
    # otherwise reveal a different child material and change unrandomized properties.
    targets = []
    explicit_targets = {}
    found = dict.fromkeys(selected_env_paths, False)
    for env_id, root in roots:
        active = {}
        iterator = iter(Usd.PrimRange(root))
        for prim in iterator:
            if prim.GetName() == "DomainRandomizationMaterials" or prim.IsA(UsdShade.Material):
                iterator.PruneChildren()
                continue
            explicit = any(marker in prim.GetName() for marker in prim_name_markers)
            settings = prim.GetCustomDataByKey(_EXPLICIT_INPUTS) or {}
            if parallel:
                # Include settings authored before cloning, but keep subsequent settings
                # at absolute, environment-specific paths outside the clone hierarchy.
                record = env.scene.stage.GetPrimAtPath(_private_target_path(prim))
                if record:
                    settings = settings | (record.GetCustomDataByKey(_EXPLICIT_INPUTS) or {})
            inherited_target = active.get(prim.GetParent().GetPath())
            target = prim if explicit else (None if shader_input in settings else inherited_target)
            active[prim.GetPath()] = target
            if target is None:
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            shader_inputs = _material_inputs(material, shader_input)
            for material_input in shader_inputs:
                if material_input.GetTypeName().type not in compatible_types:
                    expected = "an RGB" if input_kind == "color" else "a scalar"
                    raise TypeError(
                        f"material_{input_kind}s requires {expected} shader input, but "
                        f"{material_input.GetAttr().GetPath()} has USD type {material_input.GetTypeName()}."
                    )
                if validate_scalar_range is not None:
                    validate_scalar_range(env_id, material_input)
            if shader_inputs:
                assert not prim.IsInstanceProxy(), "Material DR requires non-instanceable target visual prims."
                targets.append((env_id, root, prim, material, explicit))
                explicit_targets[target.GetPath()] = target
                found[env_id] = True
    for env_id, has_input in found.items():
        if not has_input:
            raise RuntimeError(
                f"No material shader input {shader_input!r} found on prims matching {prim_name_markers!r} "
                f"for scene asset {asset_cfg.name!r} in environment {env_id}."
            )

    for prim in explicit_targets.values():
        if parallel:
            prim = env.scene.stage.DefinePrim(_private_target_path(prim))
        settings = prim.GetCustomDataByKey(_EXPLICIT_INPUTS) or {}
        prim.SetCustomDataByKey(_EXPLICIT_INPUTS, settings | {shader_input: True})

    inputs_by_env = {env_id: {} for env_id in selected_env_paths}
    edited_materials = {}
    for env_id, root, prim, material, explicit in targets:
        source = material
        target_path = _private_target_path(prim) if parallel else prim.GetPath()
        material_prim = material.GetPrim()
        owner = material_prim.GetCustomDataByKey(_OVERRIDE_TARGET)
        # Keep ownership relative to the asset in both namespaces so nested scene
        # assets can update the same override and single-environment assets can be copied.
        owns_material = (
            material_prim.GetCustomDataByKey(_OVERRIDE_MARKER)
            and owner is not None
            and material.GetPath().GetParentPath().GetParentPath().AppendPath(owner) == target_path
        )
        parent = edited_materials.get(prim.GetParent().GetPath())
        if not explicit and parent and parent[0].GetPath() == source.GetPath():
            # Unchanged inherited materials can share an ancestor's override. An
            # explicit child event takes the separate-override path below.
            material = parent[1]
        elif not owns_material:
            original = material
            while original.GetPrim().GetCustomDataByKey(_OVERRIDE_MARKER):
                original = original.GetBaseMaterial()
            identifier = hashlib.sha256(f"{prim.GetPath()}|{original.GetPath()}".encode()).hexdigest()[:16]
            storage_root = _private_target_path(root) if parallel else root.GetPath()
            local_path = storage_root.AppendChild("DomainRandomizationMaterials").AppendChild(f"Material_{identifier}")
            local_material = UsdShade.Material.Get(env.scene.stage, local_path)
            if not local_material:
                local_material = UsdShade.Material.Define(env.scene.stage, local_path)
                local_material.SetBaseMaterial(material)
                local_material.GetPrim().SetCustomDataByKey(_OVERRIDE_MARKER, True)
                local_material.GetPrim().SetCustomDataByKey(
                    _OVERRIDE_TARGET, str(prim.GetPath().MakeRelativePath(root.GetPath()))
                )
            material = local_material
        if material != source:
            if parallel:
                _bind_parallel_material(prim, material, target_path)
            else:
                _bind_material_override(prim, material)
        edited_materials[prim.GetPath()] = (source, material)
        for material_input in _material_inputs(material, shader_input):
            inputs_by_env[env_id][str(material_input.GetAttr().GetPath())] = material_input
    return {env_id: list(inputs.values()) for env_id, inputs in inputs_by_env.items()}


def _material_inputs(material: Any, name: str) -> list[Any]:
    """Find the same named shader inputs when validating a material and writing its override."""
    from pxr import Usd, UsdShade

    inputs = []
    for prim in Usd.PrimRange(material.GetPrim()):
        if prim.IsA(UsdShade.Shader):
            material_input = UsdShade.Shader(prim).GetInput(name)
            if material_input:
                inputs.append(material_input)
    return inputs


def _private_target_path(prim: Any) -> Any:
    """Mirror the scene path outside the environments so cloned prims cannot compose these opinions."""
    from pxr import Sdf

    return prim.GetPath().ReplacePrefix(Sdf.Path.absoluteRootPath, Sdf.Path(_PRIVATE_SCOPE))


def _bind_parallel_material(prim: Any, material: Any, target_path: Any) -> None:
    """Bind only this absolute path, even when other environments inherit the prim."""
    from pxr import Usd, UsdShade

    path = prim.GetPath()
    stage = prim.GetStage()
    record = stage.DefinePrim(target_path)
    collection = Usd.CollectionAPI.Apply(record, "target")
    collection.CreateExpansionRuleAttr().Set(Usd.Tokens.explicitOnly)
    collection.CreateIncludesRel().SetTargets([path])
    name = "autodata_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(collection, material, bindingName=name)
    if api.ComputeBoundMaterial()[0] == material:
        return

    # Only blocked targets need a strong collection on the top ancestor. It matches
    # exactly one prim, preserving the authored precedence everywhere else.
    owner = stage.GetPrimAtPath(path.GetPrefixes()[0])
    api = UsdShade.MaterialBindingAPI.Apply(owner)
    api.Bind(collection, material, bindingName=name, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    rel_name = api.GetCollectionBindingRel(name).GetName()
    if rel_name not in owner.GetPropertyOrder():
        owner.SetPropertyOrder([rel_name, *owner.GetPropertyOrder()])


def _bind_material_override(prim: Any, material: Any) -> None:
    """Bypass blocking bindings on one prim without changing its descendants' assignments."""
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)
    api.GetDirectBindingRel().SetCustomDataByKey(_BINDING_EXCEPTIONS, "")
    bound, binding = api.ComputeBoundMaterial()
    while bound.GetPath() != material.GetPath():
        previous = binding.GetPath()
        _exclude_material_binding(binding, bound, prim.GetPath())
        bound, binding = api.ComputeBoundMaterial()
        assert binding.GetPath() != previous, f"Could not override material binding {previous} for {prim.GetPath()}."


def _exclude_material_binding(binding: Any, material: Any, target_path: Any) -> None:
    """Retain a binding's original membership except for the prim being overridden."""
    from pxr import Sdf, Usd, UsdShade

    owner = binding.GetPrim()
    stage = owner.GetStage()
    exception_path = binding.GetCustomDataByKey(_BINDING_EXCEPTIONS)
    target_expression = Sdf.PathExpression(str(target_path))
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
