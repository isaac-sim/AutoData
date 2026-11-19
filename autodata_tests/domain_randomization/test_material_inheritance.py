# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material DR inheritance, precedence, and independent environment streams."""

import random
import torch
from types import SimpleNamespace

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from autodata_domain_randomization import materials


def make_env(binding_kind="direct", seed=23):
    torch.manual_seed(seed)
    stage = Usd.Stage.CreateInMemory()
    sources = []
    for name, color, roughness, metallic in (
        ("Body", (1.0, 1.0, 1.0), 0.4, 0.1),
        ("Trim", (0.0, 0.0, 1.0), 0.6, 0.7),
        ("Fastener", (0.0, 1.0, 0.0), 0.2, 0.9),
    ):
        material = UsdShade.Material.Define(stage, f"/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + f"/{name}Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        sources.append(material)
    root = "/World/envs/env_0/Object"
    UsdGeom.Xform.Define(stage, root)
    body = UsdGeom.Xform.Define(stage, root + "/body").GetPrim()
    trim = UsdGeom.Xform.Define(stage, root + "/body/trim").GetPrim()
    fastener = UsdGeom.Cube.Define(stage, root + "/body/trim/fastener").GetPrim()
    UsdGeom.Cube.Define(stage, root + "/body/trim/plain")
    UsdGeom.Cube.Define(stage, root + "/body/plain")
    sibling = UsdGeom.Cube.Define(stage, root + "/sibling").GetPrim()
    for prim, source in ((body, sources[0]), (trim, sources[1]), (fastener, sources[2]), (sibling, sources[1])):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(source)
    if binding_kind == "collection":
        collection = Usd.CollectionAPI.Apply(body, "painted")
        collection.CreateExpansionRuleAttr().Set(Usd.Tokens.explicitOnly)
        collection.CreateIncludesRel().SetTargets([body.GetPath(), trim.GetPath()])
        UsdShade.MaterialBindingAPI(body).Bind(collection, sources[0], bindingName="painted")
    elif binding_kind == "strong":
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(
            sources[0], bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    paths = [f"/World/envs/env_{index}" for index in range(3)]
    for path in paths[1:]:
        UsdGeom.Xform.Define(stage, path + "/Object")
        Sdf.CopySpec(stage.GetRootLayer(), root, stage.GetRootLayer(), path + "/Object")
    scene = SimpleNamespace(
        stage=stage,
        env_prim_paths=paths,
        cfg=SimpleNamespace(object=SimpleNamespace(prim_path="/World/envs/env_.*/Object")),
    )
    return SimpleNamespace(scene=scene, device="cpu", cfg=SimpleNamespace(seed=seed))


def inputs(env, suffix, env_id=0):
    path = env.scene.env_prim_paths[env_id] + "/Object/" + suffix
    material, _ = UsdShade.MaterialBindingAPI(env.scene.stage.GetPrimAtPath(path)).ComputeBoundMaterial()
    return {
        item.GetBaseName(): item
        for prim in Usd.PrimRange(material.GetPrim())
        if prim.IsA(UsdShade.Shader)
        for item in UsdShade.Shader(prim).GetInputs()
    }


def values(env, suffix, env_id=0):
    result = {name: item.Get() for name, item in inputs(env, suffix, env_id).items()}
    result["diffuseColor"] = tuple(result["diffuseColor"])
    return result


def scalar(env, marker, value_range, ids=(0,), asset="object"):
    materials.randomize_material_scalar(
        env, torch.tensor(ids), SimpleNamespace(name=asset), (marker,), "roughness", value_range
    )


def color(env, marker, palette, ids=(0,), asset="object"):
    materials.randomize_material_color(
        env, torch.tensor(ids), SimpleNamespace(name=asset), (marker,), "diffuseColor", palette
    )


@pytest.mark.parametrize("binding_kind", ["direct", "collection", "strong"])
@pytest.mark.parametrize("child_first", [False, True])
def test_parent_dr_preserves_unconfigured_child_properties(binding_kind, child_first):
    env = make_env(binding_kind)
    paths = ("body", "body/trim", "body/trim/fastener", "body/trim/plain", "body/plain")
    before = {path: values(env, path) for path in paths}
    untouched = {(env_id, path): values(env, path, env_id) for env_id, path in ((0, "sibling"), (1, "body/trim"))}
    if child_first:
        color(env, "trim", ((0.2, 0.3, 0.9),))
    scalar(env, "body", (0.8, 0.8))
    for path in paths:
        result = values(env, path)
        assert result["roughness"] == pytest.approx(0.8)
        assert result["metallic"] == before[path]["metallic"]
        expected_color = (
            (0.2, 0.3, 0.9) if child_first and path.startswith("body/trim") else before[path]["diffuseColor"]
        )
        assert result["diffuseColor"] == pytest.approx(expected_color)
    for (env_id, path), original in untouched.items():
        assert values(env, path, env_id) == original


@pytest.mark.parametrize("child_first", [False, True])
def test_nearest_dr_setting_wins_per_input_independently_of_event_order(child_first):
    env = make_env()
    operations = [("body", 0.8), ("trim", 0.3), ("fastener", 0.1)]
    for marker, value in reversed(operations) if child_first else operations:
        scalar(env, marker, (value, value))
    scalar(env, "body", (0.9, 0.9))
    assert values(env, "body/plain")["roughness"] == pytest.approx(0.9)
    assert values(env, "body/trim")["roughness"] == pytest.approx(0.3)
    assert values(env, "body/trim/plain")["roughness"] == pytest.approx(0.3)
    assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.1)
    color(env, "trim", ((0.0, 1.0, 0.0),))
    color(env, "body", ((1.0, 0.0, 0.0),))
    assert values(env, "body/plain")["diffuseColor"] == (1.0, 0.0, 0.0)
    assert values(env, "body/trim/fastener")["diffuseColor"] == (0.0, 1.0, 0.0)
    assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.1)
    before = env.scene.stage.GetRootLayer().ExportToString()
    for _ in range(3):
        scalar(env, "body", (0.9, 0.9))
        color(env, "body", ((1.0, 0.0, 0.0),))
        assert env.scene.stage.GetRootLayer().ExportToString() == before


def test_dr_precedence_crosses_nested_scene_asset_boundaries():
    env = make_env()
    env.scene.cfg.trim = SimpleNamespace(prim_path="/World/envs/env_.*/Object/body/trim")
    scalar(env, "trim", (0.2, 0.2), asset="trim")
    for value in (0.3, 0.7, 0.1):
        color(env, "body", ((value, 0.2, 0.4),))
        assert values(env, "body/trim")["roughness"] == pytest.approx(0.2)
        assert values(env, "body/trim")["diffuseColor"] == pytest.approx((value, 0.2, 0.4))
        scalar(env, "body", (0.9, 0.9))
        assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.2)


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_environment_streams_are_independent_of_batch_order_and_other_resets(event):
    def draw(env, ids):
        if event == "scalar":
            scalar(env, "body", (0.1, 0.9), ids)
        else:
            color(env, "body", ((0.0, 0.0, 0.0), (1.0, 0.2, 0.4), (0.2, 0.8, 0.1)), ids)

    batched = make_env(seed=41)
    rng = torch.get_rng_state().clone()
    draw(batched, (2, 0, 1))
    assert torch.equal(rng, torch.get_rng_state())
    first = [values(batched, "body", index) for index in range(3)]
    assert len({str(value) for value in first}) == 3
    draw(batched, (0,))
    expected_next = values(batched, "body", 0)
    separate = make_env(seed=41)
    for env_id in (1, 0, 2):
        draw(separate, (env_id,))
    assert [values(separate, "body", index) for index in range(3)] == first
    for step in range(7):
        torch.manual_seed(100 + step)
        draw(separate, (1, 2))
        torch.rand(20)
        assert values(separate, "body", 0) == first[0]
    draw(separate, (0,))
    assert values(separate, "body", 0) == expected_next


def test_material_override_count_stays_constant_across_partial_resets():
    env = make_env()
    scalar(env, "body", (0.1, 0.9), (0, 1, 2))
    color(env, "trim", ((0, 0, 0), (1, 1, 1)), (0, 1, 2))
    count = sum(prim.IsA(UsdShade.Material) for prim in env.scene.stage.Traverse())
    for ids in ((1,), (2, 0), (0, 1, 2)) * 3:
        scalar(env, "body", (0.1, 0.9), ids)
        color(env, "trim", ((0, 0, 0), (1, 1, 1)), ids)
        assert sum(prim.IsA(UsdShade.Material) for prim in env.scene.stage.Traverse()) == count


def test_unbound_asset_root_propagates_only_to_supported_descendants():
    env = make_env()
    scalar(env, "Object", (0.8, 0.8))
    for path in ("body", "body/trim", "body/trim/fastener", "sibling"):
        assert values(env, path)["roughness"] == pytest.approx(0.8)
    scalar(env, "trim", (0.2, 0.2))
    scalar(env, "Object", (0.9, 0.9))
    assert values(env, "body/plain")["roughness"] == pytest.approx(0.9)
    assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.2)


def test_roughness_dr_preserves_child_texture_connections():
    env = make_env()
    stage = env.scene.stage
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Looks/Trim/TrimShader"))
    texture = UsdShade.Shader.Define(stage, "/Looks/Trim/Texture")
    shader.GetInput("diffuseColor").ConnectToSource(texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    scalar(env, "body", (0.8, 0.8))
    trim_inputs = inputs(env, "body/trim")
    assert trim_inputs["diffuseColor"].GetConnectedSources()[0]
    assert trim_inputs["roughness"].Get() == pytest.approx(0.8)
    color(env, "body", ((1.0, 0.0, 0.0),))
    assert not trim_inputs["diffuseColor"].GetConnectedSources()[0]
    assert shader.GetInput("diffuseColor").GetConnectedSources()[0]


def test_child_override_with_different_shader_inputs_does_not_block_other_properties():
    env = make_env()
    shader = UsdShade.Shader(env.scene.stage.GetPrimAtPath("/Looks/Trim/TrimShader"))
    shader.GetPrim().RemoveProperty("inputs:roughness")
    before = values(env, "body/trim")
    scalar(env, "body", (0.8, 0.8))
    assert values(env, "body/trim") == before
    assert "roughness" not in values(env, "body/trim")
    assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.8)
    color(env, "body", ((1.0, 0.0, 0.0),))
    assert values(env, "body/trim")["diffuseColor"] == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_material_streams_leave_fixed_settings_out_of_draw_sequence(event):
    def draw(env):
        if event == "scalar":
            scalar(env, "body", (0.1, 0.9))
        else:
            color(env, "body", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        return values(env, "body")["roughness" if event == "scalar" else "diffuseColor"]

    baseline = make_env(seed=19)
    first, second = draw(baseline), draw(baseline)
    mixed = make_env(seed=19)
    assert draw(mixed) == first
    scalar(mixed, "body", (0.5, 0.5))
    color(mixed, "body", ((1.0, 0.0, 0.0),))
    assert draw(mixed) == second


@pytest.mark.parametrize("num_envs", [1, 3])
def test_global_material_targets_require_a_single_environment(num_envs):
    env = make_env()
    env.scene.env_prim_paths = env.scene.env_prim_paths[:num_envs]
    stage = env.scene.stage
    root = UsdGeom.Xform.Define(stage, "/World/GroundPlane").GetPrim()
    floor = UsdGeom.Cube.Define(stage, "/World/GroundPlane/floor").GetPrim()
    original = UsdShade.Material.Get(stage, "/Looks/Trim")
    UsdShade.MaterialBindingAPI.Apply(floor).Bind(original)
    env.scene.cfg.ground = SimpleNamespace(prim_path=str(root.GetPath()))
    before = stage.GetRootLayer().ExportToString()
    if num_envs > 1:
        with pytest.raises(AssertionError, match="Material DR.*per-environment"):
            scalar(env, "floor", (0.8, 0.8), asset="ground")
        assert stage.GetRootLayer().ExportToString() == before
    else:
        scalar(env, "floor", (0.8, 0.8), asset="ground")
        result, _ = UsdShade.MaterialBindingAPI(floor).ComputeBoundMaterial()
        shader = UsdShade.Shader(stage.GetPrimAtPath(result.GetPath().AppendChild("TrimShader")))
        assert shader.GetInput("roughness").Get() == pytest.approx(0.8)
        assert tuple(shader.GetInput("diffuseColor").Get()) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("binding_kind,override_count", [("direct", 3), ("strong", 1)])
def test_inherited_materials_are_reused_until_a_child_gets_its_own_dr(binding_kind, override_count):
    env = make_env(binding_kind)
    scalar(env, "body", (0.8, 0.8))
    assert sum(prim.IsA(UsdShade.Material) for prim in env.scene.stage.Traverse()) == 3 + override_count
    scalar(env, "trim", (0.2, 0.2))
    scalar(env, "body", (0.9, 0.9))
    assert values(env, "body/plain")["roughness"] == pytest.approx(0.9)
    assert values(env, "body/trim")["roughness"] == pytest.approx(0.2)
    assert values(env, "body/trim/plain")["roughness"] == pytest.approx(0.2)
    assert values(env, "body/trim/fastener")["roughness"] == pytest.approx(0.2)


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_environment_seed_controls_late_streams_after_global_reseeding(event):
    def draw(env, env_id):
        if event == "scalar":
            scalar(env, "body", (0.1, 0.9), (env_id,))
        else:
            color(env, "body", ((0, 0, 0), (1, 1, 1)), (env_id,))
        return values(env, "body", env_id)

    reference = make_env(seed=51)
    expected = draw(reference, 2)
    late = make_env(seed=51)
    torch.manual_seed(99)
    draw(late, 0)
    torch.manual_seed(100)
    assert draw(late, 2) == expected
    different_seed = make_env(seed=52)
    assert draw(different_seed, 2) != expected


@pytest.mark.parametrize("binding_kind", ["direct", "collection", "strong"])
def test_mixed_reset_sequences_match_nearest_property_settings(binding_kind):
    env = make_env(binding_kind)
    paths = ("body", "body/plain", "body/trim", "body/trim/plain", "body/trim/fastener", "sibling")
    original = {path: values(env, path) for path in paths}
    settings = [{} for _ in env.scene.env_prim_paths]
    rng = random.Random(41)
    for _ in range(60):
        marker = rng.choice(("body", "trim", "fastener", "plain", "sibling"))
        field = rng.choice(("roughness", "diffuseColor"))
        value = rng.choice((0.1, 0.4, 0.8)) if field == "roughness" else (rng.random(), rng.random(), rng.random())
        ids = rng.sample(range(3), rng.randint(1, 3))
        if field == "roughness":
            scalar(env, marker, (value, value), ids)
        else:
            color(env, marker, (value,), ids)
        for env_id in ids:
            for path in paths:
                if marker in path.rsplit("/", 1)[-1]:
                    settings[env_id].setdefault(path, {})[field] = value
        for env_id, environment_settings in enumerate(settings):
            for path in paths:
                result = values(env, path, env_id)
                for name, initial in original[path].items():
                    candidates = [
                        target
                        for target, properties in environment_settings.items()
                        if name in properties and (path == target or path.startswith(target + "/"))
                    ]
                    expected = environment_settings[max(candidates, key=len)][name] if candidates else initial
                    assert result[name] == pytest.approx(expected), (env_id, path, name)
