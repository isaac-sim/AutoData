# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for episodic domain-randomization interfaces."""

import torch
from types import SimpleNamespace

import pytest

from autodata_domain_randomization import (
    DomainRandomizationCfg,
    LightingRandomizationCfg,
    MaterialColorRandomizationCfg,
    MaterialScalarRandomizationCfg,
    PoseRandomizationCfg,
    apply_domain_randomization,
    color_temperature_to_rgb,
    create_domain_randomization_profile,
    lighting,
    materials,
    sample_palette_color,
)
from autodata_interfaces.env import EnvironmentProfile, apply_env_profile


def test_absent_randomization_preserves_existing_events():
    profile = EnvironmentProfile.from_dict({"name": "base", "base_env": "TestEnv-v0"})
    existing = SimpleNamespace(params={"intensity": 500.0})
    env_cfg = SimpleNamespace(events=SimpleNamespace(light=existing), scene=SimpleNamespace())
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    assert profile.domain_randomization is None
    assert vars(env_cfg.events) == {"light": existing}
    assert existing.params == {"intensity": 500.0}


def test_lighting_config_defaults_to_3000_to_8000_kelvin():
    assert LightingRandomizationCfg().color_temperature_range == (3000.0, 8000.0)


def test_training_profile_adds_episodic_lighting_event():
    lighting_cfg = LightingRandomizationCfg(
        asset_name="task_light", intensity_range=(2000.0, 4000.0), color_temperature_range=(3000.0, 7000.0)
    )
    profile = create_domain_randomization_profile("TestEnv-v0", DomainRandomizationCfg(lighting=lighting_cfg))
    event = profile.events.add["randomize_light"]
    assert event.mode == "reset"
    assert event.params == {
        "intensity_range": (2000.0, 4000.0),
        "color_temperature_range": (3000.0, 7000.0),
        "asset_cfg": "task_light",
    }


def test_training_profile_applies_as_an_isaac_lab_reset_event():
    from isaaclab.managers import SceneEntityCfg

    env_cfg = SimpleNamespace(events=SimpleNamespace(), scene=SimpleNamespace(light=object()))
    profile = create_domain_randomization_profile(
        "TestEnv-v0", DomainRandomizationCfg(lighting=LightingRandomizationCfg())
    )
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    event = env_cfg.events.randomize_light
    assert event.func is lighting.randomize_light
    assert event.mode == "reset"
    assert isinstance(event.params["asset_cfg"], SceneEntityCfg)
    assert event.params["asset_cfg"].name == "light"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intensity_range": (2.0, 1.0)},
        {"color_temperature_range": (999.0, 8000.0)},
        {"color_temperature_range": (2000.0, 40001.0)},
    ],
)
def test_lighting_config_rejects_invalid_ranges(kwargs):
    with pytest.raises(AssertionError):
        LightingRandomizationCfg(**kwargs)


def test_color_temperature_produces_plausible_warm_and_cool_colors():
    warm = color_temperature_to_rgb(2000.0)
    cool = color_temperature_to_rgb(8000.0)

    assert all(0.0 <= component <= 1.0 for component in (*warm, *cool))
    assert warm[0] > warm[2]
    assert cool[2] > cool[0]


class _Attribute:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = value

    def DisconnectSource(self):
        return True


class _LightPrim:
    def __init__(self):
        self.attributes = {"inputs:intensity": _Attribute(), "inputs:color": _Attribute()}

    def GetAttribute(self, name):
        return self.attributes[name]


def test_lighting_event_uses_seeded_torch_rng(monkeypatch):
    selected_prim = [_LightPrim()]
    monkeypatch.setattr(lighting, "_find_light_prims", lambda *_: (selected_prim, []))
    env = SimpleNamespace(device="cpu")

    torch.manual_seed(7)
    lighting.randomize_light(env, torch.tensor([0]), (100.0, 200.0), (2000.0, 8000.0), object())
    first_values = tuple(attribute.value for attribute in selected_prim[0].attributes.values())

    selected_prim[0] = _LightPrim()
    torch.manual_seed(7)
    lighting.randomize_light(env, torch.tensor([0]), (100.0, 200.0), (2000.0, 8000.0), object())
    second_values = tuple(attribute.value for attribute in selected_prim[0].attributes.values())

    assert first_values == second_values
    assert 100.0 <= first_values[0] <= 200.0


def test_training_profile_adds_episodic_material_color_event():
    material_cfg = MaterialColorRandomizationCfg(
        asset_name="apple",
        prim_name_markers=("mesh",),
        shader_input="base_color_factor",
        color_palette=((0.4, 0.1, 0.1), (0.1, 2.0, 0.1)),
    )
    profile = create_domain_randomization_profile(
        "TestEnv-v0", DomainRandomizationCfg(lighting=LightingRandomizationCfg(), material_colors=(material_cfg,))
    )
    event = profile.events.add["randomize_material_color_0"]
    assert event.mode == "reset"
    assert event.params == {
        "asset_cfg": "apple",
        "prim_name_markers": ("mesh",),
        "shader_input": "base_color_factor",
        "color_palette": ((0.4, 0.1, 0.1), (0.1, 2.0, 0.1)),
    }


def test_material_color_profile_applies_as_an_isaac_lab_reset_event():
    from isaaclab.managers import SceneEntityCfg

    env_cfg = SimpleNamespace(events=SimpleNamespace(), scene=SimpleNamespace(apple=object(), light=object()))
    material_cfg = MaterialColorRandomizationCfg(
        asset_name="apple",
        prim_name_markers=("mesh",),
        shader_input="base_color_factor",
        color_palette=((1.0, 1.0, 1.0),),
    )
    profile = create_domain_randomization_profile(
        "TestEnv-v0", DomainRandomizationCfg(lighting=LightingRandomizationCfg(), material_colors=(material_cfg,))
    )
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    event = env_cfg.events.randomize_material_color_0
    assert event.func is materials.randomize_material_color
    assert event.mode == "reset"
    assert isinstance(event.params["asset_cfg"], SceneEntityCfg)
    assert event.params["asset_cfg"].name == "apple"


def test_material_palette_sampler_is_seeded_and_interpolates():
    palette = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

    torch.manual_seed(11)
    first = sample_palette_color(palette, "cpu")
    torch.manual_seed(11)
    second = sample_palette_color(palette, "cpu")

    assert first == second
    assert 0.0 <= first[0] <= 1.0
    assert first[1] == pytest.approx(2.0 * first[0])
    assert first[2] == pytest.approx(3.0 * first[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"asset_name": ""},
        {"prim_name_markers": ()},
        {"shader_input": ""},
        {"color_palette": ()},
        {"color_palette": ((1.0, 1.0),)},
        {"color_palette": ((1.0, -0.1, 1.0),)},
    ],
)
def test_material_color_config_rejects_invalid_values(kwargs):
    defaults = {
        "asset_name": "apple",
        "prim_name_markers": ("mesh",),
        "shader_input": "base_color_factor",
        "color_palette": ((1.0, 1.0, 1.0),),
    }
    with pytest.raises(AssertionError):
        MaterialColorRandomizationCfg(**(defaults | kwargs))


def test_domain_randomization_yaml_builds_reset_profile(tmp_path):
    config_path = tmp_path / "domain_randomization.yaml"
    config_path.write_text("""
name: test_dr
base_env: TestEnv-v0
domain_randomization:
  lighting:
    asset_name: task_light
    intensity_range:
    - 300.0
    - 600.0
    color_temperature_range:
    - 3000.0
    - 7000.0
  poses:
  - event_name: apple
    position_xyz_min:
    - 0.1
    - 0.15
    - 0.775
    position_xyz_max:
    - 0.2
    - 0.25
    - 0.775
  material_colors:
  - asset_name: apple
    prim_name_markers:
    - mesh
    shader_input: base_color_factor
    color_palette:
    - - 0.38
      - 0.06
      - 0.1
    - - 0.12
      - 2.4
      - 0.1
""")
    cfg = EnvironmentProfile.from_yaml(config_path).domain_randomization
    cfg.validate_runtime(num_envs=1)
    profile = create_domain_randomization_profile("TestEnv-v0", cfg)
    assert cfg.lighting.intensity_range == (300.0, 600.0)
    assert cfg.poses[0].position_xyz_min == (0.1, 0.15, 0.775)
    assert cfg.material_colors[0].prim_name_markers == ("mesh",)
    assert profile.events.override["apple"].params["pose_range"] == {
        "x": (0.1, 0.2),
        "y": (0.15, 0.25),
        "z": (0.775, 0.775),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    assert profile.events.add["randomize_light"].params["asset_cfg"] == "task_light"
    assert profile.events.add["randomize_material_color_0"].params["asset_cfg"] == "apple"


def test_pose_profile_overrides_existing_reset_event():
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(apple=SimpleNamespace(params={"pose_range": {}})), scene=SimpleNamespace(light=object())
    )
    cfg = DomainRandomizationCfg(
        poses=(
            PoseRandomizationCfg(
                event_name="apple", position_xyz_min=(0.1, 0.15, 0.775), position_xyz_max=(0.2, 0.25, 0.775)
            ),
        ),
    )
    apply_env_profile(env_cfg, create_domain_randomization_profile("TestEnv-v0", cfg), "TestEnv-v0")
    assert env_cfg.events.apple.params["pose_range"]["x"] == (0.1, 0.2)
    assert env_cfg.events.apple.params["pose_range"]["y"] == (0.15, 0.25)


@pytest.mark.parametrize("section", ["scene", "episode", "chunk"])
def test_legacy_lifecycle_sections_are_rejected(section):
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({section: {}})


def test_material_only_randomization_supports_parallel_environments():
    cfg = DomainRandomizationCfg(
        material_scalars=(
            MaterialScalarRandomizationCfg(
                asset_name="table", prim_name_markers=("mesh",), shader_input="roughness", value_range=(0.1, 0.9)
            ),
        )
    )
    cfg.validate_runtime(num_envs=4)


def test_domain_randomization_yaml_rejects_unknown_keys():
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({"episodic": {}})


@pytest.mark.parametrize(
    "section", ["lighting", "robot_root", "robot_joints", "poses", "material_colors", "material_scalars", "cameras"]
)
@pytest.mark.parametrize("value", [True, False])
def test_randomizers_reject_enable_switches(section, value):
    entry = {"enabled": value}
    data = {section: entry if section in {"lighting", "robot_root", "robot_joints"} else [entry]}
    with pytest.raises(AssertionError, match="unknown keys.*enabled"):
        DomainRandomizationCfg.from_dict(data)


def test_apply_domain_randomization_updates_env_cfg(tmp_path):
    config_path = tmp_path / "domain_randomization.yaml"
    config_path.write_text("""
name: test_dr
base_env: TestEnv-v0
domain_randomization:
  poses:
  - event_name: apple
    position_xyz_min:
    - 0.1
    - 0.15
    - 0.775
    position_xyz_max:
    - 0.2
    - 0.25
    - 0.775
""")
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(apple=SimpleNamespace(params={"pose_range": {}})), scene=SimpleNamespace()
    )
    cfg = apply_domain_randomization(
        env_cfg, "TestEnv-v0", EnvironmentProfile.from_yaml(config_path).domain_randomization, num_envs=2
    )
    assert cfg.lighting is None
    assert env_cfg.events.apple.params["pose_range"]["x"] == (0.1, 0.2)


@pytest.mark.parametrize("root", [True, False])
def test_robot_root_presence_controls_reset_override(root):
    import math

    from isaaclab_arena.terms.events import reset_all_articulation_joints as arena_reset

    profile = EnvironmentProfile.from_dict({
        "name": "root_reset",
        "base_env": "TestEnv-v0",
        "domain_randomization": (
            {
                "robot_root": {
                    "x_range_m": [-0.1, 0.1],
                    "y_range_m": [-0.05, 0.05],
                    "yaw_range_deg": [-5, 5],
                },
            }
            if root
            else {}
        ),
    })
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(reset_all=SimpleNamespace(params={}, func=arena_reset)),
        scene=SimpleNamespace(num_envs=5),
    )
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    root_range = env_cfg.events.reset_all.params.get("robot_root_pose_range")
    if root:
        assert root_range == {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "yaw": (-math.pi / 36, math.pi / 36)}
    else:
        assert root_range is None


@pytest.mark.parametrize(
    "values",
    [
        {"x_range_m": [0.1, -0.1]},
        {"yaw_range_deg": [0, float("inf")]},
        {"y_range_m": [0]},
    ],
)
def test_robot_root_rejects_invalid_bounds(values):
    from autodata_domain_randomization import RobotRootRandomizationCfg

    with pytest.raises(AssertionError):
        RobotRootRandomizationCfg(**values)


def test_robot_root_reset_samples_world_offsets_and_preserves_partial_reset_state():
    import numpy as np
    from scipy.spatial.transform import Rotation

    import warp as wp

    from autodata_domain_randomization.robot import reset_all_articulation_joints

    class Articulation:
        def __init__(self, count):
            default = torch.zeros(count, 13)
            default[:, :3] = torch.tensor([0.2, 0.55, 0.78])
            default[:, 3:7] = torch.tensor(Rotation.from_euler("xyz", [0.15, -0.1, -1.57]).as_quat())
            self.data = SimpleNamespace(
                default_root_state=wp.from_torch(default),
                default_joint_pos=wp.from_torch(torch.ones(count, 3)),
                default_joint_vel=wp.from_torch(torch.zeros(count, 3)),
            )
            self.pose = torch.full((count, 7), -999.0)
            self.velocity = torch.full((count, 6), -999.0)
            self.joints = torch.full((count, 3), -999.0)
            self.joint_velocity = torch.full((count, 3), -999.0)

        def write_root_pose_to_sim(self, value, env_ids):
            self.pose[env_ids] = value

        def write_root_velocity_to_sim(self, value, env_ids):
            self.velocity[env_ids] = value

        def write_joint_state_to_sim(self, position, velocity, env_ids):
            self.joints[env_ids] = position
            self.joint_velocity[env_ids] = velocity

    count = 8193
    robot, other = Articulation(count), Articulation(count)
    origins = torch.zeros(count, 3)
    origins[:, 0] = torch.arange(count) % 3
    env = SimpleNamespace(
        device="cpu", scene=SimpleNamespace(articulations={"robot": robot, "other": other}, env_origins=origins)
    )
    ids = torch.arange(1, count)
    bounds = {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "yaw": (-0.0872664626, 0.0872664626)}
    torch.manual_seed(123)
    reset_all_articulation_joints(env, ids, bounds)
    first = robot.pose.clone()
    torch.manual_seed(123)
    reset_all_articulation_joints(env, ids, bounds)
    torch.testing.assert_close(robot.pose, first, rtol=0, atol=0)
    default = wp.to_torch(robot.data.default_root_state)
    delta = robot.pose[ids, :3] - origins[ids] - default[ids, :3]
    relative = Rotation.from_quat(robot.pose[ids, 3:].numpy()) * Rotation.from_quat(default[ids, 3:7].numpy()).inv()
    rotvec = relative.as_rotvec()
    np.testing.assert_allclose(rotvec[:, :2], 0, atol=1e-7)
    draws = np.column_stack([delta[:, :2].numpy(), rotvec[:, 2]])
    for i, (low, high) in enumerate(bounds.values()):
        assert draws[:, i].min() >= low - 1e-6 and draws[:, i].max() <= high + 1e-6
        assert draws[:, i].min() < low * 0.95 and draws[:, i].max() > high * 0.95
        assert abs(draws[:, i].mean()) < (high - low) * 0.02
        assert draws[:, i].std() == pytest.approx((high - low) / np.sqrt(12), rel=0.03)
    assert np.abs(np.corrcoef(draws.T) - np.eye(3)).max() < 0.04
    torch.testing.assert_close(delta[:, 2], torch.zeros(len(ids)), rtol=0, atol=0)
    for asset in (robot, other):
        assert (asset.pose[0] == -999).all() and (asset.joints[0] == -999).all()
        assert (asset.velocity[ids] == 0).all() and (asset.joints[ids] == 1).all()
        assert (asset.joint_velocity[ids] == 0).all()
    torch.testing.assert_close(other.pose[ids, :3], default[ids, :3] + origins[ids])
    torch.testing.assert_close(other.pose[ids, 3:], default[ids, 3:7])
    # Omitting the new parameter preserves the original pose and RNG behavior.
    rng = torch.get_rng_state().clone()
    reset_all_articulation_joints(env, ids)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(robot.pose, other.pose)


def test_material_scalar_yaml_extends_example_preset():
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "autodata_examples/env_profiles/g1_locomanip_dr.yaml"
    data = yaml.safe_load(path.read_text())["domain_randomization"]
    baseline = create_domain_randomization_profile("TestEnv-v0", DomainRandomizationCfg.from_dict(data))
    index = len(data["material_scalars"])
    data["material_scalars"].append({
        "asset_name": "table",
        "prim_name_markers": ["tabletop"],
        "shader_input": "specular_level",
        "value_range": [0.0, 0.5],
    })
    cfg = DomainRandomizationCfg.from_dict(data)
    cfg.validate_runtime(num_envs=1)
    profile = create_domain_randomization_profile("TestEnv-v0", cfg)
    event = profile.events.add.pop(f"randomize_material_scalar_{index}")
    assert event.mode == "reset"
    assert event.params == {
        "asset_cfg": "table",
        "prim_name_markers": ("tabletop",),
        "shader_input": "specular_level",
        "value_range": (0.0, 0.5),
    }
    assert profile == baseline


def test_material_scalar_profile_applies_as_a_reset_event():
    scalar = MaterialScalarRandomizationCfg(
        asset_name="table", prim_name_markers=("tabletop",), shader_input="roughness", value_range=(0.5, 0.5)
    )
    cfg = DomainRandomizationCfg(material_scalars=(scalar,))
    env_cfg = SimpleNamespace(events=SimpleNamespace(), scene=SimpleNamespace(table=object()))
    apply_env_profile(env_cfg, create_domain_randomization_profile("TestEnv-v0", cfg), "TestEnv-v0")
    event = env_cfg.events.randomize_material_scalar_0
    assert event.func is materials.randomize_material_scalar
    assert event.params["asset_cfg"].name == "table"
    assert event.params["value_range"] == (0.5, 0.5)


@pytest.mark.parametrize("term", [True, False])
def test_material_scalar_presence_controls_reset_event(term):
    cfg = DomainRandomizationCfg.from_dict({
        "material_scalars": (
            [{
                "asset_name": "table",
                "prim_name_markers": ["mesh"],
                "shader_input": "roughness",
                "value_range": [0.1, 0.9],
            }]
            if term
            else []
        ),
    })
    profile = create_domain_randomization_profile("TestEnv-v0", cfg)
    assert ("randomize_material_scalar_0" in profile.events.add) == term


@pytest.mark.parametrize("bounds", [(-0.1, 0.5), (0.5, 0.1), (0.1,), (0.1, float("inf")), (float("nan"), 0.5)])
def test_material_scalar_rejects_invalid_ranges(bounds):
    with pytest.raises(AssertionError):
        MaterialScalarRandomizationCfg(
            asset_name="table",
            prim_name_markers=("mesh",),
            shader_input="roughness",
            value_range=bounds,
        )


def test_material_scalar_yaml_rejects_unknown_keys():
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({"material_scalars": [{"unknown": 1}]})


def test_material_scalar_sampling_is_seeded_and_updates_all_matches(monkeypatch):
    attributes = [_Attribute(), _Attribute()]
    monkeypatch.setattr(materials, "_find_shader_inputs", lambda *args: {1: attributes})

    def sample(bounds):
        materials.randomize_material_scalar(
            SimpleNamespace(device="cpu"),
            torch.tensor([1]),
            SimpleNamespace(name="table"),
            ("tabletop",),
            "roughness",
            bounds,
        )
        assert attributes[0].value == attributes[1].value
        return attributes[0].value

    rng = torch.random.get_rng_state().clone()
    assert sample((1.0, 1.0)) == 1.0
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.manual_seed(7)
    first = [sample((0.25, 0.75)) for _ in range(20)]
    torch.manual_seed(7)
    assert [sample((0.25, 0.75)) for _ in range(20)] == first
    assert len(set(first)) > 1
    assert all(0.25 <= value <= 0.75 for value in first)


@pytest.mark.parametrize("static_asset", [False, True])
def test_material_overrides_isolate_targets_sharing_a_source_and_are_reused(static_asset):
    from isaaclab.assets import AssetBaseCfg
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Shared")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Shared/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    root = "/World/envs/env_0/Object"
    UsdGeom.Xform.Define(stage, root)
    meshes = [UsdGeom.Cube.Define(stage, f"{root}/{name}").GetPrim() for name in ("mesh_A", "mesh_B")]
    for mesh in meshes:
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    object_cfg = AssetBaseCfg(prim_path=root)
    scene = Scene(object=object_cfg if static_asset else SimpleNamespace(cfg=object_cfg))
    scene.cfg = SimpleNamespace(object=object_cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    ids = torch.tensor([0])
    asset_cfg = SimpleNamespace(name="object")

    def reset_materials():
        for marker, roughness in (("mesh_A", 0.2), ("mesh_B", 0.8)):
            materials.randomize_material_scalar(env, ids, asset_cfg, (marker,), "roughness", (roughness, roughness))
        materials.randomize_material_color(env, ids, asset_cfg, ("mesh_A",), "diffuseColor", ((0.1, 0.2, 0.3),))

    reset_materials()
    bound_materials = [UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0] for mesh in meshes]
    bound_shaders = [
        UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/Shader")) for bound in bound_materials
    ]
    assert bound_shaders[0].GetInput("roughness").Get() == pytest.approx(0.2)
    assert bound_shaders[1].GetInput("roughness").Get() == pytest.approx(0.8)
    assert bound_materials[0].GetPath() != bound_materials[1].GetPath()
    assert tuple(bound_shaders[0].GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(bound_shaders[1].GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)
    assert shader.GetInput("roughness").Get() == 0.5
    assert tuple(shader.GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)
    assert sum(prim.IsA(UsdShade.Material) for prim in stage.Traverse()) == 3

    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_materials()
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_material_randomization_replaces_connections_only_on_selected_targets(event):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    source = UsdShade.Material.Define(stage, "/World/Looks/Source")
    shader = UsdShade.Shader.Define(stage, str(source.GetPath()) + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, str(source.GetPath()) + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    source.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    env_paths = [f"/World/envs/env_{index}" for index in range(2)]
    meshes = []
    for env_path in env_paths:
        UsdGeom.Xform.Define(stage, env_path + "/Object")
        mesh = UsdGeom.Cube.Define(stage, env_path + "/Object/mesh").GetPrim()
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(source)
        meshes.append(mesh)
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = env_paths
    env = SimpleNamespace(scene=scene, device="cpu")
    asset = SimpleNamespace(name="object")

    def reset_material():
        if event == "scalar":
            materials.randomize_material_scalar(env, torch.tensor([0]), asset, ("mesh",), "roughness", (0.8, 0.8))
        else:
            materials.randomize_material_color(
                env, torch.tensor([0]), asset, ("mesh",), "diffuseColor", ((0.1, 0.2, 0.3),)
            )

    reset_material()
    bound, _ = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial()
    assert bound.GetPath() != source.GetPath()
    result_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/Shader"))
    input_name = "roughness" if event == "scalar" else "diffuseColor"
    result = result_shader.GetInput(input_name)
    assert not result.GetConnectedSources()[0]
    assert [attribute.GetPath() for attribute in result.GetValueProducingAttributes()] == [result.GetAttr().GetPath()]
    value = result.Get() if event == "scalar" else tuple(result.Get())
    assert value == pytest.approx(0.8 if event == "scalar" else (0.1, 0.2, 0.3))
    other_input = "diffuseColor" if event == "scalar" else "roughness"
    assert result_shader.GetInput(other_input).GetConnectedSources()[0]
    for name in ("roughness", "diffuseColor"):
        assert shader.GetInput(name).GetConnectedSources()[0]
        assert shader.GetInput(name).Get() is None
    assert UsdShade.MaterialBindingAPI(meshes[1]).ComputeBoundMaterial()[0].GetPath() == source.GetPath()

    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_material()
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_material_randomization_excludes_generated_materials_and_reuses_root_override(monkeypatch, event):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    source = UsdShade.Material.Define(stage, "/World/Looks/Source")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Source/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    source.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    root_path = "/World/envs/env_0/MaterialPanel"
    root = UsdGeom.Cube.Define(stage, root_path).GetPrim()
    UsdShade.MaterialBindingAPI.Apply(root).Bind(source)
    scene = Scene(panel=SimpleNamespace(cfg=SimpleNamespace(prim_path=root_path)))
    scene.cfg = SimpleNamespace(panel=scene["panel"].cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    ids = torch.tensor([0])
    asset_cfg = SimpleNamespace(name="panel")
    materials.randomize_material_scalar(env, ids, asset_cfg, ("MaterialPanel",), "roughness", (0.2, 0.2))
    initial_material, _ = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial()
    overrides_path = root.GetPath().AppendChild("DomainRandomizationMaterials")
    compute_bound_material = UsdShade.MaterialBindingAPI.ComputeBoundMaterial

    def compute_visual_material(binding_api, *args, **kwargs):
        # Fail before a regression can create overrides indefinitely during traversal.
        assert not binding_api.GetPrim().GetPath().HasPrefix(overrides_path)
        return compute_bound_material(binding_api, *args, **kwargs)

    monkeypatch.setattr(UsdShade.MaterialBindingAPI, "ComputeBoundMaterial", compute_visual_material)

    def reset_materials():
        if event == "scalar":
            materials.randomize_material_scalar(env, ids, asset_cfg, ("Material",), "roughness", (0.8, 0.8))
        else:
            materials.randomize_material_color(env, ids, asset_cfg, ("Material",), "diffuseColor", ((0.1, 0.2, 0.3),))

    reset_materials()
    bound_material, _ = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial()
    assert bound_material.GetPath() == initial_material.GetPath()
    bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound_material.GetPath()) + "/Shader"))
    if event == "scalar":
        assert bound_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    else:
        assert tuple(bound_shader.GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert sum(prim.IsA(UsdShade.Material) for prim in stage.Traverse()) == 2
    assert all(
        not prim.GetRelationship("material:binding") for prim in Usd.PrimRange(stage.GetPrimAtPath(overrides_path))
    )

    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_materials()
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    "binding_kind",
    ["direct", "collection", "explicit_collection", "expression", "descendant_collection", "descendant_expression"],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("clone_source", [False, True])
@pytest.mark.parametrize("global_ancestor", [False, True])
def test_material_dr_overrides_strong_ancestors_without_changing_siblings(
    binding_kind, nested, clone_source, global_ancestor
):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    sources = []
    for name, color, roughness in (
        ("Red", (1.0, 0.0, 0.0), 0.2),
        ("Blue", (0.0, 0.0, 1.0), 0.4),
        ("Gray", (0.5, 0.5, 0.5), 0.5),
    ):
        material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        sources.append(material)

    roots = [f"/World/envs/env_{index}/Object" for index in range(2)]

    def define_root(path):
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        branch = UsdGeom.Xform.Define(stage, path + "/branch").GetPrim()
        UsdGeom.Xform.Define(stage, path + "/branch/target")
        for suffix in (
            "/branch/target/leaf",
            "/branch/peer",
            "/sibling/protected_child",
            "/sibling/protected_child/reincluded",
            "/untouched",
        ):
            UsdGeom.Cube.Define(stage, path + suffix)
        for suffix in ("/branch/target/authored", "/sibling"):
            prim = UsdGeom.Cube.Define(stage, path + suffix).GetPrim()
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(sources[1])
        root_api = UsdShade.MaterialBindingAPI.Apply(root)
        if binding_kind == "direct":
            root_api.Bind(sources[0], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            collection = Usd.CollectionAPI.Apply(root, "sibling_override")
            collection.CreateIncludesRel().SetTargets([root.GetPath().AppendChild("sibling")])
            collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
            root_api.Bind(collection, sources[1], bindingName="sibling_override")
        else:
            descendant_only = binding_kind.startswith("descendant_")
            root_api.Bind(
                sources[0] if descendant_only else sources[2],
                bindingStrength=(
                    UsdShade.Tokens.strongerThanDescendants
                    if descendant_only and nested
                    else UsdShade.Tokens.weakerThanDescendants
                ),
            )
            collection = Usd.CollectionAPI.Apply(root, "painted")
            paths = [branch.GetPath(), root.GetPath().AppendChild("sibling")]
            if descendant_only:
                # Block children without including the selected parent in the collection.
                paths = [
                    Sdf.Path(path + suffix)
                    for suffix in ("/branch/target/leaf", "/branch/target/authored", "/branch/peer", "/sibling")
                ]
            excluded = Sdf.Path(path + "/sibling/protected_child")
            if binding_kind == "explicit_collection":
                collection.CreateExpansionRuleAttr().Set(Usd.Tokens.explicitOnly)
                collection.CreateIncludesRel().SetTargets([Sdf.Path(path + "/branch/target")])
            elif binding_kind in ("collection", "descendant_collection"):
                collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
                collection.CreateIncludesRel().SetTargets(paths + [excluded.AppendChild("reincluded")])
                collection.CreateExcludesRel().SetTargets([excluded])
            else:
                expression = Sdf.PathExpression.Nothing()
                for included in paths:
                    expression = Sdf.PathExpression.MakeOp(
                        Sdf.PathExpression.Union, expression, Sdf.PathExpression(str(included) + "//")
                    )
                expression = Sdf.PathExpression.MakeOp(
                    Sdf.PathExpression.Difference, expression, Sdf.PathExpression(str(excluded) + "//")
                )
                collection.CreateMembershipExpressionAttr().Set(expression)
            root_api.Bind(
                collection,
                sources[2] if descendant_only else sources[0],
                bindingName="painted",
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        if nested:
            UsdShade.MaterialBindingAPI.Apply(branch).Bind(
                sources[1], bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )

    define_root(roots[0])
    if clone_source:
        UsdGeom.Xform.Define(stage, roots[1])
        Sdf.CopySpec(stage.GetRootLayer(), roots[0], stage.GetRootLayer(), roots[1])
    else:
        define_root(roots[1])
    if global_ancestor:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/envs")).Bind(
            sources[0], bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = [f"/World/envs/env_{index}" for index in range(2)]
    env = SimpleNamespace(scene=scene, device="cpu")
    asset = SimpleNamespace(name="object")
    target_path = roots[0] + "/branch/target"

    def values(path):
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()
        shader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
        return material.GetPath(), tuple(shader.GetInput("diffuseColor").Get()), shader.GetInput("roughness").Get()

    outside = {
        prim.GetPath(): values(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Imageable)
        and not prim.GetPath().HasPrefix(Sdf.Path(target_path))
        and UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    }
    ids = torch.tensor([0])
    materials.randomize_material_scalar(env, ids, asset, ("target",), "roughness", (0.8, 0.8))
    assert values(target_path)[1] == (1.0, 0.0, 0.0)
    assert values(target_path)[2] == pytest.approx(0.8)
    assert values(target_path + "/leaf") == values(target_path)
    assert values(target_path + "/authored")[0] == sources[1].GetPath()
    for path, original in outside.items():
        assert values(path) == original

    materials.randomize_material_color(env, ids, asset, ("leaf",), "diffuseColor", ((0.0, 1.0, 0.0),))
    materials.randomize_material_scalar(env, ids, asset, ("target",), "roughness", (0.9, 0.9))
    assert values(target_path + "/leaf")[1] == (0.0, 1.0, 0.0)
    assert values(target_path + "/leaf")[2] == pytest.approx(0.9)
    assert values(target_path + "/authored")[0] == sources[1].GetPath()
    for path, original in outside.items():
        assert values(path) == original

    materials.randomize_material_scalar(env, torch.tensor([1]), asset, ("target",), "roughness", (0.7, 0.7))
    assert values(roots[1] + "/branch/target")[2] == pytest.approx(0.7)
    assert values(target_path)[2] == pytest.approx(0.9)
    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        materials.randomize_material_scalar(env, ids, asset, ("target",), "roughness", (0.9, 0.9))
        materials.randomize_material_scalar(env, torch.tensor([1]), asset, ("target",), "roughness", (0.7, 0.7))
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("parent_first", [True, False])
@pytest.mark.parametrize("clone_source", [True, False])
def test_material_overrides_preserve_parent_properties_across_creation_orders(parent_first, clone_source):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Shared")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Shared/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    root = "/World/envs/env_0/Object"
    UsdGeom.Xform.Define(stage, root)
    group = UsdGeom.Xform.Define(stage, f"{root}/panel_group").GetPrim()
    UsdGeom.Xform.Define(stage, f"{root}/panel_group/middle")
    UsdGeom.Cube.Define(stage, f"{root}/panel_group/middle/insert")
    direct_insert = UsdGeom.Cube.Define(stage, f"{root}/panel_group/middle/direct_insert").GetPrim()
    UsdGeom.Cube.Define(stage, f"{root}/panel_group/sibling")
    UsdGeom.Cube.Define(stage, f"{root}/panel_group/color_only")
    UsdShade.MaterialBindingAPI.Apply(group).Bind(material)
    UsdShade.MaterialBindingAPI.Apply(direct_insert).Bind(material)
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    asset_cfg = SimpleNamespace(name="object")
    materials.randomize_material_color(
        env, torch.tensor([0]), asset_cfg, ("color_only",), "diffuseColor", ((0.1, 0.2, 0.3),)
    )

    def reset_materials(env_id):
        ids = torch.tensor([env_id])
        if not parent_first:
            materials.randomize_material_scalar(env, ids, asset_cfg, ("insert",), "roughness", (0.8, 0.8))
        materials.randomize_material_scalar(env, ids, asset_cfg, ("panel_group",), "roughness", (0.2, 0.2))
        materials.randomize_material_color(env, ids, asset_cfg, ("panel_group",), "diffuseColor", ((0.9, 0.4, 0.1),))
        if parent_first:
            materials.randomize_material_scalar(env, ids, asset_cfg, ("insert",), "roughness", (0.8, 0.8))

    if clone_source:
        reset_materials(0)
        root = "/World/envs/env_1/Object"
        UsdGeom.Xform.Define(stage, root)
        Sdf.CopySpec(stage.GetRootLayer(), "/World/envs/env_0/Object", stage.GetRootLayer(), root)
        scene.env_prim_paths.append("/World/envs/env_1")
    env_id = int(clone_source)
    ids = torch.tensor([env_id])
    reset_materials(env_id)

    def bound(node):
        prim = stage.GetPrimAtPath(f"{root}/panel_group{node}")
        bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound_material.GetPath()) + "/Shader"))
        return bound_material, bound_shader

    group_material, group_shader = bound("")
    insert_material, insert_shader = bound("/middle/insert")
    assert group_material.GetPath() != insert_material.GetPath()
    assert bound("/sibling")[0].GetPath() == group_material.GetPath()
    assert insert_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    assert group_shader.GetInput("roughness").Get() == pytest.approx(0.2)
    assert tuple(insert_shader.GetInput("diffuseColor").Get()) == pytest.approx((0.9, 0.4, 0.1))
    color_only = bound("/color_only")[1]
    assert tuple(color_only.GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert color_only.GetInput("roughness").Get() == pytest.approx(0.2)
    materials.randomize_material_scalar(env, ids, asset_cfg, ("panel_group",), "roughness", (0.6, 0.6))
    assert color_only.GetInput("roughness").Get() == pytest.approx(0.6)
    assert insert_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    # An authored child binding keeps its own color even when it shares the source material.
    assert tuple(bound("/middle/direct_insert")[1].GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)

    # Inserting an override between the group and child must preserve the chain.
    materials.randomize_material_scalar(env, ids, asset_cfg, ("middle",), "roughness", (0.4, 0.4))
    assert bound("/middle")[1].GetInput("roughness").Get() == pytest.approx(0.4)
    assert insert_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    materials.randomize_material_color(env, ids, asset_cfg, ("panel_group",), "diffuseColor", ((0.0, 1.0, 0.0),))
    assert tuple(insert_shader.GetInput("diffuseColor").Get()) == (0.0, 1.0, 0.0)
    materials.randomize_material_color(env, ids, asset_cfg, ("middle",), "diffuseColor", ((0.7, 0.1, 0.9),))
    materials.randomize_material_color(env, ids, asset_cfg, ("panel_group",), "diffuseColor", ((1.0, 1.0, 0.0),))
    assert tuple(insert_shader.GetInput("diffuseColor").Get()) == pytest.approx((0.7, 0.1, 0.9))
    assert tuple(bound("/middle/direct_insert")[1].GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)

    materials.randomize_material_color(env, ids, asset_cfg, ("insert",), "diffuseColor", ((0.1, 0.2, 0.3),))
    materials.randomize_material_color(env, ids, asset_cfg, ("panel_group",), "diffuseColor", ((1.0, 0.0, 0.0),))
    assert bound("/middle/insert")[0].GetPath() == insert_material.GetPath()
    assert tuple(insert_shader.GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(group_shader.GetInput("diffuseColor").Get()) == (1.0, 0.0, 0.0)
    assert insert_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    assert shader.GetInput("roughness").Get() == 0.5
    assert tuple(shader.GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)

    reset_materials(env_id)
    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_materials(env_id)
        assert stage.GetRootLayer().ExportToString() == before
    if clone_source:
        reset_materials(0)
        assert tuple(insert_shader.GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
        assert insert_shader.GetInput("roughness").Get() == pytest.approx(0.8)


@pytest.mark.parametrize("has_base_material", [False, True])
def test_authored_materials_in_override_scope_remain_valid_sources(has_base_material):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/Object"
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Scope.Define(stage, root + "/DomainRandomizationMaterials")
    material = UsdShade.Material.Define(stage, root + "/DomainRandomizationMaterials/Paint")
    if has_base_material:
        base = UsdShade.Material.Define(stage, "/World/Looks/Base")
        base_shader = UsdShade.Shader.Define(stage, "/World/Looks/Base/Shader")
        base_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
        material.GetPrim().GetSpecializes().AddSpecialize(base.GetPath())
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Cube.Define(stage, root + "/mesh").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path=root)))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")

    def reset_material():
        materials.randomize_material_scalar(
            env, torch.tensor([0]), SimpleNamespace(name="object"), ("mesh",), "roughness", (0.8, 0.8)
        )

    reset_material()
    bound_material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert bound_material.GetPath() != material.GetPath()
    assert bound_material.GetBaseMaterialPath() == material.GetPath()
    bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound_material.GetPath()) + "/Shader"))
    assert bound_shader.GetInput("roughness").Get() == pytest.approx(0.8)
    assert tuple(bound_shader.GetInput("diffuseColor").Get()) == (1.0, 0.0, 0.0)
    assert shader.GetInput("roughness").Get() == 0.5

    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_material()
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_material_randomization_preserves_inherited_binding_when_input_is_missing(event):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    source_materials = []
    shader_input = "roughness" if event == "scalar" else "specularColor"
    for name, has_input in (("White", False), ("Supported", True)):
        path = f"/World/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
        if has_input:
            value_type = Sdf.ValueTypeNames.Float if event == "scalar" else Sdf.ValueTypeNames.Color3f
            shader.CreateInput(shader_input, value_type).Set(0.5 if event == "scalar" else Gf.Vec3f(0.5))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        source_materials.append(material)

    root = "/World/envs/env_0/Object"
    group = UsdGeom.Xform.Define(stage, root + "/panel_group").GetPrim()
    insert_a = UsdGeom.Cube.Define(stage, root + "/panel_group/insert_A").GetPrim()
    insert_b = UsdGeom.Cube.Define(stage, root + "/panel_group/insert_B").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(group).Bind(source_materials[0])
    UsdShade.MaterialBindingAPI.Apply(insert_b).Bind(source_materials[1])
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path=root)))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    ids = torch.tensor([0])
    asset_cfg = SimpleNamespace(name="object")
    materials.randomize_material_color(env, ids, asset_cfg, ("panel_group",), "diffuseColor", ((1.0, 0.0, 0.0),))
    inherited_material, _ = UsdShade.MaterialBindingAPI(insert_a).ComputeBoundMaterial()

    def reset_materials():
        if event == "scalar":
            materials.randomize_material_scalar(env, ids, asset_cfg, ("insert",), shader_input, (0.8, 0.8))
        else:
            materials.randomize_material_color(env, ids, asset_cfg, ("insert",), shader_input, ((0.1, 0.2, 0.3),))

    reset_materials()
    skipped_material, _ = UsdShade.MaterialBindingAPI(insert_a).ComputeBoundMaterial()
    assert skipped_material.GetPath() == inherited_material.GetPath()
    assert not insert_a.GetRelationship("material:binding")
    skipped_shader = UsdShade.Shader(stage.GetPrimAtPath(str(skipped_material.GetPath()) + "/Shader"))
    assert tuple(skipped_shader.GetInput("diffuseColor").Get()) == (1.0, 0.0, 0.0)
    bound_material, _ = UsdShade.MaterialBindingAPI(insert_b).ComputeBoundMaterial()
    assert bound_material.GetPath() != source_materials[1].GetPath()
    assert bound_material.GetBaseMaterialPath() == source_materials[1].GetPath()
    bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound_material.GetPath()) + "/Shader"))
    assert bound_shader.GetInput(shader_input).Get() == pytest.approx(0.8 if event == "scalar" else (0.1, 0.2, 0.3))
    assert sum(prim.IsA(UsdShade.Material) for prim in stage.Traverse()) == 4

    before = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_materials()
        assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("event", ["scalar", "color"])
def test_partial_material_resets_preserve_unselected_parent_properties(event):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Source")
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    roots = [f"/World/envs/env_{index}/Object" for index in range(2)]
    UsdGeom.Xform.Define(stage, roots[0])
    box = UsdGeom.Xform.Define(stage, roots[0] + "/box").GetPrim()
    UsdGeom.Cube.Define(stage, roots[0] + "/box/handle")
    UsdShade.MaterialBindingAPI.Apply(box).Bind(material)
    UsdGeom.Xform.Define(stage, roots[1])
    Sdf.CopySpec(stage.GetRootLayer(), roots[0], stage.GetRootLayer(), roots[1])
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = [f"/World/envs/env_{index}" for index in range(2)]
    env = SimpleNamespace(scene=scene, device="cpu")
    asset = SimpleNamespace(name="object")
    all_ids = torch.tensor([0, 1])

    def reset_material(ids, marker, value):
        if event == "scalar":
            materials.randomize_material_scalar(env, ids, asset, (marker,), "roughness", (value, value))
        else:
            materials.randomize_material_color(env, ids, asset, (marker,), "diffuseColor", ((value, 0.2, 0.1),))

    # Give each handle its own override while leaving the tested input inherited.
    if event == "scalar":
        materials.randomize_material_color(env, all_ids, asset, ("handle",), "diffuseColor", ((0.1, 0.2, 0.3),))
    else:
        materials.randomize_material_scalar(env, all_ids, asset, ("handle",), "roughness", (0.8, 0.8))
    reset_material(all_ids, "box", 0.3)

    def unselected_specs():
        layer = Sdf.Layer.CreateAnonymous()
        Sdf.CreatePrimInLayer(layer, roots[1])
        Sdf.CopySpec(stage.GetRootLayer(), roots[1], layer, roots[1])
        return layer.ExportToString()

    before = unselected_specs()
    reset_material(torch.tensor([0]), "handle", 0.9)
    assert unselected_specs() == before

    # The unselected handle must still follow its own parent's next update.
    reset_material(torch.tensor([1]), "box", 0.7)
    handle = stage.GetPrimAtPath(roots[1] + "/box/handle")
    bound, _ = UsdShade.MaterialBindingAPI(handle).ComputeBoundMaterial()
    result = UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/Shader"))
    assert result.GetInput("roughness").Get() == pytest.approx(0.7 if event == "scalar" else 0.8)
    assert tuple(result.GetInput("diffuseColor").Get()) == pytest.approx(
        (0.1, 0.2, 0.3) if event == "scalar" else (0.7, 0.2, 0.1)
    )


@pytest.mark.parametrize("event", ["scalar", "color"])
@pytest.mark.parametrize("randomize_before_cloning", [False, True])
def test_partial_material_resets_preserve_copied_environments(event, randomize_before_cloning):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Source")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Source/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.4, 0.5, 0.6))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Object").GetPrim()
    mesh = UsdGeom.Cube.Define(stage, str(root.GetPath()) + "/mesh").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    ids, asset = torch.tensor([0]), SimpleNamespace(name="object")

    def reset_material(value):
        if event == "scalar":
            materials.randomize_material_scalar(env, ids, asset, ("mesh",), "roughness", (value, value))
        else:
            materials.randomize_material_color(env, ids, asset, ("mesh",), "diffuseColor", ((value, 0.2, 0.1),))

    if randomize_before_cloning:
        reset_material(0.3)
    UsdGeom.Xform.Define(stage, "/World/envs/env_1/Object")
    Sdf.CopySpec(stage.GetRootLayer(), root.GetPath(), stage.GetRootLayer(), "/World/envs/env_1/Object")
    scene.env_prim_paths.append("/World/envs/env_1")
    copied = stage.GetPrimAtPath("/World/envs/env_1/Object/mesh")

    def values(prim):
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/Shader"))
        return bound_shader.GetInput("roughness").Get(), tuple(bound_shader.GetInput("diffuseColor").Get())

    before = values(copied)
    rng = torch.get_rng_state().clone()
    reset_material(0.8)
    assert values(copied) == before
    assert values(mesh)[0 if event == "scalar" else 1] == pytest.approx(0.8 if event == "scalar" else (0.8, 0.2, 0.1))
    stage_after_reset = stage.GetRootLayer().ExportToString()
    for _ in range(3):
        reset_material(0.8)
        assert values(copied) == before
        assert stage.GetRootLayer().ExportToString() == stage_after_reset
    assert torch.equal(rng, torch.get_rng_state())
    assert shader.GetInput("roughness").Get() == 0.5
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.4, 0.5, 0.6))


def test_material_resets_sample_independently_and_isolate_shared_source_material():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Shared")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Shared/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    env_paths = [f"/World/envs/env_{i}" for i in range(3)]
    meshes = []
    for env_path in env_paths:
        UsdGeom.Xform.Define(stage, env_path + "/Object")
        mesh = UsdGeom.Cube.Define(stage, env_path + "/Object/mesh").GetPrim()
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
        meshes.append(mesh)
    scene = Scene(object=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Object")))
    scene.cfg = SimpleNamespace(object=scene["object"].cfg)
    scene.stage = stage
    scene.env_prim_paths = env_paths
    env = SimpleNamespace(scene=scene, device="cpu")
    params = {"asset_cfg": SimpleNamespace(name="object"), "prim_name_markers": ("mesh",)}
    materials.randomize_material_color(
        env, torch.tensor([1]), **params, shader_input="diffuseColor", color_palette=((0.1, 0.2, 0.3),)
    )
    materials.randomize_material_scalar(
        env, torch.tensor([1]), **params, shader_input="roughness", value_range=(0.8, 0.8)
    )
    original, _ = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial()
    randomized, _ = UsdShade.MaterialBindingAPI(meshes[1]).ComputeBoundMaterial()
    assert original.GetPath() == material.GetPath()
    assert randomized.GetPath() != original.GetPath()
    result = UsdShade.Shader(stage.GetPrimAtPath(str(randomized.GetPath()) + "/Shader"))
    assert tuple(result.GetInput("diffuseColor").Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert result.GetInput("roughness").Get() == pytest.approx(0.8)
    assert tuple(shader.GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)
    assert shader.GetInput("roughness").Get() == 0.5
    before = stage.GetRootLayer().ExportToString()
    materials.randomize_material_scalar(
        env, torch.tensor([1]), **params, shader_input="roughness", value_range=(0.8, 0.8)
    )
    assert stage.GetRootLayer().ExportToString() == before

    def reset_materials(env_ids):
        materials.randomize_material_color(
            env, env_ids, **params, shader_input="diffuseColor", color_palette=((0, 0, 0), (1, 1, 1))
        )
        materials.randomize_material_scalar(env, env_ids, **params, shader_input="roughness", value_range=(0.1, 0.9))
        values = []
        for mesh in meshes[1:]:
            bound_material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            bound_shader = UsdShade.Shader(stage.GetPrimAtPath(str(bound_material.GetPath()) + "/Shader"))
            values.append(
                (tuple(bound_shader.GetInput("diffuseColor").Get()), bound_shader.GetInput("roughness").Get())
            )
        return values

    ids = torch.tensor([2, 1])
    torch.manual_seed(7)
    first = reset_materials(ids)
    assert first[0][0] != first[1][0]
    assert first[0][1] != first[1][1]
    assert all(0 <= component <= 1 for color, _ in first for component in color)
    assert all(0.1 <= scalar <= 0.9 for _, scalar in first)
    torch.manual_seed(7)
    assert reset_materials(ids) == first

    torch.manual_seed(11)
    second = reset_materials(torch.tensor([1]))
    assert second[0] != first[0]
    assert second[1] == first[1]
    assert tuple(shader.GetInput("diffuseColor").Get()) == (1.0, 1.0, 1.0)
    assert shader.GetInput("roughness").Get() == 0.5


@pytest.mark.parametrize("event", ["light", "color", "scalar"])
def test_visual_empty_reset_does_not_consume_rng_or_access_the_scene(event):
    env = SimpleNamespace(device="cpu")
    ids = torch.tensor([], dtype=torch.long)
    rng = torch.get_rng_state().clone()
    if event == "light":
        lighting.randomize_light(env, ids, (100, 200), (3000, 8000), object())
    elif event == "color":
        materials.randomize_material_color(env, ids, object(), ("mesh",), "diffuseColor", ((0, 0, 0), (1, 1, 1)))
    else:
        materials.randomize_material_scalar(env, ids, object(), ("mesh",), "roughness", (0.1, 0.9))
    assert torch.equal(rng, torch.get_rng_state())


def test_visual_example_applies_to_registered_g1_task():
    from pathlib import Path

    from autodata_interfaces.env import setup_env_config
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    base_env = "IsaacContrib-PickPlace-Locomanipulation-G1-Abs"
    root = Path(__file__).resolve().parents[2]
    env_cfg, _ = setup_env_config(
        base_env,
        "/tmp",
        "randomized",
        1,
        "cpu",
        GenerationPolicy(),
        env_profile=EnvironmentProfile.from_yaml(root / "autodata_examples/env_profiles/g1_locomanip_dr.yaml"),
        enable_cameras=True,
    )
    assert not hasattr(env_cfg.events, "randomize_light")
    event = env_cfg.events.randomize_camera_0
    assert event.params["asset_cfg"].name == "robot_pov_cam"
    assert event.params["focal_length_scale_range"] == (0.98, 1.02)
    assert env_cfg.scene.robot_pov_cam is not None
    assert "robot_pov_cam" in env_cfg.image_obs_list
    assert not hasattr(env_cfg.events, "reset_all")
    pose_event = env_cfg.events.reset_object_pose
    assert pose_event.params["asset_cfg"].name == "object"
    assert pose_event.params["pose_range"]["x"] == (-0.02, 0.02)
    assert pose_event.params["pose_range"]["yaw"] == (-0.15, 0.15)
    event_names = list(vars(env_cfg.events))
    assert event_names.index("reset_scene_to_default") < event_names.index("reset_object_pose")
    color_event = env_cfg.events.randomize_material_color_0
    assert color_event.params["asset_cfg"].name == "object"
    assert color_event.params["prim_name_markers"] == ("wheel_rim",)
    assert color_event.params["shader_input"] == "diffuse_tint"
    scalar_event = env_cfg.events.randomize_material_scalar_0
    assert scalar_event.params["asset_cfg"].name == "object"
    assert scalar_event.params["shader_input"] == "reflection_roughness_constant"


def test_robot_randomization_rejects_an_unrelated_reset_function(tmp_path):
    config = tmp_path / "robot.yaml"
    config.write_text("""
name: test_dr
base_env: TestEnv-v0
domain_randomization:
  robot_root: {}
""")
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(), events=SimpleNamespace(reset_all=SimpleNamespace(func=lambda: None))
    )
    with pytest.raises(AssertionError, match="requires Arena's"):
        apply_domain_randomization(env_cfg, "Test", EnvironmentProfile.from_yaml(config).domain_randomization, 1)


@pytest.mark.parametrize("config_type", [MaterialColorRandomizationCfg, MaterialScalarRandomizationCfg])
def test_material_markers_reject_strings(config_type):
    kwargs = (
        {"color_palette": ((1, 1, 1),)} if config_type is MaterialColorRandomizationCfg else {"value_range": (0, 1)}
    )
    with pytest.raises(AssertionError, match="must be a sequence"):
        config_type(asset_name="object", prim_name_markers="mesh", shader_input="input", **kwargs)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_lighting_and_pose_bounds_reject_nonfinite_values(value):
    with pytest.raises(AssertionError, match="finite"):
        LightingRandomizationCfg(intensity_range=(0, value))
    with pytest.raises(AssertionError, match="finite"):
        PoseRandomizationCfg(event_name="object", position_xyz_min=(0, 0, 0), position_xyz_max=(value, 0, 0))


def test_randomization_overrides_events_added_by_the_same_profile():
    profile = EnvironmentProfile.from_dict({
        "name": "with_new_reset",
        "base_env": "TestEnv-v0",
        "events": {
            "add": {
                "reset_object": {
                    "func": "isaaclab.envs.mdp.events:reset_root_state_uniform",
                    "mode": "reset",
                    "params": {"asset_cfg": "object", "pose_range": {"x": [0.0, 0.0]}, "velocity_range": {}},
                }
            }
        },
        "domain_randomization": {
            "poses": [{
                "event_name": "reset_object",
                "position_xyz_min": [-0.1, 0, 0],
                "position_xyz_max": [0.1, 0, 0],
            }]
        },
    })
    env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2, object=object()), events=SimpleNamespace())
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    term = env_cfg.events.reset_object
    assert term.params["pose_range"]["x"] == (-0.1, 0.1)
    assert term.params["velocity_range"] == {}
    assert term.params["asset_cfg"].name == "object"


def test_empty_randomization_does_not_add_events():
    profile = EnvironmentProfile.from_dict({"name": "empty", "base_env": "TestEnv-v0", "domain_randomization": {}})
    env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2), events=SimpleNamespace())
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    assert vars(env_cfg.events) == {}


def test_generation_requires_cameras_for_camera_randomization():
    from autodata_interfaces.env import setup_env_config
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    base_env = "IsaacContrib-Stack-Cube-Franka-IK-Rel"
    profile = EnvironmentProfile.from_dict({
        "name": "camera",
        "base_env": base_env,
        "domain_randomization": {"cameras": [{"asset_name": "cam"}]},
    })
    with pytest.raises(AssertionError, match="requires --enable_cameras"):
        setup_env_config(base_env, "/tmp", "camera", 2, "cpu", GenerationPolicy(), env_profile=profile)


@pytest.mark.parametrize("light_type", ["DomeLight", "RectLight"])
@pytest.mark.parametrize("prim_path", ["/World/Light", "/World/envs/env_0/Light"])
def test_single_environment_lighting_supports_global_and_local_lights(light_type, prim_path):
    from isaaclab.assets import AssetBaseCfg
    from pxr import Usd, UsdLux

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    light = getattr(UsdLux, light_type).Define(stage, prim_path)
    light_cfg = AssetBaseCfg(prim_path=prim_path)
    scene = Scene(light=light_cfg)
    scene.cfg = SimpleNamespace(light=light_cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0"]
    env = SimpleNamespace(scene=scene, device="cpu")
    paths = [str(prim.GetPath()) for prim in stage.Traverse()]
    lighting.randomize_light(env, torch.tensor([0]), (500.0, 500.0), (5500.0, 5500.0), SimpleNamespace(name="light"))
    assert light.GetIntensityAttr().Get() == 500.0
    assert tuple(light.GetColorAttr().Get()) == pytest.approx(color_temperature_to_rgb(5500.0))
    assert [str(prim.GetPath()) for prim in stage.Traverse()] == paths


@pytest.mark.parametrize("global_light", [True, False])
@pytest.mark.parametrize("env_ids", [[0], [0, 1]])
def test_parallel_dome_randomization_is_rejected_before_modifying_the_scene(global_light, env_ids):
    from isaaclab.assets import AssetBaseCfg
    from pxr import Usd, UsdLux

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    roots = ["/World/envs/env_0", "/World/envs/env_1"]
    for path in ["/World/Dome"] if global_light else [root + "/Dome" for root in roots]:
        UsdLux.DomeLight.Define(stage, path)
    pattern = "/World/Dome" if global_light else "/World/envs/env_.*/Dome"
    light_cfg = AssetBaseCfg(prim_path=pattern)
    scene = Scene(light=light_cfg)
    scene.cfg = SimpleNamespace(light=light_cfg)
    scene.stage = stage
    scene.env_prim_paths = roots
    env = SimpleNamespace(scene=scene, device="cpu")
    original = stage.GetRootLayer().ExportToString()
    rng = torch.get_rng_state().clone()
    with pytest.raises(AssertionError, match="Dome-light DR requires num_envs=1"):
        lighting.randomize_light(
            env, torch.tensor(env_ids), (100.0, 900.0), (3000.0, 8000.0), SimpleNamespace(name="light")
        )
    assert stage.GetRootLayer().ExportToString() == original
    assert torch.equal(rng, torch.get_rng_state())


def test_partial_light_reset_preserves_existing_inherited_light_values():
    from isaaclab.assets import AssetBaseCfg
    from pxr import Usd, UsdLux

    stage = Usd.Stage.CreateInMemory()
    roots = ["/World/envs/env_0", "/World/envs/env_1"]
    source = UsdLux.RectLight.Define(stage, roots[0] + "/Light")
    source.GetIntensityAttr().Set(200.0)
    source.GetColorAttr().Set((0.4, 0.5, 0.6))
    inherited = stage.DefinePrim(roots[1] + "/Light")
    inherited.GetInherits().AddInherit(source.GetPath())
    names = ("inputs:intensity", "inputs:color")
    initial = [inherited.GetAttribute(name).Get() for name in names]
    assert all(
        stage.GetRootLayer().GetAttributeAtPath(inherited.GetPath().AppendProperty(name)) is None for name in names
    )
    scene = SimpleNamespace(
        cfg=SimpleNamespace(light=AssetBaseCfg(prim_path="/World/envs/env_.*/Light")),
        stage=stage,
        env_prim_paths=roots,
    )
    env = SimpleNamespace(scene=scene, device="cpu")
    paths = [prim.GetPath() for prim in stage.Traverse()]

    lighting.randomize_light(env, torch.tensor([0]), (500.0, 500.0), (5500.0, 5500.0), SimpleNamespace(name="light"))

    assert source.GetIntensityAttr().Get() == 500.0
    assert tuple(source.GetColorAttr().Get()) == pytest.approx(color_temperature_to_rgb(5500.0))
    assert inherited.GetAttribute("inputs:intensity").Get() == initial[0]
    assert inherited.GetAttribute("inputs:color").Get() == initial[1]
    assert [prim.GetPath() for prim in stage.Traverse()] == paths


def test_existing_non_dome_lights_are_seeded_and_only_selected_environments_change():
    from isaaclab.assets import AssetBaseCfg
    from pxr import Usd, UsdLux

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    roots = ["/World/envs/env_0", "/World/envs/env_1"]
    lamps = [UsdLux.RectLight.Define(stage, root + "/Light").GetPrim() for root in roots]
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.GetIntensityAttr().Set(250.0)
    light_cfg = AssetBaseCfg(prim_path="/World/envs/env_.*/Light")
    scene = Scene(light=light_cfg)
    scene.cfg = SimpleNamespace(light=light_cfg)
    scene.stage = stage
    scene.env_prim_paths = roots
    env = SimpleNamespace(scene=scene, device="cpu")
    asset = SimpleNamespace(name="light")
    paths = [str(prim.GetPath()) for prim in stage.Traverse()]

    def values():
        return [
            (prim.GetAttribute("inputs:intensity").Get(), tuple(prim.GetAttribute("inputs:color").Get()))
            for prim in lamps
        ]

    torch.manual_seed(47)
    lighting.randomize_light(env, torch.tensor([0, 1]), (100.0, 900.0), (2500.0, 8500.0), asset)
    first = values()
    torch.manual_seed(47)
    lighting.randomize_light(env, torch.tensor([0, 1]), (100.0, 900.0), (2500.0, 8500.0), asset)
    assert values() == first
    assert first[0] != first[1]
    lighting.randomize_light(env, torch.tensor([1]), (500.0, 500.0), (5500.0, 5500.0), asset)
    assert values()[0] == first[0]
    assert values()[1][0] == 500.0
    assert dome.GetIntensityAttr().Get() == 250.0
    assert [str(prim.GetPath()) for prim in stage.Traverse()] == paths


def test_global_non_dome_light_cannot_be_randomized_as_a_cloned_light():
    from isaaclab.assets import AssetBaseCfg
    from pxr import Usd, UsdLux

    class Scene(dict):
        pass

    stage = Usd.Stage.CreateInMemory()
    UsdLux.RectLight.Define(stage, "/World/Light")
    light_cfg = AssetBaseCfg(prim_path="/World/Light")
    scene = Scene(light=light_cfg)
    scene.cfg = SimpleNamespace(light=light_cfg)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    env = SimpleNamespace(scene=scene, device="cpu")
    with pytest.raises(AssertionError, match="Global lights cannot be randomized independently"):
        lighting.randomize_light(env, torch.tensor([0]), (100, 200), (3000, 8000), SimpleNamespace(name="light"))


@pytest.mark.parametrize("mode", ["dome", "local_box"])
def test_lighting_mode_switch_is_not_supported(mode):
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({"lighting": {"mode": mode}})
