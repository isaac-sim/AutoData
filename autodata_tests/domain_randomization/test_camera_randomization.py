# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera DR schema, event binding, and calibration regression tests."""
import inspect
import torch
from types import SimpleNamespace

import pytest

from autodata_domain_randomization import (
    CameraRandomizationCfg,
    DomainRandomizationCfg,
    LightingRandomizationCfg,
    MaterialColorRandomizationCfg,
    camera,
    create_domain_randomization_profile,
)
from autodata_interfaces.env import EnvironmentProfile, apply_env_profile


def test_training_profile_adds_episodic_camera_event():
    camera_cfg = CameraRandomizationCfg(
        asset_name="robot_head_cam",
        focal_length_scale_range=(0.98, 1.02),
        position_xyz_min=(-0.005, -0.005, -0.005),
        position_xyz_max=(0.005, 0.005, 0.005),
        rotation_rpy_min_deg=(-1.0, -1.0, -1.0),
        rotation_rpy_max_deg=(1.0, 1.0, 1.0),
    )
    profile = create_domain_randomization_profile(
        "TestEnv-v0", DomainRandomizationCfg(lighting=LightingRandomizationCfg(), cameras=(camera_cfg,))
    )
    event = profile.events.add["randomize_camera_0"]
    assert event.mode == "reset"
    assert event.params == {
        "asset_cfg": "robot_head_cam",
        "focal_length_scale_range": (0.98, 1.02),
        "position_xyz_min": (-0.005, -0.005, -0.005),
        "position_xyz_max": (0.005, 0.005, 0.005),
        "rotation_rpy_min_deg": (-1.0, -1.0, -1.0),
        "rotation_rpy_max_deg": (1.0, 1.0, 1.0),
    }


def test_camera_profile_applies_as_an_isaac_lab_reset_event():
    from isaaclab.managers import SceneEntityCfg

    env_cfg = SimpleNamespace(events=SimpleNamespace(), scene=SimpleNamespace(light=object(), robot_head_cam=object()))
    profile = create_domain_randomization_profile(
        "TestEnv-v0",
        DomainRandomizationCfg(
            cameras=(CameraRandomizationCfg(asset_name="robot_head_cam"),),
        ),
    )
    apply_env_profile(env_cfg, profile, "TestEnv-v0")
    event = env_cfg.events.randomize_camera_0
    assert event.func is camera.randomize_camera
    assert event.mode == "reset"
    assert isinstance(event.params["asset_cfg"], SceneEntityCfg)
    assert event.params["asset_cfg"].name == "robot_head_cam"


def test_camera_extrinsics_randomization_supports_parallel_environments():
    cfg = DomainRandomizationCfg(cameras=(CameraRandomizationCfg(asset_name="robot_head_cam"),))
    cfg.validate_runtime(num_envs=5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"asset_name": ""},
        {"asset_name": "cam", "focal_length_scale_range": (1.1, 0.9)},
        {"asset_name": "cam", "focal_length_scale_range": (0.0, 1.0)},
        {"asset_name": "cam", "position_xyz_min": (0.01, 0.0, 0.0), "position_xyz_max": (0.0, 0.0, 0.0)},
        {"asset_name": "cam", "rotation_rpy_min_deg": (0.0, 1.0, 0.0), "rotation_rpy_max_deg": (0.0, 0.0, 0.0)},
        {"asset_name": "cam", "position_xyz_min": (0.0, 0.0)},
    ],
)
def test_camera_config_rejects_invalid_values(kwargs):
    with pytest.raises(AssertionError):
        CameraRandomizationCfg(**kwargs)


def test_camera_yaml_section_builds_reset_profile(tmp_path):
    config_path = tmp_path / "domain_randomization.yaml"
    config_path.write_text("""
name: test_dr
base_env: TestEnv-v0
domain_randomization:
  cameras:
  - asset_name: robot_head_cam
    focal_length_scale_range:
    - 0.98
    - 1.02
    position_xyz_min:
    - -0.005
    - -0.005
    - -0.005
    position_xyz_max:
    - 0.005
    - 0.005
    - 0.005
    rotation_rpy_min_deg:
    - -1.0
    - -1.0
    - -1.0
    rotation_rpy_max_deg:
    - 1.0
    - 1.0
    - 1.0
""")
    cfg = EnvironmentProfile.from_yaml(config_path).domain_randomization
    cfg.validate_runtime(num_envs=1)
    profile = create_domain_randomization_profile("TestEnv-v0", cfg)
    assert cfg.cameras[0].focal_length_scale_range == (0.98, 1.02)
    assert cfg.cameras[0].rotation_rpy_max_deg == (1.0, 1.0, 1.0)
    assert profile.events.add["randomize_camera_0"].params["asset_cfg"] == "robot_head_cam"


def test_camera_yaml_section_rejects_unknown_keys():
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({"cameras": [{"asset_name": "cam", "focal_length_range": [1.0, 1.0]}]})


def test_profile_event_params_bind_to_their_event_functions():
    """Guard the config-to-event-function contract that only fails at reset time otherwise."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.string import string_to_callable

    from autodata_interfaces.env.env_profile import convert_event_params

    cfg = DomainRandomizationCfg(
        lighting=LightingRandomizationCfg(),
        cameras=(CameraRandomizationCfg(asset_name="robot_head_cam"),),
        material_colors=(
            MaterialColorRandomizationCfg(
                asset_name="apple",
                prim_name_markers=("mesh",),
                shader_input="base_color_factor",
                color_palette=((1.0, 1.0, 1.0),),
            ),
        ),
    )
    profile = create_domain_randomization_profile("TestEnv-v0", cfg)
    assert set(profile.events.add) == {"randomize_camera_0", "randomize_light", "randomize_material_color_0"}
    for term_name, add_spec in profile.events.add.items():
        func = string_to_callable(add_spec.func)
        params = convert_event_params(add_spec.params, SceneEntityCfg)
        inspect.signature(func).bind(object(), torch.zeros(1, dtype=torch.long), **params)


@pytest.mark.parametrize("field", ["focal_length_scale_range", "position_xyz_max", "rotation_rpy_max_deg"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_camera_bounds_must_be_finite(field, value):
    bounds = (1.0, value) if field == "focal_length_scale_range" else (value, value, value)
    with pytest.raises(AssertionError, match="finite"):
        CameraRandomizationCfg(asset_name="cam", **{field: bounds})


@pytest.mark.parametrize("present", [True, False])
def test_camera_presence_controls_reset_event(present):
    cfg = DomainRandomizationCfg.from_dict({"cameras": [{"asset_name": "cam"}]} if present else {})
    profile = create_domain_randomization_profile("Test", cfg)
    assert set(profile.events.add) == ({"randomize_camera_0"} if present else set())


def test_duplicate_camera_entries_rejected():
    with pytest.raises(AssertionError, match="unique"):
        DomainRandomizationCfg.from_dict({"cameras": [{"asset_name": "cam"}, {"asset_name": "cam"}]})


def test_empty_reset_is_a_noop():
    env = SimpleNamespace(scene={"cam": object()})
    assert (
        camera.randomize_camera(
            env,
            torch.tensor([], dtype=torch.long),
            SimpleNamespace(name="cam"),
            (1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        )
        == {}
    )


def test_camera_resets_use_nominal_mount_and_only_selected_environment():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    paths = ["/World/envs/env_0/Camera", "/World/envs/env_1/Camera"]
    for path in paths:
        UsdGeom.Camera.Define(stage, path)
    nominal_q = Rotation.from_euler("XYZ", [41, -12, 83], degrees=True).as_quat()
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_.*/Camera",
        offset=SimpleNamespace(pos=(0.083, 0.029, 0.476), rot=tuple(nominal_q), convention="ros"),
        spawn=SimpleNamespace(focal_length=15.0, horizontal_aperture=16.0),
    )
    writes = []
    sensor = SimpleNamespace(
        cfg=cfg,
        image_shape=(240, 320),
        set_intrinsic_matrices=lambda matrices, **kwargs: writes.append((matrices.clone(), kwargs)),
    )

    class Scene(dict):
        pass

    scene = Scene(cam=sensor)
    scene.stage = stage
    scene.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    env = SimpleNamespace(scene=scene, device="cpu", cfg=SimpleNamespace(sim=SimpleNamespace(use_fabric=False)))
    ids = torch.tensor([1])
    params = dict(
        asset_cfg=SimpleNamespace(name="cam"),
        focal_length_scale_range=(1.04, 1.04),
        position_xyz_min=(0.005, -0.005, 0.002),
        position_xyz_max=(0.005, -0.005, 0.002),
        rotation_rpy_min_deg=(3, -3, 2),
        rotation_rpy_max_deg=(3, -3, 2),
    )
    before = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(paths[0])).GetLocalTransformation())
    expected_rotation = Rotation.from_quat(nominal_q) * Rotation.from_euler("XYZ", [3, -3, 2], degrees=True)
    for _ in range(8):
        result = camera.randomize_camera(env, ids, **params)
        assert set(result) == {1}
        assert result[1]["position_xyz"] == pytest.approx((0.088, 0.024, 0.478))
        assert result[1]["rotation_xyzw"] == pytest.approx(expected_rotation.as_quat(), abs=1e-6)
        assert result[1]["focal_length_px"] == pytest.approx((312.0,))
        unchanged = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(paths[0])).GetLocalTransformation())
        np.testing.assert_array_equal(before, unchanged)
        assert writes[-1][1]["env_ids"] == [1]
    assert cfg.offset.pos == (0.083, 0.029, 0.476)
    assert cfg.offset.rot == tuple(nominal_q)
    assert cfg.spawn.focal_length == 15.0
    params.update(
        focal_length_scale_range=(0.96, 1.04), position_xyz_max=(0.01, 0.0, 0.02), rotation_rpy_max_deg=(4, 0, 3)
    )
    torch.manual_seed(91)
    first = camera.randomize_camera(env, torch.tensor([0, 1]), **params)
    torch.manual_seed(91)
    assert camera.randomize_camera(env, torch.tensor([0, 1]), **params) == first
    assert first[0] != first[1]


def test_camera_yaml_preserves_renderer_configuration_and_other_sensors(tmp_path):
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    from autodata_domain_randomization import apply_domain_randomization

    renderer = IsaacRtxRendererCfg()
    cfg = SimpleNamespace(
        events=SimpleNamespace(),
        scene=SimpleNamespace(cam=SimpleNamespace(renderer_cfg=renderer), other=SimpleNamespace(renderer_cfg=renderer)),
    )
    path = tmp_path / "camera.yaml"
    path.write_text("""
name: test_dr
base_env: TestEnv-v0
domain_randomization:
  cameras:
  - asset_name: cam
""")
    apply_domain_randomization(cfg, "Test", EnvironmentProfile.from_yaml(path).domain_randomization, num_envs=5)
    assert cfg.scene.cam.renderer_cfg is renderer
    assert cfg.scene.other.renderer_cfg is renderer
    assert cfg.events.randomize_camera_0.func is camera.randomize_camera


def test_calibrated_focal_dr_preserves_principal_point_and_does_not_accumulate():
    from isaaclab.sensors import Camera
    from isaaclab.sim import OpenCvPinholeDistortionCfg, PinholeCameraCfg
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    nominal = (303.9342346191406, 303.904052734375, 159.6581573486328, 126.32707977294922)
    prefix = "omni:lensdistortion:opencvPinhole:"
    prims = []
    for env_id in range(2):
        prim = UsdGeom.Camera.Define(stage, f"/World/envs/env_{env_id}/Camera").GetPrim()
        prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
        for name, value in zip(("fx", "fy", "cx", "cy"), nominal):
            prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Float).Set(value)
        prims.append(prim)
    matrices = torch.eye(3).repeat(2, 1, 1)
    updates = []

    def update_intrinsics(env_ids):
        updates.append(env_ids)
        for env_id in env_ids:
            fx, fy, cx, cy = Camera._read_authored_opencv_intrinsics(sensor, prims[env_id], 320, 240, env_id)
            matrices[env_id] = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    distortion = OpenCvPinholeDistortionCfg(
        fx=nominal[0], fy=nominal[1], cx=nominal[2], cy=nominal[3], image_size=(320, 240)
    )
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_.*/Camera",
            spawn=PinholeCameraCfg(focal_length=15.0, horizontal_aperture=16.0, distortion=distortion),
        ),
        image_shape=(240, 320),
        _update_intrinsic_matrices=update_intrinsics,
    )
    scene = SimpleNamespace(stage=stage, env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"])
    env = SimpleNamespace(scene=scene, device="cpu")
    for scale in (1.04, 1.04, 0.96, 1.0):
        camera._randomize_intrinsics(env, [1], sensor, (scale, scale))
        expected = torch.tensor(
            [[nominal[0] * scale, 0.0, nominal[2]], [0.0, nominal[1] * scale, nominal[3]], [0.0, 0.0, 1.0]]
        )
        torch.testing.assert_close(matrices[1], expected)
        torch.testing.assert_close(matrices[0], torch.eye(3))
        for index, name in enumerate(("fx", "fy", "cx", "cy")):
            actual = prims[1].GetAttribute(prefix + name).Get()
            assert actual == pytest.approx(nominal[index] * (scale if index < 2 else 1))
            assert prims[0].GetAttribute(prefix + name).Get() == pytest.approx(nominal[index])
    assert updates == [[1], [1], [1], [1]]
    assert (distortion.fx, distortion.fy, distortion.cx, distortion.cy) == nominal


@pytest.mark.parametrize(
    "num_envs,bounds,allowed",
    [
        (1, (0.96, 1.04), True),
        (1, (1.04, 1.04), True),
        (3, (1.0, 1.0), True),
        (3, (1.04, 1.04), False),
        (3, (0.96, 1.04), False),
        (3, None, True),
    ],
)
def test_camera_intrinsics_are_restricted_to_one_environment(num_envs, bounds, allowed):
    kwargs = {} if bounds is None else {"focal_length_scale_range": bounds}
    cfg = DomainRandomizationCfg(cameras=(CameraRandomizationCfg(asset_name="cam", **kwargs),))
    if allowed:
        cfg.validate_runtime(num_envs)
    else:
        with pytest.raises(AssertionError, match="Camera intrinsics DR requires num_envs=1"):
            cfg.validate_runtime(num_envs)


@pytest.mark.parametrize("renderer_type", ["isaac_rtx", "newton"])
def test_parallel_intrinsics_are_rejected_even_with_independent_render_products(renderer_type):
    from autodata_domain_randomization import apply_domain_randomization

    renderer_cfg = SimpleNamespace(renderer_type=renderer_type, tiled_rendering=False)
    sensor_cfg = SimpleNamespace(renderer_cfg=renderer_cfg)
    env_cfg = SimpleNamespace(scene=SimpleNamespace(cam=sensor_cfg), events=SimpleNamespace())
    dr = DomainRandomizationCfg(
        cameras=(CameraRandomizationCfg(asset_name="cam", focal_length_scale_range=(0.96, 1.04)),)
    )
    with pytest.raises(AssertionError, match="Camera intrinsics DR requires num_envs=1"):
        apply_domain_randomization(env_cfg, "Test", dr, num_envs=3)
    assert vars(env_cfg.events) == {}
    assert sensor_cfg.renderer_cfg is renderer_cfg
