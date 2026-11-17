# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-profile construction for episodic domain randomization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autodata_domain_randomization.configs import DomainRandomizationCfg

if TYPE_CHECKING:
    from autodata_interfaces.env.env_profile import EnvironmentProfile

_LIGHTING_EVENT_FUNC = "autodata_domain_randomization.lighting:randomize_light"
_CAMERA_EVENT_FUNC = "autodata_domain_randomization.camera:randomize_camera"
_MATERIAL_COLOR_EVENT_FUNC = "autodata_domain_randomization.materials:randomize_material_color"
_MATERIAL_SCALAR_EVENT_FUNC = "autodata_domain_randomization.materials:randomize_material_scalar"


def create_domain_randomization_profile(
    base_env: str,
    cfg: DomainRandomizationCfg,
) -> EnvironmentProfile:
    """Create an AutoData environment profile containing enabled reset randomizers.

    Args:
        base_env: Environment id against which the profile will be applied.
        cfg: Domain-randomization settings.

    Returns:
        A profile whose event terms are sampled exactly once per Isaac Lab reset.
    """
    from autodata_interfaces.env.env_profile import EnvironmentProfile

    event_overrides = {}
    for pose in cfg.poses:
        if pose.enabled:
            event_overrides[pose.event_name] = {"params": {"pose_range": pose.as_pose_range()}}
    if cfg.robot_root is not None and cfg.robot_root.enabled:
        event_overrides.setdefault("reset_all", {"params": {}})["params"][
            "robot_root_pose_range"
        ] = cfg.robot_root.as_pose_range()
    if cfg.robot_joints is not None and cfg.robot_joints.enabled:
        # Integrate with reset_all so controller initialization sees the sampled joints.
        params = event_overrides.setdefault("reset_all", {"params": {}})["params"]
        params["robot_joint_names"] = cfg.robot_joints.joint_names
        params["robot_joint_position_range"] = cfg.robot_joints.as_position_range()
    event_additions = {}
    if cfg.lighting is not None and cfg.lighting.enabled:
        event_additions["randomize_light"] = {
            "func": _LIGHTING_EVENT_FUNC,
            "mode": "reset",
            "params": {
                "intensity_range": cfg.lighting.intensity_range,
                "color_temperature_range": cfg.lighting.color_temperature_range,
                "asset_cfg": cfg.lighting.asset_name,
            },
        }
    for index, material_scalar in enumerate(cfg.material_scalars):
        if not material_scalar.enabled:
            continue
        event_additions[f"randomize_material_scalar_{index}"] = {
            "func": _MATERIAL_SCALAR_EVENT_FUNC,
            "mode": "reset",
            "params": {
                "asset_cfg": material_scalar.asset_name,
                "prim_name_markers": material_scalar.prim_name_markers,
                "shader_input": material_scalar.shader_input,
                "value_range": material_scalar.value_range,
            },
        }
    for index, material_color in enumerate(cfg.material_colors):
        if not material_color.enabled:
            continue
        event_additions[f"randomize_material_color_{index}"] = {
            "func": _MATERIAL_COLOR_EVENT_FUNC,
            "mode": "reset",
            "params": {
                "asset_cfg": material_color.asset_name,
                "prim_name_markers": material_color.prim_name_markers,
                "shader_input": material_color.shader_input,
                "color_palette": material_color.color_palette,
            },
        }

    for index, camera in enumerate(cfg.cameras):
        if not camera.enabled:
            continue
        event_additions[f"randomize_camera_{index}"] = {
            "func": _CAMERA_EVENT_FUNC,
            "mode": "reset",
            "params": {
                "asset_cfg": camera.asset_name,
                "focal_length_scale_range": camera.focal_length_scale_range,
                "position_xyz_min": camera.position_xyz_min,
                "position_xyz_max": camera.position_xyz_max,
                "rotation_rpy_min_deg": camera.rotation_rpy_min_deg,
                "rotation_rpy_max_deg": camera.rotation_rpy_max_deg,
            },
        }

    return EnvironmentProfile.from_dict({
        "name": "episodic_domain_randomization",
        "description": "Per-reset domain randomization for visuomotor data generation.",
        "base_env": base_env,
        "events": {"override": event_overrides, "add": event_additions},
    })


def apply_domain_randomization(
    env_cfg: Any,
    base_env: str,
    cfg: DomainRandomizationCfg,
    num_envs: int,
) -> DomainRandomizationCfg:
    """Apply domain randomization before environment creation.

    Args:
        env_cfg: Parsed Isaac Lab environment configuration to modify in place.
        base_env: Environment id against which the reset-event profile is applied.
        cfg: Randomization settings from an environment profile.
        num_envs: Number of parallel environments in the process.

    Returns:
        The validated domain-randomization configuration.
    """
    cfg.validate_runtime(num_envs)
    prepare_visual_randomization(env_cfg, cfg)
    profile = create_domain_randomization_profile(base_env, cfg)

    if (cfg.robot_root is not None and cfg.robot_root.enabled) or (
        cfg.robot_joints is not None and cfg.robot_joints.enabled
    ):
        from isaaclab_arena.terms.events import reset_all_articulation_joints as arena_reset

        from autodata_domain_randomization.robot import reset_all_articulation_joints

        reset_term = getattr(env_cfg.events, "reset_all", None)
        assert (
            reset_term is not None and reset_term.func is arena_reset
        ), "Robot root/joint randomization requires Arena's reset_all_articulation_joints reset_all event."
        # Replace only the supported reset so offsets are applied before controller initialization.
        reset_term.func = reset_all_articulation_joints

    from autodata_interfaces.env.isaaclab_env_interface import apply_env_profile

    apply_env_profile(env_cfg, profile, base_env)
    return cfg


def prepare_visual_randomization(env_cfg: Any, cfg: DomainRandomizationCfg) -> None:
    """Prepare independent camera render products before environment creation.

    Args:
        env_cfg: Environment configuration to modify in place.
        cfg: Validated domain-randomization configuration.
    """
    for camera in cfg.cameras:
        if not camera.enabled:
            continue
        sensor_cfg = getattr(env_cfg.scene, camera.asset_name, None)
        assert sensor_cfg is not None, f"Camera DR requires scene sensor {camera.asset_name!r}; enable cameras."
        renderer_cfg = sensor_cfg.renderer_cfg
        if renderer_cfg.renderer_type == "isaac_rtx":
            # Tiled RTX can reuse the first view's projection for all environments.
            sensor_cfg.renderer_cfg = renderer_cfg.replace(tiled_rendering=False)
