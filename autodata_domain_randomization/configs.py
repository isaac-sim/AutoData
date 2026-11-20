# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for domain randomization."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass(kw_only=True)
class LightingRandomizationCfg:
    """Randomization settings for existing lights.

    Attributes:
        asset_name: Scene-asset name of the light. Global and dome lights require one environment.
        intensity_range: Uniform USD emitter intensity range.
        color_temperature_range: Uniform color-temperature range [K].
    """

    asset_name: str = "light"
    intensity_range: tuple[float, float] = (250.0, 750.0)
    color_temperature_range: tuple[float, float] = (3000.0, 8000.0)

    def __post_init__(self) -> None:
        assert self.asset_name, "Lighting randomization requires a non-empty scene asset name."
        self.intensity_range = tuple(self.intensity_range)
        self.color_temperature_range = tuple(self.color_temperature_range)
        _validate_range(self.intensity_range, "intensity_range", minimum=0.0)
        _validate_range(
            self.color_temperature_range,
            "color_temperature_range",
            minimum=1000.0,
            maximum=40000.0,
        )


@dataclass(kw_only=True)
class MaterialColorRandomizationCfg:
    """Material color-multiplier settings.

    Colors are sampled continuously along the piecewise-linear gradient formed by
    ``color_palette``. Values may exceed one because shader color multipliers often need
    to brighten a dark source texture.

    Attributes:
        asset_name: Scene-asset name containing the target prims.
        prim_name_markers: Substrings that identify target visual prim names.
        shader_input: Material shader input that receives the sampled color.
        color_palette: RGB multiplier anchors defining the sampled color gradient.
    """

    asset_name: str
    prim_name_markers: tuple[str, ...]
    shader_input: str
    color_palette: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        assert not isinstance(self.prim_name_markers, str), "Material color prim name markers must be a sequence."
        self.prim_name_markers = tuple(self.prim_name_markers)
        self.color_palette = tuple(tuple(color) for color in self.color_palette)
        assert self.asset_name, "Material color randomization requires a non-empty scene asset name."
        assert self.prim_name_markers, "Material color randomization requires at least one prim name marker."
        assert all(
            isinstance(marker, str) and marker for marker in self.prim_name_markers
        ), "Material color prim name markers must be non-empty strings."
        assert self.shader_input, "Material color randomization requires a shader input name."
        assert self.color_palette, "Material color randomization requires at least one palette color."
        for color in self.color_palette:
            assert len(color) == 3, "Material color palette entries must contain exactly three RGB values."
            assert all(
                math.isfinite(component) and component >= 0.0 for component in color
            ), "Material color values must be finite and non-negative."


@dataclass(kw_only=True)
class MaterialScalarRandomizationCfg:
    """Scalar shader-property settings, such as roughness or specular level.

    Floating-point inputs are sampled continuously from ``value_range``. If any matched
    input in an environment is integral, sample one of the representable integers between
    the bounds, inclusive. Equal bounds pin the input to a fixed value.

    Attributes:
        asset_name: Scene-asset name containing the target prims.
        prim_name_markers: Substrings that identify target visual prim names.
        shader_input: Material shader input that receives the sampled value.
        value_range: Uniform sampling range for the scalar.
    """

    asset_name: str
    prim_name_markers: tuple[str, ...]
    shader_input: str
    value_range: tuple[float, float]

    def __post_init__(self) -> None:
        assert not isinstance(self.prim_name_markers, str), "Material scalar prim name markers must be a sequence."
        self.prim_name_markers = tuple(self.prim_name_markers)
        self.value_range = tuple(self.value_range)
        assert self.asset_name, "Material scalar randomization requires a non-empty scene asset name."
        assert self.prim_name_markers, "Material scalar randomization requires at least one prim name marker."
        assert all(
            isinstance(marker, str) and marker for marker in self.prim_name_markers
        ), "Material scalar prim name markers must be non-empty strings."
        assert self.shader_input, "Material scalar randomization requires a shader input name."
        assert all(math.isfinite(value) for value in self.value_range), "Material scalar bounds must be finite."
        _validate_range(self.value_range, "value_range", minimum=0.0)


@dataclass(kw_only=True)
class CameraRandomizationCfg:
    """Camera intrinsics/extrinsics settings applied to a mounted camera sensor.

    Perturbations are relative to the sensor's nominal spawn calibration, so they read directly as
    the calibration error being modelled and never accumulate across resets.

    Attributes:
        asset_name: Scene-asset name of the camera sensor.
        focal_length_scale_range: Uniform multiplier range applied to the nominal focal length.
            The native pinhole sensor uses square pixels and a centered principal point.
        position_xyz_min: Lower bound of the mount-translation offset [m], in the parent-link frame.
        position_xyz_max: Upper bound of the mount-translation offset [m], in the parent-link frame.
        rotation_rpy_min_deg: Lower bound of the rotation offset [deg] about the camera optical
            frame's x (right), y (down), and z (forward) axes.
        rotation_rpy_max_deg: Upper bound of the rotation offset [deg] about the same axes.
    """

    asset_name: str
    focal_length_scale_range: tuple[float, float] = (1.0, 1.0)
    position_xyz_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_xyz_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_min_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_max_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        assert self.asset_name, "Camera randomization requires a non-empty scene asset name."
        self.focal_length_scale_range = tuple(self.focal_length_scale_range)
        assert all(
            math.isfinite(value) for value in self.focal_length_scale_range
        ), "Camera focal bounds must be finite."
        _validate_range(self.focal_length_scale_range, "focal_length_scale_range", minimum=1e-3)
        self.position_xyz_min = tuple(self.position_xyz_min)
        self.position_xyz_max = tuple(self.position_xyz_max)
        self.rotation_rpy_min_deg = tuple(self.rotation_rpy_min_deg)
        self.rotation_rpy_max_deg = tuple(self.rotation_rpy_max_deg)
        for name, lower, upper in (
            ("position_xyz", self.position_xyz_min, self.position_xyz_max),
            ("rotation_rpy_deg", self.rotation_rpy_min_deg, self.rotation_rpy_max_deg),
        ):
            assert len(lower) == 3 and len(upper) == 3, f"Camera {name} bounds must each contain three values."
            assert all(math.isfinite(value) for value in (*lower, *upper)), f"Camera {name} bounds must be finite."
            assert all(
                low <= high for low, high in zip(lower, upper)
            ), f"Camera {name} lower bounds must not exceed upper bounds."


@dataclass(kw_only=True)
class PoseRandomizationCfg:
    """Pose range applied through an existing Isaac Lab reset event.

    Attributes:
        event_name: Existing reset event accepting a ``pose_range`` parameter.
        position_xyz_min: Lower position bounds [m], in the event's reference frame.
        position_xyz_max: Upper position bounds [m], in the event's reference frame.
        rpy_min: Lower roll/pitch/yaw bounds [rad].
        rpy_max: Upper roll/pitch/yaw bounds [rad].
    """

    event_name: str
    position_xyz_min: tuple[float, float, float]
    position_xyz_max: tuple[float, float, float]
    rpy_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        assert self.event_name, "Pose randomization requires a reset event name."
        self.position_xyz_min = tuple(self.position_xyz_min)
        self.position_xyz_max = tuple(self.position_xyz_max)
        self.rpy_min = tuple(self.rpy_min)
        self.rpy_max = tuple(self.rpy_max)
        for name, lower, upper in (
            ("position_xyz", self.position_xyz_min, self.position_xyz_max),
            ("rpy", self.rpy_min, self.rpy_max),
        ):
            assert len(lower) == 3 and len(upper) == 3, f"{name} bounds must each contain three values."
            assert all(math.isfinite(value) for value in (*lower, *upper)), f"{name} bounds must be finite."
            assert all(
                low <= high for low, high in zip(lower, upper)
            ), f"{name} lower bounds must not exceed upper bounds."

    def as_pose_range(self) -> dict[str, tuple[float, float]]:
        """Return the range shape consumed by Isaac Lab pose-reset events."""
        return {
            "x": (self.position_xyz_min[0], self.position_xyz_max[0]),
            "y": (self.position_xyz_min[1], self.position_xyz_max[1]),
            "z": (self.position_xyz_min[2], self.position_xyz_max[2]),
            "roll": (self.rpy_min[0], self.rpy_max[0]),
            "pitch": (self.rpy_min[1], self.rpy_max[1]),
            "yaw": (self.rpy_min[2], self.rpy_max[2]),
        }


@dataclass(kw_only=True)
class RobotRootRandomizationCfg:
    """Independent uniform offsets from the default robot root, in world axes."""

    x_range_m: tuple[float, float] = (0.0, 0.0)
    y_range_m: tuple[float, float] = (0.0, 0.0)
    yaw_range_deg: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("x_range_m", "y_range_m", "yaw_range_deg"):
            bounds = tuple(getattr(self, name))
            assert len(bounds) == 2 and all(
                math.isfinite(value) for value in bounds
            ), f"Robot root {name} must contain two finite values."
            assert bounds[0] <= bounds[1], f"Robot root {name} lower bound exceeds upper bound."
            setattr(self, name, bounds)

    def as_pose_range(self) -> dict[str, tuple[float, float]]:
        """Return world X/Y offsets [m] and world-Z yaw offsets [rad]."""
        return {
            "x": self.x_range_m,
            "y": self.y_range_m,
            "yaw": tuple(math.radians(value) for value in self.yaw_range_deg),
        }


@dataclass(kw_only=True)
class RobotJointRandomizationCfg:
    """Independent uniform joint offsets from the nominal reset position."""

    joint_names: tuple[str, ...] = ()
    delta_range_deg: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        assert not isinstance(self.joint_names, str), "Robot joint_names must be a sequence of names or regex patterns."
        self.joint_names = tuple(self.joint_names)
        assert all(
            isinstance(name, str) and name for name in self.joint_names
        ), "Joint names must be non-empty strings."
        assert self.joint_names, "Robot joint DR requires joint_names."
        self.delta_range_deg = tuple(self.delta_range_deg)
        assert len(self.delta_range_deg) == 2 and all(
            math.isfinite(value) for value in self.delta_range_deg
        ), "Robot joint delta_range_deg must contain two finite values."
        assert self.delta_range_deg[0] <= self.delta_range_deg[1], "Robot joint lower bound exceeds upper bound."

    def as_position_range(self) -> tuple[float, float]:
        """Return additive joint-position bounds [rad]."""
        return tuple(math.radians(value) for value in self.delta_range_deg)


@dataclass(kw_only=True)
class DomainRandomizationCfg:
    """Optional randomization settings embedded in an environment profile.

    Only sections present in the profile are applied. Settings are sampled once per episode reset.

    Attributes:
        lighting: Optional lighting settings.
        robot_root: Optional robot root offsets.
        robot_joints: Optional robot joint offsets.
        poses: Pose ranges for existing reset events.
        material_colors: Material color multipliers.
        material_scalars: Scalar material properties.
        cameras: Camera calibration offsets.
    """

    lighting: LightingRandomizationCfg | None = None
    robot_root: RobotRootRandomizationCfg | None = None
    robot_joints: RobotJointRandomizationCfg | None = None
    poses: tuple[PoseRandomizationCfg, ...] = ()
    material_colors: tuple[MaterialColorRandomizationCfg, ...] = ()
    material_scalars: tuple[MaterialScalarRandomizationCfg, ...] = ()
    cameras: tuple[CameraRandomizationCfg, ...] = ()

    def __post_init__(self) -> None:
        self.poses = tuple(self.poses)
        pose_names = [pose.event_name for pose in self.poses]
        assert len(pose_names) == len(set(pose_names)), "Pose event names must be unique."
        self.material_colors = tuple(self.material_colors)
        self.material_scalars = tuple(self.material_scalars)
        self.cameras = tuple(self.cameras)
        camera_names = [camera.asset_name for camera in self.cameras]
        assert len(camera_names) == len(set(camera_names)), "Camera asset names must be unique."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainRandomizationCfg:
        """Parse the optional ``domain_randomization`` mapping in an environment profile.

        Args:
            data: Randomization settings without an episode or lifecycle wrapper.
        """
        _validate_keys(data, "domain_randomization", {item.name for item in fields(cls)})
        kwargs: dict[str, Any] = {}
        for name, config_type in (
            ("lighting", LightingRandomizationCfg),
            ("robot_root", RobotRootRandomizationCfg),
            ("robot_joints", RobotJointRandomizationCfg),
        ):
            if name in data:
                _validate_keys(data[name], f"domain_randomization.{name}", {item.name for item in fields(config_type)})
                kwargs[name] = config_type(**data[name])
        for name, config_type in (
            ("poses", PoseRandomizationCfg),
            ("material_colors", MaterialColorRandomizationCfg),
            ("material_scalars", MaterialScalarRandomizationCfg),
            ("cameras", CameraRandomizationCfg),
        ):
            entries = data.get(name, [])
            assert isinstance(entries, list), f"domain_randomization.{name} must be a list."
            allowed = {item.name for item in fields(config_type)}
            configs = []
            for index, entry in enumerate(entries):
                _validate_keys(entry, f"domain_randomization.{name}[{index}]", allowed)
                configs.append(config_type(**entry))
            kwargs[name] = tuple(configs)
        return cls(**kwargs)

    def validate_runtime(self, num_envs: int) -> None:
        """Validate the environment count and single-environment intrinsics restriction.

        Args:
            num_envs: Number of parallel environments in the process.
        """
        assert num_envs > 0, "num_envs must be positive."
        for camera in self.cameras:
            assert num_envs == 1 or camera.focal_length_scale_range == (1.0, 1.0), (
                "Camera intrinsics DR requires num_envs=1. For multi-environment extrinsics DR, "
                "keep focal_length_scale_range at [1.0, 1.0]."
            )


def _validate_keys(data: Any, name: str, allowed_keys: set[str]) -> None:
    assert isinstance(data, dict), f"{name} must be a mapping."
    unknown_keys = set(data) - allowed_keys
    assert not unknown_keys, f"{name} has unknown keys: {sorted(unknown_keys)}."


def _validate_range(
    value_range: tuple[float, float],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    assert len(value_range) == 2, f"{name} must contain exactly two values."
    assert all(math.isfinite(value) for value in value_range), f"{name} bounds must be finite."
    low, high = value_range
    assert low <= high, f"{name} lower bound must not exceed its upper bound."
    assert low >= minimum, f"{name} lower bound must be at least {minimum}."
    if maximum is not None:
        assert high <= maximum, f"{name} upper bound must not exceed {maximum}."
