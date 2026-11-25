# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-randomization configuration and episode reset events."""

from autodata_domain_randomization.camera import randomize_camera
from autodata_domain_randomization.configs import (
    CameraRandomizationCfg,
    DomainRandomizationCfg,
    LightingRandomizationCfg,
    MaterialColorRandomizationCfg,
    MaterialScalarRandomizationCfg,
    PoseRandomizationCfg,
    RobotJointRandomizationCfg,
    RobotRootRandomizationCfg,
)
from autodata_domain_randomization.lighting import color_temperature_to_rgb, randomize_light
from autodata_domain_randomization.materials import (
    randomize_material_color,
    randomize_material_scalar,
    sample_palette_color,
)
from autodata_domain_randomization.profiles import apply_domain_randomization, create_domain_randomization_profile

__all__ = [
    "CameraRandomizationCfg",
    "randomize_camera",
    "DomainRandomizationCfg",
    "LightingRandomizationCfg",
    "MaterialColorRandomizationCfg",
    "MaterialScalarRandomizationCfg",
    "PoseRandomizationCfg",
    "RobotJointRandomizationCfg",
    "RobotRootRandomizationCfg",
    "apply_domain_randomization",
    "color_temperature_to_rgb",
    "create_domain_randomization_profile",
    "randomize_light",
    "randomize_material_color",
    "randomize_material_scalar",
    "sample_palette_color",
]
