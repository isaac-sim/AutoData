# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete :class:`EmbodimentAdapter` implementations, organized by morphology.

Single-arm and bimanual adapters provide morphology-specific control layouts.
:class:`AbsolutePoseEmbodimentAdapter` provides configurable absolute-pose and
passthrough channels. YAML construction goes through :mod:`factory`.
"""

from autodata_interfaces.embodiments.absolute_pose_embodiment_adapter import AbsolutePoseEmbodimentAdapter
from autodata_interfaces.embodiments.bimanual_embodiment_adapter import (
    AbsolutePoseWholeBodyBimanualAdapter,
    BimanualEefConfig,
    BimanualEmbodimentAdapter,
)
from autodata_interfaces.embodiments.embodiment_types import PoseObsKeys
from autodata_interfaces.embodiments.factory import (
    EMBODIMENT_TYPE_REGISTRY,
    embodiment_adapter_from_dict,
    embodiment_adapter_from_yaml,
)
from autodata_interfaces.embodiments.single_arm_embodiment_adapter import (
    DeltaPoseIKSingleArmAdapter,
    SingleArmEmbodimentAdapter,
)

__all__ = [
    "AbsolutePoseEmbodimentAdapter",
    "AbsolutePoseWholeBodyBimanualAdapter",
    "BimanualEefConfig",
    "BimanualEmbodimentAdapter",
    "DeltaPoseIKSingleArmAdapter",
    "EMBODIMENT_TYPE_REGISTRY",
    "PoseObsKeys",
    "SingleArmEmbodimentAdapter",
    "embodiment_adapter_from_dict",
    "embodiment_adapter_from_yaml",
]
