# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data generation runtime for AutoData.

Public entrypoints:

* :class:`DataGenerator` — orchestrates a single async generation job.
* :class:`DataGenInfoPool` — source-demo container; built via :meth:`DataGenInfoPool.from_hdf5`.
* :class:`GenerationAlgorithm` and concrete subclasses — pluggable algorithm surface.
* :func:`get_algorithm`, :func:`iter_algorithms` — algorithm registry lookup.
"""

from autodata_core.algorithms import (
    DexMimicGen,
    GenerationAlgorithm,
    MimicGen,
    SkillGen,
    get_algorithm,
    iter_algorithms,
)
from autodata_core.data_generator import DataGenerator, GenerationResult
from autodata_core.datagen_info import DatagenInfo
from autodata_core.pool import DataGenInfoPool
from autodata_core.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

__all__ = [
    "DataGenInfoPool",
    "DataGenerator",
    "DatagenInfo",
    "DexMimicGen",
    "GenerationAlgorithm",
    "GenerationResult",
    "MimicGen",
    "MultiWaypoint",
    "SkillGen",
    "Waypoint",
    "WaypointSequence",
    "WaypointTrajectory",
    "get_algorithm",
    "iter_algorithms",
]
