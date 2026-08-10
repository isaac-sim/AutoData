# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data generation runtime for Isaac Auto Data.

Public entrypoints:

* :class:`DataGenerator` — orchestrates a single async generation job.
* :class:`DataGenInfoPool` — source-demo container; built via :meth:`DataGenInfoPool.from_hdf5`.
* :class:`GenerationAlgorithm` and concrete subclasses — pluggable algorithm surface.
* :func:`get_algorithm`, :func:`iter_algorithms` — algorithm registry lookup.
"""

from isaac_autodata_core.algorithms import (
    DexMimicGen,
    GenerationAlgorithm,
    MimicGen,
    SkillGen,
    get_algorithm,
    iter_algorithms,
)
from isaac_autodata_core.data_generator import DataGenerator, GenerationResult
from isaac_autodata_core.datagen_info import DatagenInfo
from isaac_autodata_core.pool import DataGenInfoPool
from isaac_autodata_core.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

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
