# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planner backends for SkillGen.

This branch ships the cuRobo v1 backend only. The package exposes the abstract
:class:`MotionPlannerBase` and the v1 :class:`CuroboPlanner` / :class:`CuroboPlannerCfg`.

The planner class is imported lazily so the configuration dataclass and the abstract base can be
loaded in sim-free contexts (e.g. unit tests or CLI argument parsing) without pulling in cuRobo
or Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autodata_interfaces.motion_planners.curobo import CuroboPlannerCfg
from autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

if TYPE_CHECKING:
    from autodata_interfaces.motion_planners.curobo import CuroboPlanner

__all__ = [
    "MotionPlannerBase",
    "CuroboPlanner",
    "CuroboPlannerCfg",
]


def __getattr__(name: str):
    if name == "CuroboPlanner":
        from autodata_interfaces.motion_planners.curobo import CuroboPlanner as _CuroboPlanner

        return _CuroboPlanner
    raise AttributeError(f"module 'motion_planners' has no attribute {name!r}")
