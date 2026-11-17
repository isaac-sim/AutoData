# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planner backends for SkillGen and ScheduleGen.

The package provides the cuRobo v1 backend and exposes the abstract
:class:`MotionPlannerBase` and the v1 :class:`CuroboPlanner` / :class:`CuroboPlannerCfg`.

The planner class is imported lazily so the configuration dataclass and the abstract base can be
loaded in sim-free contexts (e.g. unit tests or CLI argument parsing) without pulling in cuRobo
or Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

if TYPE_CHECKING:
    from autodata_interfaces.motion_planners.curobo import CuroboPlanner, CuroboPlannerCfg

__all__ = [
    "MotionPlannerBase",
    "CuroboPlanner",
    "CuroboPlannerCfg",
]


def __getattr__(name: str):
    """Load planner symbols lazily so ordinary imports do not require cuRobo."""
    if name in {"CuroboPlanner", "CuroboPlannerCfg"}:
        from autodata_interfaces.motion_planners import curobo

        return getattr(curobo, name)
    raise AttributeError(f"module 'motion_planners' has no attribute {name!r}")
