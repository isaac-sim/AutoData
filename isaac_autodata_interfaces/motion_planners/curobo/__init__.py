# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo v1 backend for the SkillGen motion planner.

Imports the planner lazily so the configuration dataclass can be loaded in sim-free
contexts (e.g. unit tests or factory selection) without bringing in Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner

from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

__all__ = [
    "CuroboPlanner",
    "CuroboPlannerCfg",
]


def __getattr__(name: str):
    if name == "CuroboPlanner":
        from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner as _CuroboPlanner

        return _CuroboPlanner
    raise AttributeError(f"module 'curobo' has no attribute {name!r}")
