# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo v2 backend for the SkillGen motion planner.

Provides an implementation of :class:`MotionPlannerBase` against cuRobo v2's redesigned
public API (:class:`curobo.motion_planner.MotionPlanner`). The public surface mirrors the
v1 backend in :mod:`isaac_autodata_interfaces.motion_planners.curobo` so callers can switch
between versions without code changes.

Imports are lazy so the config dataclass can be loaded in sim-free contexts (e.g. unit
tests or factory selection) without bringing in Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaac_autodata_interfaces.motion_planners.curobo_v2.curobo_v2_planner import CuroboV2Planner

from isaac_autodata_interfaces.motion_planners.curobo_v2.curobo_v2_planner_cfg import CuroboV2PlannerCfg

__all__ = [
    "CuroboV2Planner",
    "CuroboV2PlannerCfg",
]


def __getattr__(name: str):
    if name == "CuroboV2Planner":
        from isaac_autodata_interfaces.motion_planners.curobo_v2.curobo_v2_planner import (
            CuroboV2Planner as _CuroboV2Planner,
        )

        return _CuroboV2Planner
    raise AttributeError(f"module 'curobo_v2' has no attribute {name!r}")
