# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuRobo v2 motion-planner backend.

Implements :class:`MotionPlannerBase` on :class:`curobo.motion_planner.MotionPlanner`.

The planner is imported lazily so the configuration can be loaded without a simulator, for
backend selection or CLI argument parsing. Both cuRobo versions install under the ``curobo``
package name, so only the selected backend is ever imported.
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
