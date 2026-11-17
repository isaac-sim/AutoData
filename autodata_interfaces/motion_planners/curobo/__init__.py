# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuRobo v1 planning, collision geometry and robot-model utilities.

Imports the planner lazily so the configuration dataclass can be loaded in sim-free
contexts (e.g. unit tests or factory selection) without bringing in Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autodata_interfaces.motion_planners.curobo.collision_spheres import ObjectCollisionSpheres
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
    from autodata_interfaces.motion_planners.curobo.planner_factory import (
        CuroboPlanningContext,
        curobo_group_planners_create,
        curobo_planning_context_create,
    )

__all__ = [
    "CuroboPlanner",
    "CuroboPlannerCfg",
    "CuroboPlanningContext",
    "ObjectCollisionSpheres",
    "curobo_group_planners_create",
    "curobo_planning_context_create",
]


def __getattr__(name: str):
    """Load the requested cuRobo planner or robot-model symbol on demand."""
    if name == "ObjectCollisionSpheres":
        from autodata_interfaces.motion_planners.curobo.collision_spheres import (
            ObjectCollisionSpheres as _ObjectCollisionSpheres,
        )

        return _ObjectCollisionSpheres
    if name == "CuroboPlanner":
        from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner as _CuroboPlanner

        return _CuroboPlanner
    if name == "CuroboPlannerCfg":
        from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg as _CuroboPlannerCfg

        return _CuroboPlannerCfg
    if name == "curobo_group_planners_create":
        from autodata_interfaces.motion_planners.curobo.planner_factory import (
            curobo_group_planners_create as _curobo_group_planners_create,
        )

        return _curobo_group_planners_create
    if name in {"CuroboPlanningContext", "curobo_planning_context_create"}:
        from autodata_interfaces.motion_planners.curobo import planner_factory

        return getattr(planner_factory, name)
    raise AttributeError(f"module 'curobo' has no attribute {name!r}")
