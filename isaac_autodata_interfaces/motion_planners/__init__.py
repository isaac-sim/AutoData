# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Motion-planner interfaces used by the SkillGen generation algorithm.

Two concrete backends are provided:

* :mod:`isaac_autodata_interfaces.motion_planners.curobo` — cuRobo v1
  (``curobo.wrap.reacher.motion_gen.MotionGen``-based).
* :mod:`isaac_autodata_interfaces.motion_planners.curobo_v2` — cuRobo v2
  (``curobo.motion_planner.MotionPlanner``-based).

Both satisfy the :class:`MotionPlannerBase` protocol documented in
``docs/agents/phase-2-datagenerator/data_generator.md`` so the SkillGen algorithm is unchanged.

Use :func:`detect_curobo_version` to discover which version is installed in the active env;
use :func:`get_curobo_planner_classes` to fetch the corresponding planner + config classes.
The factory imports are deferred so importing this module does not require either cuRobo
version to be installed.
"""

from __future__ import annotations

from isaac_autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

__all__ = [
    "MotionPlannerBase",
    "detect_curobo_version",
    "get_curobo_planner_classes",
]


def detect_curobo_version() -> str:
    """Return ``"v2"`` if cuRobo v2 is installed, otherwise ``"v1"`` if cuRobo v1 is installed.

    Detection is import-based:

    * v2: ``curobo.motion_planner.MotionPlanner`` is importable.
    * v1: ``curobo.wrap.reacher.motion_gen.MotionGen`` is importable.

    Returns:
        ``"v2"`` or ``"v1"``.

    Raises:
        ImportError: If neither version is installed.
    """
    try:
        from curobo.motion_planner import MotionPlanner  # noqa: F401

        return "v2"
    except ImportError:
        pass

    try:
        from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401

        return "v1"
    except ImportError as e:
        raise ImportError(
            "No cuRobo installation found. Install either v1 (legacy MotionGen) or v2 "
            "(MotionPlanner) before constructing a CuRobo planner."
        ) from e


def get_curobo_planner_classes(version: str = "auto") -> tuple[type, type]:
    """Return ``(PlannerClass, PlannerCfgClass)`` for the requested cuRobo version.

    Args:
        version: One of ``"v1"``, ``"v2"``, or ``"auto"``. ``"auto"`` defers to
            :func:`detect_curobo_version`.

    Returns:
        A tuple ``(planner_cls, planner_cfg_cls)`` ready to instantiate.

    Raises:
        ValueError: If ``version`` is not one of the recognized values.
        ImportError: If the requested version is not installed.
    """
    if version == "auto":
        version = detect_curobo_version()

    if version == "v1":
        from isaac_autodata_interfaces.motion_planners.curobo import CuroboPlanner, CuroboPlannerCfg

        return CuroboPlanner, CuroboPlannerCfg

    if version == "v2":
        from isaac_autodata_interfaces.motion_planners.curobo_v2 import CuroboV2Planner, CuroboV2PlannerCfg

        return CuroboV2Planner, CuroboV2PlannerCfg

    raise ValueError(f"Unknown cuRobo version {version!r}; expected one of 'v1', 'v2', 'auto'.")
