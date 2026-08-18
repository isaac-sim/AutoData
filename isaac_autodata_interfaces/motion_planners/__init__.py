# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planner backends for SkillGen.

Two cuRobo backends ship here, selected by name through :func:`get_planner_backend`:

* ``"curobo"`` — :class:`CuroboPlanner` / :class:`CuroboPlannerCfg`, built on cuRobo v1.
* ``"curobo_v2"`` — :class:`CuroboV2Planner` / :class:`CuroboV2PlannerCfg`, built on cuRobo v2.

Both implement :class:`MotionPlannerBase` and accept the same profile names, so an entry point
picks a backend without changing any other logic.

Nothing is imported at module load. Both cuRobo versions install under the ``curobo`` package
name at incompatible versions and live in separate environments, so importing a backend that is
not installed would fail. Resolution waits until a backend is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaac_autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

if TYPE_CHECKING:
    from isaac_autodata_interfaces.motion_planners.curobo import CuroboPlanner, CuroboPlannerCfg
    from isaac_autodata_interfaces.motion_planners.curobo_v2 import CuroboV2Planner, CuroboV2PlannerCfg

__all__ = [
    "MotionPlannerBase",
    "CuroboPlanner",
    "CuroboPlannerCfg",
    "CuroboV2Planner",
    "CuroboV2PlannerCfg",
    "PLANNER_BACKENDS",
    "get_planner_backend",
]

# Backend name -> (subpackage, planner class name, config class name).
_BACKEND_SPECS: dict[str, tuple[str, str, str]] = {
    "curobo": ("curobo", "CuroboPlanner", "CuroboPlannerCfg"),
    "curobo_v2": ("curobo_v2", "CuroboV2Planner", "CuroboV2PlannerCfg"),
}

PLANNER_BACKENDS: tuple[str, ...] = tuple(_BACKEND_SPECS)
"""Names of the selectable planner backends, in registration order.

``"curobo"`` is the cuRobo v1 backend and the default; ``"curobo_v2"`` is the cuRobo v2 backend.
"""


def get_planner_backend(name: str) -> tuple[type[MotionPlannerBase], type]:
    """Resolve a backend name to its planner and configuration classes.

    Importing the backend pulls in the matching cuRobo version, so only the requested backend
    is loaded.

    Args:
        name: Backend name; one of :data:`PLANNER_BACKENDS`.

    Returns:
        The ``(planner_class, config_class)`` pair for the backend. The planner takes
        ``(datastream, config, env_id)`` and the config exposes ``from_profile`` /
        ``from_task_name``.
    """
    assert name in _BACKEND_SPECS, f"Unknown planner backend {name!r}. Available: {list(PLANNER_BACKENDS)}"
    import importlib

    subpackage, planner_name, config_name = _BACKEND_SPECS[name]
    module = importlib.import_module(f"{__name__}.{subpackage}")
    return getattr(module, planner_name), getattr(module, config_name)


def __getattr__(name: str) -> Any:
    for subpackage, planner_name, config_name in _BACKEND_SPECS.values():
        if name in (planner_name, config_name):
            import importlib

            return getattr(importlib.import_module(f"{__name__}.{subpackage}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
