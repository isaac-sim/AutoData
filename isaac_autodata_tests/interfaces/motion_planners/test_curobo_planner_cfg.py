# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :data:`PLANNER_PROFILES` planner-profile registry.

The registry is asserted directly; the factories are never invoked so no robot assets are
downloaded. Skipped where ``curobo`` is unavailable.
"""

import pytest

# Skip on the specific submodule: a bare "curobo" can resolve to an unrelated namespace package.
pytest.importorskip("curobo.geom.sdf.world")

from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import (  # noqa: E402
    PLANNER_PROFILES,
    CuroboPlannerCfg,
)


def test_planner_profile_registry_contents():
    assert set(PLANNER_PROFILES) == {"franka", "franka_stack_cube", "franka_stack_cube_bin"}
    assert all(callable(factory) for factory in PLANNER_PROFILES.values())


def test_from_profile_rejects_unknown_profile_name():
    with pytest.raises(AssertionError, match="Unknown planner profile"):
        CuroboPlannerCfg.from_profile("franka_stack_cube_bin_typo")
