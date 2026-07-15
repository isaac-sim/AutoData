# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :meth:`CuroboPlannerCfg.from_profile` planner-profile registry.

Factory methods are monkeypatched to sentinels so no robot assets are downloaded; skipped where
``curobo`` is unavailable.
"""

import pytest

# Skip on the specific submodule: a bare "curobo" can resolve to an unrelated namespace package.
pytest.importorskip("curobo.geom.sdf.world")

from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg  # noqa: E402

_PROFILE_FACTORIES = {
    "franka": "franka_config",
    "franka_stack_cube": "franka_stack_cube_config",
    "franka_stack_cube_bin": "franka_stack_cube_bin_config",
}


def test_from_profile_dispatches_to_registered_factories(monkeypatch):
    for profile_name, factory_name in _PROFILE_FACTORIES.items():
        sentinel = object()
        monkeypatch.setattr(CuroboPlannerCfg, factory_name, classmethod(lambda cls, _s=sentinel: _s))
        assert CuroboPlannerCfg.from_profile(profile_name) is sentinel


def test_from_profile_rejects_unknown_profile_name():
    with pytest.raises(AssertionError, match="Unknown planner profile"):
        CuroboPlannerCfg.from_profile("franka_stack_cube_bin_typo")
