# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :data:`PLANNER_PROFILES` planner-profile registry.

The registry is asserted directly; the factories are never invoked so no robot assets are
downloaded. Skipped where ``curobo`` is unavailable.
"""

import torch
from types import SimpleNamespace

import pytest

# Skip on the specific submodule: a bare "curobo" can resolve to an unrelated namespace package.
pytest.importorskip("curobo.geom.sdf.world")

from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import (  # noqa: E402
    PLANNER_PROFILES,
    CuroboPlannerCfg,
)


def test_planner_profile_registry_contents():
    # Bound-classmethod equality verifies each name dispatches to the expected factory
    # without invoking it (no robot-asset downloads).
    assert PLANNER_PROFILES == {
        "franka": CuroboPlannerCfg.franka_config,
        "franka_stack_cube": CuroboPlannerCfg.franka_stack_cube_config,
        "franka_stack_cube_bin": CuroboPlannerCfg.franka_stack_cube_bin_config,
    }


def test_from_profile_rejects_unknown_profile_name():
    with pytest.raises(AssertionError, match="Unknown planner profile"):
        CuroboPlannerCfg.from_profile("franka_stack_cube_bin_typo")


@pytest.mark.parametrize("tool_override, expected", [(None, "panda_hand"), ("custom_tool", "custom_tool")])
def test_attachment_capture_resolves_the_default_tool(tool_override, expected):
    """Attachment metadata uses the model's tool unless an explicit override is configured."""
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner

    pose = SimpleNamespace(
        inverse=lambda: SimpleNamespace(transform_points=lambda points: points), get_matrix=lambda: torch.eye(4)
    )
    geometry = SimpleNamespace(
        get_link_spheres=lambda link: torch.tensor([[0.0, 0.0, 0.0, 0.01], [0.0, 0.0, 0.0, -10.0]])
    )
    planner = SimpleNamespace(
        config=SimpleNamespace(
            attached_object_link_name="attached_object", ee_link_name=tool_override, surface_sphere_radius=0.01
        ),
        robot_cfg={"kinematics": {"ee_link": "panda_hand"}},
        motion_gen=SimpleNamespace(kinematics=SimpleNamespace(kinematics_config=geometry)),
        _object_collision_spheres={},
    )
    captured = CuroboPlanner._object_collision_spheres_capture(
        planner, "cube", "/World/cube", SimpleNamespace(pose=pose)
    )

    assert captured.tool_frame == expected
    assert len(captured.spheres) == 1
