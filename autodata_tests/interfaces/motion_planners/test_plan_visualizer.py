# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning playback uses one clock and keeps whole-robot checks separate."""

import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("rerun")
pytest.importorskip("curobo.geom.sdf.world")

from autodata_interfaces.motion_planners.curobo import plan_visualizer as module  # noqa: E402


@pytest.fixture
def visualizer(monkeypatch):
    """Build a small kinematic preview without starting a simulator or viewer."""
    value = object.__new__(module.PlanVisualizer)
    value.robot_name = "left"
    value.debug = False
    value.save_path = None
    value._base_pose_in_env = np.eye(4)
    value._world_scene = None
    value._current_frame = 0
    value._plan_frame = 0
    value._schedule_timelines = {}
    value._active_plan_root = "world/plans"
    value._sphere_entities = {"robot": [], "attached": [], "target": []}
    value._motion_gen_ref = SimpleNamespace(
        kinematics=SimpleNamespace(
            get_robot_as_spheres=lambda q: [[SimpleNamespace(position=[float(q[0]), 0.0, 0.0], radius=0.01)]]
        )
    )
    monkeypatch.setattr(module.rr, "log", Mock())
    monkeypatch.setattr(module.rr, "set_time", Mock())
    monkeypatch.setattr(module.rr, "send_blueprint", Mock())
    return value


@pytest.mark.parametrize("save_path", [None, "plans.rrd"])
def test_consecutive_plans_share_playback_and_clear_previous_entities(visualizer, save_path):
    """Live and saved previews animate both arms without stale markers or separate clocks."""
    visualizer.save_path = save_path
    plan = SimpleNamespace(position=torch.tensor([[0.0], [1.0]]))
    for arm in ("left", "right"):
        visualizer.robot_name = arm
        visualizer.visualize_plan(plan, torch.eye(4), ee_positions=np.zeros((2, 3)))
        visualizer.animate_spheres_along_path(plan, robot_spheres_at_start=[object()], interpolation_steps=1)
    assert visualizer._plan_frame == 4
    times = module.rr.set_time.call_args_list
    assert {call.args[0] for call in times} == {"planning"}
    assert {call.kwargs["sequence"] for call in times} == {0, 1, 2, 3}
    logs = module.rr.log.call_args_list
    assert not any(call.kwargs.get("static", False) for call in logs)
    assert any(call.args[0] == "world/plans/0000_left" and isinstance(call.args[1], module.rr.Clear) for call in logs)
    assert sum(call.args[0].endswith("/robot_animation") for call in logs) == 4
    module.rr.send_blueprint.assert_not_called()


def test_scheduled_robot_does_not_switch_planning_view(visualizer):
    """The full-state preview keeps its own timeline without changing the selected view."""
    spheres = torch.tensor([[[0.0, 0.0, 0.0, 0.01]], [[1.0, 0.0, 0.0, 0.01]]])
    index = visualizer.animate_collision_spheres(spheres)
    assert index == 0
    assert visualizer._plan_frame == 0
    assert {call.args[0] for call in module.rr.set_time.call_args_list} == {"full_state_schedule_0"}
    module.rr.send_blueprint.assert_not_called()


def test_end_effector_markers_share_the_sphere_animation_interval(visualizer):
    visualizer.animate_plan(np.zeros((3, 3)))
    assert visualizer._plan_frame == 0
    visualizer.animate_spheres_along_path(
        SimpleNamespace(position=torch.zeros(3, 1)), robot_spheres_at_start=[object()], interpolation_steps=1
    )
    assert visualizer._plan_frame == 3
    visualizer.mark_idle()
    assert visualizer._plan_frame == 4


def test_object_overlay_uses_the_custom_robot_timeline(visualizer):
    spheres = torch.tensor([[[0.0, 0.0, 0.0, 0.01]]])
    index = visualizer.animate_collision_spheres(spheres, timeline_prefix="inspection")
    visualizer.animate_object_collision_spheres(spheres, schedule_index=index)
    assert {call.args[0] for call in module.rr.set_time.call_args_list} == {"inspection_0"}


def test_idle_clear_follows_a_static_only_preview(visualizer):
    times, clears = [], []
    module.rr.set_time.side_effect = lambda *args, **kwargs: times.append(kwargs["sequence"])
    module.rr.log.side_effect = lambda path, value, **kwargs: (
        clears.append(times[-1]) if isinstance(value, module.rr.Clear) else None
    )
    visualizer.visualize_plan(SimpleNamespace(position=torch.zeros(2, 1)), torch.eye(4))
    preview_tick = max(times)
    clears.clear()
    visualizer.mark_idle()
    assert clears and min(clears) > preview_tick
