# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SkillGen's shared visualization options are forwarded without starting a simulator."""

import ast
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths


def test_skillgen_saves_rerun_only_for_environment_zero(monkeypatch):
    pytest.importorskip("curobo.geom.sdf.world")
    from autodata_interfaces.motion_planners.curobo import curobo_planner
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    script = Path(TestPaths.scripts_dir) / "generate_dataset.py"
    tree = ast.parse(script.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_motion_planners"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    monkeypatch.setattr(CuroboPlannerCfg, "from_profile", lambda profile: CuroboPlannerCfg())
    monkeypatch.setattr(curobo_planner, "CuroboPlanner", lambda **kwargs: kwargs["config"])
    planners = namespace["_build_motion_planners"](
        None, 2, "example", planner_profile="example", visualize_plan=True, rerun_output="plans.rrd"
    )
    assert planners[0].rerun_output == "plans.rrd" and planners[0].visualize_plan
    assert planners[1].rerun_output is None and not planners[1].visualize_plan
