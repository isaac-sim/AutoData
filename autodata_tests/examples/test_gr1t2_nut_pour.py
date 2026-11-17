# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native mimic settings for the GR1T2 example."""

import pytest

pytest.importorskip("isaaclab")
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

from autodata_examples.gr1t2_nut_pour import _mimic_joints_newton  # noqa: E402


@pytest.mark.parametrize(
    ("frequency", "damping_ratio", "expected_reference", "expected_impedance"),
    [
        (25.0, 0.005, (2.2877142857, 0.0174923488), 0.0008742350),
        (17184.0, 558.47998046875, (0.0020001026, 4.0619149239), 0.9999486920),
    ],
)
def test_mimic_settings_preserve_coefficients_and_implicit_response(
    frequency, damping_ratio, expected_reference, expected_impedance
):
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/robot", "Xform")
    leader = UsdPhysics.RevoluteJoint.Define(stage, "/robot/leader")
    follower = UsdPhysics.RevoluteJoint.Define(stage, "/robot/follower").GetPrim()
    follower.AddAppliedSchema("PhysxMimicJointAPI:rotZ")
    source = {"naturalFrequency": frequency, "dampingRatio": damping_ratio, "gearing": -1.128, "offset": 30.0}
    for name, value in source.items():
        follower.CreateAttribute(f"physxMimicJoint:rotZ:{name}", Sdf.ValueTypeNames.Float).Set(value)
    follower.CreateRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets([leader.GetPath()])

    _mimic_joints_newton(root, solver_dt=0.001)

    assert "MjcEqualityJointAPI" in follower.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert follower.GetRelationship("newton:mimicJoint").GetTargets() == [leader.GetPath()]
    assert follower.GetAttribute("newton:mimicCoef0").Get() == -30.0
    assert follower.GetAttribute("newton:mimicCoef1").Get() == pytest.approx(1.128)
    assert follower.GetAttribute("mjc:solref").Get() == pytest.approx(expected_reference)
    assert follower.GetAttribute("mjc:solref").Get()[0] >= 0.002
    assert follower.GetAttribute("mjc:solimp").Get()[:2] == pytest.approx([expected_impedance] * 2)
    for name, value in source.items():
        assert follower.GetAttribute(f"physxMimicJoint:rotZ:{name}").Get() == pytest.approx(value)
