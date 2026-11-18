# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PhysX-to-URDF mimic conversion without constructing a simulator."""

import math
import numpy as np
import torch
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("isaaclab")
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory  # noqa: E402
from autodata_interfaces.env.pink_ik_action import (  # noqa: E402
    PinkInverseKinematicsAction,
    PinkInverseKinematicsActionSharedUrdf,
    _urdf_mimic_joints_sync,
)


def test_hand_action_order_does_not_follow_backend_joint_order(monkeypatch):
    monkeypatch.setattr(PinkInverseKinematicsAction, "_initialize_joint_info", lambda self: None)
    names = ["right_finger", "left_finger"]

    def find_joints(requested, *, preserve_order):
        assert requested == names and preserve_order
        return [4, 2], names

    action = object.__new__(PinkInverseKinematicsActionSharedUrdf)
    action.cfg = SimpleNamespace(hand_joint_names=names)
    action._asset = SimpleNamespace(find_joints=find_joints)
    action._isaaclab_controlled_joint_ids = [0, 1]
    action._isaaclab_controlled_joint_names = ["arm_a", "arm_b"]

    action._initialize_joint_info()

    assert action._controlled_joint_ids == [0, 1, 4, 2]
    assert action._controlled_joint_names == ["arm_a", "arm_b", *names]


@pytest.mark.parametrize("valid, failed_seed", [(True, False), (False, False), (True, True)])
def test_coupled_action_restores_targets_after_success_or_rejection(valid, failed_seed, monkeypatch):
    pin = pytest.importorskip("pinocchio")
    frame = SimpleNamespace(frame="tool", transform_target_to_base=pin.SE3(np.eye(3), np.ones(3)))

    def set_frame_target(target):
        frame.transform_target_to_base = target.copy()

    def frame_error(configuration):
        return np.zeros(6)

    frame.set_target = set_frame_target
    frame.compute_error = frame_error
    posture = SimpleNamespace(target_q=np.array([0.3]))

    def set_posture_target(target):
        posture.target_q = target.copy()

    posture.set_target = set_posture_target
    configuration = SimpleNamespace(
        q=np.zeros(1),
        model=SimpleNamespace(lowerPositionLimit=np.array([-1.0]), upperPositionLimit=np.array([1.0])),
        controlled_joint_names_pinocchio_order=("joint",),
    )

    def update_configuration(q):
        configuration.q = q.copy()

    configuration.update = update_configuration
    controller = SimpleNamespace(
        cfg=SimpleNamespace(
            variable_input_tasks=(frame, posture),
            joint_names=("joint",),
            base_link_name="base",
            fail_on_joint_limit_violation=True,
        ),
        pink_configuration=configuration,
        isaac_lab_to_pink_ordering=np.array([0]),
        compute=lambda q, dt: torch.tensor(q.copy()),
        update_null_space_joint_targets=posture.set_target,
        init_joint_positions=np.zeros(1),
    )
    action = object.__new__(PinkInverseKinematicsActionSharedUrdf)
    action.cfg = SimpleNamespace(controller=SimpleNamespace(base_link_name="base"))
    action._asset = SimpleNamespace(joint_names=["joint"])
    action._ik_controllers = [controller]
    action._isaaclab_controlled_joint_names = ["joint"]
    action._isaaclab_controlled_joint_ids = [0]
    action._num_frame_tasks = 1
    action._sim_dt = 0.01
    action._env = SimpleNamespace(cfg=SimpleNamespace(decimation=1))

    def base_link_pose():
        return torch.eye(4).unsqueeze(0)

    action._get_base_link_frame_transform = base_link_pose
    initial = NamedJointTrajectory(("joint",), torch.zeros(1, 1))
    seeds = NamedJointTrajectory(("joint",), torch.zeros(2, 1))
    if failed_seed:
        from pink.exceptions import NoSolutionFound

        error = NoSolutionFound.__new__(NoSolutionFound)
        Exception.__init__(error, "invalid seed")
        solver = Mock(side_effect=error)
        monkeypatch.setattr("pink.solve_ik", solver)
        frame.compute_error = lambda configuration: np.zeros(6) if configuration.q[0] > 0.5 else np.ones(6)
        seeds = NamedJointTrajectory(("joint",), torch.ones(2, 1))
    args = ({"tool": torch.eye(4).repeat(2, 1, 1)}, {"tool": seeds}, initial, 0.01, 0.02)

    assert action.get_action_target_frame_names() == ("tool",)
    if valid:
        result = action.solve_joint_trajectory(*args, lambda names, positions, sample: True)
        assert result.joint_names == ("joint",) and torch.equal(result.positions, seeds.positions)
    else:
        with pytest.raises(AssertionError, match="Pink could not solve"):
            action.solve_joint_trajectory(*args, lambda names, positions, sample: False)
    assert np.array_equal(frame.transform_target_to_base.translation, np.ones(3))
    assert np.array_equal(posture.target_q, np.array([0.3]))
    if failed_seed:
        solver.assert_called_once()


@pytest.mark.parametrize(
    ("joint_type", "axis", "offset", "expected_offset"),
    [("revolute", "rotZ", 30.0, -math.pi / 6.0), ("prismatic", "transX", 25.0, -0.25)],
)
def test_mimic_coefficients_match_physx_equation(tmp_path, joint_type, axis, offset, expected_offset):
    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    prim = stage.DefinePrim("/robot/follower")
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([f"PhysxMimicJointAPI:{axis}"]))
    prim.CreateAttribute(f"physxMimicJoint:{axis}:gearing", Sdf.ValueTypeNames.Float).Set(-1.128)
    prim.CreateAttribute(f"physxMimicJoint:{axis}:offset", Sdf.ValueTypeNames.Float).Set(offset)
    stage.GetRootLayer().Save()
    root = ET.fromstring(
        f'<robot><joint name="driver" type="{joint_type}"/><joint name="follower" type="{joint_type}">'
        '<mimic joint="driver" multiplier="-1.128" offset="0"/></joint></robot>'
    )

    _urdf_mimic_joints_sync(root, str(path))
    mimic = root.find("joint[@name='follower']/mimic")
    assert float(mimic.get("multiplier")) == pytest.approx(1.128)
    assert float(mimic.get("offset")) == pytest.approx(expected_offset)
    first_result = ET.tostring(root)
    _urdf_mimic_joints_sync(root, str(path))
    assert ET.tostring(root) == first_result
