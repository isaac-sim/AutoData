# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial joint DR configuration and reset invariants."""
import math
import torch
from types import SimpleNamespace

import pytest
import warp as wp

from autodata_domain_randomization import (
    DomainRandomizationCfg,
    RobotJointRandomizationCfg,
    create_domain_randomization_profile,
)
from autodata_domain_randomization.robot import reset_all_articulation_joints


@pytest.mark.parametrize(
    "values,error",
    [
        ({}, "requires joint_names"),
        ({"joint_names": "arm"}, "must be a sequence"),
        ({"joint_names": [""]}, "non-empty strings"),
        ({"joint_names": ["arm"], "delta_range_deg": (5, -5)}, "lower bound exceeds upper bound"),
        ({"joint_names": ["arm"], "delta_range_deg": (0,)}, "must contain two finite values"),
        ({"joint_names": ["arm"], "delta_range_deg": (0, float("inf"))}, "must contain two finite values"),
        ({"joint_names": ["arm"], "delta_range_deg": (float("nan"), 5)}, "must contain two finite values"),
    ],
)
def test_invalid_joint_dr_configuration(values, error):
    with pytest.raises(AssertionError, match=error):
        RobotJointRandomizationCfg(**values)


def test_yaml_overrides_native_reset_in_degrees_and_combines_with_root_dr():
    cfg = DomainRandomizationCfg.from_dict({
        "robot_root": {"x_range_m": [-0.1, 0.1]},
        "robot_joints": {"joint_names": [".*_shoulder_.*"], "delta_range_deg": [-5, 5]},
    })
    profile = create_domain_randomization_profile("Test", cfg)
    params = profile.events.override["reset_all"].params
    assert params["robot_joint_names"] == (".*_shoulder_.*",)
    assert params["robot_joint_position_range"] == pytest.approx((-math.pi / 36, math.pi / 36))
    assert params["robot_root_pose_range"]["x"] == (-0.1, 0.1)


def test_omitting_joint_dr_leaves_the_original_reset():
    cfg = DomainRandomizationCfg.from_dict({})
    assert "reset_all" not in create_domain_randomization_profile("Test", cfg).events.override


def test_unknown_joint_dr_yaml_key_rejected():
    with pytest.raises(AssertionError, match="unknown keys"):
        DomainRandomizationCfg.from_dict({"robot_joints": {"range_deg": [-5, 5]}})


class Articulation:
    def __init__(self, count):
        default = torch.zeros(count, 13)
        default[:, 6] = 1
        self.data = SimpleNamespace(
            default_root_state=wp.from_torch(default),
            default_joint_pos=wp.from_torch(torch.zeros(count, 5)),
            default_joint_vel=wp.from_torch(torch.zeros(count, 5)),
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([-2.0, 2.0]).expand(count, 5, 2).clone()),
        )
        self.pose = torch.full((count, 7), -999.0)
        self.velocity = torch.full((count, 6), -999.0)
        self.joints = torch.full((count, 5), -999.0)
        self.joint_velocity = torch.full((count, 5), -999.0)
        self.targets = torch.full((count, 5), -999.0)

    def find_joints(self, names):
        assert names == ("arms",)
        return [1, 2, 3], ["shoulder", "elbow", "wrist"]

    def write_root_pose_to_sim(self, value, env_ids):
        self.pose[env_ids] = value

    def write_root_velocity_to_sim(self, value, env_ids):
        self.velocity[env_ids] = value

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        self.joints[env_ids] = position
        self.joint_velocity[env_ids] = velocity

    def set_joint_position_target(self, value, joint_ids, env_ids):
        self.targets[env_ids[:, None], joint_ids] = value


def test_joint_offsets_are_independent_seeded_and_relative_to_defaults_on_partial_resets():
    count = 4097
    robot, other = Articulation(count), Articulation(count)
    env = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(articulations={"robot": robot, "other": other}, env_origins=torch.zeros(count, 3)),
    )
    ids = torch.arange(1, count)
    bounds = (-math.pi / 36, math.pi / 36)
    previous = None
    for _ in range(2):
        torch.manual_seed(41)
        reset_all_articulation_joints(env, ids, robot_joint_names=("arms",), robot_joint_position_range=bounds)
        current = robot.joints.clone()
        if _:
            torch.testing.assert_close(current, previous, atol=0, rtol=0)
        previous = current
    draws = robot.joints[ids, 1:4]
    assert draws.min() >= bounds[0] and draws.max() <= bounds[1]
    assert (draws.mean(dim=0).abs() < 0.003).all()
    expected_std = (bounds[1] - bounds[0]) / math.sqrt(12)
    torch.testing.assert_close(draws.std(dim=0), torch.full((3,), expected_std), rtol=0.04, atol=0)
    assert (torch.corrcoef(draws.T) - torch.eye(3)).abs().max() < 0.05
    assert (robot.joints[0] == -999).all() and (robot.targets[0] == -999).all()
    assert (robot.joints[ids][:, [0, 4]] == 0).all()
    assert (robot.targets[ids][:, [0, 4]] == -999).all()
    torch.testing.assert_close(robot.targets[ids, 1:4], draws, rtol=0, atol=0)
    assert (other.joints[ids] == 0).all() and (other.targets == -999).all()
    assert (robot.joint_velocity[ids] == 0).all() and (robot.velocity[ids] == 0).all()
    assert (wp.to_torch(robot.data.default_joint_pos) == 0).all()
    rng = torch.get_rng_state().clone()
    reset_all_articulation_joints(
        env, torch.tensor([], dtype=torch.long), robot_joint_names=("arms",), robot_joint_position_range=bounds
    )
    assert torch.equal(rng, torch.get_rng_state())
    reset_all_articulation_joints(env, ids)
    assert torch.equal(rng, torch.get_rng_state())
    assert (robot.joints[ids] == 0).all()


def test_joint_offsets_are_clamped_to_physical_limits():
    robot = Articulation(2)
    wp.to_torch(robot.data.default_joint_pos)[:, 1:4] = 1.99
    env = SimpleNamespace(
        device="cpu", scene=SimpleNamespace(articulations={"robot": robot}, env_origins=torch.zeros(2, 3))
    )
    reset_all_articulation_joints(
        env,
        torch.tensor([1]),
        robot_joint_names=("arms",),
        robot_joint_position_range=(math.radians(5), math.radians(5)),
    )
    assert (robot.joints[1, 1:4] == 2.0).all()
    assert (robot.targets[1, 1:4] == 2.0).all()
    assert (wp.to_torch(robot.data.default_joint_pos)[:, 1:4] == 1.99).all()
