# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.waypoint` (the non-async data structures)."""

import torch

import pytest

from isaac_autodata_core.waypoint import Waypoint, WaypointSequence, WaypointTrajectory


def _pose(pos) -> torch.Tensor:
    p = torch.eye(4)
    p[:3, 3] = torch.tensor(pos, dtype=torch.float32)
    return p


def _wp(pos, passthrough=None) -> Waypoint:
    return Waypoint(pose=_pose(pos), passthrough_action=passthrough if passthrough is not None else {})


# ---------------------------------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------------------------------
def test_waypoint_stores_fields():
    pose = _pose([1.0, 2.0, 3.0])
    passthrough = {"franka": torch.zeros(1)}
    wp = Waypoint(pose=pose, passthrough_action=passthrough, noise=0.5)
    assert torch.equal(wp.pose, pose)
    assert wp.passthrough_action is passthrough
    assert wp.noise == 0.5
    assert wp.joint_seed is None


# ---------------------------------------------------------------------------------------------------
# WaypointSequence
# ---------------------------------------------------------------------------------------------------
def test_sequence_from_poses_slices_passthrough_and_broadcasts_scalar_noise():
    poses = torch.stack([_pose([0.0, 0.0, 0.0]), _pose([1.0, 1.0, 1.0]), _pose([2.0, 2.0, 2.0])])
    passthrough = {"franka": torch.arange(3, dtype=torch.float32).reshape(3, 1)}
    seq = WaypointSequence.from_poses(poses=poses, passthrough_actions=passthrough, action_noise=0.1)
    assert len(seq) == 3
    assert torch.equal(seq[1].pose, _pose([1.0, 1.0, 1.0]))
    assert torch.allclose(seq[1].passthrough_action["franka"], torch.tensor([1.0]))
    assert torch.allclose(seq[1].noise, torch.tensor(0.1))


def test_sequence_from_poses_per_step_tensor_noise():
    poses = torch.stack([_pose([0.0, 0.0, 0.0]), _pose([1.0, 1.0, 1.0])])
    passthrough = {"franka": torch.zeros(2, 1)}
    seq = WaypointSequence.from_poses(
        poses=poses, passthrough_actions=passthrough, action_noise=torch.tensor([0.2, 0.3])
    )
    assert torch.allclose(seq[0].noise, torch.tensor(0.2))
    assert torch.allclose(seq[1].noise, torch.tensor(0.3))


def test_sequence_rejects_non_waypoint():
    with pytest.raises(AssertionError):
        WaypointSequence(sequence=[object()])


def test_sequence_deepcopies_input():
    wp = _wp([0.0, 0.0, 0.0])
    seq = WaypointSequence(sequence=[wp])
    seq[0].pose[0, 3] = 9.0
    assert wp.pose[0, 3] == 0.0  # original untouched


def test_sequence_add_concatenates():
    combined = WaypointSequence([_wp([0.0, 0.0, 0.0])]) + WaypointSequence([_wp([1.0, 1.0, 1.0])])
    assert len(combined) == 2


def test_sequence_split():
    seq = WaypointSequence([_wp([0.0, 0.0, 0.0]), _wp([1.0, 1.0, 1.0]), _wp([2.0, 2.0, 2.0])])
    prefix, suffix = seq.split(1)
    assert len(prefix) == 1
    assert len(suffix) == 2


def test_sequence_last_waypoint_is_copy():
    seq = WaypointSequence([_wp([0.0, 0.0, 0.0]), _wp([1.0, 1.0, 1.0])])
    last = seq.last_waypoint
    last.pose[0, 3] = 9.0
    assert seq[-1].pose[0, 3] == 1.0


# ---------------------------------------------------------------------------------------------------
# WaypointTrajectory
# ---------------------------------------------------------------------------------------------------
def test_trajectory_len_and_flattened_indexing():
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0]), _wp([1.0, 1.0, 1.0])]))
    traj.add_waypoint_sequence(WaypointSequence([_wp([2.0, 2.0, 2.0])]))
    assert len(traj) == 3
    assert torch.equal(traj[0].pose, _pose([0.0, 0.0, 0.0]))
    assert torch.equal(traj[2].pose, _pose([2.0, 2.0, 2.0]))  # crosses into the second sequence


def test_trajectory_index_out_of_range():
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0])]))
    with pytest.raises(AssertionError):
        traj[5]


def test_trajectory_target_pose_skip_interpolation():
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence_for_target_pose(
        pose=_pose([5.0, 5.0, 5.0]), passthrough_action={"franka": torch.zeros(1)}, num_steps=3, skip_interpolation=True
    )
    assert len(traj) == 3
    assert torch.equal(traj[0].pose, _pose([5.0, 5.0, 5.0]))
    assert torch.equal(traj.last_waypoint.pose, _pose([5.0, 5.0, 5.0]))


def test_trajectory_interpolation_requires_nonempty():
    traj = WaypointTrajectory()
    with pytest.raises(AssertionError, match="cannot interpolate from an empty trajectory"):
        traj.add_waypoint_sequence_for_target_pose(
            pose=_pose([1.0, 1.0, 1.0]),
            passthrough_action={"franka": torch.zeros(1)},
            num_steps=3,
            skip_interpolation=False,
        )


def test_trajectory_pop_first():
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0]), _wp([1.0, 1.0, 1.0])]))
    first = traj.pop_first()
    assert len(first) == 1
    assert len(traj) == 1
    assert torch.equal(traj[0].pose, _pose([1.0, 1.0, 1.0]))


def test_trajectory_merge_without_bridge_concatenates():
    a = WaypointTrajectory()
    a.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0])]))
    b = WaypointTrajectory()
    b.add_waypoint_sequence(WaypointSequence([_wp([1.0, 1.0, 1.0])]))
    a.merge(b)
    assert len(a) == 2


def test_trajectory_merge_with_fixed_hold():
    a = WaypointTrajectory()
    a.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0], {"franka": torch.zeros(1)})]))
    b = WaypointTrajectory()
    b.add_waypoint_sequence(
        WaypointSequence(
            [_wp([1.0, 1.0, 1.0], {"franka": torch.zeros(1)}), _wp([2.0, 2.0, 2.0], {"franka": torch.zeros(1)})]
        )
    )
    a.merge(b, num_steps_fixed=2)
    # pop b's first waypoint, hold it for (num_steps_fixed + 1) steps (no interp segment), then
    # append b's remaining waypoint: 1 (original) + 3 (fixed hold) + 1 (remaining) = 5.
    assert len(a) == 5


def test_trajectory_get_full_sequence_flattens():
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence([_wp([0.0, 0.0, 0.0]), _wp([1.0, 1.0, 1.0])]))
    traj.add_waypoint_sequence(WaypointSequence([_wp([2.0, 2.0, 2.0])]))
    flat = traj.get_full_sequence()
    assert isinstance(flat, WaypointSequence)
    assert len(flat) == 3
