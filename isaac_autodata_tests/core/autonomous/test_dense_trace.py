# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaac_autodata_core.autonomous.dense_trace import (
    DenseAttachmentEvent,
    DensePlanTrace,
    task_motion_plan_from_dense_trace,
)
from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    GoalPredicate,
    GripperCommandMode,
    GripperCommandSegment,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def test_dense_trace_splits_gripper_changes_and_preserves_joint_seeds():
    trace = DensePlanTrace(
        eef_name="franka",
        frame="robot_base",
        poses=(IDENTITY, IDENTITY, IDENTITY, IDENTITY),
        gripper_values=(1.0, 1.0, -1.0, -1.0),
        step_dt_s=0.05,
        joint_names=("j1", "j2"),
        joint_positions=((0.0, 0.0), (0.1, 0.0), (0.1, 0.0), (0.3, 0.0)),
        attachment_events=(DenseAttachmentEvent(2, "attach", "cube"),),
    )

    plan = task_motion_plan_from_dense_trace(
        trace,
        request_digest="request",
        snapshot_digest="snapshot",
        backend="schedulestream_custream",
        backend_version="test",
        seed=0,
        goal=(GoalPredicate("on", "cube", "table"),),
    )

    gripper_segments = [segment for segment in plan.segments if isinstance(segment, GripperCommandSegment)]
    cartesian_segments = [segment for segment in plan.segments if isinstance(segment, CartesianTrajectorySegment)]
    attach_segments = [segment for segment in plan.segments if isinstance(segment, AttachIntentSegment)]
    assert [segment.command for segment in gripper_segments] == [
        GripperCommandMode.OPEN,
        GripperCommandMode.CLOSE,
    ]
    assert len(cartesian_segments) == 2
    assert [segment.duration_s for segment in cartesian_segments] == [0.1, 0.1]
    assert cartesian_segments[0].joint_seed_names == ("j1", "j2")
    assert cartesian_segments[1].joint_seeds[-1] == (0.3, 0.0)
    assert attach_segments[0].object_name == "cube"
    assert all(
        not segment.depends_on or segment.depends_on[0] == plan.segments[index - 1].segment_id
        for index, segment in enumerate(plan.segments)
    )


def test_continuous_gripper_value_becomes_position_command():
    trace = DensePlanTrace(
        eef_name="hand",
        frame="base",
        poses=(IDENTITY,),
        gripper_values=(0.25,),
        step_dt_s=0.1,
    )
    plan = task_motion_plan_from_dense_trace(
        trace,
        request_digest="request",
        snapshot_digest="snapshot",
        backend="test",
        backend_version="1",
        seed=0,
        goal=(),
    )

    command = plan.segments[0]
    assert isinstance(command, GripperCommandSegment)
    assert command.command is GripperCommandMode.POSITION
    assert command.value == 0.25


def test_interaction_split_rejects_nonduplicated_pose():
    shifted = tuple(
        tuple(value + (0.01 if row == 0 and column == 3 else 0.0) for column, value in enumerate(values))
        for row, values in enumerate(IDENTITY)
    )

    with pytest.raises(ValueError, match="duplicate the preceding pose"):
        DensePlanTrace(
            eef_name="hand",
            frame="base",
            poses=(IDENTITY, shifted),
            gripper_values=(1.0, -1.0),
            step_dt_s=0.1,
        )


def test_interaction_split_rejects_nonduplicated_joint_seed():
    with pytest.raises(ValueError, match="duplicate the preceding joint seed"):
        DensePlanTrace(
            eef_name="hand",
            frame="base",
            poses=(IDENTITY, IDENTITY),
            gripper_values=(1.0, -1.0),
            step_dt_s=0.1,
            joint_names=("joint",),
            joint_positions=((0.0,), (0.1,)),
        )


def test_attachment_event_rejects_initial_sample_without_preceding_target():
    with pytest.raises(ValueError, match="preceding collision-checked dense sample"):
        DensePlanTrace(
            eef_name="hand",
            frame="base",
            poses=(IDENTITY,),
            gripper_values=(-1.0,),
            step_dt_s=0.1,
            attachment_events=(DenseAttachmentEvent(0, "attach", "cube"),),
        )
