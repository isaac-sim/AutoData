# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    ConcurrentGroupSegment,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionOutcome,
    GoalPredicate,
    GripperCommandMode,
    GripperCommandSegment,
    RobotStateSnapshot,
    SceneObjectSnapshot,
    SceneSnapshot,
    TaskMotionPlan,
    WaitSegment,
    make_stable_id,
    matrix4,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _scene_snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        snapshot_id="snapshot-1",
        captured_at_s=4.5,
        env_id=0,
        robot=RobotStateSnapshot(
            robot_id="franka",
            joint_names=("joint_1", "joint_2"),
            joint_positions=(0.1, -0.2),
            eef_poses={"franka": IDENTITY},
            held_objects={"franka": None},
        ),
        objects=(
            SceneObjectSnapshot(
                semantic_id="cube",
                scene_id="cube_1",
                pose=IDENTITY,
                roles=("support", "graspable", "support"),
            ),
        ),
        metadata={"seed": 7},
    )


def _plan() -> TaskMotionPlan:
    move = CartesianTrajectorySegment(
        segment_id="move-1",
        eef_name="franka",
        frame="robot_base",
        poses=(IDENTITY,),
        joint_seed_names=("joint_1", "joint_2"),
        joint_seeds=((0.1, -0.2),),
        expected_postconditions=(GoalPredicate("near", "franka", "cube_1"),),
    )
    close = GripperCommandSegment(
        segment_id="close-1",
        depends_on=(move.segment_id,),
        eef_name="franka",
        command=GripperCommandMode.CLOSE,
        settle_steps=4,
    )
    attach = AttachIntentSegment(
        segment_id="attach-1",
        depends_on=(close.segment_id,),
        eef_name="franka",
        object_name="cube_1",
        verifier="contact_and_relative_motion_v1",
    )
    return TaskMotionPlan(
        plan_id="plan-1",
        request_digest="request-digest",
        snapshot_digest="snapshot-digest",
        backend="schedulestream_custream",
        backend_version="test",
        seed=7,
        segments=(move, close, attach),
        goal=(GoalPredicate("on", "cube_1", "table"),),
        metadata={"motion_backend": "curobo_v1"},
    )


def test_stable_id_is_deterministic_and_namespaced():
    assert make_stable_id("segment", "a", 1) == make_stable_id("segment", "a", 1)
    assert make_stable_id("segment", "a", 1).startswith("segment-")
    assert make_stable_id("segment", "a", 1) != make_stable_id("segment", "a", 2)


@pytest.mark.parametrize(
    "pose,match",
    [
        (((1.0, 0.0),), "shape"),
        (
            (
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            "unit norm",
        ),
        (
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            "determinant",
        ),
    ],
)
def test_matrix4_rejects_non_rigid_transforms(pose, match):
    with pytest.raises(ValueError, match=match):
        matrix4(pose)


def test_scene_snapshot_round_trip_and_digest_are_deterministic():
    snapshot = _scene_snapshot()
    rebuilt = SceneSnapshot.from_dict(snapshot.to_dict())

    assert rebuilt == snapshot
    assert rebuilt.digest == snapshot.digest
    assert rebuilt.objects[0].roles == ("graspable", "support")


def test_task_motion_plan_round_trip_preserves_symbolic_segments():
    plan = _plan()
    rebuilt = TaskMotionPlan.from_dict(json.loads(plan.canonical_json()))

    assert rebuilt == plan
    assert rebuilt.digest == plan.digest
    assert rebuilt.segments[2].kind.value == "attach_intent"
    assert isinstance(rebuilt.segments[2], AttachIntentSegment)


def test_plan_rejects_unknown_dependencies():
    segment = WaitSegment(segment_id="wait", depends_on=("missing",), steps=1)
    with pytest.raises(ValueError, match="unknown dependencies"):
        TaskMotionPlan(
            plan_id="plan",
            request_digest="request",
            snapshot_digest="snapshot",
            backend="test",
            backend_version="1",
            seed=0,
            segments=(segment,),
            goal=(),
        )


def test_plan_rejects_dependency_cycles():
    first = WaitSegment(segment_id="first", depends_on=("second",), steps=1)
    second = WaitSegment(segment_id="second", depends_on=("first",), steps=1)
    with pytest.raises(ValueError, match="dependency cycle"):
        TaskMotionPlan(
            plan_id="plan",
            request_digest="request",
            snapshot_digest="snapshot",
            backend="test",
            backend_version="1",
            seed=0,
            segments=(first, second),
            goal=(),
        )


def test_plan_rejects_unknown_concurrent_members():
    group = ConcurrentGroupSegment(segment_id="group", member_segment_ids=("missing",))
    with pytest.raises(ValueError, match="unknown members"):
        TaskMotionPlan(
            plan_id="plan",
            request_digest="request",
            snapshot_digest="snapshot",
            backend="test",
            backend_version="1",
            seed=0,
            segments=(group,),
            goal=(),
        )


def test_execution_event_is_bounded_json_contract():
    event = ExecutionEvent(
        event_id="event-1",
        attempt_id="attempt-1",
        plan_id="plan-1",
        segment_id="attach-1",
        event_type=ExecutionEventType.GRASP_INTENT,
        outcome=ExecutionOutcome.PENDING,
        monotonic_time_s=3.2,
        verifier="contact_and_relative_motion_v1",
        metadata={"candidate_id": "grasp-2"},
    )

    assert event.to_dict()["event_type"] == "grasp_intent"
    assert event.to_dict()["outcome"] == "pending"


def test_metadata_rejects_non_finite_numbers():
    with pytest.raises(ValueError, match="finite"):
        _scene_snapshot().__class__(
            snapshot_id="snapshot",
            captured_at_s=0.0,
            env_id=0,
            robot=_scene_snapshot().robot,
            objects=(),
            metadata={"bad": float("nan")},
        )
