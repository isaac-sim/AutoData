# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math

import pytest

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    DetachIntentSegment,
    GoalPredicate,
    GripperCommandMode,
    GripperCommandSegment,
    TaskMotionPlan,
)
from isaac_autodata_interfaces.autonomous.pick_place_success import PickPlaceSuccessTracker

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _geometry_record(object_id, asset_name, dimensions, center_offset=(0.0, 0.0, 0.0)):
    transform = [list(row) for row in IDENTITY]
    for index, value in enumerate(center_offset):
        transform[index][3] = value
    return {
        "aabb_center_in_converted_object_origin_m": list(center_offset),
        "aabb_center_in_isaac_rigid_root_m": list(center_offset),
        "aabb_dimensions_m": list(dimensions),
        "aabb_kind": "converted_mesh_local_axis_aligned_bounding_box",
        "asset_name": asset_name,
        "converted_object_origin_from_aabb_center": transform,
        "isaac_rigid_root_from_aabb_center": transform,
        "object_id": object_id,
    }


def _placement_geometry(*, subject_offset=(0.0, 0.0, 0.0), target_offset=(0.0, 0.0, 0.0)):
    return {
        "attested": True,
        "destination": _geometry_record("bowl", "bowl_ycb_robolab", (0.16, 0.16, 0.05), target_offset),
        "frame_convention": "parent_from_child_homogeneous_4x4",
        "general_inside_semantics": False,
        "limitations": ["name_pinned_local_aabb_geometry_not_container_interior_geometry"],
        "placement_model": "destination_local_aabb_top_plane_shifted_downward",
        "predicted_aabb_center_offset_m": [0.0, 0.0, 0.03],
        "profile": "franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1",
        "relation": "on",
        "sampled_surface_extent_m": [0.0, 0.0, 0.0],
        "schema_version": 1,
        "subject": _geometry_record(
            "cube",
            "rubiks_cube_hot3d_robolab",
            (0.06, 0.06, 0.06),
            subject_offset,
        ),
        "surface_config": {
            "implementation": "schedulestream.applications.custream.object.SurfaceConfig",
            "xy_extend_m": -0.16,
            "z_offset_m": -0.025,
        },
        "vertical_evidence_corridor_m": 0.04,
    }


def _multiply4(left, right):
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)] for row in range(4)
    ]


def _inverse_translation(transform):
    result = [list(row) for row in IDENTITY]
    for index in range(3):
        result[index][3] = -transform[index][3]
    return result


def _grasp_geometry(object_from_aabb):
    primitives = []
    for yaw in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        primitives.append([
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    aabb_from_object = _inverse_translation(object_from_aabb)
    return {
        "asset_name": "rubiks_cube_hot3d_robolab",
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "converted_object_origin_from_aabb_center": object_from_aabb,
        "link_from_object_transforms": [_multiply4(primitive, aabb_from_object) for primitive in primitives],
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": 4,
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": "cube",
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "primitive_link_from_aabb_center_transforms": primitives,
        "profile": "franka_rubiks_cube_offcenter_cuboid_top_v1",
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }


def _plan(*, include_detach: bool = True, destination_placement=None, grasp_geometry=None) -> TaskMotionPlan:
    destination_placement = destination_placement or _placement_geometry()
    grasp_geometry = grasp_geometry or _grasp_geometry(
        destination_placement["subject"]["converted_object_origin_from_aabb_center"]
    )
    segments = [
        GripperCommandSegment(segment_id="initial_open", eef_name="tool", command=GripperCommandMode.OPEN),
        CartesianTrajectorySegment(
            segment_id="approach",
            depends_on=("initial_open",),
            eef_name="tool",
            frame="env_origin",
            poses=(IDENTITY,),
        ),
        GripperCommandSegment(
            segment_id="close",
            depends_on=("approach",),
            eef_name="tool",
            command=GripperCommandMode.CLOSE,
        ),
        AttachIntentSegment(
            segment_id="attach",
            depends_on=("close",),
            eef_name="tool",
            object_name="cube",
            verifier="contact_and_relative_motion_v1",
        ),
        CartesianTrajectorySegment(
            segment_id="transport",
            depends_on=("attach",),
            eef_name="tool",
            frame="env_origin",
            poses=(IDENTITY,),
        ),
        GripperCommandSegment(
            segment_id="release",
            depends_on=("transport",),
            eef_name="tool",
            command=GripperCommandMode.OPEN,
        ),
    ]
    if include_detach:
        segments.append(
            DetachIntentSegment(
                segment_id="detach",
                depends_on=("release",),
                eef_name="tool",
                object_name="cube",
                verifier="contact_and_relative_motion_v1",
            )
        )
    return TaskMotionPlan(
        plan_id="plan",
        request_digest="request",
        snapshot_digest="snapshot",
        backend="schedulestream_custream",
        backend_version="test",
        seed=3,
        segments=tuple(segments),
        goal=(GoalPredicate("on", "cube", "bowl"),),
        metadata={
            "arena_success_contract": {"attested": True},
            "schedulestream": {
                "attachment_events_preserved": True,
                "grasp_geometry": grasp_geometry,
                "destination_placement": destination_placement,
            },
        },
    )


def _sample(
    *,
    cube=(0.0, 0.0, 0.0),
    bowl=(0.1, 0.0, 0.03),
    eef=(0.0, 0.0, 0.0),
    aperture=0.08,
    linear_speed=0.0,
    angular_speed=0.0,
):
    eef_pose = [list(row) for row in IDENTITY]
    for index, value in enumerate(eef):
        eef_pose[index][3] = value
    return {
        "eef_poses_env": {"tool": eef_pose},
        "joint_positions": {
            "panda_finger_joint1": aperture / 2,
            "panda_finger_joint2": aperture / 2,
        },
        "objects": {
            "bowl": {
                "angular_speed_rad_s": 0.0,
                "linear_speed_m_s": 0.0,
                "position_env_m": list(bowl),
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "cube": {
                "angular_speed_rad_s": angular_speed,
                "linear_speed_m_s": linear_speed,
                "position_env_m": list(cube),
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }


def _successful_tracker() -> PickPlaceSuccessTracker:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for index in range(1, 11):
        position = (0.01 * index, 0.0, 0.004 * index)
        tracker.observe_step(_sample(cube=position, eef=position, aperture=0.056))
    release = _sample(cube=(0.1, 0.0, 0.04), eef=(0.1, 0.0, 0.04), aperture=0.056)
    tracker.begin_release("release", release)
    tracker.detach("detach", "cube", "tool", release)
    final = _sample(
        cube=(0.1, 0.0, 0.03),
        eef=(0.2, 0.0, 0.10),
        aperture=0.08,
        linear_speed=0.01,
        angular_speed=0.1,
    )
    for _ in range(5):
        tracker.observe_final(final, goal_satisfied=True)
    return tracker


def _release_ready_tracker() -> PickPlaceSuccessTracker:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for index in range(1, 11):
        position = (0.01 * index, 0.0, 0.004 * index)
        tracker.observe_step(_sample(cube=position, eef=position, aperture=0.056))
    return tracker


def test_pick_place_success_accepts_observed_grasp_transport_release_and_stable_goal() -> None:
    report = _successful_tracker().report(logical_held_object=None)

    assert report["passed"] is True
    assert report["closed_transport"]["samples"] == 10
    assert report["closed_transport"]["maximum_lift_m"] == pytest.approx(0.04)
    assert all(check["passed"] for check in report["checks"])


def test_release_contact_gate_accepts_finite_attested_placement_and_transport() -> None:
    gate = _release_ready_tracker().release_contact_gate(
        "release",
        _sample(cube=(0.1, 0.0, 0.04), eef=(0.0905, 0.0, 0.04), aperture=0.056),
    )

    assert gate["passed"] is True
    assert gate["current_physical_state_finite"] is True
    placement = gate["current_prerelease_placement"]
    assert placement["subject_target_horizontal_radius_m"] == pytest.approx(0.0)
    assert placement["subject_target_vertical_offset_abs_m"] == pytest.approx(0.01)
    assert placement["subject_eef_attachment_distance_m"] == pytest.approx(0.0095)
    assert placement["grasp_aperture_m"] == pytest.approx(0.056)
    assert all(check["passed"] for check in gate["checks"])


def test_release_contact_gate_rejects_accumulated_destination_drift_even_when_current_placement_matches() -> None:
    gate = _release_ready_tracker().release_contact_gate(
        "release",
        _sample(
            cube=(0.121, 0.0, 0.03),
            bowl=(0.121, 0.0, 0.03),
            eef=(0.1, 0.0, 0.04),
            aperture=0.056,
        ),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["passed"] is False
    assert checks["prerelease_subject_target_horizontal_radius_m"]["passed"] is True
    assert checks["prerelease_maximum_destination_drift_m"]["observed"] == pytest.approx(0.021)
    assert checks["prerelease_maximum_destination_drift_m"]["passed"] is False


@pytest.mark.parametrize(
    ("cube", "failed_check"),
    [
        ((0.129, 0.0, 0.03), "prerelease_subject_target_horizontal_radius_m"),
        ((0.1, 0.0, 0.071), "prerelease_subject_target_vertical_offset_abs_m"),
    ],
)
def test_release_contact_gate_rejects_prerelease_placement_outside_aabb_center_corridor(cube, failed_check) -> None:
    gate = _release_ready_tracker().release_contact_gate(
        "release",
        _sample(cube=cube, eef=(0.1, 0.0, 0.04), aperture=0.056),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["passed"] is False
    assert checks[failed_check]["passed"] is False


def test_release_contact_gate_rejects_insufficient_closed_transport_samples() -> None:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for index in range(1, 10):
        position = (0.1 if index == 9 else 0.01 * index, 0.0, 0.04 if index == 9 else 0.004 * index)
        tracker.observe_step(_sample(cube=position, eef=position, aperture=0.056))

    gate = tracker.release_contact_gate(
        "release",
        _sample(cube=(0.1, 0.0, 0.04), eef=(0.1, 0.0, 0.04), aperture=0.056),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["passed"] is False
    assert checks["prerelease_closed_transport_samples"]["observed"] == 9
    assert checks["prerelease_closed_transport_samples"]["passed"] is False


def test_release_contact_gate_rejects_closed_transport_relative_drift() -> None:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for index in range(1, 10):
        position = (0.01 * index, 0.0, 0.004 * index)
        tracker.observe_step(_sample(cube=position, eef=position, aperture=0.056))
    tracker.observe_step(_sample(cube=(0.1, 0.0, 0.04), eef=(0.075, 0.0, 0.04), aperture=0.056))

    gate = tracker.release_contact_gate(
        "release",
        _sample(cube=(0.1, 0.0, 0.04), eef=(0.075, 0.0, 0.04), aperture=0.056),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["passed"] is False
    relative = checks["prerelease_closed_transport_relative_translation_drift_m"]
    assert relative["observed"] == pytest.approx(0.025)
    assert relative["passed"] is False


@pytest.mark.parametrize(
    ("eef", "aperture", "failed_check"),
    [
        ((0.151, 0.0, 0.04), 0.056, "prerelease_subject_eef_attachment_distance_m"),
        ((0.1, 0.0, 0.04), 0.080, "prerelease_grasp_aperture_max_m"),
        ((0.1, 0.0, 0.04), 0.039, "prerelease_grasp_aperture_min_m"),
    ],
)
def test_release_contact_gate_rejects_lost_prerelease_grasp(eef, aperture, failed_check) -> None:
    gate = _release_ready_tracker().release_contact_gate(
        "release",
        _sample(cube=(0.1, 0.0, 0.04), eef=eef, aperture=aperture),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["passed"] is False
    assert checks[failed_check]["passed"] is False


def test_release_contact_gate_fails_closed_on_nonfinite_current_state() -> None:
    sample = copy.deepcopy(_sample(cube=(0.1, 0.0, 0.04), eef=(0.1, 0.0, 0.04), aperture=0.056))
    sample["eef_poses_env"]["tool"][0][3] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        _release_ready_tracker().release_contact_gate("release", sample)


def test_pick_place_success_measures_final_placement_between_validated_aabb_centers() -> None:
    placement = _placement_geometry(subject_offset=(0.03, 0.0, 0.0))
    tracker = PickPlaceSuccessTracker.from_plan(_plan(destination_placement=placement), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for index in range(1, 11):
        position = (0.007 * index, 0.0, 0.004 * index)
        tracker.observe_step(_sample(cube=position, eef=position, aperture=0.056))
    release = _sample(cube=(0.07, 0.0, 0.03), eef=(0.07, 0.0, 0.03), aperture=0.056)
    tracker.begin_release("release", release)
    tracker.detach("detach", "cube", "tool", release)
    final = _sample(cube=(0.07, 0.0, 0.03), eef=(0.2, 0.0, 0.1))
    for _ in range(5):
        tracker.observe_final(final, goal_satisfied=True)

    report = tracker.report(logical_held_object=None)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["final_subject_target_horizontal_radius_m"]["observed"] == pytest.approx(0.0)
    assert checks["final_subject_target_horizontal_radius_m"]["passed"] is True


def test_pick_place_success_rejects_proximity_only_attachment_with_closed_empty_aperture() -> None:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    missed_grasp = _sample(aperture=0.006)

    assert tracker.attachment_is_plausible(missed_grasp) is False
    with pytest.raises(ValueError, match="plausible physical aperture/distance"):
        tracker.attach("attach", "cube", "tool", missed_grasp)

    report = tracker.report(logical_held_object=None)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["attach_observed"]["passed"] is False
    assert checks["grasp_aperture_min_m"]["observed"] == pytest.approx(0.006)
    assert checks["grasp_aperture_min_m"]["passed"] is False


def test_pick_place_success_rejects_incomplete_plan_lifecycle_before_execution() -> None:
    with pytest.raises(ValueError, match="exactly one attach and one detach"):
        PickPlaceSuccessTracker.from_plan(_plan(include_detach=False), ("tool",))


def test_pick_place_success_rejects_missing_grasp_geometry_before_execution() -> None:
    with pytest.raises(ValueError, match="exact grasp geometry"):
        PickPlaceSuccessTracker.from_plan(_plan(grasp_geometry={"attested": True}), ("tool",))


def test_pick_place_success_rejects_tampered_grasp_transform_before_execution() -> None:
    grasp_geometry = _grasp_geometry([list(row) for row in IDENTITY])
    grasp_geometry["link_from_object_transforms"][0][0][3] += 0.001

    with pytest.raises(ValueError, match="violates its composition formula"):
        PickPlaceSuccessTracker.from_plan(_plan(grasp_geometry=grasp_geometry), ("tool",))


def test_pick_place_success_rejects_grasp_and_placement_aabb_frame_mismatch() -> None:
    placement = _placement_geometry(subject_offset=(0.03, 0.0, 0.0))
    grasp_geometry = _grasp_geometry([list(row) for row in IDENTITY])

    with pytest.raises(ValueError, match="grasp and placement AABB transforms do not match"):
        PickPlaceSuccessTracker.from_plan(
            _plan(destination_placement=placement, grasp_geometry=grasp_geometry),
            ("tool",),
        )


def test_pick_place_success_rejects_stationary_cube_even_when_arena_goal_is_true() -> None:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    tracker.settle_initial("initial_open", _sample(), goal_satisfied=False)
    tracker.attach("attach", "cube", "tool", _sample(aperture=0.056))
    for _ in range(10):
        tracker.observe_step(_sample(aperture=0.056))
    tracker.begin_release("release", _sample(aperture=0.056))
    tracker.detach("detach", "cube", "tool", _sample(aperture=0.056))
    final = _sample(cube=(0.1, 0.0, 0.03), eef=(0.2, 0.0, 0.1))
    for _ in range(5):
        tracker.observe_final(final, goal_satisfied=True)

    report = tracker.report(logical_held_object=None)

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "closed_transport_lift_m" in failed
    assert "closed_transport_subject_displacement_m" in failed
    assert report["passed"] is False


def test_opening_settle_motion_is_excluded_from_closed_relative_drift() -> None:
    tracker = _successful_tracker()
    before = tracker.report(logical_held_object=None)["closed_transport"]

    slipped = _sample(cube=(0.1, 0.0, -0.5), eef=(0.5, 0.0, 0.5))
    tracker.observe_step(slipped)
    after = tracker.report(logical_held_object=None)["closed_transport"]

    assert after == before


def test_pick_place_success_fails_closed_on_nonfinite_live_observation() -> None:
    tracker = PickPlaceSuccessTracker.from_plan(_plan(), ("tool",))
    assert tracker is not None
    sample = copy.deepcopy(_sample())
    sample["objects"]["cube"]["linear_speed_m_s"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        tracker.settle_initial("initial_open", sample, goal_satisfied=False)
