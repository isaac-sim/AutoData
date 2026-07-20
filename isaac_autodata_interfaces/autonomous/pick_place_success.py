# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live pick-and-place success checks for the reviewed Franka cube-into-bowl profile."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import takewhile
from typing import Any

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    DetachIntentSegment,
    GripperCommandMode,
    GripperCommandSegment,
    TaskMotionPlan,
)

_SCHEDULESTREAM_BACKENDS = frozenset({"schedulestream_custream", "schedulestream_custream2"})
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
_DESTINATION_PLACEMENT_PROFILE = "franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1"
_GRASP_GEOMETRY_PROFILE = "franka_rubiks_cube_offcenter_cuboid_top_v1"
_REVIEWED_SUBJECT_ASSET = "rubiks_cube_hot3d_robolab"
_REVIEWED_DESTINATION_ASSET = "bowl_ycb_robolab"


@dataclass(frozen=True)
class PickPlaceSuccessThresholds:
    """Thresholds for the name-pinned v1 cube/bowl physical-success profile.

    The geometry corridor is intentionally asset-specific and conservative. The admitted
    capability pins the Rubik and bowl asset identities, while the provider attests their actual
    converted AABBs inside reviewed drift envelopes. These thresholds are not general container
    semantics; Arena's attested success term remains authoritative.
    """

    attachment_distance_m: float = 0.05
    grasp_aperture_min_m: float = 0.040
    grasp_aperture_max_m: float = 0.075
    minimum_closed_samples: int = 10
    minimum_lift_m: float = 0.030
    minimum_transport_m: float = 0.050
    maximum_relative_translation_drift_m: float = 0.020
    maximum_relative_rotation_drift_rad: float = 0.35
    minimum_success_streak: int = 5
    maximum_final_linear_speed_m_s: float = 0.05
    maximum_final_angular_speed_rad_s: float = 0.5
    maximum_final_horizontal_radius_m: float = 0.028
    maximum_final_vertical_offset_m: float = 0.040
    maximum_destination_drift_m: float = 0.020
    minimum_final_eef_separation_m: float = 0.060
    minimum_final_aperture_m: float = 0.075


@dataclass(frozen=True)
class _PhysicalSample:
    subject_position: tuple[float, float, float]
    subject_aabb_center_position: tuple[float, float, float]
    subject_rotation: tuple[tuple[float, float, float], ...]
    subject_linear_speed: float
    subject_angular_speed: float
    target_position: tuple[float, float, float]
    target_aabb_center_position: tuple[float, float, float]
    eef_position: tuple[float, float, float]
    eef_rotation: tuple[tuple[float, float, float], ...]
    finger_aperture: float


class PickPlaceSuccessTracker:
    """Accumulate live observations and check one physical pick-and-place execution."""

    schema_version = 3
    profile = "franka_rubiks_cube_into_ycb_bowl_v1"

    def __init__(
        self,
        *,
        subject: str,
        target: str,
        eef_name: str,
        initial_open_segment_id: str,
        close_segment_id: str,
        attach_segment_id: str,
        release_open_segment_id: str,
        detach_segment_id: str,
        success_contract: Mapping[str, Any],
        grasp_geometry: Mapping[str, Any],
        placement_geometry: Mapping[str, Any],
        subject_aabb_center_offset_m: tuple[float, float, float],
        target_aabb_center_offset_m: tuple[float, float, float],
        thresholds: PickPlaceSuccessThresholds | None = None,
    ) -> None:
        self.subject = subject
        self.target = target
        self.eef_name = eef_name
        self.initial_open_segment_id = initial_open_segment_id
        self.close_segment_id = close_segment_id
        self.attach_segment_id = attach_segment_id
        self.release_open_segment_id = release_open_segment_id
        self.detach_segment_id = detach_segment_id
        self.success_contract = dict(success_contract)
        self.grasp_geometry = dict(grasp_geometry)
        self.placement_geometry = dict(placement_geometry)
        self.subject_aabb_center_offset_m = subject_aabb_center_offset_m
        self.target_aabb_center_offset_m = target_aabb_center_offset_m
        self.thresholds = thresholds or PickPlaceSuccessThresholds()
        self._baseline: _PhysicalSample | None = None
        self._baseline_goal_satisfied: bool | None = None
        self._attachment_candidate: _PhysicalSample | None = None
        self._attach: _PhysicalSample | None = None
        self._release: _PhysicalSample | None = None
        self._last: _PhysicalSample | None = None
        self._attached = False
        self._release_started = False
        self._detached = False
        self._closed_samples = 0
        self._maximum_lift_m = 0.0
        self._maximum_subject_displacement_m = 0.0
        self._maximum_eef_displacement_m = 0.0
        self._maximum_relative_translation_drift_m = 0.0
        self._maximum_relative_rotation_drift_rad = 0.0
        self._maximum_destination_drift_m = 0.0
        self._final_samples: list[tuple[_PhysicalSample, bool]] = []

    @classmethod
    def from_plan(
        cls,
        plan: TaskMotionPlan,
        eef_names: Sequence[str],
        *,
        thresholds: PickPlaceSuccessThresholds | None = None,
    ) -> PickPlaceSuccessTracker | None:
        """Return a tracker for a ScheduleStream place plan, validating its full lifecycle."""

        if plan.backend not in _SCHEDULESTREAM_BACKENDS:
            return None
        if len(plan.goal) != 1:
            raise ValueError("the reviewed ScheduleStream profile requires exactly one semantic goal")
        goal = plan.goal[0]
        if goal.relation.lower() != "on" or goal.target is None:
            raise ValueError("the reviewed ScheduleStream profile requires exactly one binary 'on' goal")
        metadata = plan.metadata.get("schedulestream")
        if not isinstance(metadata, dict) or metadata.get("attachment_events_preserved") is not True:
            raise ValueError("ScheduleStream plan does not attest preserved attachment events")
        grasp_geometry = _validate_grasp_geometry(metadata.get("grasp_geometry"), goal.subject)
        placement_geometry = _validate_placement_geometry(
            metadata.get("destination_placement"), goal.subject, goal.target
        )
        if grasp_geometry["object_from_aabb"] != placement_geometry["subject_object_from_aabb"]:
            raise ValueError("ScheduleStream grasp and placement AABB transforms do not match")
        success_contract = plan.metadata.get("arena_success_contract")
        if not isinstance(success_contract, dict) or success_contract.get("attested") is not True:
            raise ValueError("ScheduleStream plan does not carry the attested Arena success contract")

        indexed = list(enumerate(plan.segments))
        grippers = [(index, segment) for index, segment in indexed if isinstance(segment, GripperCommandSegment)]
        attaches = [(index, segment) for index, segment in indexed if isinstance(segment, AttachIntentSegment)]
        detaches = [(index, segment) for index, segment in indexed if isinstance(segment, DetachIntentSegment)]
        states = tuple(segment.command for _, segment in grippers)
        if states != (GripperCommandMode.OPEN, GripperCommandMode.CLOSE, GripperCommandMode.OPEN):
            raise ValueError("ScheduleStream place plan must contain exactly open-close-open gripper commands")
        if len(attaches) != 1 or len(detaches) != 1:
            raise ValueError("ScheduleStream place plan must contain exactly one attach and one detach intent")

        initial_index, initial_open = grippers[0]
        close_index, close = grippers[1]
        release_index, release_open = grippers[2]
        attach_index, attach = attaches[0]
        detach_index, detach = detaches[0]
        if not initial_index < close_index < attach_index < release_index < detach_index:
            raise ValueError("ScheduleStream place lifecycle must be ordered open-close-attach-open-detach")
        if not any(
            isinstance(segment, CartesianTrajectorySegment)
            for segment in plan.segments[attach_index + 1 : release_index]
        ):
            raise ValueError("ScheduleStream place plan must transport the object between attach and release")
        if attach.object_name != goal.subject or detach.object_name != goal.subject:
            raise ValueError("ScheduleStream attachment intents must bind exactly to the goal subject")
        lifecycle_eefs = {
            initial_open.eef_name,
            close.eef_name,
            attach.eef_name,
            release_open.eef_name,
            detach.eef_name,
        }
        if len(lifecycle_eefs) != 1 or lifecycle_eefs != set(eef_names):
            raise ValueError("ScheduleStream place lifecycle must bind exactly to the sole live end effector")
        if attach.verifier != "contact_and_relative_motion_v1" or detach.verifier != "contact_and_relative_motion_v1":
            raise ValueError("ScheduleStream attachment lifecycle uses an unreviewed physical verifier")

        return cls(
            subject=goal.subject,
            target=goal.target,
            eef_name=attach.eef_name,
            initial_open_segment_id=initial_open.segment_id,
            close_segment_id=close.segment_id,
            attach_segment_id=attach.segment_id,
            release_open_segment_id=release_open.segment_id,
            detach_segment_id=detach.segment_id,
            success_contract=success_contract,
            grasp_geometry=grasp_geometry["geometry"],
            placement_geometry=placement_geometry["geometry"],
            subject_aabb_center_offset_m=placement_geometry["subject_offset"],
            target_aabb_center_offset_m=placement_geometry["target_offset"],
            thresholds=thresholds,
        )

    def settle_initial(self, segment_id: str, sample: Mapping[str, Any], *, goal_satisfied: bool) -> None:
        """Capture the post-open settled baseline and prove the task was not initially complete."""

        if segment_id != self.initial_open_segment_id or self._baseline is not None:
            return
        self._baseline = self._read_sample(sample)
        self._last = self._baseline
        self._baseline_goal_satisfied = goal_satisfied

    def attach(self, segment_id: str, object_name: str, eef_name: str, sample: Mapping[str, Any]) -> None:
        """Start the physical closed-transport window after aperture/distance verification."""

        if segment_id != self.attach_segment_id or object_name != self.subject or eef_name != self.eef_name:
            raise ValueError("executed attach intent does not match the attested semantic lifecycle")
        if self._baseline is None or self._attach is not None:
            raise ValueError("attach intent occurred without one settled initial baseline")
        self._attach = self._read_sample(sample)
        self._attachment_candidate = self._attach
        if not self._attachment_sample_is_plausible(self._attach):
            self._attach = None
            raise ValueError("attach intent lacks plausible physical aperture/distance evidence")
        self._last = self._attach
        self._attached = True

    def attachment_is_plausible(self, sample: Mapping[str, Any]) -> bool:
        """Return whether a prospective attach has finite, asset-profiled grasp evidence."""

        self._attachment_candidate = self._read_sample(sample)
        return self._attachment_sample_is_plausible(self._attachment_candidate)

    def _attachment_sample_is_plausible(self, sample: _PhysicalSample) -> bool:
        distance = _distance(sample.subject_position, sample.eef_position)
        return (
            distance <= self.thresholds.attachment_distance_m
            and self.thresholds.grasp_aperture_min_m <= sample.finger_aperture <= self.thresholds.grasp_aperture_max_m
        )

    def observe_step(self, sample: Mapping[str, Any]) -> None:
        """Observe one completed simulator step, including closed-transport extrema."""

        current = self._read_sample(sample)
        self._last = current
        if self._baseline is not None:
            self._maximum_destination_drift_m = max(
                self._maximum_destination_drift_m,
                _distance(current.target_aabb_center_position, self._baseline.target_aabb_center_position),
            )
        if not self._attached or self._release_started:
            return
        assert self._attach is not None
        self._closed_samples += 1
        self._maximum_lift_m = max(
            self._maximum_lift_m,
            current.subject_position[2] - self._attach.subject_position[2],
        )
        self._maximum_subject_displacement_m = max(
            self._maximum_subject_displacement_m,
            _distance(current.subject_position, self._attach.subject_position),
        )
        self._maximum_eef_displacement_m = max(
            self._maximum_eef_displacement_m,
            _distance(current.eef_position, self._attach.eef_position),
        )
        attach_translation, attach_rotation = _relative_object_pose(self._attach)
        current_translation, current_rotation = _relative_object_pose(current)
        self._maximum_relative_translation_drift_m = max(
            self._maximum_relative_translation_drift_m,
            _distance(current_translation, attach_translation),
        )
        self._maximum_relative_rotation_drift_rad = max(
            self._maximum_relative_rotation_drift_rad,
            _rotation_distance(current_rotation, attach_rotation),
        )

    def begin_release(self, segment_id: str, sample: Mapping[str, Any]) -> None:
        """Close the transport window before the gripper begins opening."""

        if segment_id != self.release_open_segment_id or not self._attached or self._release_started:
            raise ValueError("release command does not match the attested semantic lifecycle")
        self._release = self._read_sample(sample)
        self._last = self._release
        self._release_started = True

    def release_contact_gate(self, segment_id: str, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Prove semantic placement and closed transport before the gripper may open.

        This gate is evaluated at the attested release-open boundary while the gripper is still
        closed. It intentionally does not apply the final settled speed thresholds: contact with
        the destination can leave the subject moving before release, and final verification owns
        the stable-state decision.
        """

        if segment_id != self.release_open_segment_id or not self._attached or self._release_started:
            raise ValueError("release contact gate does not match the attested semantic lifecycle")
        if self._baseline is None or self._attach is None:
            raise ValueError("release contact gate requires settled baseline and attachment evidence")
        current = self._read_sample(sample)
        delta = tuple(
            current.subject_aabb_center_position[index] - current.target_aabb_center_position[index]
            for index in range(3)
        )
        horizontal_radius = math.hypot(delta[0], delta[1])
        vertical_offset_abs = abs(delta[2])
        current_destination_drift = _distance(
            current.target_aabb_center_position,
            self._baseline.target_aabb_center_position,
        )
        accumulated_destination_drift = max(self._maximum_destination_drift_m, current_destination_drift)
        current_attachment_distance = _distance(current.subject_position, current.eef_position)
        thresholds = self.thresholds
        checks = (
            _maximum_check(
                "prerelease_subject_target_horizontal_radius_m",
                horizontal_radius,
                thresholds.maximum_final_horizontal_radius_m,
            ),
            _maximum_check(
                "prerelease_subject_target_vertical_offset_abs_m",
                vertical_offset_abs,
                thresholds.maximum_final_vertical_offset_m,
            ),
            _maximum_check(
                "prerelease_maximum_destination_drift_m",
                accumulated_destination_drift,
                thresholds.maximum_destination_drift_m,
            ),
            _maximum_check(
                "prerelease_subject_eef_attachment_distance_m",
                current_attachment_distance,
                thresholds.attachment_distance_m,
            ),
            _minimum_check(
                "prerelease_grasp_aperture_min_m",
                current.finger_aperture,
                thresholds.grasp_aperture_min_m,
            ),
            _maximum_check(
                "prerelease_grasp_aperture_max_m",
                current.finger_aperture,
                thresholds.grasp_aperture_max_m,
            ),
            _minimum_check(
                "prerelease_closed_transport_samples", self._closed_samples, thresholds.minimum_closed_samples
            ),
            _minimum_check("prerelease_closed_transport_lift_m", self._maximum_lift_m, thresholds.minimum_lift_m),
            _minimum_check(
                "prerelease_closed_transport_subject_displacement_m",
                self._maximum_subject_displacement_m,
                thresholds.minimum_transport_m,
            ),
            _minimum_check(
                "prerelease_closed_transport_eef_displacement_m",
                self._maximum_eef_displacement_m,
                thresholds.minimum_transport_m,
            ),
            _maximum_check(
                "prerelease_closed_transport_relative_translation_drift_m",
                self._maximum_relative_translation_drift_m,
                thresholds.maximum_relative_translation_drift_m,
            ),
            _maximum_check(
                "prerelease_closed_transport_relative_rotation_drift_rad",
                self._maximum_relative_rotation_drift_rad,
                thresholds.maximum_relative_rotation_drift_rad,
            ),
        )
        return {
            "checks": list(checks),
            "closed_transport": {
                "maximum_eef_displacement_m": self._maximum_eef_displacement_m,
                "maximum_lift_m": self._maximum_lift_m,
                "maximum_relative_rotation_drift_rad": self._maximum_relative_rotation_drift_rad,
                "maximum_relative_translation_drift_m": self._maximum_relative_translation_drift_m,
                "maximum_subject_displacement_m": self._maximum_subject_displacement_m,
                "samples": self._closed_samples,
            },
            "current_physical_state_finite": True,
            "current_prerelease_placement": {
                "grasp_aperture_m": current.finger_aperture,
                "subject_eef_attachment_distance_m": current_attachment_distance,
                "subject_target_horizontal_radius_m": horizontal_radius,
                "subject_target_vertical_offset_abs_m": vertical_offset_abs,
            },
            "maximum_destination_drift_m": accumulated_destination_drift,
            "passed": all(check["passed"] for check in checks),
            "segment_id": segment_id,
        }

    def detach(self, segment_id: str, object_name: str, eef_name: str, sample: Mapping[str, Any]) -> None:
        """Record completion of the logical detach after physical opening settles."""

        if (
            segment_id != self.detach_segment_id
            or object_name != self.subject
            or eef_name != self.eef_name
            or not self._release_started
        ):
            raise ValueError("executed detach intent does not match the attested semantic lifecycle")
        self._last = self._read_sample(sample)
        self._attached = False
        self._detached = True

    def observe_final(self, sample: Mapping[str, Any], *, goal_satisfied: bool) -> None:
        """Record one final-state verification sample."""

        current = self._read_sample(sample)
        self._last = current
        self._final_samples.append((current, goal_satisfied))
        if self._baseline is not None:
            self._maximum_destination_drift_m = max(
                self._maximum_destination_drift_m,
                _distance(current.target_aabb_center_position, self._baseline.target_aabb_center_position),
            )

    def report(self, *, logical_held_object: str | None) -> dict[str, Any]:
        """Return bounded evidence, checks, and the final pass/fail decision."""

        thresholds = self.thresholds
        final = self._final_samples[-1][0] if self._final_samples else self._last
        final_success_streak = sum(1 for _ in takewhile(lambda item: item[1], reversed(self._final_samples)))

        checks: list[dict[str, Any]] = []

        def maximum(name: str, observed: float | int | None, threshold: float | int) -> None:
            passed = observed is not None and observed <= threshold
            checks.append(
                {"name": name, "observed": observed, "operator": "<=", "passed": passed, "threshold": threshold}
            )

        def minimum(name: str, observed: float | int | None, threshold: float | int) -> None:
            passed = observed is not None and observed >= threshold
            checks.append(
                {"name": name, "observed": observed, "operator": ">=", "passed": passed, "threshold": threshold}
            )

        def exact(name: str, observed: Any, expected: Any) -> None:
            checks.append({
                "name": name,
                "observed": observed,
                "operator": "==",
                "passed": observed == expected,
                "threshold": expected,
            })

        attach_distance = None
        attach_aperture = None
        grasp_sample = self._attach if self._attach is not None else self._attachment_candidate
        if grasp_sample is not None:
            attach_distance = _distance(grasp_sample.subject_position, grasp_sample.eef_position)
            attach_aperture = grasp_sample.finger_aperture
        final_linear_speed = None
        final_angular_speed = None
        final_horizontal_radius = None
        final_vertical_offset = None
        final_eef_separation = None
        final_aperture = None
        final_subject_displacement = None
        if final is not None:
            final_linear_speed = max(
                (sample.subject_linear_speed for sample, _ in self._final_samples),
                default=final.subject_linear_speed,
            )
            final_angular_speed = max(
                (sample.subject_angular_speed for sample, _ in self._final_samples),
                default=final.subject_angular_speed,
            )
            delta = tuple(
                final.subject_aabb_center_position[index] - final.target_aabb_center_position[index]
                for index in range(3)
            )
            final_horizontal_radius = math.hypot(delta[0], delta[1])
            final_vertical_offset = delta[2]
            final_eef_separation = _distance(final.subject_position, final.eef_position)
            final_aperture = final.finger_aperture
            if self._baseline is not None:
                final_subject_displacement = _distance(final.subject_position, self._baseline.subject_position)

        exact("initial_goal_false", self._baseline_goal_satisfied, False)
        exact("attach_observed", self._attach is not None, True)
        maximum("attach_distance_m", attach_distance, thresholds.attachment_distance_m)
        minimum("grasp_aperture_min_m", attach_aperture, thresholds.grasp_aperture_min_m)
        maximum("grasp_aperture_max_m", attach_aperture, thresholds.grasp_aperture_max_m)
        minimum("closed_transport_samples", self._closed_samples, thresholds.minimum_closed_samples)
        minimum("closed_transport_lift_m", self._maximum_lift_m, thresholds.minimum_lift_m)
        minimum(
            "closed_transport_subject_displacement_m",
            self._maximum_subject_displacement_m,
            thresholds.minimum_transport_m,
        )
        minimum("closed_transport_eef_displacement_m", self._maximum_eef_displacement_m, thresholds.minimum_transport_m)
        maximum(
            "closed_transport_relative_translation_drift_m",
            self._maximum_relative_translation_drift_m,
            thresholds.maximum_relative_translation_drift_m,
        )
        maximum(
            "closed_transport_relative_rotation_drift_rad",
            self._maximum_relative_rotation_drift_rad,
            thresholds.maximum_relative_rotation_drift_rad,
        )
        exact("release_observed", self._release is not None, True)
        exact("detach_observed", self._detached, True)
        minimum("final_arena_success_streak", final_success_streak, thresholds.minimum_success_streak)
        maximum("final_subject_linear_speed_m_s", final_linear_speed, thresholds.maximum_final_linear_speed_m_s)
        maximum("final_subject_angular_speed_rad_s", final_angular_speed, thresholds.maximum_final_angular_speed_rad_s)
        maximum(
            "final_subject_target_horizontal_radius_m",
            final_horizontal_radius,
            thresholds.maximum_final_horizontal_radius_m,
        )
        maximum(
            "final_subject_target_vertical_offset_abs_m",
            None if final_vertical_offset is None else abs(final_vertical_offset),
            thresholds.maximum_final_vertical_offset_m,
        )
        maximum(
            "maximum_destination_drift_m",
            self._maximum_destination_drift_m,
            thresholds.maximum_destination_drift_m,
        )
        minimum("final_eef_subject_separation_m", final_eef_separation, thresholds.minimum_final_eef_separation_m)
        minimum("final_gripper_aperture_m", final_aperture, thresholds.minimum_final_aperture_m)
        minimum("final_subject_displacement_m", final_subject_displacement, thresholds.minimum_transport_m)
        exact("logical_attachment_empty", logical_held_object, None)

        return {
            "checks": checks,
            "closed_transport": {
                "maximum_eef_displacement_m": self._maximum_eef_displacement_m,
                "maximum_lift_m": self._maximum_lift_m,
                "maximum_relative_rotation_drift_rad": self._maximum_relative_rotation_drift_rad,
                "maximum_relative_translation_drift_m": self._maximum_relative_translation_drift_m,
                "maximum_subject_displacement_m": self._maximum_subject_displacement_m,
                "samples": self._closed_samples,
            },
            "geometry_models": {
                "grasp_geometry": dict(self.grasp_geometry),
                "placement_geometry": dict(self.placement_geometry),
                "position_metric": "rotated_local_aabb_centers_in_env_origin",
                "source": "attested_schedulestream_converted_geometry",
            },
            "arena_success_contract": dict(self.success_contract),
            "goal": {"relation": "on", "subject": self.subject, "target": self.target},
            "maximum_destination_drift_m": self._maximum_destination_drift_m,
            "passed": all(check["passed"] for check in checks),
            "plan_lifecycle": {
                "attach_segment_id": self.attach_segment_id,
                "close_segment_id": self.close_segment_id,
                "detach_segment_id": self.detach_segment_id,
                "initial_open_segment_id": self.initial_open_segment_id,
                "release_open_segment_id": self.release_open_segment_id,
            },
            "profile": self.profile,
            "schema_version": self.schema_version,
            "thresholds": {key: value for key, value in vars(thresholds).items()},
        }

    def _read_sample(self, sample: Mapping[str, Any]) -> _PhysicalSample:
        objects = _mapping(sample.get("objects"), "objects")
        subject = _mapping(objects.get(self.subject), f"objects.{self.subject}")
        target = _mapping(objects.get(self.target), f"objects.{self.target}")
        eef_poses = _mapping(sample.get("eef_poses_env"), "eef_poses_env")
        eef_pose = _matrix4(eef_poses.get(self.eef_name), f"eef_poses_env.{self.eef_name}")
        joint_positions = _mapping(sample.get("joint_positions"), "joint_positions")
        aperture = sum(_finite(joint_positions.get(name), f"joint_positions.{name}") for name in _FINGER_JOINT_NAMES)
        subject_position = _vector3(subject.get("position_env_m"), f"objects.{self.subject}.position_env_m")
        subject_rotation = _quaternion_rotation(
            subject.get("quaternion_xyzw"),
            f"objects.{self.subject}.quaternion_xyzw",
        )
        target_position = _vector3(target.get("position_env_m"), f"objects.{self.target}.position_env_m")
        target_rotation = _quaternion_rotation(
            target.get("quaternion_xyzw"),
            f"objects.{self.target}.quaternion_xyzw",
        )
        return _PhysicalSample(
            subject_position=subject_position,
            subject_aabb_center_position=_apply_local_offset(
                subject_position,
                subject_rotation,
                self.subject_aabb_center_offset_m,
            ),
            subject_rotation=subject_rotation,
            subject_linear_speed=_nonnegative(
                subject.get("linear_speed_m_s"),
                f"objects.{self.subject}.linear_speed_m_s",
            ),
            subject_angular_speed=_nonnegative(
                subject.get("angular_speed_rad_s"),
                f"objects.{self.subject}.angular_speed_rad_s",
            ),
            target_position=target_position,
            target_aabb_center_position=_apply_local_offset(
                target_position,
                target_rotation,
                self.target_aabb_center_offset_m,
            ),
            eef_position=(eef_pose[0][3], eef_pose[1][3], eef_pose[2][3]),
            eef_rotation=tuple(tuple(row[column] for column in range(3)) for row in eef_pose[:3]),
            finger_aperture=aperture,
        )


def _validate_grasp_geometry(value: Any, subject_id: str) -> dict[str, Any]:
    """Validate the finite, name-pinned grasp geometry for the off-center mesh."""

    expected_scalars = {
        "asset_name": _REVIEWED_SUBJECT_ASSET,
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": 4,
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": subject_id,
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "profile": _GRASP_GEOMETRY_PROFILE,
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }
    transform_keys = {
        "converted_object_origin_from_aabb_center",
        "link_from_object_transforms",
        "primitive_link_from_aabb_center_transforms",
    }
    if not isinstance(value, Mapping) or set(value) != set(expected_scalars) | transform_keys:
        raise ValueError("ScheduleStream plan has no exact grasp geometry")
    if any(value.get(key) != expected for key, expected in expected_scalars.items()):
        raise ValueError("ScheduleStream grasp geometry is incompatible")
    object_from_aabb = _matrix4(
        value.get("converted_object_origin_from_aabb_center"),
        "grasp_geometry.converted_object_origin_from_aabb_center",
    )
    primitive_values = value.get("primitive_link_from_aabb_center_transforms")
    link_from_object_values = value.get("link_from_object_transforms")
    if not isinstance(primitive_values, list) or not isinstance(link_from_object_values, list):
        raise ValueError("ScheduleStream grasp transforms must be materialized lists")
    if len(primitive_values) != 4 or len(link_from_object_values) != 4:
        raise ValueError("ScheduleStream grasp geometry must contain exactly four transforms")
    primitive = tuple(
        _matrix4(item, f"grasp_geometry.primitive[{index}]") for index, item in enumerate(primitive_values)
    )
    link_from_object = tuple(
        _matrix4(item, f"grasp_geometry.link_from_object[{index}]")
        for index, item in enumerate(link_from_object_values)
    )
    if len(set(primitive)) != 4:
        raise ValueError("ScheduleStream grasp primitives must be unique")
    aabb_from_object = _rigid_inverse4(object_from_aabb)
    for index, (primitive_pose, link_from_object_pose) in enumerate(zip(primitive, link_from_object)):
        expected = _matrix_multiply4(primitive_pose, aabb_from_object)
        if any(
            not math.isclose(expected[row][column], link_from_object_pose[row][column], abs_tol=1e-7)
            for row in range(4)
            for column in range(4)
        ):
            raise ValueError(f"ScheduleStream grasp transform {index} violates its composition formula")
    return {"geometry": dict(value), "object_from_aabb": object_from_aabb}


def _validate_placement_geometry(value: Any, subject_id: str, target_id: str) -> dict[str, Any]:
    """Validate the exact name-pinned v1 placement geometry before execution."""

    if not isinstance(value, Mapping):
        raise ValueError("ScheduleStream plan has no destination placement geometry")
    if (
        value.get("attested") is not True
        or value.get("schema_version") != 1
        or value.get("profile") != _DESTINATION_PLACEMENT_PROFILE
        or value.get("relation") != "on"
        or value.get("general_inside_semantics") is not False
        or value.get("frame_convention") != "parent_from_child_homogeneous_4x4"
        or value.get("placement_model") != "destination_local_aabb_top_plane_shifted_downward"
    ):
        raise ValueError("ScheduleStream destination placement geometry is incompatible")
    surface = value.get("surface_config")
    if not isinstance(surface, Mapping):
        raise ValueError("ScheduleStream destination placement has no SurfaceConfig geometry")
    xy_extend = _finite(surface.get("xy_extend_m"), "destination_placement.surface_config.xy_extend_m")
    z_offset = _finite(surface.get("z_offset_m"), "destination_placement.surface_config.z_offset_m")
    sampled_extent = _vector3(
        value.get("sampled_surface_extent_m"),
        "destination_placement.sampled_surface_extent_m",
    )
    if any(abs(component) > 1e-9 for component in sampled_extent):
        raise ValueError("ScheduleStream destination placement must sample the exact bowl AABB center")
    predicted_offset = _vector3(
        value.get("predicted_aabb_center_offset_m"),
        "destination_placement.predicted_aabb_center_offset_m",
    )
    vertical_corridor = _nonnegative(
        value.get("vertical_evidence_corridor_m"),
        "destination_placement.vertical_evidence_corridor_m",
    )
    if math.hypot(predicted_offset[0], predicted_offset[1]) > 1e-9 or abs(predicted_offset[2]) > vertical_corridor:
        raise ValueError("ScheduleStream destination placement prediction exceeds its evidence corridor")

    offsets = {}
    object_from_aabb = {}
    dimensions_by_role = {}
    expected_records = {
        "subject": (subject_id, _REVIEWED_SUBJECT_ASSET, ((0.05, 0.065),) * 3),
        "destination": (
            target_id,
            _REVIEWED_DESTINATION_ASSET,
            ((0.14, 0.17), (0.14, 0.17), (0.04, 0.07)),
        ),
    }
    for role, (object_id, asset_name, dimension_bounds) in expected_records.items():
        geometry = value.get(role)
        if not isinstance(geometry, Mapping):
            raise ValueError(f"ScheduleStream destination placement has no {role} geometry")
        if (
            geometry.get("object_id") != object_id
            or geometry.get("asset_name") != asset_name
            or geometry.get("aabb_kind") != "converted_mesh_local_axis_aligned_bounding_box"
        ):
            raise ValueError(f"ScheduleStream destination placement {role} identity is incompatible")
        dimensions = _vector3(
            geometry.get("aabb_dimensions_m"),
            f"destination_placement.{role}.aabb_dimensions_m",
        )
        if any(not lower <= observed <= upper for observed, (lower, upper) in zip(dimensions, dimension_bounds)):
            raise ValueError(f"ScheduleStream destination placement {role} dimensions are outside the reviewed profile")
        offset = _vector3(
            geometry.get("aabb_center_in_isaac_rigid_root_m"),
            f"destination_placement.{role}.aabb_center_in_isaac_rigid_root_m",
        )
        transform = _matrix4(
            geometry.get("isaac_rigid_root_from_aabb_center"),
            f"destination_placement.{role}.isaac_rigid_root_from_aabb_center",
        )
        if any(not math.isclose(transform[index][3], offset[index], abs_tol=1e-8) for index in range(3)):
            raise ValueError(f"ScheduleStream destination placement {role} center transform is inconsistent")
        offsets[role] = offset
        object_from_aabb[role] = _matrix4(
            geometry.get("converted_object_origin_from_aabb_center"),
            f"destination_placement.{role}.converted_object_origin_from_aabb_center",
        )
        dimensions_by_role[role] = dimensions

    subject_dimensions = dimensions_by_role["subject"]
    destination_dimensions = dimensions_by_role["destination"]
    expected_xy_extend = -max(destination_dimensions[0], destination_dimensions[1])
    expected_z_offset = 0.03 - 0.5 * (destination_dimensions[2] + subject_dimensions[2])
    if not math.isclose(xy_extend, expected_xy_extend, abs_tol=1e-8) or not math.isclose(
        z_offset,
        expected_z_offset,
        abs_tol=1e-8,
    ):
        raise ValueError("ScheduleStream destination placement SurfaceConfig is not derived from attested geometry")
    expected_surface_extent = (
        max(0.0, destination_dimensions[0] + xy_extend),
        max(0.0, destination_dimensions[1] + xy_extend),
        0.0,
    )
    if any(
        not math.isclose(observed, expected, abs_tol=1e-8)
        for observed, expected in zip(sampled_extent, expected_surface_extent)
    ):
        raise ValueError("ScheduleStream destination placement sampled extent is inconsistent with SurfaceConfig")
    if any(
        not math.isclose(observed, expected, abs_tol=1e-8) for observed, expected in zip(predicted_offset, (0, 0, 0.03))
    ):
        raise ValueError("ScheduleStream destination placement prediction is not the reviewed center offset")
    return {
        "geometry": dict(value),
        "subject_object_from_aabb": object_from_aabb["subject"],
        "subject_offset": offsets["subject"],
        "target_offset": offsets["destination"],
    }


def _apply_local_offset(
    position: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], ...],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        position[row] + sum(rotation[row][column] * offset[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _maximum_check(name: str, observed: float | int, threshold: float | int) -> dict[str, Any]:
    """Return one bounded maximum check for the task-success report."""

    return {
        "name": name,
        "observed": observed,
        "operator": "<=",
        "passed": observed <= threshold,
        "threshold": threshold,
    }


def _minimum_check(name: str, observed: float | int, threshold: float | int) -> dict[str, Any]:
    """Return one bounded minimum check for the task-success report."""

    return {
        "name": name,
        "observed": observed,
        "operator": ">=",
        "passed": observed >= threshold,
        "threshold": threshold,
    }


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"physical evidence {field_name} must be a mapping")
    return value


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"physical evidence {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"physical evidence {field_name} must be finite")
    return result


def _nonnegative(value: Any, field_name: str) -> float:
    result = _finite(value, field_name)
    if result < 0:
        raise ValueError(f"physical evidence {field_name} must be non-negative")
    return result


def _vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"physical evidence {field_name} must be a 3-vector")
    result = tuple(_finite(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _matrix4(value: Any, field_name: str) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"physical evidence {field_name} must be a 4x4 matrix")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError(f"physical evidence {field_name} must be a 4x4 matrix")
        rows.append(tuple(_finite(item, f"{field_name}[{row_index}][]") for item in row))
    if any(abs(rows[3][index] - expected) > 1e-5 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"physical evidence {field_name} must be a homogeneous transform")
    rotation = tuple(tuple(rows[row][column] for column in range(3)) for row in range(3))
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[index][left] * rotation[index][right] for index in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1e-4:
                raise ValueError(f"physical evidence {field_name} rotation must be orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError(f"physical evidence {field_name} rotation determinant must be +1")
    return tuple(rows)  # type: ignore[return-value]


def _rigid_inverse4(
    value: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    rotation_transpose = tuple(tuple(value[column][row] for column in range(3)) for row in range(3))
    translation = tuple(value[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_transpose[row][column] * translation[column] for column in range(3)) for row in range(3)
    )
    return tuple(
        tuple(rotation_transpose[row][column] for column in range(3)) + (inverse_translation[row],) for row in range(3)
    ) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply4(
    left: tuple[tuple[float, float, float, float], ...],
    right: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _quaternion_rotation(value: Any, field_name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"physical evidence {field_name} must be a quaternion")
    x, y, z, w = (_finite(item, f"{field_name}[]") for item in value)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError(f"physical evidence {field_name} must have nonzero norm")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _relative_object_pose(
    sample: _PhysicalSample,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    eef_transpose = tuple(zip(*sample.eef_rotation))
    delta = tuple(sample.subject_position[index] - sample.eef_position[index] for index in range(3))
    translation = tuple(sum(row[index] * delta[index] for index in range(3)) for row in eef_transpose)
    rotation = tuple(
        tuple(
            sum(eef_transpose[row][index] * sample.subject_rotation[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    return translation, rotation


def _rotation_distance(
    first: tuple[tuple[float, float, float], ...],
    second: tuple[tuple[float, float, float], ...],
) -> float:
    trace = sum(first[index][axis] * second[index][axis] for index in range(3) for axis in range(3))
    cosine = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    return math.acos(cosine)
