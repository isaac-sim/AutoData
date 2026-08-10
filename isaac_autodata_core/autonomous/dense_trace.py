# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lower aligned backend samples into the typed task-motion plan IR."""

from __future__ import annotations

from dataclasses import dataclass

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    DetachIntentSegment,
    GripperCommandMode,
    GripperCommandSegment,
    Matrix4,
    TaskMotionPlan,
    TaskMotionSegment,
    make_stable_id,
    matrix4,
)
from isaac_autodata_interfaces.tasks.task_goal import GoalPredicate


@dataclass(frozen=True)
class DenseAttachmentEvent:
    """Symbolic attachment transition aligned to one dense sample index."""

    sample_index: int
    operation: str
    object_name: str
    verifier: str = "contact_and_relative_motion_v1"

    def __post_init__(self) -> None:
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if self.operation not in ("attach", "detach"):
            raise ValueError("attachment operation must be 'attach' or 'detach'")
        if not self.object_name:
            raise ValueError("attachment object_name must not be empty")
        if not self.verifier:
            raise ValueError("attachment verifier must not be empty")


@dataclass(frozen=True)
class DensePlanTrace:
    """Backend-neutral aligned poses, joint seeds, gripper values, and symbolic events."""

    eef_name: str
    frame: str
    poses: tuple[Matrix4, ...]
    gripper_values: tuple[float, ...]
    step_dt_s: float
    gripper_settle_steps: int = 1
    joint_names: tuple[str, ...] = ()
    joint_positions: tuple[tuple[float, ...], ...] = ()
    attachment_events: tuple[DenseAttachmentEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.eef_name or not self.frame:
            raise ValueError("eef_name and frame must not be empty")
        if not self.poses:
            raise ValueError("dense plan trace must contain at least one pose")
        object.__setattr__(
            self,
            "poses",
            tuple(matrix4(pose, f"poses[{index}]") for index, pose in enumerate(self.poses)),
        )
        if len(self.gripper_values) != len(self.poses):
            raise ValueError("gripper_values must align one-to-one with poses")
        values = tuple(float(value) for value in self.gripper_values)
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError("gripper_values must be finite")
        object.__setattr__(self, "gripper_values", values)
        if self.step_dt_s <= 0:
            raise ValueError("step_dt_s must be positive")
        if (
            isinstance(self.gripper_settle_steps, bool)
            or not isinstance(self.gripper_settle_steps, int)
            or not 1 <= self.gripper_settle_steps <= 10_000
        ):
            raise ValueError("gripper_settle_steps must be an integer in [1, 10000]")
        if self.joint_positions:
            if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
                raise ValueError("joint_names must be non-empty and unique when joint positions are present")
            if len(self.joint_positions) != len(self.poses):
                raise ValueError("joint_positions must align one-to-one with poses")
            if any(len(position) != len(self.joint_names) for position in self.joint_positions):
                raise ValueError("joint position widths must match joint_names")
        elif self.joint_names:
            raise ValueError("joint_names must be empty when joint_positions are absent")
        if any(event.sample_index >= len(self.poses) for event in self.attachment_events):
            raise ValueError("attachment event sample index is outside the dense trace")
        if any(event.sample_index == 0 for event in self.attachment_events):
            raise ValueError("attachment events require a preceding collision-checked dense sample")
        event_keys = [(event.sample_index, event.operation, event.object_name) for event in self.attachment_events]
        if len(event_keys) != len(set(event_keys)):
            raise ValueError("dense trace attachment events must be unique")
        split_indices = {
            index
            for index in range(1, len(self.gripper_values))
            if abs(self.gripper_values[index] - self.gripper_values[index - 1]) > 1e-6
        }
        split_indices.update(event.sample_index for event in self.attachment_events if event.sample_index > 0)
        for index in sorted(split_indices):
            if self.poses[index] != self.poses[index - 1]:
                raise ValueError(
                    f"dense trace split at sample {index} must duplicate the preceding pose before interaction"
                )
            if self.joint_positions and self.joint_positions[index] != self.joint_positions[index - 1]:
                raise ValueError(
                    f"dense trace split at sample {index} must duplicate the preceding joint seed before interaction"
                )


def task_motion_plan_from_dense_trace(
    trace: DensePlanTrace,
    *,
    request_digest: str,
    snapshot_digest: str,
    backend: str,
    backend_version: str,
    seed: int,
    goal: tuple[GoalPredicate, ...],
    metadata: dict | None = None,
) -> TaskMotionPlan:
    """Split a dense trace at gripper/attachment changes and build a sequential typed plan."""

    split_indices = {0, len(trace.poses)}
    for index in range(1, len(trace.gripper_values)):
        if abs(trace.gripper_values[index] - trace.gripper_values[index - 1]) > 1e-6:
            split_indices.add(index)
    for event in trace.attachment_events:
        split_indices.add(event.sample_index)
    boundaries = sorted(split_indices)
    events_by_index: dict[int, list[DenseAttachmentEvent]] = {}
    for event in trace.attachment_events:
        events_by_index.setdefault(event.sample_index, []).append(event)

    segments: list[TaskMotionSegment] = []
    previous_id: str | None = None

    def append(segment: TaskMotionSegment) -> None:
        nonlocal previous_id
        segments.append(segment)
        previous_id = segment.segment_id

    previous_gripper: float | None = None
    for start, end in zip(boundaries, boundaries[1:]):
        gripper = trace.gripper_values[start]
        if previous_gripper is None or abs(gripper - previous_gripper) > 1e-6:
            command, value = _gripper_command(gripper)
            segment_id = make_stable_id("gripper", request_digest, start, command.value, value)
            append(
                GripperCommandSegment(
                    segment_id=segment_id,
                    depends_on=(() if previous_id is None else (previous_id,)),
                    eef_name=trace.eef_name,
                    command=command,
                    value=value,
                    settle_steps=trace.gripper_settle_steps,
                )
            )
            previous_gripper = gripper

        for event in sorted(events_by_index.get(start, ()), key=lambda item: (item.operation, item.object_name)):
            segment_id = make_stable_id("attachment", request_digest, start, event.operation, event.object_name)
            event_kwargs = {
                "segment_id": segment_id,
                "depends_on": () if previous_id is None else (previous_id,),
                "eef_name": trace.eef_name,
                "object_name": event.object_name,
                "verifier": event.verifier,
            }
            if event.operation == "attach":
                append(AttachIntentSegment(**event_kwargs))
            else:
                append(DetachIntentSegment(**event_kwargs))

        if end <= start:
            continue
        segment_id = make_stable_id("cartesian", request_digest, start, end)
        joint_seeds = trace.joint_positions[start:end] if trace.joint_positions else ()
        append(
            CartesianTrajectorySegment(
                segment_id=segment_id,
                depends_on=(() if previous_id is None else (previous_id,)),
                duration_s=(end - start) * trace.step_dt_s,
                eef_name=trace.eef_name,
                frame=trace.frame,
                poses=trace.poses[start:end],
                joint_seed_names=trace.joint_names,
                joint_seeds=joint_seeds,
                metadata={"sample_end_exclusive": end, "sample_start": start, "step_dt_s": trace.step_dt_s},
            )
        )

    plan_id = make_stable_id("plan", request_digest, snapshot_digest, backend, seed)
    return TaskMotionPlan(
        plan_id=plan_id,
        request_digest=request_digest,
        snapshot_digest=snapshot_digest,
        backend=backend,
        backend_version=backend_version,
        seed=seed,
        segments=tuple(segments),
        goal=goal,
        metadata={} if metadata is None else metadata,
    )


def _gripper_command(value: float) -> tuple[GripperCommandMode, float | None]:
    if value >= 1.0 - 1e-6:
        return GripperCommandMode.OPEN, None
    if value <= -1.0 + 1e-6:
        return GripperCommandMode.CLOSE, None
    return GripperCommandMode.POSITION, value
