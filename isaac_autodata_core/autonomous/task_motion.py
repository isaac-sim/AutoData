# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned, planner-neutral contracts for autonomous generation.

The types in this module deliberately depend only on the Python standard library. Planner and
simulator adapters convert their native tensors and commands at this boundary; native cuRobo,
ScheduleStream, Isaac Lab, or Pydantic objects must not leak into a serialized plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Matrix4: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

TASK_MOTION_PLAN_SCHEMA_VERSION = 1
SCENE_SNAPSHOT_SCHEMA_VERSION = 1
EXECUTION_EVENT_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def make_stable_id(namespace: str, *parts: Any) -> str:
    """Return a compact deterministic identifier for canonical JSON-compatible inputs.

    Args:
        namespace: Human-readable ID prefix.
        parts: Values contributing to the content hash.
    """

    assert namespace and namespace.strip(), "namespace must not be empty"
    digest = hashlib.sha256(_canonical_json(list(parts)).encode("utf-8")).hexdigest()[:20]
    return f"{namespace}-{digest}"


def _require_text(value: str, field_name: str, *, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")


def _finite_float(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _number_tuple(values: Sequence[int | float], field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a numeric sequence")
    return tuple(_finite_float(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def matrix4(value: Sequence[Sequence[int | float]], field_name: str = "pose") -> Matrix4:
    """Validate and normalize one approximately rigid homogeneous transform.

    Args:
        value: Four rows with four finite numeric values each.
        field_name: Field path used in validation messages.
    """

    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field_name} must have shape [4, 4]")
    rows = tuple(_number_tuple(row, f"{field_name}[{index}]") for index, row in enumerate(value))
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"{field_name} must have shape [4, 4]")
    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(rows[3], expected_last_row)):
        raise ValueError(f"{field_name} must have homogeneous last row [0, 0, 0, 1]")

    rotation = tuple(row[:3] for row in rows[:3])
    for column in range(3):
        norm = sum(rotation[row][column] ** 2 for row in range(3))
        if abs(norm - 1.0) > 2e-3:
            raise ValueError(f"{field_name} rotation columns must have unit norm")
        for other in range(column + 1, 3):
            dot = sum(rotation[row][column] * rotation[row][other] for row in range(3))
            if abs(dot) > 2e-3:
                raise ValueError(f"{field_name} rotation columns must be orthogonal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 2e-3:
        raise ValueError(f"{field_name} rotation determinant must be +1")
    return rows  # type: ignore[return-value]


def _json_value(value: Any, field_name: str = "metadata", depth: int = 0) -> JsonValue:
    if depth > 10:
        raise ValueError(f"{field_name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_float(value, field_name)
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            result[key] = _json_value(value[key], f"{field_name}.{key}", depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, f"{field_name}[{index}]", depth + 1) for index, item in enumerate(value)]
    raise ValueError(f"{field_name} contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True, order=True)
class GoalPredicate:
    """One planner-neutral relational predicate."""

    relation: str
    subject: str
    target: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.relation, "relation", maximum=128)
        _require_text(self.subject, "subject")
        if self.target is not None:
            _require_text(self.target, "target")

    def to_dict(self) -> dict[str, str]:
        result = {"relation": self.relation, "subject": self.subject}
        if self.target is not None:
            result["target"] = self.target
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoalPredicate:
        return cls(relation=value["relation"], subject=value["subject"], target=value.get("target"))


@dataclass(frozen=True)
class SceneObjectSnapshot:
    """Planner-neutral state for one semantic scene object."""

    semantic_id: str
    scene_id: str
    pose: Matrix4
    geometry_ref: str | None = None
    geometry_digest: str | None = None
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.semantic_id, "semantic_id")
        _require_text(self.scene_id, "scene_id")
        object.__setattr__(self, "pose", matrix4(self.pose, f"objects[{self.semantic_id}].pose"))
        if self.geometry_ref is not None:
            _require_text(self.geometry_ref, "geometry_ref", maximum=2048)
        if self.geometry_digest is not None:
            _require_text(self.geometry_digest, "geometry_digest", maximum=256)
        object.__setattr__(self, "roles", tuple(sorted(set(self.roles))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_digest": self.geometry_digest,
            "geometry_ref": self.geometry_ref,
            "pose": [list(row) for row in self.pose],
            "roles": list(self.roles),
            "scene_id": self.scene_id,
            "semantic_id": self.semantic_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneObjectSnapshot:
        return cls(
            semantic_id=value["semantic_id"],
            scene_id=value["scene_id"],
            pose=value["pose"],
            geometry_ref=value.get("geometry_ref"),
            geometry_digest=value.get("geometry_digest"),
            roles=tuple(value.get("roles", ())),
        )


@dataclass(frozen=True)
class RobotStateSnapshot:
    """Planner-neutral embodiment state at one instant."""

    robot_id: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    eef_poses: Mapping[str, Matrix4]
    held_objects: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.robot_id, "robot_id")
        if not self.joint_names or len(self.joint_names) != len(self.joint_positions):
            raise ValueError("joint_names and joint_positions must be non-empty and have equal length")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        for name in self.joint_names:
            _require_text(name, "joint_name")
        object.__setattr__(self, "joint_positions", _number_tuple(self.joint_positions, "joint_positions"))
        if not self.eef_poses:
            raise ValueError("eef_poses must not be empty")
        normalized_poses: dict[str, Matrix4] = {}
        for name, pose in sorted(self.eef_poses.items()):
            _require_text(name, "eef_name")
            normalized_poses[name] = matrix4(pose, f"eef_poses[{name}]")
        object.__setattr__(self, "eef_poses", normalized_poses)
        normalized_holds: dict[str, str | None] = {}
        for name, held_object in sorted(self.held_objects.items()):
            if name not in normalized_poses:
                raise ValueError(f"held_objects references unknown EEF {name!r}")
            if held_object is not None:
                _require_text(held_object, f"held_objects[{name}]")
            normalized_holds[name] = held_object
        object.__setattr__(self, "held_objects", normalized_holds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eef_poses": {name: [list(row) for row in pose] for name, pose in self.eef_poses.items()},
            "held_objects": dict(self.held_objects),
            "joint_names": list(self.joint_names),
            "joint_positions": list(self.joint_positions),
            "robot_id": self.robot_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotStateSnapshot:
        return cls(
            robot_id=value["robot_id"],
            joint_names=tuple(value["joint_names"]),
            joint_positions=tuple(value["joint_positions"]),
            eef_poses=value["eef_poses"],
            held_objects=value.get("held_objects", {}),
        )


@dataclass(frozen=True)
class SceneSnapshot:
    """Complete planner-neutral state used for one planning attempt."""

    snapshot_id: str
    captured_at_s: float
    env_id: int
    robot: RobotStateSnapshot
    objects: tuple[SceneObjectSnapshot, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = SCENE_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.snapshot_id, "snapshot_id")
        object.__setattr__(self, "captured_at_s", _finite_float(self.captured_at_s, "captured_at_s"))
        if self.captured_at_s < 0:
            raise ValueError("captured_at_s must be non-negative")
        if isinstance(self.env_id, bool) or not isinstance(self.env_id, int) or self.env_id < 0:
            raise ValueError("env_id must be a non-negative integer")
        if self.schema_version != SCENE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported scene snapshot schema version {self.schema_version}")
        ids = [item.semantic_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("scene object semantic IDs must be unique")
        object.__setattr__(self, "objects", tuple(sorted(self.objects, key=lambda item: item.semantic_id)))
        metadata = _json_value(self.metadata)
        assert isinstance(metadata, dict)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at_s": self.captured_at_s,
            "env_id": self.env_id,
            "metadata": dict(self.metadata),
            "objects": [item.to_dict() for item in self.objects],
            "robot": self.robot.to_dict(),
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneSnapshot:
        return cls(
            snapshot_id=value["snapshot_id"],
            captured_at_s=value["captured_at_s"],
            env_id=value["env_id"],
            robot=RobotStateSnapshot.from_dict(value["robot"]),
            objects=tuple(SceneObjectSnapshot.from_dict(item) for item in value["objects"]),
            metadata=value.get("metadata", {}),
            schema_version=value.get("schema_version", SCENE_SNAPSHOT_SCHEMA_VERSION),
        )


class SegmentKind(StrEnum):
    CARTESIAN_TRAJECTORY = "cartesian_trajectory"
    JOINT_TRAJECTORY = "joint_trajectory"
    GRIPPER_COMMAND = "gripper_command"
    ATTACH_INTENT = "attach_intent"
    DETACH_INTENT = "detach_intent"
    WAIT = "wait"
    BARRIER = "barrier"
    CONCURRENT_GROUP = "concurrent_group"


class GripperCommandMode(StrEnum):
    OPEN = "open"
    CLOSE = "close"
    POSITION = "position"
    EFFORT = "effort"


@dataclass(frozen=True, kw_only=True)
class PlanSegment:
    """Fields shared by every task-motion segment."""

    kind: ClassVar[SegmentKind]
    segment_id: str
    depends_on: tuple[str, ...] = ()
    start_time_s: float | None = None
    duration_s: float | None = None
    assumptions: tuple[GoalPredicate, ...] = ()
    expected_postconditions: tuple[GoalPredicate, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "segment_id")
        for dependency in self.depends_on:
            _require_text(dependency, "depends_on")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"segment {self.segment_id!r} has duplicate dependencies")
        if self.start_time_s is not None:
            value = _finite_float(self.start_time_s, "start_time_s")
            if value < 0:
                raise ValueError("start_time_s must be non-negative")
            object.__setattr__(self, "start_time_s", value)
        if self.duration_s is not None:
            value = _finite_float(self.duration_s, "duration_s")
            if value < 0:
                raise ValueError("duration_s must be non-negative")
            object.__setattr__(self, "duration_s", value)
        metadata = _json_value(self.metadata)
        assert isinstance(metadata, dict)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, kw_only=True)
class CartesianTrajectorySegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.CARTESIAN_TRAJECTORY
    eef_name: str
    frame: str
    poses: tuple[Matrix4, ...]
    joint_seed_names: tuple[str, ...] = ()
    joint_seeds: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(self.eef_name, "eef_name")
        _require_text(self.frame, "frame")
        if not self.poses:
            raise ValueError("Cartesian trajectory must contain at least one pose")
        object.__setattr__(
            self,
            "poses",
            tuple(
                matrix4(pose, f"segments[{self.segment_id}].poses[{index}]") for index, pose in enumerate(self.poses)
            ),
        )
        normalized_seeds = tuple(
            _number_tuple(seed, f"segments[{self.segment_id}].joint_seeds[{index}]")
            for index, seed in enumerate(self.joint_seeds)
        )
        if normalized_seeds and len(normalized_seeds) != len(self.poses):
            raise ValueError("joint_seeds must be empty or align one-to-one with Cartesian poses")
        if normalized_seeds:
            if not self.joint_seed_names or len(set(self.joint_seed_names)) != len(self.joint_seed_names):
                raise ValueError("joint_seed_names must be non-empty and unique when joint_seeds are present")
            for name in self.joint_seed_names:
                _require_text(name, "joint_seed_name")
            if any(len(seed) != len(self.joint_seed_names) for seed in normalized_seeds):
                raise ValueError("every joint seed must match joint_seed_names length")
        elif self.joint_seed_names:
            raise ValueError("joint_seed_names must be empty when joint_seeds are absent")
        object.__setattr__(self, "joint_seeds", normalized_seeds)


@dataclass(frozen=True, kw_only=True)
class JointTrajectorySegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.JOINT_TRAJECTORY
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    timestamps_s: tuple[float, ...] = ()
    attached_object: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        for name in self.joint_names:
            _require_text(name, "joint_name")
        if not self.positions:
            raise ValueError("joint trajectory must contain at least one position")
        normalized_positions = tuple(
            _number_tuple(position, f"segments[{self.segment_id}].positions[{index}]")
            for index, position in enumerate(self.positions)
        )
        if any(len(position) != len(self.joint_names) for position in normalized_positions):
            raise ValueError("every joint position must match joint_names length")
        object.__setattr__(self, "positions", normalized_positions)
        timestamps = _number_tuple(self.timestamps_s, "timestamps_s")
        if timestamps and len(timestamps) != len(normalized_positions):
            raise ValueError("timestamps_s must be empty or align with joint positions")
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps_s must be monotonic")
        if timestamps and timestamps[0] < 0:
            raise ValueError("timestamps_s must be non-negative")
        object.__setattr__(self, "timestamps_s", timestamps)
        if self.attached_object is not None:
            _require_text(self.attached_object, "attached_object")


@dataclass(frozen=True, kw_only=True)
class GripperCommandSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.GRIPPER_COMMAND
    eef_name: str
    command: GripperCommandMode
    value: float | None = None
    settle_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(self.eef_name, "eef_name")
        if not isinstance(self.command, GripperCommandMode):
            object.__setattr__(self, "command", GripperCommandMode(self.command))
        if self.command in (GripperCommandMode.POSITION, GripperCommandMode.EFFORT) and self.value is None:
            raise ValueError(f"gripper command {self.command.value!r} requires value")
        if self.value is not None:
            object.__setattr__(self, "value", _finite_float(self.value, "gripper value"))
        if isinstance(self.settle_steps, bool) or not isinstance(self.settle_steps, int) or self.settle_steps < 1:
            raise ValueError("settle_steps must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class AttachIntentSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.ATTACH_INTENT
    eef_name: str
    object_name: str
    verifier: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(self.eef_name, "eef_name")
        _require_text(self.object_name, "object_name")
        _require_text(self.verifier, "verifier")


@dataclass(frozen=True, kw_only=True)
class DetachIntentSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.DETACH_INTENT
    eef_name: str
    object_name: str
    verifier: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(self.eef_name, "eef_name")
        _require_text(self.object_name, "object_name")
        _require_text(self.verifier, "verifier")


@dataclass(frozen=True, kw_only=True)
class WaitSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.WAIT
    steps: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.steps is None and self.duration_s is None:
            raise ValueError("wait segment requires steps or duration_s")
        if self.steps is not None and (
            isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1
        ):
            raise ValueError("wait steps must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class BarrierSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.BARRIER
    participants: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.participants or len(set(self.participants)) != len(self.participants):
            raise ValueError("barrier participants must be non-empty and unique")
        for participant in self.participants:
            _require_text(participant, "barrier participant")


@dataclass(frozen=True, kw_only=True)
class ConcurrentGroupSegment(PlanSegment):
    kind: ClassVar[SegmentKind] = SegmentKind.CONCURRENT_GROUP
    member_segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.member_segment_ids or len(set(self.member_segment_ids)) != len(self.member_segment_ids):
            raise ValueError("concurrent group members must be non-empty and unique")
        for member in self.member_segment_ids:
            _require_text(member, "concurrent group member")


TaskMotionSegment: TypeAlias = (
    CartesianTrajectorySegment
    | JointTrajectorySegment
    | GripperCommandSegment
    | AttachIntentSegment
    | DetachIntentSegment
    | WaitSegment
    | BarrierSegment
    | ConcurrentGroupSegment
)


def _segment_base_dict(segment: PlanSegment) -> dict[str, Any]:
    return {
        "assumptions": [predicate.to_dict() for predicate in segment.assumptions],
        "depends_on": list(segment.depends_on),
        "duration_s": segment.duration_s,
        "expected_postconditions": [predicate.to_dict() for predicate in segment.expected_postconditions],
        "kind": segment.kind.value,
        "metadata": dict(segment.metadata),
        "segment_id": segment.segment_id,
        "start_time_s": segment.start_time_s,
    }


def segment_to_dict(segment: TaskMotionSegment) -> dict[str, Any]:
    result = _segment_base_dict(segment)
    if isinstance(segment, CartesianTrajectorySegment):
        result.update({
            "eef_name": segment.eef_name,
            "frame": segment.frame,
            "joint_seed_names": list(segment.joint_seed_names),
            "joint_seeds": [list(seed) for seed in segment.joint_seeds],
            "poses": [[list(row) for row in pose] for pose in segment.poses],
        })
    elif isinstance(segment, JointTrajectorySegment):
        result.update({
            "attached_object": segment.attached_object,
            "joint_names": list(segment.joint_names),
            "positions": [list(position) for position in segment.positions],
            "timestamps_s": list(segment.timestamps_s),
        })
    elif isinstance(segment, GripperCommandSegment):
        result.update({
            "command": segment.command.value,
            "eef_name": segment.eef_name,
            "settle_steps": segment.settle_steps,
            "value": segment.value,
        })
    elif isinstance(segment, (AttachIntentSegment, DetachIntentSegment)):
        result.update({
            "eef_name": segment.eef_name,
            "object_name": segment.object_name,
            "verifier": segment.verifier,
        })
    elif isinstance(segment, WaitSegment):
        result["steps"] = segment.steps
    elif isinstance(segment, BarrierSegment):
        result["participants"] = list(segment.participants)
    elif isinstance(segment, ConcurrentGroupSegment):
        result["member_segment_ids"] = list(segment.member_segment_ids)
    else:
        raise TypeError(f"unsupported plan segment {type(segment).__name__}")
    return result


def segment_from_dict(value: Mapping[str, Any]) -> TaskMotionSegment:
    kind = SegmentKind(value["kind"])
    common = {
        "segment_id": value["segment_id"],
        "depends_on": tuple(value.get("depends_on", ())),
        "start_time_s": value.get("start_time_s"),
        "duration_s": value.get("duration_s"),
        "assumptions": tuple(GoalPredicate.from_dict(item) for item in value.get("assumptions", ())),
        "expected_postconditions": tuple(
            GoalPredicate.from_dict(item) for item in value.get("expected_postconditions", ())
        ),
        "metadata": value.get("metadata", {}),
    }
    if kind is SegmentKind.CARTESIAN_TRAJECTORY:
        return CartesianTrajectorySegment(
            **common,
            eef_name=value["eef_name"],
            frame=value["frame"],
            poses=tuple(value["poses"]),
            joint_seed_names=tuple(value.get("joint_seed_names", ())),
            joint_seeds=tuple(tuple(seed) for seed in value.get("joint_seeds", ())),
        )
    if kind is SegmentKind.JOINT_TRAJECTORY:
        return JointTrajectorySegment(
            **common,
            joint_names=tuple(value["joint_names"]),
            positions=tuple(tuple(position) for position in value["positions"]),
            timestamps_s=tuple(value.get("timestamps_s", ())),
            attached_object=value.get("attached_object"),
        )
    if kind is SegmentKind.GRIPPER_COMMAND:
        return GripperCommandSegment(
            **common,
            eef_name=value["eef_name"],
            command=GripperCommandMode(value["command"]),
            value=value.get("value"),
            settle_steps=value.get("settle_steps", 1),
        )
    if kind is SegmentKind.ATTACH_INTENT:
        return AttachIntentSegment(
            **common,
            eef_name=value["eef_name"],
            object_name=value["object_name"],
            verifier=value["verifier"],
        )
    if kind is SegmentKind.DETACH_INTENT:
        return DetachIntentSegment(
            **common,
            eef_name=value["eef_name"],
            object_name=value["object_name"],
            verifier=value["verifier"],
        )
    if kind is SegmentKind.WAIT:
        return WaitSegment(**common, steps=value.get("steps"))
    if kind is SegmentKind.BARRIER:
        return BarrierSegment(**common, participants=tuple(value["participants"]))
    if kind is SegmentKind.CONCURRENT_GROUP:
        return ConcurrentGroupSegment(**common, member_segment_ids=tuple(value["member_segment_ids"]))
    raise AssertionError(f"unhandled segment kind {kind}")


@dataclass(frozen=True)
class TaskMotionPlan:
    """Executable task-and-motion plan with preserved symbolic and temporal structure."""

    plan_id: str
    request_digest: str
    snapshot_digest: str
    backend: str
    backend_version: str
    seed: int
    segments: tuple[TaskMotionSegment, ...]
    goal: tuple[GoalPredicate, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = TASK_MOTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("plan_id", "request_digest", "snapshot_digest", "backend", "backend_version"):
            _require_text(getattr(self, field_name), field_name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.schema_version != TASK_MOTION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported task-motion plan schema version {self.schema_version}")
        if not self.segments:
            raise ValueError("task-motion plan must contain at least one segment")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("task-motion plan segment IDs must be unique")
        known = set(segment_ids)
        dependencies: dict[str, set[str]] = {}
        for segment in self.segments:
            missing = set(segment.depends_on) - known
            if missing:
                raise ValueError(f"segment {segment.segment_id!r} has unknown dependencies {sorted(missing)}")
            if segment.segment_id in segment.depends_on:
                raise ValueError(f"segment {segment.segment_id!r} cannot depend on itself")
            dependencies[segment.segment_id] = set(segment.depends_on)
            if isinstance(segment, ConcurrentGroupSegment):
                missing_members = set(segment.member_segment_ids) - known
                if missing_members:
                    raise ValueError(
                        f"concurrent group {segment.segment_id!r} has unknown members {sorted(missing_members)}"
                    )
                if segment.segment_id in segment.member_segment_ids:
                    raise ValueError(f"concurrent group {segment.segment_id!r} cannot contain itself")
        _assert_acyclic(dependencies)
        metadata = _json_value(self.metadata)
        assert isinstance(metadata, dict)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": {"name": self.backend, "version": self.backend_version},
            "goal": [predicate.to_dict() for predicate in self.goal],
            "metadata": dict(self.metadata),
            "plan_id": self.plan_id,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "segments": [segment_to_dict(segment) for segment in self.segments],
            "snapshot_digest": self.snapshot_digest,
        }

    def canonical_json(self) -> str:
        """Return deterministic, finite JSON suitable for hashing and run records."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskMotionPlan:
        backend = value["backend"]
        return cls(
            plan_id=value["plan_id"],
            request_digest=value["request_digest"],
            snapshot_digest=value["snapshot_digest"],
            backend=backend["name"],
            backend_version=backend["version"],
            seed=value["seed"],
            segments=tuple(segment_from_dict(segment) for segment in value["segments"]),
            goal=tuple(GoalPredicate.from_dict(predicate) for predicate in value.get("goal", ())),
            metadata=value.get("metadata", {}),
            schema_version=value.get("schema_version", TASK_MOTION_PLAN_SCHEMA_VERSION),
        )


def _assert_acyclic(dependencies: Mapping[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(segment_id: str) -> None:
        if segment_id in visiting:
            raise ValueError(f"task-motion plan dependency cycle includes {segment_id!r}")
        if segment_id in visited:
            return
        visiting.add(segment_id)
        for dependency in dependencies[segment_id]:
            visit(dependency)
        visiting.remove(segment_id)
        visited.add(segment_id)

    for segment_id in dependencies:
        visit(segment_id)


class ExecutionEventType(StrEnum):
    PLAN_STARTED = "plan_started"
    PLAN_COMPLETED = "plan_completed"
    PLAN_FAILED = "plan_failed"
    SEGMENT_STARTED = "segment_started"
    SEGMENT_COMPLETED = "segment_completed"
    SEGMENT_FAILED = "segment_failed"
    GRASP_INTENT = "grasp_intent"
    GRASP_VERIFIED = "grasp_verified"
    GRASP_REJECTED = "grasp_rejected"
    TASK_VERIFIED = "task_verified"
    TASK_REJECTED = "task_rejected"
    RECORDING_COMPLETED = "recording_completed"
    RECORDING_FAILED = "recording_failed"


class ExecutionOutcome(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ExecutionEvent:
    """One bounded observation or lifecycle event from plan execution."""

    event_id: str
    attempt_id: str
    plan_id: str | None
    segment_id: str | None
    event_type: ExecutionEventType
    outcome: ExecutionOutcome
    monotonic_time_s: float
    verifier: str | None = None
    failure_code: str | None = None
    message: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = EXECUTION_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.attempt_id, "attempt_id")
        for field_name in ("plan_id", "segment_id", "verifier", "failure_code"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if self.message is not None:
            _require_text(self.message, "message", maximum=2048)
        if not isinstance(self.event_type, ExecutionEventType):
            object.__setattr__(self, "event_type", ExecutionEventType(self.event_type))
        if not isinstance(self.outcome, ExecutionOutcome):
            object.__setattr__(self, "outcome", ExecutionOutcome(self.outcome))
        timestamp = _finite_float(self.monotonic_time_s, "monotonic_time_s")
        if timestamp < 0:
            raise ValueError("monotonic_time_s must be non-negative")
        object.__setattr__(self, "monotonic_time_s", timestamp)
        if self.schema_version != EXECUTION_EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported execution event schema version {self.schema_version}")
        metadata = _json_value(self.metadata)
        assert isinstance(metadata, dict)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "failure_code": self.failure_code,
            "message": self.message,
            "metadata": dict(self.metadata),
            "monotonic_time_s": self.monotonic_time_s,
            "outcome": self.outcome.value,
            "plan_id": self.plan_id,
            "schema_version": self.schema_version,
            "segment_id": self.segment_id,
            "verifier": self.verifier,
        }
