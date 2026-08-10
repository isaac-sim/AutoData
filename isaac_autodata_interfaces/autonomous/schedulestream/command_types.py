# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure contracts for the optional ScheduleStream compatibility boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from isaac_autodata_core.autonomous.task_motion import JsonValue
from isaac_autodata_interfaces.tasks.task_goal import GoalPredicate


class ScheduleStreamBoundaryError(RuntimeError):
    """Base class for safe, classified ScheduleStream boundary failures."""

    code = "schedulestream_boundary_error"


class ScheduleStreamImportError(ScheduleStreamBoundaryError):
    """Raised when a selected ScheduleStream application cannot be loaded."""

    code = "schedulestream_import_error"


class ScheduleStreamClosedError(ScheduleStreamBoundaryError):
    """Raised when a closed boundary or planner provider is reused."""

    code = "schedulestream_closed"


class ScheduleStreamCommandError(ScheduleStreamBoundaryError):
    """Base class for a malformed or unsupported native command."""

    code = "schedulestream_command_error"

    def __init__(self, message: str, *, path: tuple[int, ...] = ()) -> None:
        self.path = path
        path_text = "$" + "".join(f"[{index}]" for index in path)
        super().__init__(f"{path_text}: {message}")


class MalformedScheduleStreamCommandError(ScheduleStreamCommandError):
    """Raised when a recognized native command violates its API contract."""

    code = "malformed_schedulestream_command"


class ScheduleStreamLimitError(ScheduleStreamCommandError):
    """Raised before an untrusted native stream exceeds a configured resource limit."""

    code = "schedulestream_limit_exceeded"


class ScheduleStreamTimingError(ScheduleStreamCommandError):
    """Raised when native command timing is invalid or disagrees with the executor tick."""

    code = "schedulestream_timing_mismatch"


class ScheduleStreamProviderError(ScheduleStreamBoundaryError):
    """Raised when a concrete runtime provider cannot create or run its planner."""

    code = "schedulestream_provider_error"


@dataclass(frozen=True)
class ScheduleStreamLoweringLimits:
    """Hard bounds applied before a native dense controller is materialized."""

    max_samples_per_segment: int = 100_000
    max_total_samples: int = 1_000_000
    max_joints: int = 256
    max_duration_s: float = 86_400.0

    def __post_init__(self) -> None:
        integer_fields = (
            "max_samples_per_segment",
            "max_total_samples",
            "max_joints",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.max_duration_s, bool)
            or not isinstance(self.max_duration_s, (int, float))
            or not math.isfinite(float(self.max_duration_s))
            or self.max_duration_s <= 0
        ):
            raise ValueError("max_duration_s must be a positive finite number")
        object.__setattr__(self, "max_duration_s", float(self.max_duration_s))


@dataclass(frozen=True)
class ScheduleStreamLoweringContext:
    """Planner-neutral identity, timing, and name bindings for one native command stream."""

    request_digest: str
    snapshot_digest: str
    seed: int
    goal: tuple[GoalPredicate, ...]
    eef_name: str
    frame: str
    step_dt_s: float
    initial_gripper_value: float = 1.0
    eef_by_arm: Mapping[str, str] = field(default_factory=dict)
    eef_by_link: Mapping[str, str] = field(default_factory=dict)
    object_name_map: Mapping[str, str] = field(default_factory=dict)
    initial_attachments_by_link: Mapping[str, str] = field(default_factory=dict)
    attachment_verifier: str = "contact_and_relative_motion_v1"
    backend_version: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "request_digest",
            "snapshot_digest",
            "eef_name",
            "frame",
            "attachment_verifier",
        ):
            _require_text(getattr(self, field_name), field_name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        step_dt_s = _positive_finite(self.step_dt_s, "step_dt_s")
        gripper = _finite(self.initial_gripper_value, "initial_gripper_value")
        object.__setattr__(self, "step_dt_s", step_dt_s)
        object.__setattr__(self, "initial_gripper_value", gripper)
        object.__setattr__(self, "goal", tuple(self.goal))
        for predicate in self.goal:
            if not isinstance(predicate, GoalPredicate):
                raise TypeError("goal entries must be GoalPredicate instances")
        for field_name in (
            "eef_by_arm",
            "eef_by_link",
            "object_name_map",
            "initial_attachments_by_link",
        ):
            value = getattr(self, field_name)
            normalized: dict[str, str] = {}
            for key, item in sorted(value.items()):
                _require_text(key, f"{field_name} key")
                _require_text(item, f"{field_name}[{key}]")
                normalized[key] = item
            object.__setattr__(self, field_name, normalized)
        if len(set(self.initial_attachments_by_link.values())) != len(self.initial_attachments_by_link):
            raise ValueError("one initial object cannot be attached to multiple links")
        if self.backend_version is not None:
            _require_text(self.backend_version, "backend_version")
        object.__setattr__(self, "metadata", dict(self.metadata))


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > 512 or "\x00" in value:
        raise ValueError(f"{field_name} must be at most 512 characters and contain no NUL")
    return value


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_finite(value: Any, field_name: str) -> float:
    result = _finite(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result
