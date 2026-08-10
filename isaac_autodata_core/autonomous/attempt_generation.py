# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Source-demo-free attempt lifecycle independent of simulator and planner backends."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from isaac_autodata_core.autonomous.run_log import (
    RunLogWriter,
    RunLogWriteUncertainError,
    normalize_run_record,
    safe_exception_record,
)
from isaac_autodata_core.autonomous.task_motion import (
    ExecutionEvent,
    JsonValue,
    SceneSnapshot,
    TaskMotionPlan,
    make_stable_id,
)
from isaac_autodata_interfaces.tasks.task_goal import GoalPredicate

MAX_INLINE_PLAN_RECORD_BYTES = 512_000
MAX_RESET_EVIDENCE_RECORD_BYTES = 64_000


class FailureStage(StrEnum):
    RESET = "reset"
    SNAPSHOT = "snapshot"
    PLANNING = "planning"
    LOWERING = "lowering"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    RECORDING = "recording"


class AttemptGenerationError(RuntimeError):
    """Expected, classified failure in one autonomous generation attempt."""

    def __init__(self, stage: FailureStage, code: str, message: str, *, recoverable: bool = True) -> None:
        self.stage = stage
        self.code = code
        self.recoverable = recoverable
        super().__init__(message)


@dataclass(frozen=True)
class ExecutionResult:
    """Final, evidence-based executor result for one task-motion plan."""

    success: bool
    events: tuple[ExecutionEvent, ...]
    final_observation: Mapping[str, JsonValue] = field(default_factory=dict)
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.success and self.failure_code is not None:
            raise ValueError("successful execution cannot have failure_code")
        if self.success and self.failure_stage is not None:
            raise ValueError("successful execution cannot have failure_stage")
        if not self.success and not self.failure_code:
            raise ValueError("failed execution requires failure_code")
        if type(self.recoverable) is not bool:
            raise ValueError("execution recoverable must be a boolean")


@dataclass(frozen=True)
class AttemptRequest:
    """Resolved immutable inputs for exactly one environment attempt."""

    request_digest: str
    attempt_index: int
    seed: int
    env_id: int
    goal: tuple[GoalPredicate, ...]
    keep_failed: bool
    expected_plan_backend: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_digest, str) or not self.request_digest.strip():
            raise ValueError("request_digest must be a non-empty string")
        for field_name in ("attempt_index", "seed", "env_id"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not self.goal or any(not isinstance(item, GoalPredicate) for item in self.goal):
            raise ValueError("goal must contain GoalPredicate instances")
        if type(self.keep_failed) is not bool:
            raise ValueError("keep_failed must be a boolean")
        if self.expected_plan_backend is not None and (
            not isinstance(self.expected_plan_backend, str) or not self.expected_plan_backend.strip()
        ):
            raise ValueError("expected_plan_backend must be null or a non-empty string")

    @property
    def attempt_id(self) -> str:
        return make_stable_id("attempt", self.request_digest, self.attempt_index, self.seed, self.env_id)


@dataclass(frozen=True)
class AttemptResult:
    """Complete result of one source-demo-free attempt, including no-action failures."""

    attempt_id: str
    success: bool
    initial_state: Mapping[str, Any]
    snapshot: SceneSnapshot | None
    plan: TaskMotionPlan | None
    execution: ExecutionResult | None
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    recoverable: bool = True


class AttemptRuntime(Protocol):
    """Simulator/recorder operations required by the autonomous lifecycle."""

    async def reset_attempt(self, env_id: int) -> Mapping[str, Any]:
        pass

    def capture_scene_snapshot(self, env_id: int, *, snapshot_id: str) -> SceneSnapshot:
        pass

    async def finish_attempt(self, env_id: int, *, success: bool, keep_failed: bool) -> None:
        pass


class EpisodePlanner(Protocol):
    """Planner backend boundary. Native planner types stay behind this interface."""

    def plan(self, request: AttemptRequest, snapshot: SceneSnapshot) -> TaskMotionPlan:
        pass

    def close(self) -> None:
        pass


class EpisodeExecutor(Protocol):
    """Execution boundary for one typed task-motion plan."""

    async def execute(self, request: AttemptRequest, plan: TaskMotionPlan) -> ExecutionResult:
        pass


class AttemptGenerator:
    """Run bounded, source-demo-free planning attempts with durable run_log."""

    def __init__(
        self,
        runtime: AttemptRuntime,
        planner: EpisodePlanner,
        executor: EpisodeExecutor,
        *,
        run_log_writer: RunLogWriter | None = None,
        include_debug_tracebacks: bool = False,
    ) -> None:
        self.runtime = runtime
        self.planner = planner
        self.executor = executor
        self.run_log_writer = run_log_writer
        self.include_debug_tracebacks = include_debug_tracebacks

    async def generate_attempt(self, request: AttemptRequest) -> AttemptResult:
        """Run one attempt and finalize recorder state even when the task is cancelled."""

        try:
            return await self._generate_attempt(request)
        except BaseException as exc:
            if isinstance(exc, Exception):
                raise
            await _finish_attempt_before_reraise(self.runtime, request)
            raise

    async def _generate_attempt(self, request: AttemptRequest) -> AttemptResult:
        """Reset, snapshot, plan, execute, verify, record, and account for one attempt."""

        initial_state: Mapping[str, Any] = {}
        snapshot: SceneSnapshot | None = None
        plan: TaskMotionPlan | None = None
        execution: ExecutionResult | None = None
        failure: AttemptGenerationError | None = None
        started_at = time.time()

        try:
            try:
                initial_state = await self.runtime.reset_attempt(request.env_id)
            except Exception as exc:
                raise _classify(exc, FailureStage.RESET, "reset_failed") from exc

            try:
                snapshot_id = make_stable_id("snapshot", request.attempt_id, request.seed)
                snapshot = self.runtime.capture_scene_snapshot(request.env_id, snapshot_id=snapshot_id)
            except Exception as exc:
                raise _classify(exc, FailureStage.SNAPSHOT, "snapshot_failed") from exc

            try:
                plan = self.planner.plan(request, snapshot)
            except Exception as exc:
                raise _classify(exc, FailureStage.PLANNING, "planning_failed") from exc
            if plan.request_digest != request.request_digest:
                raise AttemptGenerationError(
                    FailureStage.LOWERING,
                    "request_digest_mismatch",
                    "planner returned a plan for a different request digest",
                    recoverable=False,
                )
            if plan.snapshot_digest != snapshot.digest:
                raise AttemptGenerationError(
                    FailureStage.LOWERING,
                    "snapshot_digest_mismatch",
                    "planner returned a plan for a different scene snapshot",
                    recoverable=False,
                )
            if plan.seed != request.seed:
                raise AttemptGenerationError(
                    FailureStage.LOWERING,
                    "seed_mismatch",
                    "planner returned a plan generated with a different attempt seed",
                    recoverable=False,
                )
            if plan.goal != request.goal:
                raise AttemptGenerationError(
                    FailureStage.LOWERING,
                    "goal_mismatch",
                    "planner returned a plan for different goal predicates",
                    recoverable=False,
                )
            if request.expected_plan_backend is not None and plan.backend != request.expected_plan_backend:
                raise AttemptGenerationError(
                    FailureStage.LOWERING,
                    "plan_backend_mismatch",
                    "planner returned a plan from an unexpected backend",
                    recoverable=False,
                )

            try:
                execution = await self.executor.execute(request, plan)
            except Exception as exc:
                raise _classify(exc, FailureStage.EXECUTION, "execution_failed") from exc
            if not execution.success:
                failure = AttemptGenerationError(
                    execution.failure_stage or FailureStage.VERIFICATION,
                    execution.failure_code or "task_not_verified",
                    execution.failure_message or "final task postconditions were not verified",
                    recoverable=execution.recoverable,
                )
        except AttemptGenerationError as exc:
            failure = exc

        success = failure is None and execution is not None and execution.success
        result = _attempt_result(
            request,
            success=success,
            initial_state=initial_state,
            snapshot=snapshot,
            plan=plan,
            execution=execution,
            failure=failure,
        )
        run_log_prepared = False
        run_log_write_uncertainty: RunLogWriteUncertainError | None = None
        try:
            self._write_attempt_prepared(
                request,
                result,
                started_at=started_at,
                prepared_at=time.time(),
            )
            run_log_prepared = self.run_log_writer is not None
        except RunLogWriteUncertainError as exc:
            run_log_write_uncertainty = exc
            failure = AttemptGenerationError(
                FailureStage.RECORDING,
                "run_log_write_uncertain",
                str(exc),
                recoverable=False,
            )
            success = False
        except Exception as exc:
            safe = safe_exception_record(exc)
            failure = AttemptGenerationError(
                FailureStage.RECORDING,
                "run_log_prepare_failed",
                safe["message"] or safe["exception_type"],
                recoverable=False,
            )
            success = False

        try:
            await self.runtime.finish_attempt(request.env_id, success=success, keep_failed=request.keep_failed)
        except Exception as exc:
            safe = safe_exception_record(exc)
            failure = AttemptGenerationError(
                FailureStage.RECORDING,
                "dataset_finalize_failed",
                safe["message"] or safe["exception_type"],
                recoverable=False,
            )
            success = False

        if run_log_write_uncertainty is not None:
            raise run_log_write_uncertainty

        result = _attempt_result(
            request,
            success=success,
            initial_state=initial_state,
            snapshot=snapshot,
            plan=plan,
            execution=execution,
            failure=failure,
        )
        if run_log_prepared:
            try:
                self._write_attempt_committed(request, result, completed_at=time.time())
            except RunLogWriteUncertainError:
                raise
            except Exception as exc:
                safe = safe_exception_record(exc)
                failure = AttemptGenerationError(
                    FailureStage.RECORDING,
                    "run_log_commit_failed",
                    safe["message"] or safe["exception_type"],
                    recoverable=False,
                )
                result = _attempt_result(
                    request,
                    success=False,
                    initial_state=initial_state,
                    snapshot=snapshot,
                    plan=plan,
                    execution=execution,
                    failure=failure,
                )
        return result

    def close(self) -> None:
        """Release planner-owned GPU and service resources."""

        self.planner.close()

    def _write_attempt_prepared(
        self,
        request: AttemptRequest,
        result: AttemptResult,
        *,
        started_at: float,
        prepared_at: float,
    ) -> None:
        if self.run_log_writer is None:
            return
        normalized_reset_record = normalize_run_record(
            {"reset_evidence": result.initial_state},
            max_serialized_bytes=MAX_RESET_EVIDENCE_RECORD_BYTES,
        )
        assert isinstance(normalized_reset_record, dict)
        reset_evidence = normalized_reset_record["reset_evidence"]
        if not isinstance(reset_evidence, dict):
            raise ValueError("attempt reset evidence must be a JSON-compatible mapping")
        self.run_log_writer.append({
            "attempt_id": result.attempt_id,
            "attempt_index": request.attempt_index,
            "prepared_at_unix_s": prepared_at,
            "record_type": "attempt_prepared",
            "env_id": request.env_id,
            "events": [] if result.execution is None else [event.to_dict() for event in result.execution.events],
            "failure": (
                None
                if result.failure_code is None
                else {
                    "code": result.failure_code,
                    "message": result.failure_message,
                    "recoverable": result.recoverable,
                    "stage": result.failure_stage.value if result.failure_stage is not None else None,
                }
            ),
            "final_observation": {} if result.execution is None else dict(result.execution.final_observation),
            "goal": [predicate.to_dict() for predicate in request.goal],
            "plan": _plan_record(result.plan),
            "request_digest": request.request_digest,
            "reset_evidence": reset_evidence,
            "seed": request.seed,
            "snapshot": None if result.snapshot is None else result.snapshot.to_dict(),
            "started_at_unix_s": started_at,
            "success": result.success,
        })

    def _write_attempt_committed(
        self,
        request: AttemptRequest,
        result: AttemptResult,
        *,
        completed_at: float,
    ) -> None:
        assert self.run_log_writer is not None
        self.run_log_writer.append({
            "attempt_id": result.attempt_id,
            "attempt_index": request.attempt_index,
            "completed_at_unix_s": completed_at,
            "failure_code": result.failure_code,
            "record_type": "attempt_recorded",
            "request_digest": request.request_digest,
            "success": result.success,
        })


async def _finish_attempt_before_reraise(runtime: AttemptRuntime, request: AttemptRequest) -> None:
    """Retain and await one finalizer despite repeated cancellation of the caller."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # GUI generation is driven inline because Kit owns the main-thread event loop. The reviewed
        # live finalizer does not suspend, so it can still preserve recorder state before an
        # operator interrupt propagates. A finalization failure must not hide that interrupt.
        with suppress(BaseException):
            await runtime.finish_attempt(
                request.env_id,
                success=False,
                keep_failed=request.keep_failed,
            )
        return

    finalizer = asyncio.create_task(
        runtime.finish_attempt(
            request.env_id,
            success=False,
            keep_failed=request.keep_failed,
        )
    )
    while not finalizer.done():
        try:
            await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            continue
        except BaseException:
            return
    try:
        finalizer.result()
    except BaseException:
        # Preserve the original operator cancellation/interrupt. Runtime finalization is
        # idempotent and the process-level terminal run log records interrupted execution.
        return


def _attempt_result(
    request: AttemptRequest,
    *,
    success: bool,
    initial_state: Mapping[str, Any],
    snapshot: SceneSnapshot | None,
    plan: TaskMotionPlan | None,
    execution: ExecutionResult | None,
    failure: AttemptGenerationError | None,
) -> AttemptResult:
    return AttemptResult(
        attempt_id=request.attempt_id,
        success=success,
        initial_state=initial_state,
        snapshot=snapshot,
        plan=plan,
        execution=execution,
        failure_stage=None if failure is None else failure.stage,
        failure_code=None if failure is None else failure.code,
        failure_message=None if failure is None else str(failure),
        recoverable=True if failure is None else failure.recoverable,
    )


def _plan_record(plan: TaskMotionPlan | None) -> Mapping[str, Any] | None:
    if plan is None:
        return None
    canonical = plan.canonical_json()
    summary: dict[str, Any] = {
        "backend": {"name": plan.backend, "version": plan.backend_version},
        "digest": plan.digest,
        "plan_id": plan.plan_id,
        "segment_count": len(plan.segments),
        "serialized_bytes": len(canonical.encode("utf-8")),
    }
    if summary["serialized_bytes"] <= MAX_INLINE_PLAN_RECORD_BYTES:
        summary["representation"] = "inline"
        summary["value"] = plan.to_dict()
    else:
        summary["representation"] = "digest_only"
        summary["segments"] = [
            {
                "kind": segment.kind.value,
                "segment_id": segment.segment_id,
            }
            for segment in plan.segments
        ]
    return summary


def _classify(exc: Exception, default_stage: FailureStage, default_code: str) -> AttemptGenerationError:
    if isinstance(exc, AttemptGenerationError):
        return exc
    safe = safe_exception_record(exc)
    return AttemptGenerationError(default_stage, default_code, safe["message"] or safe["exception_type"])
