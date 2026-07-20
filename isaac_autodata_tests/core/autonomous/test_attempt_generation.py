# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from isaac_autodata_core.autonomous.attempt_generation import (
    AttemptGenerationError,
    AttemptGenerator,
    AttemptRequest,
    ExecutionResult,
    FailureStage,
)
from isaac_autodata_core.autonomous.run_log import RunLogWriter, RunLogWriteUncertainError
from isaac_autodata_core.autonomous.task_motion import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionOutcome,
    GoalPredicate,
    RobotStateSnapshot,
    SceneSnapshot,
    TaskMotionPlan,
    WaitSegment,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Runtime:
    def __init__(self) -> None:
        self.finished: list[tuple[bool, bool]] = []
        self.snapshot = SceneSnapshot(
            snapshot_id="placeholder",
            captured_at_s=0.0,
            env_id=0,
            robot=RobotStateSnapshot(
                robot_id="robot",
                joint_names=("joint",),
                joint_positions=(0.0,),
                eef_poses={"eef": IDENTITY},
            ),
            objects=(),
        )

    async def reset_attempt(self, env_id: int):
        return {"env_id": env_id}

    def capture_scene_snapshot(self, env_id: int, *, snapshot_id: str):
        return SceneSnapshot(
            snapshot_id=snapshot_id,
            captured_at_s=0.0,
            env_id=env_id,
            robot=self.snapshot.robot,
            objects=(),
        )

    async def finish_attempt(self, env_id: int, *, success: bool, keep_failed: bool):
        self.finished.append((success, keep_failed))


class _CancellingRuntime(_Runtime):
    async def reset_attempt(self, env_id: int):
        raise asyncio.CancelledError


class _ResetEvidenceRuntime(_Runtime):
    def __init__(self, reset_evidence) -> None:
        super().__init__()
        self.reset_evidence = reset_evidence

    async def reset_attempt(self, env_id: int):
        del env_id
        return self.reset_evidence


class _SlowFinalizingCancellingRuntime(_CancellingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.finalizer_started = asyncio.Event()
        self.release_finalizer = asyncio.Event()

    async def finish_attempt(self, env_id: int, *, success: bool, keep_failed: bool):
        self.finalizer_started.set()
        await self.release_finalizer.wait()
        await super().finish_attempt(env_id, success=success, keep_failed=keep_failed)


class _Planner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    def plan(self, request, snapshot):
        if self.fail:
            raise AttemptGenerationError(FailureStage.PLANNING, "no_plan", "no feasible plan")
        return TaskMotionPlan(
            plan_id="plan",
            request_digest=request.request_digest,
            snapshot_digest=snapshot.digest,
            backend="fake",
            backend_version="1",
            seed=request.seed,
            segments=(WaitSegment(segment_id="wait", steps=1),),
            goal=request.goal,
        )

    def close(self):
        self.closed = True


class _Executor:
    def __init__(self, *, success: bool = True, recoverable: bool = True) -> None:
        self.success = success
        self.recoverable = recoverable
        self.calls = 0

    async def execute(self, request, plan):
        self.calls += 1
        event = ExecutionEvent(
            event_id="event",
            attempt_id=request.attempt_id,
            plan_id=plan.plan_id,
            segment_id=None,
            event_type=(ExecutionEventType.TASK_VERIFIED if self.success else ExecutionEventType.TASK_REJECTED),
            outcome=(ExecutionOutcome.SUCCEEDED if self.success else ExecutionOutcome.FAILED),
            monotonic_time_s=0.1,
        )
        return ExecutionResult(
            success=self.success,
            events=(event,),
            failure_code=None if self.success else "task_not_stable",
            recoverable=self.recoverable,
        )


class _TamperedPlanner(_Planner):
    def __init__(self, **changes) -> None:
        super().__init__()
        self.changes = changes

    def plan(self, request, snapshot):
        return replace(super().plan(request, snapshot), **self.changes)


class _FailingWriter:
    def append(self, _record):
        raise OSError("ledger unavailable")


class _AmbiguousWriter:
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on
        self.record_types: list[str] = []

    def append(self, record):
        record_type = record["record_type"]
        self.record_types.append(record_type)
        if record_type == self.fail_on:
            raise RunLogWriteUncertainError("ledger tail is ambiguous")


def _request() -> AttemptRequest:
    return AttemptRequest(
        request_digest="request",
        attempt_index=0,
        seed=7,
        env_id=0,
        goal=(GoalPredicate("on", "cube", "table"),),
        keep_failed=True,
    )


def test_successful_attempt_finishes_recorder_and_writes_run_log(tmp_path):
    runtime = _Runtime()
    planner = _Planner()
    executor = _Executor()
    path = tmp_path / "attempts.jsonl"
    generator = AttemptGenerator(
        runtime,
        planner,
        executor,
        run_log_writer=RunLogWriter(path),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    assert result.success
    assert runtime.finished == [(True, True)]
    assert executor.calls == 1
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["record_type"] == "attempt_prepared"
    assert records[0]["reset_evidence"] == {"env_id": 0}
    assert records[1]["record_type"] == "attempt_recorded"


def test_prepared_run_log_preserves_bounded_reset_settling_evidence(tmp_path):
    reset_evidence = {
        "env_id": 0,
        "recorder_excludes_reset_settling": True,
        "reset_completed": True,
        "reset_settle_duration_s": 0.2,
        "reset_settle_steps": 10,
    }
    runtime = _ResetEvidenceRuntime(reset_evidence)
    path = tmp_path / "attempts.jsonl"
    generator = AttemptGenerator(
        runtime,
        _Planner(),
        _Executor(),
        run_log_writer=RunLogWriter(path),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    prepared = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert result.success
    assert prepared["record_type"] == "attempt_prepared"
    assert prepared["reset_evidence"] == reset_evidence


@pytest.mark.parametrize(
    "reset_evidence",
    [
        {1: "non-string key"},
        {"unsupported": object()},
        ["not", "a", "mapping"],
    ],
)
def test_malformed_reset_evidence_fails_run_log_before_dataset_export(tmp_path, reset_evidence):
    runtime = _ResetEvidenceRuntime(reset_evidence)
    path = tmp_path / "attempts.jsonl"
    generator = AttemptGenerator(
        runtime,
        _Planner(),
        _Executor(),
        run_log_writer=RunLogWriter(path),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.RECORDING
    assert result.failure_code == "run_log_prepare_failed"
    assert runtime.finished == [(False, True)]
    assert path.read_bytes() == b""


def test_oversized_reset_evidence_fails_run_log_before_dataset_export(tmp_path):
    runtime = _ResetEvidenceRuntime({"payload": "x" * 70_000})
    path = tmp_path / "attempts.jsonl"
    generator = AttemptGenerator(
        runtime,
        _Planner(),
        _Executor(),
        run_log_writer=RunLogWriter(path),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.RECORDING
    assert result.failure_code == "run_log_prepare_failed"
    assert runtime.finished == [(False, True)]
    assert path.read_bytes() == b""


def test_no_action_planning_failure_is_a_completed_failed_attempt():
    runtime = _Runtime()
    executor = _Executor()
    generator = AttemptGenerator(runtime, _Planner(fail=True), executor)

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.PLANNING
    assert result.failure_code == "no_plan"
    assert executor.calls == 0
    assert runtime.finished == [(False, True)]


def test_final_verification_failure_cannot_be_labeled_successful():
    runtime = _Runtime()
    generator = AttemptGenerator(runtime, _Planner(), _Executor(success=False))

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "task_not_stable"
    assert runtime.finished == [(False, True)]


def test_executor_nonrecoverable_failure_survives_attempt_classification():
    runtime = _Runtime()
    generator = AttemptGenerator(
        runtime,
        _Planner(),
        _Executor(success=False, recoverable=False),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "task_not_stable"
    assert not result.recoverable
    assert runtime.finished == [(False, True)]


def test_close_releases_planner_resources():
    planner = _Planner()
    generator = AttemptGenerator(_Runtime(), planner, _Executor())

    generator.close()

    assert planner.closed


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"seed": 8}, "seed_mismatch"),
        ({"goal": (GoalPredicate("in", "cube", "bowl"),)}, "goal_mismatch"),
        ({"backend": "unexpected"}, "plan_backend_mismatch"),
    ],
)
def test_plan_attestation_rejects_tampering_before_execution(changes, expected_code):
    runtime = _Runtime()
    executor = _Executor()
    generator = AttemptGenerator(runtime, _TamperedPlanner(**changes), executor)
    request = replace(_request(), expected_plan_backend="fake")

    result = asyncio.run(generator.generate_attempt(request))

    assert not result.success
    assert result.failure_stage is FailureStage.LOWERING
    assert result.failure_code == expected_code
    assert not result.recoverable
    assert executor.calls == 0
    assert runtime.finished == [(False, True)]


def test_cancellation_finalizes_attempt_as_failed_before_propagating():
    runtime = _CancellingRuntime()
    generator = AttemptGenerator(runtime, _Planner(), _Executor())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generator.generate_attempt(_request()))

    assert runtime.finished == [(False, True)]


def test_cancellation_without_running_loop_finalizes_inline_before_propagating():
    runtime = _CancellingRuntime()
    generator = AttemptGenerator(runtime, _Planner(), _Executor())
    operation = generator.generate_attempt(_request())

    with pytest.raises(asyncio.CancelledError):
        operation.send(None)

    assert runtime.finished == [(False, True)]
    assert operation.cr_frame is None


def test_repeated_cancellation_does_not_detach_attempt_finalizer():
    async def run_scenario():
        runtime = _SlowFinalizingCancellingRuntime()
        generator = AttemptGenerator(runtime, _Planner(), _Executor())
        task = asyncio.create_task(generator.generate_attempt(_request()))
        await runtime.finalizer_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        runtime.release_finalizer.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runtime

    runtime = asyncio.run(run_scenario())

    assert runtime.finished == [(False, True)]


def test_run_log_prepare_failure_prevents_successful_dataset_export():
    runtime = _Runtime()
    generator = AttemptGenerator(
        runtime,
        _Planner(),
        _Executor(),
        run_log_writer=_FailingWriter(),
    )

    result = asyncio.run(generator.generate_attempt(_request()))

    assert not result.success
    assert result.failure_stage is FailureStage.RECORDING
    assert result.failure_code == "run_log_prepare_failed"
    assert not result.recoverable
    assert runtime.finished == [(False, True)]


def test_ambiguous_prepare_append_finalizes_failure_then_propagates() -> None:
    runtime = _Runtime()
    writer = _AmbiguousWriter("attempt_prepared")
    generator = AttemptGenerator(runtime, _Planner(), _Executor(), run_log_writer=writer)

    with pytest.raises(RunLogWriteUncertainError, match="ambiguous"):
        asyncio.run(generator.generate_attempt(_request()))

    assert writer.record_types == ["attempt_prepared"]
    assert runtime.finished == [(False, True)]


def test_ambiguous_commit_append_propagates_without_later_ledger_write() -> None:
    runtime = _Runtime()
    writer = _AmbiguousWriter("attempt_recorded")
    generator = AttemptGenerator(runtime, _Planner(), _Executor(), run_log_writer=writer)

    with pytest.raises(RunLogWriteUncertainError, match="ambiguous"):
        asyncio.run(generator.generate_attempt(_request()))

    assert writer.record_types == ["attempt_prepared", "attempt_recorded"]
    assert runtime.finished == [(True, True)]
