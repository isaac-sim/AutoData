# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded dataset-level run loop for source-free autonomous generation."""

from __future__ import annotations

from dataclasses import dataclass

from isaac_autodata_core.autonomous.attempt_generation import AttemptGenerator, AttemptRequest, AttemptResult
from isaac_autodata_core.autonomous.task_motion import GoalPredicate

MAX_ATTEMPT_SEED = 2**63 - 1


@dataclass(frozen=True)
class DatasetGenerationRequest:
    """Immutable dataset-level controls shared by all generation attempts."""

    request_digest: str
    goal: tuple[GoalPredicate, ...]
    successful_episodes: int
    max_attempts: int
    base_seed: int
    num_envs: int = 1
    keep_failed: bool = False
    expected_plan_backend: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_digest, str) or not self.request_digest.strip():
            raise ValueError("request_digest must be a non-empty string")
        if not self.goal or any(not isinstance(item, GoalPredicate) for item in self.goal):
            raise ValueError("goal must contain GoalPredicate instances")
        for field_name in ("successful_episodes", "max_attempts", "base_seed", "num_envs"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if self.successful_episodes <= 0:
            raise ValueError("successful_episodes must be positive")
        if self.max_attempts < self.successful_episodes:
            raise ValueError("max_attempts must be at least successful_episodes")
        if not 0 <= self.base_seed <= MAX_ATTEMPT_SEED:
            raise ValueError(f"base_seed must be in [0, {MAX_ATTEMPT_SEED}]")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if type(self.keep_failed) is not bool:
            raise ValueError("keep_failed must be a boolean")
        if self.expected_plan_backend is not None and (
            not isinstance(self.expected_plan_backend, str) or not self.expected_plan_backend.strip()
        ):
            raise ValueError("expected_plan_backend must be null or a non-empty string")


@dataclass(frozen=True)
class DatasetGenerationSummary:
    """Terminal counters and reason for one bounded generation run."""

    request_digest: str
    requested_successful_episodes: int
    attempts: int
    successes: int
    failures: int
    stop_reason: str
    target_reached: bool
    last_result: AttemptResult | None

    def __post_init__(self) -> None:
        if self.successes + self.failures != self.attempts:
            raise ValueError("successes and failures must sum to attempts")

    def to_dict(self) -> dict[str, int | str | bool | None]:
        """Return a compact JSON-compatible run summary."""

        return {
            "attempts": self.attempts,
            "failures": self.failures,
            "last_attempt_id": None if self.last_result is None else self.last_result.attempt_id,
            "request_digest": self.request_digest,
            "requested_successful_episodes": self.requested_successful_episodes,
            "stop_reason": self.stop_reason,
            "successes": self.successes,
            "target_reached": self.target_reached,
        }


async def generate_dataset(
    generator: AttemptGenerator,
    request: DatasetGenerationRequest,
    *,
    close: bool = True,
) -> DatasetGenerationSummary:
    """Run deterministic attempts until the requested target or a hard stop is reached.

    Failed attempts always count toward ``max_attempts``. Only attempts that pass execution and
    task-success validation count toward ``successful_episodes``. A classified non-recoverable
    failure ends the run immediately. Planner resources are closed in a ``finally`` block by
    default.

    Args:
        generator: Configured source-free attempt orchestrator.
        request: Dataset-level generation controls.
        close: Whether to close planner-owned resources before returning or propagating.
    """

    attempts = 0
    successes = 0
    failures = 0
    last_result: AttemptResult | None = None
    stop_reason = "max_attempts"
    target_reached = False

    try:
        while attempts < request.max_attempts:
            if successes >= request.successful_episodes:
                target_reached = True
                stop_reason = "requested_successes"
                break

            attempt_request = AttemptRequest(
                request_digest=request.request_digest,
                attempt_index=attempts,
                seed=_attempt_seed(request.base_seed, attempts),
                env_id=attempts % request.num_envs,
                goal=request.goal,
                keep_failed=request.keep_failed,
                expected_plan_backend=request.expected_plan_backend,
            )
            last_result = await generator.generate_attempt(attempt_request)
            attempts += 1
            if last_result.success:
                successes += 1
            else:
                failures += 1
                if not last_result.recoverable:
                    stop_reason = "unrecoverable_failure"
                    break
        else:
            stop_reason = "max_attempts"

        if not target_reached:
            if successes >= request.successful_episodes:
                target_reached = True
                stop_reason = "requested_successes"

        return DatasetGenerationSummary(
            request_digest=request.request_digest,
            requested_successful_episodes=request.successful_episodes,
            attempts=attempts,
            successes=successes,
            failures=failures,
            stop_reason=stop_reason,
            target_reached=target_reached,
            last_result=last_result,
        )
    finally:
        if close:
            generator.close()


def _attempt_seed(base_seed: int, attempt_index: int) -> int:
    """Derive a deterministic signed-int64-compatible attempt seed without overflow."""

    return (base_seed + attempt_index) % (MAX_ATTEMPT_SEED + 1)
