# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from isaac_autodata_core.autonomous.attempt_generation import AttemptResult, FailureStage
from isaac_autodata_core.autonomous.dataset_generation import (
    MAX_ATTEMPT_SEED,
    DatasetGenerationRequest,
    generate_dataset,
)
from isaac_autodata_core.autonomous.task_motion import GoalPredicate


class _FakeGenerator:
    def __init__(self, outcomes: list[tuple[bool, bool]]) -> None:
        self.outcomes = outcomes
        self.requests = []
        self.closed = False

    async def generate_attempt(self, request):
        self.requests.append(request)
        success, recoverable = self.outcomes[len(self.requests) - 1]
        return AttemptResult(
            attempt_id=request.attempt_id,
            success=success,
            initial_state={},
            snapshot=None,
            plan=None,
            execution=None,
            failure_stage=None if success else FailureStage.PLANNING,
            failure_code=None if success else "no_plan",
            failure_message=None if success else "no plan",
            recoverable=recoverable,
        )

    def close(self):
        self.closed = True


def _request(**overrides):
    values = {
        "request_digest": "abc",
        "goal": (GoalPredicate("on", "cube", "table"),),
        "successful_episodes": 2,
        "max_attempts": 5,
        "base_seed": 11,
        "num_envs": 2,
        "keep_failed": False,
    }
    values.update(overrides)
    return DatasetGenerationRequest(**values)


def test_failed_attempts_do_not_count_toward_success_target():
    generator = _FakeGenerator([(False, True), (True, True), (True, True)])

    summary = asyncio.run(generate_dataset(generator, _request()))

    assert summary.target_reached
    assert (summary.attempts, summary.successes, summary.failures) == (3, 2, 1)
    assert summary.stop_reason == "requested_successes"
    assert [request.seed for request in generator.requests] == [11, 12, 13]
    assert [request.env_id for request in generator.requests] == [0, 1, 0]
    assert generator.closed


def test_hard_attempt_budget_is_terminal():
    generator = _FakeGenerator([(False, True), (False, True), (False, True)])

    summary = asyncio.run(generate_dataset(generator, _request(successful_episodes=2, max_attempts=3)))

    assert not summary.target_reached
    assert summary.stop_reason == "max_attempts"
    assert summary.attempts == 3


def test_unrecoverable_failure_stops_immediately():
    generator = _FakeGenerator([(False, False), (True, True), (True, True)])

    summary = asyncio.run(generate_dataset(generator, _request()))

    assert not summary.target_reached
    assert summary.stop_reason == "unrecoverable_failure"
    assert summary.attempts == 1


def test_close_can_be_owned_by_caller():
    generator = _FakeGenerator([(True, True), (True, True)])

    asyncio.run(generate_dataset(generator, _request(), close=False))

    assert not generator.closed


def test_attempt_seed_wraps_within_signed_int64_contract():
    generator = _FakeGenerator([(False, True), (True, True), (True, True)])

    asyncio.run(generate_dataset(generator, _request(base_seed=MAX_ATTEMPT_SEED)))

    assert [request.seed for request in generator.requests] == [MAX_ATTEMPT_SEED, 0, 1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"successful_episodes": True}, "successful_episodes must be an integer"),
        ({"base_seed": MAX_ATTEMPT_SEED + 1}, "base_seed must be in"),
        ({"keep_failed": 1}, "keep_failed must be a boolean"),
        ({"expected_plan_backend": ""}, "expected_plan_backend"),
    ],
)
def test_generation_request_rejects_invalid_public_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _request(**overrides)
