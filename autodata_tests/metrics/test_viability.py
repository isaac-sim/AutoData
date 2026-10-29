# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests of the viability statistics."""

from types import SimpleNamespace

import pytest

from autodata_metrics.viability import attempts_before_stop, time_to_dataset, wilson_interval


def test_wilson_interval_matches_known_values():
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def _attempt(env_id: int, success: bool, start: float, duration: float) -> dict:
    return {"env_id": env_id, "success": success, "start_time_s": start, "end_time_s": start + duration}


def test_attempts_before_stop_keeps_attempts_started_before_the_first_env_finished():
    long_successes = [_attempt(0, True, 4.0 * k, 4.0) for k in range(3)]  # env 0 last finishes at 12 s
    short_failures = [_attempt(1, False, 1.6 * k, 1.6) for k in range(10)]  # one runs from 11.2 to 12.8 s

    kept = attempts_before_stop(long_successes + short_failures, num_envs=2)

    assert len(kept) == 11 and sum(attempt["success"] for attempt in kept) == 3
    assert attempts_before_stop(long_successes[:2] + short_failures, num_envs=2) is None
    assert attempts_before_stop(short_failures, num_envs=2) is None


def test_time_to_dataset_counts_automated_steps_and_keeps_manual_annotation_apart():
    run = {"setup_time_s": 20.0, "generation_time_s": 80.0}

    def measured(annotation: dict):
        return time_to_dataset(
            SimpleNamespace(source=SimpleNamespace(env_args={"autodata": {"annotation": annotation}}), run=run)
        )

    automatic = measured({"mode": "auto", "setup_time_s": 15.0, "time_s": 60.0})
    manual = measured({"mode": "manual", "setup_time_s": 15.0, "time_s": 600.0})

    assert automatic.value == 175.0
    assert (manual.value, manual.details["manual_annotation_s"]) == (115.0, 600.0)
