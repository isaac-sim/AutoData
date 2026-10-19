# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known-answer tests for the dataset KPI functions."""

import numpy as np

import pytest

from autodata_utils import dataset_metrics


def test_wilson_interval_matches_known_values():
    assert dataset_metrics.wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert dataset_metrics.wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_eef_jerk_of_cubic_trajectory_is_six():
    t = np.arange(10) * 0.05
    positions = np.stack([t**3, np.zeros_like(t), np.zeros_like(t)], axis=1)
    np.testing.assert_allclose(dataset_metrics.eef_jerk(positions, 0.05), 6.0, rtol=1e-6)


def test_saturated_fraction_counts_entries_at_the_limit():
    pose_actions = np.array([[1.0, -1.0, 0.5, 0.0, 0.2, -0.3], [0.9, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert dataset_metrics.saturated_fraction(pose_actions) == pytest.approx(3 / 12)


def test_path_deviation_ignores_lag_along_the_path():
    intended = np.stack([np.linspace(0.0, 1.0, 50), np.zeros(50), np.zeros(50)], axis=1)
    lagging = intended[np.maximum(np.arange(50) - 3, 0)] + [0.0, 0.005, 0.0]
    np.testing.assert_allclose(dataset_metrics.path_deviation(lagging, intended), 0.005, atol=1e-9)


def test_tracking_error_skips_tracking_only_channels():
    path = np.linspace(0.0, 0.1, 30)[:, None] * np.array([1.0, 0.0, 0.0])
    beside = path + np.array([0.0, 0.002, 0.0])
    # A tracking-only channel commands no pose; its adapter writes an identity placeholder.
    episode = {
        "eef_pose": {"left": beside, "body": path},
        "target_eef_pose": {"left": path, "body": np.zeros_like(path)},
    }

    kpi = dataset_metrics._tracking_error([episode], [episode])

    assert kpi["value"]["p95"] == pytest.approx(2.0) and kpi["source_baseline"]["p95"] == pytest.approx(2.0)


def _attempt(env_id: int, success: bool, start: float, duration: float) -> dict:
    return {"env_id": env_id, "success": success, "start_time_s": start, "end_time_s": start + duration}


def test_attempts_before_stop_keeps_attempts_started_before_the_first_env_finished():
    long_successes = [_attempt(0, True, 4.0 * k, 4.0) for k in range(3)]  # env 0 last finishes at 12 s
    short_failures = [_attempt(1, False, 1.6 * k, 1.6) for k in range(10)]  # one runs from 11.2 to 12.8 s

    kept = dataset_metrics.attempts_before_stop(long_successes + short_failures, num_envs=2)

    assert len(kept) == 11 and sum(attempt["success"] for attempt in kept) == 3
    assert dataset_metrics.attempts_before_stop(long_successes[:2] + short_failures, num_envs=2) is None
    assert dataset_metrics.attempts_before_stop(short_failures, num_envs=2) is None


def test_coverage_ratio_drops_when_successes_leave_a_region_empty():
    rng = np.random.default_rng(1)
    attempts = rng.uniform(0.0, 1.0, size=(400, 2))
    spread = attempts[rng.choice(400, 100, replace=False)]
    left_half_only = attempts[attempts[:, 0] < 0.5][:100]
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    assert dataset_metrics.coverage_ratio(spread, attempts, bounds)["ratio"] == pytest.approx(1.0, abs=0.1)
    assert dataset_metrics.coverage_ratio(left_half_only, attempts, bounds)["ratio"] == pytest.approx(0.5, abs=0.1)


def _attempts(x_positions: np.ndarray, successes: np.ndarray) -> list[dict]:
    # The table only settles by about a millimetre, so it does not count as a varying object.
    return [
        {
            "success": bool(success),
            "initial_object_positions": {"cube": [float(x), 0.0], "table": [1e-3 * (i % 2), 0.0]},
        }
        for i, (x, success) in enumerate(zip(x_positions, successes))
    ]


def test_success_by_configuration_flags_only_a_truly_weak_region():
    rng = np.random.default_rng(2)
    x_positions = rng.uniform(0.0, 0.3, size=600)
    uniform = _attempts(x_positions, rng.uniform(size=600) < 0.7)
    weak_left = _attempts(x_positions, (rng.uniform(size=600) < 0.7) & (x_positions > 0.1))
    # 9 of 11 at an overall rate of 65/67: unremarkable by an exact test, though a Wilson bound reads it as low.
    near_perfect = _attempts(
        np.array([0.05] * 11 + [0.15] * 28 + [0.25] * 28), np.array([True] * 9 + [False] * 2 + [True] * 56)
    )

    uniform_kpi = dataset_metrics._success_by_configuration(uniform, corrected=False)
    assert uniform_kpi["value"] == 0 and set(uniform_kpi["per_asset"]) == {"cube"}
    assert dataset_metrics._success_by_configuration(weak_left, corrected=False)["value"] == 1
    assert dataset_metrics._success_by_configuration(near_perfect, corrected=False)["value"] == 0
