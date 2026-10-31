# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viability metrics: what producing the dataset cost, and diagnostics of why attempts failed."""

from __future__ import annotations

import math
import numpy as np
from collections import Counter
from typing import Any

from autodata_metrics.diversity import MIN_START_EXTENT_M
from autodata_metrics.evidence import Evidence
from autodata_metrics.registry import NotApplicable, Value, metric

Z_95 = 1.959963984540054
MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION = 3
"""Below this many logged attempts in some env, the stop correction is itself biased."""
BIAS_BINS = 3
"""Cells per moving axis of the start-position grid that ``generation_bias`` tests."""
SPARSE_CELL_ATTEMPTS = 10
"""Fewest attempts a start-position cell needs for ``generation_bias`` to test its success rate."""


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    assert trials > 0, "trials must be positive"
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def binomial_cdf(successes: int, trials: int, rate: float) -> float:
    """Probability of at most ``successes`` successes in ``trials`` trials that each succeed at ``rate``."""
    if successes >= trials:
        return 1.0
    assert 0.0 < rate < 1.0, "rate must lie strictly between 0 and 1 when some trial fails"
    # Log space: binomial coefficients overflow a float beyond about a thousand trials.
    log_terms = (
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + k * math.log(rate)
        + (trials - k) * math.log1p(-rate)
        for k in range(successes + 1)
    )
    return min(1.0, sum(math.exp(term) for term in log_terms))


def attempts_before_stop(attempts: list[dict[str, Any]], num_envs: int) -> list[dict[str, Any]] | None:
    """Attempts that started before the first env completed its last logged attempt.

    Attempts running when generation stops are cancelled unlogged, and long attempts are the likelier to be running.
    Each env's cancelled attempt started after that env's last completion, so this start-time subset was fully
    logged. None when some env logged fewer than :data:`MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION` attempts.
    """
    last_end: dict[int, float] = {}
    logged: Counter[int] = Counter()
    for attempt in attempts:
        last_end[attempt["env_id"]] = max(last_end.get(attempt["env_id"], 0.0), attempt["end_time_s"])
        logged[attempt["env_id"]] += 1
    if len(logged) < num_envs or min(logged.values()) < MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION:
        return None
    cutoff = min(last_end.values())
    return [attempt for attempt in attempts if attempt["start_time_s"] < cutoff]


def _counted_attempts(ev: Evidence) -> tuple[list[dict[str, Any]], bool]:
    """The stop-corrected attempts when the log allows the correction, else all attempts; and which one it is."""
    corrected = attempts_before_stop(ev.attempts, ev.run["num_envs"]) if "num_envs" in ev.run else None
    return (corrected, True) if corrected is not None else (ev.attempts, False)


@metric(
    "generation_success_rate",
    area="viability",
    signal="generation performance",
    scope="run",
    unit="fraction",
    sample="attempts",
    better="higher",
    requires=["attempts"],
)
def generation_success_rate(ev: Evidence) -> Value:
    """Successful attempts over attempts, stop-corrected when the attempt log allows, with a Wilson 95% interval."""
    attempts, corrected = _counted_attempts(ev)
    successes = sum(attempt["success"] for attempt in attempts)
    low, high = wilson_interval(successes, len(attempts))
    raw_rate = sum(attempt["success"] for attempt in ev.attempts) / len(ev.attempts)
    details = {"interval_95": [low, high], "raw_rate": raw_rate, "stop_corrected": corrected}
    return Value(successes / len(attempts), n=len(attempts), details=details)


@metric(
    "demos_per_gpu_hour",
    area="viability",
    signal="generation performance",
    scope="run",
    unit="demos/GPU-h",
    sample="demos",
    better="higher",
    requires=["run.generation_time_s", "run.num_gpus", "env_args.autodata.exported_episodes"],
)
def demos_per_gpu_hour(ev: Evidence) -> Value:
    """Exported demonstrations per GPU-hour of generation time."""
    if ev.run["num_gpus"] == 0:
        raise NotApplicable("the run used no GPU")
    gpu_hours = ev.run["generation_time_s"] / 3600 * ev.run["num_gpus"]
    exported = ev.entry["exported_episodes"]
    details = {key: ev.run.get(key) for key in ("gpu", "cpu", "device", "num_envs")}
    return Value(exported / gpu_hours, n=exported, details=details)


@metric(
    "time_to_dataset",
    area="viability",
    signal="time to dataset",
    scope="run",
    unit="s",
    sample="runs",
    better="lower",
    requires=["run.setup_time_s", "run.generation_time_s"],
    version=2,
)
def time_to_dataset(ev: Evidence) -> Value:
    """Wall time of the automated steps, each process timed from its start: annotating the source and generating.

    Annotation that was manual, or recorded without its mode, is reported in ``uncounted_annotation_s`` instead.
    """
    annotation = ev.source.entry.get("annotation", {}) if ev.source else {}
    parts = {"setup_s": ev.run["setup_time_s"], "generation_s": ev.run["generation_time_s"]}
    uncounted = None
    if annotation:
        automatic = annotation.get("mode") == "auto"
        parts["annotation_s"] = annotation.get("setup_time_s", 0.0) + (annotation["time_s"] if automatic else 0.0)
        uncounted = None if automatic else annotation["time_s"]
    details = {**parts, "annotation_mode": annotation.get("mode"), "uncounted_annotation_s": uncounted}
    return Value(sum(parts.values()), n=1, details=details)


@metric(
    "annotation_pass_rate",
    area="viability",
    signal="time to dataset",
    scope="run",
    unit="fraction",
    sample="recordings",
    better="higher",
    requires=["source.env_args.autodata.annotation"],
)
def annotation_pass_rate(ev: Evidence) -> Value:
    """Share of processed recordings that annotation exported into the source dataset."""
    annotation = ev.source.entry["annotation"]
    if annotation["processed_episodes"] == 0:
        raise NotApplicable("annotation processed no episodes")
    rate = annotation["annotated_episodes"] / annotation["processed_episodes"]
    return Value(rate, n=annotation["processed_episodes"], details={"input_episodes": annotation["input_episodes"]})


@metric(
    "planning_failure_rate",
    area="diagnostics",
    signal="failure causes",
    scope="run",
    unit="fraction of attempts",
    sample="attempts",
    better="lower",
    requires=["attempts.extras.planning_failed"],
)
def planning_failure_rate(ev: Evidence) -> Value:
    """Share of attempts whose motion planning failed, stop-corrected when the attempt log allows."""
    attempts, corrected = _counted_attempts(ev)
    failed = sum(attempt["extras"]["planning_failed"] for attempt in attempts)
    return Value(failed / len(attempts), n=len(attempts), details={"stop_corrected": corrected})


@metric(
    "planning_time",
    area="diagnostics",
    signal="generation performance",
    scope="run",
    unit="s per attempt",
    sample="attempts",
    better="lower",
    primary="p95",
    requires=["attempts.extras.planning_time_s"],
)
def planning_time(ev: Evidence) -> Value:
    """p50 and p95 motion-planning time per attempt, and its share of generation time."""
    times = np.array([attempt["extras"]["planning_time_s"] for attempt in ev.attempts])
    p50, p95 = np.percentile(times, [50, 95])
    generation_time = ev.run.get("generation_time_s")
    share = float(times.sum() / generation_time) if generation_time else None
    return Value({"p50": float(p50), "p95": float(p95)}, n=len(times), details={"share_of_generation_time": share})


@metric(
    "generation_bias",
    area="diagnostics",
    signal="failure causes",
    scope="run",
    unit="low start cells",
    sample="attempts",
    better="lower",
    requires=["attempts.initial_object_positions"],
)
def generation_bias(ev: Evidence) -> Value:
    """Start-position cells whose success rate is significantly below the overall rate.

    Each moving object's attempted starts are split into :data:`BIAS_BINS` cells per moving axis. Every cell with
    :data:`SPARSE_CELL_ATTEMPTS` or more attempts gets a one-sided exact binomial test against the overall rate,
    stop-corrected when the attempt log allows, Bonferroni-corrected at 2.5%. With nine cells tested, a cell that never
    succeeds is flagged from 10 attempts at an overall rate of 50%, and from 56 at 10%. The details name the worst
    tested cell, flagged or not.
    """
    attempts, corrected = _counted_attempts(ev)
    success = np.array([attempt["success"] for attempt in attempts])
    if not success.any():
        raise NotApplicable("no attempt succeeded")
    rate = float(success.mean())
    names = set.intersection(*(set(attempt["initial_object_positions"]) for attempt in attempts))
    if not names:
        raise NotApplicable("no object's start position is in every attempt")
    tested = []
    for name in sorted(names):
        starts = np.array([attempt["initial_object_positions"][name] for attempt in attempts])
        moving = np.ptp(starts, axis=0) >= MIN_START_EXTENT_M
        if not moving.any():
            continue
        axes = [axis for axis, moves in zip("xy", moving) if moves]
        low, extent = starts[:, moving].min(axis=0), np.ptp(starts[:, moving], axis=0)
        bins = np.clip(((starts[:, moving] - low) / extent * BIAS_BINS).astype(int), 0, BIAS_BINS - 1)
        for cell in np.unique(bins, axis=0):
            members = (bins == cell).all(axis=1)
            count, successes = int(members.sum()), int(success[members].sum())
            if count >= SPARSE_CELL_ATTEMPTS:
                p_value = binomial_cdf(successes, count, rate)
                label = dict(zip(axes, cell.tolist()))
                tested.append(
                    {"object": name, "cell": label, "attempts": count, "successes": successes, "p_value": p_value}
                )
    if not tested:
        raise NotApplicable(f"no moving object's start cell has {SPARSE_CELL_ATTEMPTS} attempts")
    low_cells = [entry for entry in tested if entry["p_value"] < 0.025 / len(tested)]
    details = {
        "success_rate": rate,
        "stop_corrected": corrected,
        "tested_cells": len(tested),
        "worst_cell": min(tested, key=lambda entry: entry["p_value"]),
        "low_cells": low_cells,
    }
    return Value(len(low_cells), n=len(attempts), details=details)
