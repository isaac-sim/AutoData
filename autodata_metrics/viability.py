# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viability metrics: what producing the dataset cost, and diagnostics of why attempts failed."""

from __future__ import annotations

import math
import numpy as np
from collections import Counter
from typing import Any

from autodata_metrics.evidence import Evidence
from autodata_metrics.registry import NotApplicable, Value, metric

Z_95 = 1.959963984540054
MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION = 3
"""Below this many logged attempts in some env, the stop correction is itself biased."""


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    assert trials > 0, "trials must be positive"
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


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
    exported = ev.env_args["autodata"]["exported_episodes"]
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
    """Wall time of the automated steps, each process timed from its start: annotating the source and generating."""
    annotation = ev.source.env_args.get("autodata", {}).get("annotation", {}) if ev.source else {}
    manual = annotation.get("mode") == "manual"
    parts = {"setup_s": ev.run["setup_time_s"], "generation_s": ev.run["generation_time_s"]}
    if annotation:
        parts["annotation_s"] = annotation.get("setup_time_s", 0.0) + (0.0 if manual else annotation["time_s"])
    details = {**parts, "manual_annotation_s": annotation["time_s"] if manual else None}
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
    annotation = ev.source.env_args["autodata"]["annotation"]
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
