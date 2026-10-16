# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPIs of a generated AutoData dataset.

The report reads the generated HDF5, the generation result JSON and its attempt log, the source
dataset, the embodiment YAML and an optional annotation summary. It needs only h5py, numpy and PyYAML,
so it runs without the simulator. Every KPI is either a value or ``not_applicable`` with a reason;
missing inputs never read as zero.
"""

from __future__ import annotations

import h5py
import json
import math
import numpy as np
import yaml
from collections import Counter
from pathlib import Path
from typing import Any

from autodata_utils.generation_result import file_sha256

REPORT_SCHEMA_VERSION = 1
Z_95 = 1.959963984540054
"""Two-sided 95% standard-normal quantile."""

SPARSE_CELL_ATTEMPTS = 10
"""Grid cells with fewer attempts are left out of the success-by-configuration headline."""

MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION = 3
"""Below this many completed attempts in some env, the correction for attempts cancelled at the stop is itself biased."""

SAMPLES_PER_COVERAGE_CELL = 5
"""The coverage grid is sized so each cell holds about this many successful starts."""

PATH_DEVIATION_WINDOW = 20
"""Achieved steps are matched to intended path segments at most this many steps away."""

RANGE_TOLERANCE = 1e-3
"""Starts farther than this [m] outside their reset range mean the range frame was resolved wrongly."""

# Isaac Lab reset functions whose pose_range semantics are known.
_OFFSET_RANGE_FUNCS = ("reset_root_state_uniform",)
_ABSOLUTE_RANGE_FUNCS = ("randomize_object_pose",)

# Whether each embodiment adapter type clips the pose part (first six entries) of its action. The report
# stays torch-free, so it cannot import the adapters themselves.
_CLIPPING_BY_EMBODIMENT_TYPE = {
    "absolute_pose": False,
    "delta_pose_ik_single_arm": True,
    "absolute_pose_whole_body_bimanual": False,
}


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------


def binomial_cdf(successes: int, trials: int, rate: float) -> float:
    """Probability of at most ``successes`` successes in ``trials`` Bernoulli trials with success ``rate``."""
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


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    assert trials > 0, "trials must be positive"
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def eef_jerk(positions: np.ndarray, step_dt: float) -> np.ndarray:
    """Linear jerk magnitude [m/s³] along an end-effector position trajectory.

    Args:
        positions: ``[T, 3]`` positions [m] sampled every ``step_dt``.
        step_dt: Control step [s].
    """
    if len(positions) < 4:
        return np.empty(0)
    return np.linalg.norm(np.diff(positions, n=3, axis=0), axis=1) / step_dt**3


def saturated_fraction(pose_actions: np.ndarray) -> float:
    """Share of pose-action entries clipped to ±1."""
    return float(np.mean(np.isclose(np.abs(pose_actions), 1.0, rtol=0.0, atol=1e-6)))


def path_deviation(achieved: np.ndarray, intended: np.ndarray, window: int = PATH_DEVIATION_WINDOW) -> np.ndarray:
    """Distance [m] from each achieved position to the intended path near the same step.

    Args:
        achieved: ``[T, 3]`` achieved end-effector positions [m].
        intended: ``[T, 3]`` intended end-effector positions [m], step-aligned with ``achieved``.
        window: Steps on either side whose intended segments are candidates.
    """
    assert len(achieved) == len(intended), "achieved and intended positions must be step-aligned"
    if len(intended) < 2:
        return np.linalg.norm(achieved - intended, axis=1)
    starts, ends = intended[:-1], intended[1:]
    deviations = np.empty(len(achieved))
    for step, point in enumerate(achieved):
        lo, hi = max(0, step - window), min(len(starts), step + window + 1)
        segment_start, segment = starts[lo:hi], ends[lo:hi] - starts[lo:hi]
        length_sq = np.einsum("ij,ij->i", segment, segment)
        t = np.einsum("ij,ij->i", point - segment_start, segment) / np.where(length_sq > 0, length_sq, 1.0)
        closest = segment_start + np.clip(t, 0.0, 1.0)[:, None] * segment
        deviations[step] = np.linalg.norm(closest - point, axis=1).min()
    return deviations


def grid_cells(points: np.ndarray, bounds: list[tuple[float, float]], bins: int) -> np.ndarray:
    """Flat cell index of each point on a ``bins``-per-dimension grid over ``bounds``."""
    index = np.zeros(len(points), dtype=int)
    for dim, (lo, hi) in enumerate(bounds):
        cell = np.clip(((points[:, dim] - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
        index = index * bins + cell
    return index


def min_coverage_successes(num_dims: int) -> int:
    """Fewest successful starts that give the coverage grid at least two bins per dimension."""
    return SAMPLES_PER_COVERAGE_CELL * 2**num_dims


def coverage_ratio(
    success_points: np.ndarray, attempt_points: np.ndarray, bounds: list[tuple[float, float]], seed: int = 0
) -> dict[str, float]:
    """Cells occupied by successful starts over cells occupied by as many attempted starts.

    The grid is sized to about :data:`SAMPLES_PER_COVERAGE_CELL` successes per cell, so raw sample count
    does not drive the ratio.

    Args:
        success_points: ``[S, D]`` starts of successful attempts, in range coordinates, with
            ``S >= min_coverage_successes(D)``.
        attempt_points: ``[A, D]`` starts of all attempts, ``A >= S``.
        bounds: Per-dimension ``(lo, hi)`` of the reset range.
        seed: Seed of the random attempt subsets.
    """
    num_success = len(success_points)
    assert num_success >= min_coverage_successes(len(bounds)), "too few successes for a two-bin grid"
    rng = np.random.default_rng(seed)
    bins = round((num_success / SAMPLES_PER_COVERAGE_CELL) ** (1 / len(bounds)))

    def occupied(points: np.ndarray) -> int:
        return len(np.unique(grid_cells(points, bounds, bins)))

    attempt_occupancy = np.mean(
        [occupied(attempt_points[rng.choice(len(attempt_points), num_success, replace=False)]) for _ in range(50)]
    )
    return {"ratio": occupied(success_points) / attempt_occupancy, "bins_per_dim": bins}


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def embodiment_clips_pose_actions(path: str) -> bool:
    """Whether the embodiment's adapter clips the pose part (first six entries) of its action to ±1."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    embodiment_type = data.get("type")
    assert (
        embodiment_type in _CLIPPING_BY_EMBODIMENT_TYPE
    ), f"Embodiment type {embodiment_type!r} is unknown to the KPI report; add it to _CLIPPING_BY_EMBODIMENT_TYPE."
    return _CLIPPING_BY_EMBODIMENT_TYPE[embodiment_type] and bool(
        data.get("action_layout", {}).get("clip_pose_action_to_unit", True)
    )


def load_episodes(path: str) -> tuple[list[dict], dict]:
    """Per-episode arrays needed by the KPIs, plus the dataset's ``env_args``.

    End-effector positions (``eef_pos``, ``[T, E, 3]``) come from ``obs/datagen_info/eef_pose``, the control-link
    pose that annotation and generation both record before each step, so source and generated data share a frame
    and a step.
    """
    episodes = []
    with h5py.File(path, "r") as h5_file:
        data = h5_file["data"]
        env_args_raw = data.attrs.get("env_args", "{}")
        env_args = json.loads(env_args_raw.decode() if isinstance(env_args_raw, bytes) else env_args_raw)
        for name in data:
            demo = data[name]
            episode: dict[str, Any] = {"actions": demo["actions"][:]}
            datagen_info = demo["obs"].get("datagen_info")
            if datagen_info is not None:
                for group in ("intended_eef_pose", "target_eef_pose", "eef_pose"):
                    if group in datagen_info:
                        episode[group] = {eef: datagen_info[group][eef][:][:, :3, 3] for eef in datagen_info[group]}
                if "eef_pose" in episode:
                    episode["eef_pos"] = np.stack([episode["eef_pose"][eef] for eef in sorted(episode["eef_pose"])], 1)
            episodes.append(episode)
    return episodes, env_args


def _step_dt_from_env_args(env_args: dict) -> float | None:
    sim_args = env_args.get("sim_args") or {}
    if "dt" in sim_args and "decimation" in sim_args:
        return float(sim_args["dt"]) * int(sim_args["decimation"])
    return None


def _read_attempts(result_file: str, result: dict) -> list[dict] | None:
    attempts_file = result.get("run", {}).get("attempts_file")
    if attempts_file is None:
        return None
    path = Path(result_file).parent / attempts_file
    assert path.exists(), f"{result_file} names the attempt log {path}, which is missing."
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ----------------------------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------------------------


def _kpi(value: Any, unit: str, n: int, **details: Any) -> dict[str, Any]:
    return {"value": value, "unit": unit, "n": n, **details}


def _not_applicable(reason: str) -> dict[str, str]:
    return {"not_applicable": reason}


def _percentiles(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    p50, p95 = np.percentile(values, [50, 95]) * scale
    return {"p50": float(p50), "p95": float(p95)}


def attempts_before_stop(attempts: list[dict], num_envs: int) -> list[dict] | None:
    """Attempts that started before the first env completed its last logged attempt.

    Attempts still running when generation stops are cancelled and never logged, and long attempts are the likelier
    to be running, so the full log under-represents long outcomes. Each env's cancelled attempt started at or after
    that env's last completion, so every attempt that started before the earliest such completion was logged. This
    subset is selected by start time, not duration. None when some env completed fewer than
    :data:`MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION` attempts, where neither the subset nor the full log is unbiased.
    """
    last_end: dict[int, float] = {}
    completed: Counter[int] = Counter()
    for attempt in attempts:
        last_end[attempt["env_id"]] = max(last_end.get(attempt["env_id"], 0.0), attempt["end_time_s"])
        completed[attempt["env_id"]] += 1
    if len(completed) < num_envs or min(completed.values()) < MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION:
        return None
    cutoff = min(last_end.values())
    return [attempt for attempt in attempts if attempt["start_time_s"] < cutoff]


_UNCORRECTED_NOTE = (
    "uncorrected: attempts cancelled at the stop can bias it (correcting needs an attempt log with "
    f"{MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION}+ completed attempts from every env)"
)
_CORRECTED_NOTE = "counts only attempts that started before the first env's last completion"


def _success_rate(result: dict | None, stop_attempts: list[dict] | None) -> dict:
    if result is None:
        return _not_applicable("needs --result_file")
    successes, attempts = result["num_success"], result["num_attempts"]
    if attempts == 0:
        return _not_applicable("the run made no attempts")
    raw_rate = successes / attempts
    note = _UNCORRECTED_NOTE
    if stop_attempts is not None:
        successes, attempts = sum(attempt["success"] for attempt in stop_attempts), len(stop_attempts)
        note = _CORRECTED_NOTE
    lo, hi = wilson_interval(successes, attempts)
    return _kpi(
        successes / attempts,
        "fraction",
        attempts,
        interval_95=[lo, hi],
        raw_rate=raw_rate,
        stop_rule=result.get("run", {}).get("stop_rule"),
        note=note,
    )


def _demos_per_hour(result: dict | None) -> dict:
    run = (result or {}).get("run", {})
    if "generation_time_s" not in run:
        return _not_applicable("needs a --result_file with run timing")
    return _kpi(
        result["num_success"] / (run["generation_time_s"] / 3600),
        "demos/h",
        result["num_success"],
        generation_time_s=run["generation_time_s"],
        setup_time_s=run["setup_time_s"],
        device=run["device"],
        num_envs=run["num_envs"],
    )


def _demos_per_human_demo(result: dict | None, source_episodes: list[dict] | None) -> dict:
    if result is None:
        return _not_applicable("needs --result_file")
    num_source = result.get("run", {}).get("source", {}).get("num_episodes")
    if num_source is None and source_episodes is not None:
        num_source = len(source_episodes)
    if not num_source:
        return _not_applicable("needs the source episode count (--result_file run metadata or --source)")
    note = "input ratio: about requested demos / source demos when the run stops at a success count"
    return _kpi(result["num_success"] / num_source, "ratio", num_source, note=note)


def _annotation_yield(annotation_summary: dict | None) -> dict:
    if annotation_summary is None:
        return _not_applicable("needs --annotation_summary")
    processed = annotation_summary["num_processed"]
    if processed == 0:
        return _not_applicable("no episodes were processed")
    return _kpi(
        annotation_summary["num_exported"] / processed,
        "fraction",
        processed,
        rejections=annotation_summary["rejections"],
        manual_remarks=annotation_summary["manual_remarks"],
        mode=annotation_summary["mode"],
    )


def _pooled_path_deviation(episodes: list[dict], intended_key: str, eefs: set[str] | None = None) -> np.ndarray:
    deviations = [
        path_deviation(episode["eef_pose"][eef], episode[intended_key][eef])
        for episode in episodes
        if intended_key in episode and "eef_pose" in episode
        for eef in episode[intended_key]
        if eefs is None or eef in eefs
    ]
    return np.concatenate(deviations) if deviations else np.empty(0)


def _tracking_error(episodes: list[dict], source_episodes: list[dict] | None) -> dict:
    if not episodes:
        return _not_applicable("the dataset has no episodes")
    deviations = _pooled_path_deviation(episodes, "intended_eef_pose")
    if not len(deviations):
        return _not_applicable("dataset has no obs/datagen_info/intended_eef_pose")
    baseline = None
    if source_episodes is not None:
        # Generation records intended poses only for commanded EEFs; the source's target poses include
        # placeholders for tracking-only channels.
        commanded = {eef for episode in episodes for eef in episode.get("intended_eef_pose", {})}
        source_deviations = _pooled_path_deviation(source_episodes, "target_eef_pose", commanded)
        baseline = _percentiles(source_deviations, 1000.0) if len(source_deviations) else None
    return _kpi(
        _percentiles(deviations, 1000.0),
        "mm from the noise-free intended path (position)",
        len(deviations),
        source_baseline=baseline,
        note="generated values include injected action noise; the source baseline is against executed commands",
    )


def _action_saturation(episodes: list[dict], source_episodes: list[dict] | None, clips: bool | None) -> dict:
    if clips is None:
        return _not_applicable("needs --embodiment")
    if not clips:
        return _not_applicable("the embodiment adapter does not clip pose actions")
    if not episodes:
        return _not_applicable("the dataset has no episodes")
    pose_actions = np.concatenate([episode["actions"][:, :6] for episode in episodes])
    baseline = None
    if source_episodes is not None:
        baseline = saturated_fraction(np.concatenate([episode["actions"][:, :6] for episode in source_episodes]))
    return _kpi(saturated_fraction(pose_actions), "fraction", pose_actions.size, source_baseline=baseline)


def _pooled_jerk(episodes: list[dict], step_dt: float) -> np.ndarray:
    jerks = [
        eef_jerk(episode["eef_pos"][:, eef], step_dt)
        for episode in episodes
        if "eef_pos" in episode
        for eef in range(episode["eef_pos"].shape[1])
    ]
    return np.concatenate(jerks) if jerks else np.empty(0)


def _motion_smoothness(
    episodes: list[dict],
    step_dt: float | None,
    source_episodes: list[dict] | None,
    source_step_dt: float | None,
) -> dict:
    if step_dt is None:
        return _not_applicable("no step_dt in the result file or dataset sim_args")
    if not episodes:
        return _not_applicable("the dataset has no episodes")
    jerk = _pooled_jerk(episodes, step_dt)
    if not len(jerk):
        return _not_applicable("dataset has no obs/datagen_info/eef_pose")
    baseline = None
    if source_episodes is not None:
        source_jerk = _pooled_jerk(source_episodes, source_step_dt or step_dt)
        baseline = _percentiles(source_jerk) if len(source_jerk) else None
    return _kpi(
        _percentiles(jerk),
        "m/s^3 end-effector jerk",
        len(jerk),
        source_baseline=baseline,
        note=(
            None
            if source_episodes is None or source_step_dt is not None
            else "source has no sim_args; its baseline assumes the generated step_dt"
        ),
    )


def _range_points(reset_range: dict, asset: str, attempts: list[dict]) -> tuple | str:
    """Starts of ``asset`` in reset-range coordinates over the range's non-zero-width x/y axes."""
    func_name = reset_range["func"].rsplit(":", 1)[-1]
    if func_name in _OFFSET_RANGE_FUNCS:
        origin = reset_range["default_positions"].get(asset)
        if origin is None:
            return f"no default position recorded for {asset}"
    elif func_name in _ABSOLUTE_RANGE_FUNCS:
        origin = [0.0, 0.0, 0.0]
    else:
        return f"unknown pose_range semantics of {reset_range['func']}"
    axes = [
        (index, axis)
        for index, axis in enumerate(("x", "y"))
        if axis in reset_range["pose_range"] and reset_range["pose_range"][axis][1] > reset_range["pose_range"][axis][0]
    ]
    if not axes:
        return f"{asset} has no x/y variation"
    starts = np.array([attempt["initial_object_poses"][asset][:2] for attempt in attempts]) - np.array(origin[:2])
    points = starts[:, [index for index, _ in axes]]
    bounds = [tuple(reset_range["pose_range"][axis]) for _, axis in axes]
    in_range = np.all(
        [
            (points[:, dim] >= lo - RANGE_TOLERANCE) & (points[:, dim] <= hi + RANGE_TOLERANCE)
            for dim, (lo, hi) in enumerate(bounds)
        ],
        axis=0,
    )
    if not in_range.all():
        return f"only {in_range.mean():.0%} of {asset} starts lie inside its reset range, so its frame is wrong"
    return points, bounds, [axis for _, axis in axes]


def _configuration_inputs(result: dict | None, attempts: list[dict] | None) -> tuple[list, list[str]] | str:
    if result is None or attempts is None:
        return "needs --result_file with an attempt log"
    if not attempts:
        return "the attempt log is empty"
    reset_ranges = result.get("run", {}).get("reset_ranges", [])
    entries, skipped = [], []
    for reset_range in reset_ranges:
        for asset in reset_range["assets"]:
            if not all(asset in attempt["initial_object_poses"] for attempt in attempts):
                skipped.append(f"{asset} is not a rigid object")
                continue
            resolved = _range_points(reset_range, asset, attempts)
            if isinstance(resolved, str):
                skipped.append(resolved)
            else:
                entries.append((asset, *resolved))
    if not entries:
        return "no reset range with x/y variation" + (f" ({'; '.join(skipped)})" if skipped else "")
    return entries, skipped


def _success_by_configuration(result: dict | None, attempts: list[dict] | None, corrected: bool) -> dict:
    resolved = _configuration_inputs(result, attempts)
    if isinstance(resolved, str):
        return _not_applicable(resolved)
    entries, skipped = resolved
    success = np.array([attempt["success"] for attempt in attempts])
    overall_rate = float(success.mean())
    if overall_rate == 0.0:
        return _not_applicable("no attempt succeeded")
    per_asset = {}
    for asset, points, bounds, axes in entries:
        cells = grid_cells(points, bounds, 3)
        table = []
        for cell in range(3 ** len(bounds)):
            in_cell = cells == cell
            n = int(in_cell.sum())
            table.append({"cell": cell, "attempts": n, "successes": int(success[in_cell].sum())})
        per_asset[asset] = {"axes": axes, "range": [list(bound) for bound in bounds], "cells": table}
    dense = [
        entry for grid in per_asset.values() for entry in grid["cells"] if entry["attempts"] >= SPARSE_CELL_ATTEMPTS
    ]
    if not dense:
        reasons = [f"no grid cell has {SPARSE_CELL_ATTEMPTS} or more attempts", *skipped]
        return _not_applicable("; ".join(reasons))
    for entry in dense:
        entry["rate"] = entry["successes"] / entry["attempts"]
        # One-sided exact test: how likely so few successes are if the cell had the overall rate.
        entry["p_value"] = binomial_cdf(entry["successes"], entry["attempts"], overall_rate)
    rate_label = "corrected" if corrected else "uncorrected"
    return _kpi(
        sum(entry["p_value"] < 0.025 / len(dense) for entry in dense),
        "cells significantly below the overall success rate",
        len(attempts),
        note=(
            f"of {len(dense)} cells with {SPARSE_CELL_ATTEMPTS}+ attempts; "
            f"overall success rate {overall_rate:.3f} ({rate_label})"
        ),
        overall_rate=overall_rate,
        per_asset=per_asset,
        skipped=skipped,
    )


def _initial_state_coverage(result: dict | None, attempts: list[dict] | None) -> dict:
    resolved = _configuration_inputs(result, attempts)
    if isinstance(resolved, str):
        return _not_applicable(resolved)
    entries, skipped = resolved
    success = np.array([attempt["success"] for attempt in attempts])
    per_asset = {}
    for asset, points, bounds, axes in entries:
        if success.sum() < min_coverage_successes(len(bounds)):
            skipped.append(f"{asset} needs {min_coverage_successes(len(bounds))} successful attempts")
            continue
        per_asset[asset] = {
            "axes": axes,
            "range": [list(bound) for bound in bounds],
            **coverage_ratio(points[success], points, bounds),
        }
    if not per_asset:
        return _not_applicable("; ".join(skipped))
    worst = min(entry["ratio"] for entry in per_asset.values())
    return _kpi(worst, "worst per-object coverage ratio", int(success.sum()), per_asset=per_asset, skipped=skipped)


def _failure_share(attempts: list[dict] | None, stop_attempts: list[dict] | None) -> dict:
    if attempts is None:
        return _not_applicable("needs --result_file with an attempt log")
    note = _UNCORRECTED_NOTE
    if stop_attempts is not None:
        attempts = stop_attempts
        note = _CORRECTED_NOTE
    failures = [attempt for attempt in attempts if not attempt["success"]]
    if not failures:
        return _kpi({}, "fraction of failures", 0, note=note)
    by_cause = Counter(attempt["failure_cause"] or "unknown" for attempt in failures)
    by_subtask: dict[str, dict[str, dict[str, int]]] = {}
    for attempt in failures:
        for eef, subtask in (attempt["failed_subtask"] or {}).items():
            cause_table = by_subtask.setdefault(attempt["failure_cause"] or "unknown", {}).setdefault(eef, {})
            cause_table[str(subtask)] = cause_table.get(str(subtask), 0) + 1
    return _kpi(
        {cause: count / len(failures) for cause, count in by_cause.items()},
        "fraction of failures",
        len(failures),
        by_subtask=by_subtask,
        note=note,
    )


def _planner_calls(attempts: list[dict] | None) -> list[dict]:
    return [call for attempt in attempts or [] for call in attempt.get("planner_calls", [])]


def _planner_failure_types(attempts: list[dict] | None) -> dict:
    calls = _planner_calls(attempts)
    if not calls:
        return _not_applicable("no motion-planner calls in the attempt log")
    failed = [call for call in calls if not call["success"]]
    return _kpi(
        {status: count / len(failed) for status, count in Counter(call["status"] for call in failed).items()},
        "fraction of failed plans",
        len(failed),
        num_calls=len(calls),
    )


def _planning_time(result: dict | None, attempts: list[dict] | None) -> dict:
    calls = _planner_calls(attempts)
    if not calls:
        return _not_applicable("no motion-planner calls in the attempt log")
    per_attempt = np.array([
        sum(call["wall_time_s"] for call in attempt["planner_calls"])
        for attempt in attempts
        if attempt["planner_calls"]
    ])
    generation_time = (result or {}).get("run", {}).get("generation_time_s")
    share = float(per_attempt.sum() / generation_time) if generation_time else None
    return _kpi(_percentiles(per_attempt), "s per attempt", len(per_attempt), share_of_generation_time=share)


def build_report(
    dataset: str,
    result_file: str | None = None,
    source: str | None = None,
    embodiment: str | None = None,
    annotation_summary: str | None = None,
) -> dict[str, Any]:
    """Compute every KPI for a generated dataset.

    Args:
        dataset: Generated HDF5 dataset holding the run's successful demos.
        result_file: Generation result JSON (``--result_file`` of ``generate_dataset.py``).
        source: Annotated source HDF5 dataset the generation started from; enables source baselines.
        embodiment: Embodiment YAML used for generation; needed for action saturation.
        annotation_summary: Annotation summary JSON (``--summary_file`` of ``annotate_demos.py``).
    """
    episodes, env_args = load_episodes(dataset)
    source_episodes, source_env_args = load_episodes(source) if source else (None, {})
    result = None
    attempts = None
    stop_attempts = None
    if result_file:
        with open(result_file, encoding="utf-8") as handle:
            result = json.load(handle)
        assert len(episodes) == result["num_success"], (
            f"{dataset} holds {len(episodes)} demos but {result_file} counts {result['num_success']} successes;"
            " they come from different runs."
        )
        attempts = _read_attempts(result_file, result)
        assert attempts is None or len(attempts) == result["num_attempts"], (
            f"The attempt log has {len(attempts)} lines but {result_file} counts {result['num_attempts']} attempts;"
            " they come from different runs."
        )
        num_envs = result.get("run", {}).get("num_envs")
        if attempts and num_envs:
            stop_attempts = attempts_before_stop(attempts, num_envs)
        recorded_source = result.get("run", {}).get("source", {})
        assert source is None or recorded_source.get("sha256") in (
            None,
            file_sha256(source),
        ), f"--source {source} is not the dataset this run generated from ({recorded_source.get('path')})."
    annotation = None
    if annotation_summary:
        with open(annotation_summary, encoding="utf-8") as handle:
            annotation = json.load(handle)

    step_dt = (result or {}).get("run", {}).get("step_dt") or _step_dt_from_env_args(env_args)
    source_step_dt = _step_dt_from_env_args(source_env_args)
    kpis = {
        "success_rate": _success_rate(result, stop_attempts),
        "demos_per_hour": _demos_per_hour(result),
        "demos_per_human_demo": _demos_per_human_demo(result, source_episodes),
        "annotation_yield": _annotation_yield(annotation),
        "tracking_error": _tracking_error(episodes, source_episodes),
        "action_saturation": _action_saturation(
            episodes, source_episodes, embodiment_clips_pose_actions(embodiment) if embodiment else None
        ),
        "motion_smoothness": _motion_smoothness(episodes, step_dt, source_episodes, source_step_dt),
        "initial_state_coverage": _initial_state_coverage(result, stop_attempts or attempts),
        "success_by_initial_configuration": _success_by_configuration(
            result, stop_attempts or attempts, corrected=stop_attempts is not None
        ),
        "failure_share": _failure_share(attempts, stop_attempts),
        "planner_failure_types": _planner_failure_types(attempts),
        "planning_time": _planning_time(result, attempts),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": dataset,
        "num_episodes": len(episodes),
        "num_source_episodes": len(source_episodes) if source_episodes is not None else None,
        "kpis": kpis,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items()) or "none"
    return str(value)


def format_report(report: dict[str, Any]) -> str:
    """Render the report as an aligned text table."""
    rows = []
    for name, kpi in report["kpis"].items():
        if "not_applicable" in kpi:
            rows.append((name, "n/a", "", "", kpi["not_applicable"]))
            continue
        details = []
        if "interval_95" in kpi:
            details.append(f"95% CI {kpi['interval_95'][0]:.3f}-{kpi['interval_95'][1]:.3f}")
        if kpi.get("source_baseline") is not None:
            details.append(f"source {_format_value(kpi['source_baseline'])}")
        if kpi.get("note"):
            details.append(kpi["note"])
        rows.append((name, _format_value(kpi["value"]), kpi["unit"], str(kpi["n"]), "; ".join(details)))
    headers = ("KPI", "Value", "Unit", "n", "Baseline / note")
    widths = [max(len(row[col]) for row in [headers, *rows]) for col in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [headers, *rows]]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)
