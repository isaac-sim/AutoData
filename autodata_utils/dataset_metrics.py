# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPIs of a generated AutoData dataset, computed offline from generation and annotation outputs.

Every KPI is either a value or ``not_applicable`` with a reason; missing inputs never read as zero.
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

Z_95 = 1.959963984540054
SPARSE_CELL_ATTEMPTS = 10
"""Grid cells with fewer attempts are left out of the success-by-configuration headline."""
MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION = 3
"""Below this many completed attempts in some env, the stop correction is itself biased."""
SAMPLES_PER_COVERAGE_CELL = 5
PATH_DEVIATION_WINDOW = 20
"""Achieved steps are matched to commanded path segments at most this many steps away."""
MIN_START_EXTENT = 0.01
"""Objects whose attempted starts span less than this [m] along an axis do not vary along it."""


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


def path_deviation(achieved: np.ndarray, commanded: np.ndarray, window: int = PATH_DEVIATION_WINDOW) -> np.ndarray:
    """Distance [m] from each achieved position to the commanded path near the same step.

    Args:
        achieved: ``[T, 3]`` achieved end-effector positions [m].
        commanded: ``[T, 3]`` commanded end-effector positions [m], step-aligned with ``achieved``.
        window: Steps on either side whose commanded segments are candidates.
    """
    assert len(achieved) == len(commanded), "achieved and commanded positions must be step-aligned"
    if len(commanded) < 2:
        return np.linalg.norm(achieved - commanded, axis=1)
    starts, ends = commanded[:-1], commanded[1:]
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
        success_points: ``[S, D]`` starts of successful attempts [m], with ``S >= min_coverage_successes(D)``.
        attempt_points: ``[A, D]`` starts of all attempts [m], ``A >= S``.
        bounds: Per-dimension ``(lo, hi)`` of the attempted starts [m].
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


def embodiment_clips_pose_actions(path: str) -> bool:
    """Whether the embodiment's adapter clips the pose part (first six entries) of its action to ±1.

    Only the single-arm delta-pose IK adapter clips; the report stays torch-free, so it reads the YAML.
    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data.get("type") == "delta_pose_ik_single_arm" and bool(
        data.get("action_layout", {}).get("clip_pose_action_to_unit", True)
    )


def load_episodes(path: str) -> tuple[list[dict], dict]:
    """Per-episode actions and ``obs/datagen_info`` positions, plus the dataset's ``env_args``.

    Annotation and generation record ``obs/datagen_info`` with the same recorder term before each step, so
    source and generated data share a frame and a step.
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
                for group in ("target_eef_pose", "eef_pose"):
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


def _kpi(value: Any, unit: str, n: int, **details: Any) -> dict[str, Any]:
    return {"value": value, "unit": unit, "n": n, **details}


def _not_applicable(reason: str) -> dict[str, str]:
    return {"not_applicable": reason}


def _percentiles(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    p50, p95 = np.percentile(values, [50, 95]) * scale
    return {"p50": float(p50), "p95": float(p95)}


def attempts_before_stop(attempts: list[dict], num_envs: int) -> list[dict] | None:
    """Attempts that started before the first env completed its last logged attempt.

    Attempts running at the stop are cancelled unlogged, and long ones are likelier to be running. Each env's
    cancelled attempt started after that env's last completion, so this start-time subset was fully logged. None
    when some env completed fewer than :data:`MIN_ATTEMPTS_PER_ENV_FOR_STOP_CORRECTION` attempts.
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
    return _kpi(successes / attempts, "fraction", attempts, interval_95=[lo, hi], raw_rate=raw_rate, note=note)


def _demos_per_hour(result: dict | None) -> dict:
    run = (result or {}).get("run", {})
    if "generation_time_s" not in run:
        return _not_applicable("needs a --result_file with run timing")
    gpu, cpu = run["gpu"] or "unknown GPU", run["cpu"] or "unknown CPU"
    hardware = f"on {gpu}" if run["device"] != "cpu" else f"CPU physics on {cpu}, with {gpu}"
    return _kpi(
        result["num_success"] / (run["generation_time_s"] / 3600),
        "demos/h",
        result["num_success"],
        note=f"{hardware}, {run['num_envs']} envs; compare only on the same hardware and env count",
        generation_time_s=run["generation_time_s"],
        device=run["device"],
        gpu=run["gpu"],
        cpu=run["cpu"],
        num_envs=run["num_envs"],
    )


def _demos_per_human_demo(result: dict | None) -> dict:
    num_source = (result or {}).get("run", {}).get("num_source_episodes")
    if not num_source:
        return _not_applicable("needs a --result_file with the source episode count")
    note = "input ratio: about requested demos / source demos when the run stops at a success count"
    return _kpi(result["num_success"] / num_source, "ratio", num_source, note=note)


def _annotation_yield(source_episodes: list[dict] | None, recorded_dataset: str | None) -> dict:
    if source_episodes is None or recorded_dataset is None:
        return _not_applicable("needs --source and --recorded_dataset")
    with h5py.File(recorded_dataset, "r") as h5_file:
        num_recorded = len(h5_file["data"])
    if num_recorded == 0:
        return _not_applicable("the recorded dataset has no episodes")
    note = "assumes --source is the annotation output of --recorded_dataset"
    return _kpi(len(source_episodes) / num_recorded, "fraction", num_recorded, note=note)


def _pooled_path_deviation(episodes: list[dict]) -> np.ndarray:
    deviations = [
        path_deviation(episode["eef_pose"][eef], target)
        for episode in episodes
        if "target_eef_pose" in episode and "eef_pose" in episode
        for eef, target in episode["target_eef_pose"].items()
        # Tracking-only channels command no pose; their adapter writes an identity placeholder.
        if np.any(target)
    ]
    return np.concatenate(deviations) if deviations else np.empty(0)


def _tracking_error(episodes: list[dict], source_episodes: list[dict] | None) -> dict:
    if not episodes:
        return _not_applicable("the dataset has no episodes")
    deviations = _pooled_path_deviation(episodes)
    if not len(deviations):
        return _not_applicable("dataset has no obs/datagen_info/target_eef_pose")
    source_deviations = _pooled_path_deviation(source_episodes) if source_episodes is not None else np.empty(0)
    baseline = _percentiles(source_deviations, 1000.0) if len(source_deviations) else None
    return _kpi(
        _percentiles(deviations, 1000.0),
        "mm from the commanded path (position)",
        len(deviations),
        source_baseline=baseline,
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
        return _not_applicable("the dataset has no sim_args")
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
            else "source has no sim_args; its baseline assumes the generated control step"
        ),
    )


def _varying_starts(attempts: list[dict] | None) -> list[tuple] | str:
    """Per object, its attempted start positions along the x/y axes on which they vary, with their bounds."""
    if not attempts:
        return "needs --result_file with a non-empty attempt log"
    entries = []
    for asset in attempts[0]["initial_object_positions"]:
        starts = np.array([attempt["initial_object_positions"][asset] for attempt in attempts])
        axes = [axis for axis in range(2) if np.ptp(starts[:, axis]) >= MIN_START_EXTENT]
        if axes:
            points = starts[:, axes]
            bounds = [(float(points[:, dim].min()), float(points[:, dim].max())) for dim in range(len(axes))]
            entries.append((asset, points, bounds, ["xy"[axis] for axis in axes]))
    return entries or f"no object's start position varies by {MIN_START_EXTENT} m or more"


def _success_by_configuration(attempts: list[dict] | None, corrected: bool) -> dict:
    entries = _varying_starts(attempts)
    if isinstance(entries, str):
        return _not_applicable(entries)
    success = np.array([attempt["success"] for attempt in attempts])
    overall_rate = float(success.mean())
    if overall_rate == 0.0:
        return _not_applicable("no attempt succeeded")
    per_asset = {}
    for asset, points, bounds, axes in entries:
        cells = grid_cells(points, bounds, 3)
        table = [
            {"cell": cell, "attempts": int((cells == cell).sum()), "successes": int(success[cells == cell].sum())}
            for cell in range(3 ** len(bounds))
        ]
        per_asset[asset] = {"axes": axes, "bounds": bounds, "cells": table}
    dense = [
        entry for grid in per_asset.values() for entry in grid["cells"] if entry["attempts"] >= SPARSE_CELL_ATTEMPTS
    ]
    if not dense:
        return _not_applicable(f"no grid cell has {SPARSE_CELL_ATTEMPTS} or more attempts")
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
    )


def _initial_state_coverage(attempts: list[dict] | None) -> dict:
    entries = _varying_starts(attempts)
    if isinstance(entries, str):
        return _not_applicable(entries)
    success = np.array([attempt["success"] for attempt in attempts])
    per_asset, skipped = {}, []
    for asset, points, bounds, axes in entries:
        if success.sum() < min_coverage_successes(len(bounds)):
            skipped.append(f"{asset} needs {min_coverage_successes(len(bounds))} successful attempts")
            continue
        per_asset[asset] = {"axes": axes, "bounds": bounds, **coverage_ratio(points[success], points, bounds)}
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
    by_cause = Counter(attempt["failure_cause"] for attempt in failures)
    return _kpi(
        {cause: count / len(failures) for cause, count in by_cause.items()},
        "fraction of failures",
        len(failures),
        note=note,
    )


def _planning_time(result: dict | None, attempts: list[dict] | None) -> dict:
    planning = np.array([attempt["planning_time_s"] for attempt in attempts or [] if attempt["planning_time_s"] > 0])
    if not len(planning):
        return _not_applicable("the run did no motion planning")
    generation_time = (result or {}).get("run", {}).get("generation_time_s")
    share = float(planning.sum() / generation_time) if generation_time else None
    return _kpi(_percentiles(planning), "s per attempt", len(planning), share_of_generation_time=share)


def build_report(
    dataset: str,
    result_file: str | None = None,
    source: str | None = None,
    embodiment: str | None = None,
    recorded_dataset: str | None = None,
) -> dict[str, Any]:
    """Compute every KPI for a generated dataset.

    Args:
        dataset: Generated HDF5 dataset holding the run's successful demos.
        result_file: Generation result JSON (``--result_file`` of ``generate_dataset.py``).
        source: Annotated source HDF5 dataset the generation started from; enables source baselines.
        embodiment: Embodiment YAML used for generation; needed for action saturation.
        recorded_dataset: Teleoperated HDF5 dataset that was annotated into ``source``; enables annotation yield.
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

    kpis = {
        "success_rate": _success_rate(result, stop_attempts),
        "demos_per_hour": _demos_per_hour(result),
        "demos_per_human_demo": _demos_per_human_demo(result),
        "annotation_yield": _annotation_yield(source_episodes, recorded_dataset),
        "tracking_error": _tracking_error(episodes, source_episodes),
        "action_saturation": _action_saturation(
            episodes, source_episodes, embodiment_clips_pose_actions(embodiment) if embodiment else None
        ),
        "motion_smoothness": _motion_smoothness(
            episodes, _step_dt_from_env_args(env_args), source_episodes, _step_dt_from_env_args(source_env_args)
        ),
        "initial_state_coverage": _initial_state_coverage(stop_attempts or attempts),
        "success_by_initial_configuration": _success_by_configuration(
            stop_attempts or attempts, corrected=stop_attempts is not None
        ),
        "failure_share": _failure_share(attempts, stop_attempts),
        "planning_time": _planning_time(result, attempts),
    }
    return {"dataset": dataset, "num_episodes": len(episodes), "kpis": kpis}


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
