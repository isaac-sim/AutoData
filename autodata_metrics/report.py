# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the registered metrics of a dataset and its run, render them, and check them against a gate."""

from __future__ import annotations

import contextlib
import h5py
import json
import math
import numbers
import yaml
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autodata_metrics.evidence import Evidence, read_env_args
from autodata_metrics.registry import AREAS, REGISTERED_CHECKS, REGISTERED_METRICS, CheckSpec, MetricSpec, NotApplicable
from autodata_utils.generation_result import attempts_file_for, code_revision, input_record

SCHEMA_VERSION = 1


def read_run(
    result_file: str, run_id: str | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, list[str]]:
    """The generation result and attempt log of run ``run_id``, each None when missing, and any warnings.

    A rerun with the same result file leaves the earlier result in place until it completes, so a result from
    another run is set aside and the run is reported from its own attempt log.
    """
    result_path = Path(result_file)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    warnings = []
    if result is not None and run_id is not None and result.get("run", {}).get("run_id") != run_id:
        warnings.append(f"{result_file} belongs to another run, so the run is reported from its attempt log")
        result = None
    if result is not None and "attempts_file" in result.get("run", {}):
        attempts_path = result_path.parent / result["run"]["attempts_file"]
    elif run_id is not None:
        attempts_path = Path(attempts_file_for(result_file, run_id))
    else:
        return result, None, warnings
    attempts = None
    if attempts_path.exists():
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
        attempts = [json.loads(line) for line in lines if line.strip()]
    return result, attempts, warnings


def check_same_run(evidence: Evidence) -> list[str]:
    """Refuse a dataset and run record that come from different runs; return warnings for unverifiable data.

    Run ids alone are not enough: merging datasets keeps the first file's ``env_args``, so the exported episode
    count must match too.

    Raises:
        ValueError: If the dataset, result and attempt log disagree on their run or their counts.
    """
    num_episodes = len(evidence.all_episodes)
    result = evidence.result
    if result is not None and num_episodes != result["num_success"]:
        raise ValueError(
            f"The dataset holds {num_episodes} episodes but the run counts {result['num_success']} successes."
        )
    if result is not None and evidence.attempts is not None:
        logged = len(evidence.attempts), sum(line["success"] for line in evidence.attempts)
        if logged != (result["num_attempts"], result["num_success"]):
            raise ValueError(
                f"The attempt log holds {logged[0]} attempts and {logged[1]} successes but the result counts"
                f" {result['num_attempts']} and {result['num_success']}."
            )
    entry = evidence.env_args.get("autodata")
    if entry is None:
        return ["the dataset has no autodata entry, so it is matched to its run by episode count only"]
    if entry["exported_episodes"] != num_episodes:
        raise ValueError(
            f"The dataset holds {num_episodes} episodes but its run exported {entry['exported_episodes']};"
            " it was merged or edited after export."
        )
    run_ids = {entry["run_id"]} | {line["run_id"] for line in evidence.attempts or []}
    if "run_id" in evidence.run:
        run_ids.add(evidence.run["run_id"])
    if len(run_ids) != 1:
        raise ValueError(f"The dataset, result and attempt log come from different runs: {sorted(run_ids)}")
    return []


def run_checks(evidence: Evidence, checks: Mapping[str, CheckSpec]) -> None:
    """Run the episode checks and keep only the episodes that pass all of them in ``evidence.episodes``."""
    for spec in checks.values():
        if any(not evidence.has(key) for key in spec.requires):
            continue
        failed: dict[str, str] = {}
        applied = False
        for episode in evidence.all_episodes:
            try:
                reason = spec.check(episode, evidence)
            except NotApplicable:
                continue
            except Exception as exc:  # A broken check fails its episodes visibly instead of sinking the report.
                reason = f"{type(exc).__name__}: {exc}"
            applied = True
            if reason is not None:
                failed[episode.name] = reason
        if applied:
            evidence.failures[spec.name] = failed
            evidence.check_groups[spec.name] = spec.group
    failing = {name for failed in evidence.failures.values() for name in failed}
    evidence.episodes = [episode for episode in evidence.all_episodes if episode.name not in failing]


def evaluate(spec: MetricSpec, evidence: Evidence) -> dict[str, Any]:
    """One report row: the metric's value, or why it is not applicable or failed."""
    row = {
        "area": spec.area,
        "signal": spec.signal,
        "scope": spec.scope,
        "unit": spec.unit,
        "better": spec.better,
        "primary": spec.primary,
        "version": spec.version,
    }
    if spec.scope == "dataset" and not evidence.episodes:
        return {**row, "status": "not_applicable", "reason": "no valid episodes"}
    missing = next((key for key in spec.requires if not evidence.has(key)), None)
    if missing is not None:
        return {**row, "status": "not_applicable", "reason": f"missing {missing}"}
    try:
        value = spec.compute(evidence)
    except NotApplicable as exc:
        return {**row, "status": "not_applicable", "reason": str(exc)}
    except Exception as exc:  # A failing metric must not sink the report of every other metric.
        return {**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    if not _is_finite(value.value):
        return {**row, "status": "error", "reason": f"non-finite value {value.value}"}
    row = {**row, "status": "ok", "value": value.value, "n": value.n, "details": _finite_or_none(value.details)}
    if spec.baseline and evidence.source is not None:
        row["baseline"] = evaluate(spec, evidence.source).get("value")
    return row


def _is_finite(value: Any) -> bool:
    """Whether ``value`` is a finite number, or a mapping of finite numbers."""
    if isinstance(value, Mapping):
        return all(_is_finite(item) for item in value.values())
    return isinstance(value, numbers.Real) and math.isfinite(value)


def _finite_or_none(value: Any) -> Any:
    """``value`` with every non-finite number replaced by None, so the report stays valid JSON."""
    value = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, Mapping):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_or_none(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def build_report(
    dataset: str,
    result_file: str | None = None,
    source: str | None = None,
    metrics: Mapping[str, MetricSpec] | None = None,
    checks: Mapping[str, CheckSpec] | None = None,
) -> dict[str, Any]:
    """Compute every metric for a dataset and, when ``result_file`` is given, its generation run.

    Args:
        dataset: HDF5 dataset.
        result_file: Generation result (``generate_dataset.py --result_file``); its attempt log is read beside it.
        source: Source dataset the run started from.
        metrics: Metrics to compute; defaults to every registered metric.
        checks: Episode checks to run first; defaults to every registered check.

    Raises:
        ValueError: If the dataset, result, attempt log and source do not all come from one run.
    """
    with contextlib.ExitStack() as stack:
        h5_file = stack.enter_context(h5py.File(dataset, "r"))
        run_id = read_env_args(h5_file).get("autodata", {}).get("run_id")
        result, attempts, warnings = read_run(result_file, run_id) if result_file else (None, None, [])
        source_evidence = Evidence(stack.enter_context(h5py.File(source, "r"))) if source else None
        evidence = Evidence(h5_file, result, attempts, source_evidence)
        warnings += check_same_run(evidence)
        source_record = input_record(source)
        recorded_source = evidence.run.get("inputs", {}).get("source")
        if source_record and recorded_source and source_record["sha256"] != recorded_source["sha256"]:
            raise ValueError(f"The source {source} is not the one the run started from, {recorded_source['path']}.")
        for checked in (evidence, source_evidence):
            if checked is not None:
                run_checks(checked, REGISTERED_CHECKS if checks is None else checks)
        metrics = REGISTERED_METRICS if metrics is None else metrics
        rows = {name: evaluate(spec, evidence) for name, spec in metrics.items()}
        num_episodes = len(evidence.all_episodes)
        num_valid = len(evidence.episodes)
        failures = {
            name: {"failed": len(failed), "examples": dict(list(failed.items())[:3])}
            for name, failed in evidence.failures.items()
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "dataset": input_record(dataset),
            "result_file": result_file,
            "source": source_record,
            "run_id": run_id,
            "revision": code_revision(),
        },
        "num_episodes": num_episodes,
        "num_valid_episodes": num_valid,
        "check_failures": failures,
        "warnings": warnings,
        "metrics": rows,
    }


def gate_violations(report: Mapping[str, Any], gate_file: str) -> list[str]:
    """Bounds in ``gate_file`` that the report violates.

    The gate maps metric names to ``min``/``max`` bounds, an optional ``key`` into a dict value (default: the
    metric's primary entry) and an optional ``version``. A gated metric that is missing, not applicable, errored or
    at another version is a violation.

    Raises:
        ValueError: If a gate entry has a bound that is not a finite number, or bounds a dict value without a
            ``key`` and the metric has no primary entry.
    """
    violations = []
    for name, bound in yaml.safe_load(Path(gate_file).read_text(encoding="utf-8")).items():
        limits = [bound[side] for side in ("min", "max") if side in bound]
        if not all(_is_bound(limit) for limit in limits):
            raise ValueError(f"Gate entry {name!r} has a bound that is not a finite number: {limits}.")
        row = report["metrics"].get(name)
        if row is None or row["status"] != "ok":
            violations.append(f"{name}: {'not reported' if row is None else row['status']}")
            continue
        if "version" in bound and bound["version"] != row["version"]:
            violations.append(f"{name}: version {row['version']}, gate written for version {bound['version']}")
            continue
        key = bound.get("key", row["primary"])
        value = row["value"].get(key) if key else row["value"]
        if isinstance(value, Mapping):
            raise ValueError(f"Gate entry {name!r} needs a key into {sorted(value)}.")
        if not _is_finite(value):
            violations.append(f"{name}: {value!r} is not a finite number")
            continue
        if "min" in bound and value < bound["min"]:
            violations.append(f"{name}: {value:.4g} is below {bound['min']}")
        if "max" in bound and value > bound["max"]:
            violations.append(f"{name}: {value:.4g} is above {bound['max']}")
    return violations


def _is_bound(limit: Any) -> bool:
    """Whether a gate bound is a finite number; YAML reads ``true`` and ``false`` as bools, which are not bounds."""
    return isinstance(limit, int | float) and not isinstance(limit, bool) and math.isfinite(limit)


def _format_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items()) or "none"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def format_report(report: Mapping[str, Any]) -> str:
    """The report as an aligned text table, grouped by area."""
    rows = [("Metric", "Area", "Value", "Unit", "n")]
    for name, row in sorted(report["metrics"].items(), key=lambda item: (AREAS.index(item[1]["area"]), item[0])):
        if row["status"] == "ok":
            rows.append((name, row["area"], _format_value(row["value"]), row["unit"], str(row["n"])))
        else:
            rows.append((name, row["area"], f"{row['status']}: {row['reason']}", "", ""))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    failed = [f"{name}={entry['failed']}" for name, entry in report["check_failures"].items() if entry["failed"]]
    lines.append(
        f"episodes: {report['num_valid_episodes']} valid of {report['num_episodes']}"
        + (f"; failed checks: {', '.join(failed)}" if failed else "")
    )
    lines += [f"warning: {warning}" for warning in report["warnings"]]
    return "\n".join(lines)
