# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the registered metrics of a dataset and its run, render them, and check them against a gate."""

from __future__ import annotations

import contextlib
import h5py
import json
import math
import numbers
import re
import textwrap
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
    if entry is None and result is not None:
        return [
            "the dataset was recorded without --record_metric_inputs, so it is matched to its run by episode count only"
        ]
    if entry is None:
        return []
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
            evidence.skipped_checks[spec.name] = spec.group
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
        else:
            evidence.skipped_checks[spec.name] = spec.group
    failing = {name for failed in evidence.failures.values() for name in failed}
    evidence.episodes = [episode for episode in evidence.all_episodes if episode.name not in failing]


def evaluate(spec: MetricSpec, evidence: Evidence) -> dict[str, Any]:
    """One report row: the metric's value, or why it is not applicable or failed."""
    row = {
        "area": spec.area,
        "signal": spec.signal,
        "scope": spec.scope,
        "unit": spec.unit,
        "sample": spec.sample,
        "better": spec.better,
        "primary": spec.primary,
        "version": spec.version,
    }
    if spec.scope == "dataset" and not evidence.episodes:
        return {**row, "status": "not_applicable", "reason": "no valid episodes"}
    missing = next((key for key in spec.requires if not evidence.has(key)), None)
    if missing is not None:
        return {**row, "status": "not_applicable", "reason": f"missing {missing}", "missing": missing}
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


def gate_results(report: Mapping[str, Any], gate_file: str) -> dict[str, str | None]:
    """Per metric in ``gate_file``, None when the report meets its bounds, else why it does not.

    The gate maps metric names to ``min``/``max`` bounds, an optional ``key`` into a dict value (default: the
    metric's primary entry) and an optional ``version``. A gated metric that is missing, not applicable, errored or
    at another version fails.

    Raises:
        ValueError: If a gate entry has a bound that is not a finite number, or bounds a dict value without a
            ``key`` and the metric has no primary entry.
    """
    results = {}
    for name, bound in yaml.safe_load(Path(gate_file).read_text(encoding="utf-8")).items():
        limits = [bound[side] for side in ("min", "max") if side in bound]
        if not all(_is_bound(limit) for limit in limits):
            raise ValueError(f"Gate entry {name!r} has a bound that is not a finite number: {limits}.")
        results[name] = _gate_failure(name, report["metrics"].get(name), bound)
    return results


def _gate_failure(name: str, row: Mapping[str, Any] | None, bound: Mapping[str, Any]) -> str | None:
    if row is None or row["status"] != "ok":
        return "not reported" if row is None else row["status"]
    if "version" in bound and bound["version"] != row["version"]:
        return f"version {row['version']}, gate written for version {bound['version']}"
    key = bound.get("key", row["primary"])
    value = row["value"].get(key) if key else row["value"]
    if isinstance(value, Mapping):
        raise ValueError(f"Gate entry {name!r} needs a key into {sorted(value)}.")
    if not _is_finite(value):
        return f"{value!r} is not a finite number"
    if "min" in bound and value < bound["min"]:
        return f"{value:.4g} is below {bound['min']}"
    if "max" in bound and value > bound["max"]:
        return f"{value:.4g} is above {bound['max']}"
    return None


def _is_bound(limit: Any) -> bool:
    """Whether a gate bound is a finite number; YAML reads ``true`` and ``false`` as bools, which are not bounds."""
    return isinstance(limit, int | float) and not isinstance(limit, bool) and math.isfinite(limit)


CONSOLE_WIDTH = 100
"""Width the console summary wraps its lists of metrics to."""


def format_report(report: Mapping[str, Any], gate: Mapping[str, str | None] | None = None) -> str:
    """The report as a console summary: a verdict, the computed metrics by area, and what was not computed and why.

    The JSON report holds every detail; the summary keeps a metric's primary value, its source baseline, its sample
    count and, for rates, the 95% interval.

    Args:
        report: A report from :func:`build_report`.
        gate: Results from :func:`gate_results`, to mark each gated metric as met or failed.
    """
    inputs, metrics = report["inputs"], report["metrics"]
    computed = {name: row for name, row in metrics.items() if row["status"] == "ok"}
    errors = {name: row for name, row in metrics.items() if row["status"] == "error"}
    skipped = {name: row for name, row in metrics.items() if row["status"] == "not_applicable"}

    title = [f"AutoData metrics · {Path(inputs['dataset']['path']).name}"]
    title += [f"run {inputs['run_id'][:8]}"] if inputs["run_id"] else []
    title += [f"source {Path(inputs['source']['path']).name}"] if inputs["source"] else []
    verdict = [f"{report['num_valid_episodes']}/{report['num_episodes']} episodes valid", f"{len(computed)} computed"]
    verdict += [f"{len(skipped)} not computed"] if skipped else []
    verdict += [_count(len(errors), "errors")] if errors else []
    if gate is not None:
        failed = sum(reason is not None for reason in gate.values())
        verdict.append(
            f"gate failed on {failed} of {len(gate)}" if failed else f"gate passed on {len(gate)} of {len(gate)}"
        )
    lines = [" · ".join(title), " · ".join(verdict)]
    for name, entry in report["check_failures"].items():
        if entry["failed"]:
            more = ", …" if entry["failed"] > len(entry["examples"]) else ""
            lines.append(
                f"  ✗ {name} failed {_count(entry['failed'], 'episodes')}: {', '.join(entry['examples'])}{more}"
            )

    rows = []
    for name, row in computed.items():
        mark = " " if gate is None or name not in gate else ("✗" if gate[name] else "✓")
        baseline = row.get("baseline")
        source = "" if baseline is None else _console_value(baseline, row, labelled=False)
        rows.append((row["area"], mark, name, _console_value(row["value"], row), source, _based_on(row)))
    if rows:
        has_source = any(row[4] for row in rows)
        name_width = max(len(row[2]) for row in rows)
        value_width = max(len(row[3]) for row in rows + [("", "", "", "Value")])
        source_width = max(len(row[4]) for row in rows + [("", "", "", "", "Source")])

        def line(mark: str, name: str, value: str, source: str, based_on: str) -> str:
            cells = [f"{mark} {name.ljust(name_width)}", value.rjust(value_width)]
            cells += [source.rjust(source_width)] if has_source else []
            return "  ".join(cells + [based_on]).rstrip()

        lines += ["", line(" ", "", "Value", "Source", "Based on")]
        for area in AREAS:
            area_rows = sorted((row for row in rows if row[0] == area), key=lambda row: row[2])
            if area_rows:
                lines.append(area.upper())
                lines += [line(*row[1:]) for row in area_rows]
    if errors:
        lines += ["", "ERRORS"] + [f"  {name}: {row['reason']}" for name, row in errors.items()]
    if skipped:
        groups: dict[str, list[str]] = {}
        for name, row in skipped.items():
            groups.setdefault(_not_computed_because(row, inputs), []).append(name)
        lines += ["", "NOT COMPUTED"]
        for reason, names in sorted(groups.items(), key=lambda item: -len(item[1])):
            lines += textwrap.wrap(
                ", ".join(sorted(names)), CONSOLE_WIDTH, initial_indent=f"  {reason}: ", subsequent_indent="      "
            )
    if report["warnings"]:
        lines += [""] + [f"Warning: {warning}" for warning in report["warnings"]]
    return "\n".join(lines)


def _count(count: int, noun: str) -> str:
    return f"{count:,} {noun.removesuffix('s') if count == 1 else noun}"


def _based_on(row: Mapping[str, Any]) -> str:
    """The sample count, with a rate's 95% interval and a pass rate's check coverage."""
    parts = [_count(row["n"], row["sample"])]
    interval = row["details"].get("interval_95")
    if interval:
        low, high = (_console_number(bound, row["unit"]) for bound in interval)
        parts.append(f"95% CI {low.removesuffix('%')}–{high}")
    if row["details"].get("checks_skipped"):
        run, skipped = len(row["details"]["checks_run"]), len(row["details"]["checks_skipped"])
        parts.append(f"{run} of {run + skipped} checks")
    return " · ".join(parts)


def _console_value(value: Any, row: Mapping[str, Any], labelled: bool = True) -> str:
    """A value, or a dict value's primary entry (every entry without one), as the console shows it."""
    if not isinstance(value, Mapping):
        return _console_number(value, row["unit"])
    keys = [row["primary"]] if row["primary"] else list(value)
    return ", ".join((f"{key} " if labelled else "") + _console_number(value[key], row["unit"]) for key in keys)


def _console_number(value: float, unit: str) -> str:
    """Fractions as percentages, seconds as durations; other units follow the number unless they hold a space, which
    marks a description of the quantity rather than a unit."""
    if unit.startswith("fraction"):
        return f"{100 * value:.3g}%"
    if unit == "s" or unit.startswith("s "):
        return _duration(value)
    text = f"{value:,.0f}" if abs(value) >= 100 else re.sub(r"e\+?(-?)0*(\d)", r"e\1\2", f"{value:.3g}")
    return text if " " in unit else f"{text} {unit}"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.3g} s"
    minutes, rest = divmod(round(seconds), 60)
    if minutes < 120:
        return f"{minutes} min {rest:02d} s"
    return f"{minutes // 60} h {minutes % 60:02d} min"


_METRIC_INPUT_KEYS = ("env_args.autodata", "step_success", "obs/datagen_info/")
"""Evidence keys a generated dataset holds only when recorded with ``--record_metric_inputs``."""


def _not_computed_because(row: Mapping[str, Any], inputs: Mapping[str, Any]) -> str:
    """What would make a not-applicable metric computable, or its own reason."""
    missing = row.get("missing")
    if missing is None:
        return row["reason"]
    if missing.startswith("attempts.extras."):
        return "not reported by this algorithm"
    if missing.partition(".")[0] == "source":
        return "needs --source" if inputs["source"] is None else "needs a --source recorded with --record_metric_inputs"
    if missing == "attempts" or missing.startswith("run."):
        return "needs --result_file" if inputs["result_file"] is None else "needs --record_metric_inputs"
    if missing.startswith(_METRIC_INPUT_KEYS):
        return "needs --record_metric_inputs"
    return row["reason"]
