# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the registered metrics of a dataset and its run, render them, and check them against a gate."""

from __future__ import annotations

import contextlib
import h5py
import json
import math
import re
import textwrap
import yaml
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

from autodata_metrics.evidence import Evidence, read_env_args
from autodata_metrics.records import ATTEMPT, RESULT, InvalidRecord, is_finite_number, parse
from autodata_metrics.registry import (
    AREAS,
    REGISTERED_CHECKS,
    REGISTERED_METRICS,
    CheckSpec,
    MetricSpec,
    NotApplicable,
    Value,
)
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
    text = _read_text(result_path)
    result = None if text is None else parse(_json(text, result_path), RESULT, result_file)
    warnings = [] if text is not None else [f"{result_file} does not exist"]
    if result is not None and run_id is not None and result.get("run", {}).get("run_id") != run_id:
        warnings.append(f"{result_file} belongs to another run, so the run is reported from its attempt log")
        result = None
    if result is not None and "attempts_file" in result.get("run", {}):
        attempts_path = result_path.parent / result["run"]["attempts_file"]
    elif run_id is not None:
        attempts_path = Path(attempts_file_for(result_file, run_id))
    else:
        return result, None, warnings
    text = _read_text(attempts_path)
    attempts = None
    if text is not None:
        attempts = [
            parse(_json(line, attempts_path), ATTEMPT, f"{attempts_path} line {number}")
            for number, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
    return result, attempts, warnings


def _read_text(path: Path) -> str | None:
    """The file's text, or None when there is no such file."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise InvalidRecord(f"{path} cannot be read: {exc}") from exc


def _json(text: str, path: Path) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise InvalidRecord(f"{path} must be JSON: {exc}") from exc


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
    """Run the episode checks and keep only the episodes that pass all of them in ``evidence.episodes``.

    Episodes are checked one at a time, each listed once for all its checks and released after them.
    """
    for spec in checks.values():
        evidence.check_groups[spec.name] = spec.group
    applicable = {spec.name: spec for spec in checks.values() if all(evidence.has(key) for key in spec.requires)}
    failed: dict[str, dict[str, str]] = {name: {} for name in applicable}
    applied: set[str] = set()
    for episode in evidence.all_episodes:
        with episode.listed():
            for name, spec in applicable.items():
                try:
                    reason = spec.check(episode, evidence)
                except NotApplicable:
                    continue
                except Exception as exc:  # A broken check fails its episodes visibly instead of sinking the report.
                    reason = f"{type(exc).__name__}: {exc}"
                applied.add(name)
                if reason is not None:
                    failed[name][episode.name] = reason
    evidence.failures.update({name: failed[name] for name in applicable if name in applied})
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
        head = missing.partition(".")[0]
        missing = head if head in ("attempts", "source") and not evidence.has(head) else missing
        return {**row, "status": "not_applicable", "reason": f"missing {missing}", "missing": missing}
    try:
        value = spec.compute(evidence)
    except NotApplicable as exc:
        return {**row, "status": "not_applicable", "reason": str(exc)}
    except Exception as exc:  # A failing metric must not sink the report of every other metric.
        return {**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(value, Value):
        return {**row, "status": "error", "reason": f"returned {type(value).__name__}, not a Value"}
    row = {
        **row,
        "status": "ok",
        "value": _finite_or_none(value.value),
        "n": int(value.n),
        "details": _finite_or_none(value.details),
    }
    if spec.baseline and evidence.source is not None:
        row["baseline"] = evaluate(spec, evidence.source.as_baseline_for(evidence)).get("value")
    return row


def _finite_or_none(value: Any) -> Any:
    """``value`` with every non-finite number replaced by None, so the report stays valid JSON."""
    value = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
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
        if source_evidence is not None:
            warnings += _set_aside_merged_annotation(source_evidence, source)
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
        checked = _check_summary(evidence)
        source_checked = _check_summary(source_evidence) if source_evidence is not None else None
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "dataset": input_record(dataset),
            "result_file": result_file,
            "source": source_record,
            "run_id": run_id,
            "revision": code_revision(),
        },
        **checked,
        "source_checks": source_checked,
        "warnings": warnings,
        "metrics": rows,
    }


def _set_aside_merged_annotation(source: Evidence, path: str) -> list[str]:
    """Drop the annotation record of a source merged or edited after annotation, since it counts other episodes."""
    entry = source.env_args.get("autodata")
    if entry is None or "annotation" not in entry or entry["exported_episodes"] == len(source.all_episodes):
        return []
    source.env_args = {**source.env_args, "autodata": {key: item for key, item in entry.items() if key != "annotation"}}
    return [
        f"{path} holds {len(source.all_episodes)} episodes but was annotated with {entry['exported_episodes']}, so its"
        " annotation record is set aside"
    ]


def _check_summary(evidence: Evidence) -> dict[str, Any]:
    return {
        "num_episodes": len(evidence.all_episodes),
        "num_valid_episodes": len(evidence.episodes),
        "check_failures": {
            name: {"failed": len(failed), "examples": dict(list(failed.items())[:3])}
            for name, failed in evidence.failures.items()
        },
    }


_GATE_FIELDS = {"min", "max", "key", "version"}


class _GateLoader(yaml.SafeLoader):
    """Safe YAML loading that refuses a repeated key, which plain YAML silently resolves to its last value."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [self.construct_object(key_node, deep=deep) for key_node, _ in node.value]
        repeated = sorted({repr(key) for key in keys if isinstance(key, Hashable) and keys.count(key) > 1})
        if repeated:
            raise yaml.constructor.ConstructorError(None, None, f"repeated keys {', '.join(repeated)}", node.start_mark)
        return super().construct_mapping(node, deep=deep)


def load_gate(gate_file: str) -> dict[str, dict[str, Any]]:
    """The gate in ``gate_file``: per metric, optional ``min``/``max`` bounds, a ``key`` into a dict value (default: the
    metric's primary entry) and a ``version``.

    Raises:
        ValueError: If the file does not map metric names to these fields, a bound is not a finite number, the key is
            not a name or the version not an integer.
    """
    try:
        gate = yaml.load(Path(gate_file).read_text(encoding="utf-8"), Loader=_GateLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{gate_file} is not a readable YAML gate: {exc}") from exc
    if not isinstance(gate, Mapping):
        raise ValueError(f"{gate_file} must map metric names to bounds, such as 'name: {{min: 0.5}}'.")
    entries = {}
    for name, bound in gate.items():
        bound = {} if bound is None else bound
        if not isinstance(bound, Mapping) or not set(bound) <= _GATE_FIELDS:
            raise ValueError(f"Gate entry {name!r} must map some of {sorted(_GATE_FIELDS)}, not {bound!r}.")
        if not isinstance(bound.get("key", ""), str):
            raise ValueError(f"Gate entry {name!r} has a key that is not an entry name such as p95: {bound['key']!r}.")
        if isinstance(bound.get("version", 1), bool) or not isinstance(bound.get("version", 1), int):
            raise ValueError(f"Gate entry {name!r} has a version that is not an integer: {bound['version']!r}.")
        entries[name] = {**bound, **{side: _bound(name, bound[side]) for side in ("min", "max") if side in bound}}
    return entries


def _bound(name: str, limit: Any) -> float:
    """A gate bound as a number: YAML reads ``1e-3`` as a string, and ``true`` as a bool, which is not a bound."""
    number = limit
    if isinstance(limit, str):
        with contextlib.suppress(ValueError):
            number = float(limit)
    if not is_finite_number(number):
        raise ValueError(f"Gate entry {name!r} has a bound that is not a finite number: {limit!r}.")
    return number


def gate_results(report: Mapping[str, Any], gate: Mapping[str, Mapping[str, Any]]) -> dict[str, str | None]:
    """Per gated metric, None when the report meets its bounds, else why it does not.

    A gated metric that is missing, not applicable, errored or at another version fails.

    Args:
        report: A report from :func:`build_report`.
        gate: A gate from :func:`load_gate`.
    """
    return {name: _gate_failure(report["metrics"].get(name), bound) for name, bound in gate.items()}


def _gate_failure(row: Mapping[str, Any] | None, bound: Mapping[str, Any]) -> str | None:
    if row is None or row["status"] != "ok":
        return "not reported" if row is None else row["status"]
    if "version" in bound and bound["version"] != row["version"]:
        return f"version {row['version']}, gate written for version {bound['version']}"
    value = row["value"]
    if isinstance(value, Mapping):
        key = bound.get("key", row["primary"])
        if key not in value:
            return f"needs a key into {sorted(value)}"
        value = value[key]
    elif "key" in bound:
        return f"key {bound['key']!r} given for a value without entries"
    if "min" in bound and value < bound["min"]:
        return f"{value:.4g} is below {bound['min']}"
    if "max" in bound and value > bound["max"]:
        return f"{value:.4g} is above {bound['max']}"
    return None


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
    if inputs["source"]:
        valid, total = report["source_checks"]["num_valid_episodes"], report["source_checks"]["num_episodes"]
        title.append(
            f"source {Path(inputs['source']['path']).name}" + (f" ({valid}/{total} valid)" if valid < total else "")
        )
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
            lines.append(
                f"  ✗ {name} failed {_count(entry['failed'], 'episodes')}, e.g. {', '.join(entry['examples'])}"
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
    interval, run, skipped = (row["details"].get(key) for key in ("interval_95", "checks_run", "checks_skipped"))
    if isinstance(interval, list) and len(interval) == 2 and all(isinstance(bound, int | float) for bound in interval):
        low, high = (_console_number(bound, row["unit"]) for bound in interval)
        parts.append(f"95% CI {low.removesuffix('%')}–{high}")
    if isinstance(run, list) and isinstance(skipped, list) and skipped:
        parts.append(f"{len(run)} of {len(run) + len(skipped)} checks")
    return " · ".join(parts)


def _console_value(value: Any, row: Mapping[str, Any], labelled: bool = True) -> str:
    """A value, or a dict value's primary entry (every entry without one), as the console shows it."""
    if not isinstance(value, Mapping):
        return _console_number(value, row["unit"])
    keys = [row["primary"]] if row["primary"] in value else list(value)
    return ", ".join((f"{key} " if labelled else "") + _console_number(value[key], row["unit"]) for key in keys)


def _console_number(value: float, unit: str) -> str:
    """Fractions as percentages, seconds as durations; other units follow the number unless they hold a space, which
    marks a description of the quantity rather than a unit."""
    if unit.startswith("fraction"):
        percent = f"{100 * value:.3g}%"
        return ">99.9%" if percent == "100%" and value < 1 else percent
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
    if missing == "source":
        return "needs --source"
    if missing.startswith("source."):
        return "needs a --source annotated with --record_metric_inputs"
    if missing == "attempts" or missing.startswith("run."):
        return "needs --result_file" if inputs["result_file"] is None else "needs --record_metric_inputs"
    if missing.startswith("attempts.extras."):
        return "not reported by this algorithm"
    if missing.startswith(_METRIC_INPUT_KEYS):
        return "needs --record_metric_inputs"
    return row["reason"]
