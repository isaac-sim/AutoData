# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of how the report runs metrics, refuses mismatched runs and gates values."""

import h5py
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autodata_metrics import MetricSpec, Value, build_report, gate_violations


def _spec(name: str, compute, requires: tuple[str, ...] = ()) -> MetricSpec:
    return MetricSpec(
        name=name,
        compute=compute,
        area="viability",
        signal="test",
        scope="run",
        unit="fraction",
        better="higher",
        requires=requires,
        primary=None,
        version=1,
    )


def _dataset(path: Path, num_episodes: int = 1, entry: dict | None = None) -> str:
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test-v0", **({"autodata": entry} if entry else {})})
        for index in range(num_episodes):
            data.create_group(f"demo_{index}").create_dataset("actions", data=[[0.0, 1.0]])
    return str(path)


def _log(tmp_path: Path, run_id: str, successes: list[bool]) -> None:
    lines = [json.dumps({"run_id": run_id, "success": success}) for success in successes]
    (tmp_path / f"result.{run_id}.attempts.jsonl").write_text("\n".join(lines) + "\n")


def _run(tmp_path: Path, run_id: str, source_sha256: str | None = None) -> str:
    """A completed run: its result file and attempt log."""
    result_file = tmp_path / "result.json"
    run = {"run_id": run_id, "attempts_file": f"result.{run_id}.attempts.jsonl"}
    if source_sha256 is not None:
        run["inputs"] = {"source": {"path": "source.hdf5", "sha256": source_sha256}}
    result_file.write_text(json.dumps({"num_success": 1, "num_attempts": 1, "run": run}))
    _log(tmp_path, run_id, [True])
    return str(result_file)


def test_missing_evidence_is_not_applicable_and_names_the_key(tmp_path):
    report = build_report(
        _dataset(tmp_path / "d.hdf5"), metrics={"m": _spec("m", lambda ev: Value(1.0, n=1), ("step_success",))}
    )

    assert report["metrics"]["m"]["status"] == "not_applicable"
    assert report["metrics"]["m"]["reason"] == "missing step_success"


def test_failing_metric_is_an_error_and_the_rest_are_still_computed(tmp_path):
    metrics = {"broken": _spec("broken", lambda ev: 1 / 0), "fine": _spec("fine", lambda ev: Value(0.5, n=2))}

    rows = build_report(_dataset(tmp_path / "d.hdf5"), metrics=metrics)["metrics"]

    assert rows["broken"]["status"] == "error" and "ZeroDivisionError" in rows["broken"]["reason"]
    assert rows["fine"]["status"] == "ok" and rows["fine"]["value"] == 0.5


def test_interrupted_rerun_is_reported_from_its_own_attempt_log(tmp_path):
    result_file = _run(tmp_path, "run-a")
    _log(tmp_path, "run-b", [True, False])
    dataset = _dataset(tmp_path / "d.hdf5", entry={"run_id": "run-b", "exported_episodes": 1})
    count = {"attempts": _spec("attempts", lambda ev: Value(len(ev.attempts), n=1), ("attempts",))}

    report = build_report(dataset, result_file=result_file, metrics=count)

    assert report["metrics"]["attempts"]["value"] == 2
    assert "belongs to another run" in report["warnings"][0]


def test_report_refuses_inputs_that_disagree_with_their_run(tmp_path):
    entry = {"run_id": "run-a", "exported_episodes": 1}
    with pytest.raises(ValueError, match="merged or edited"):
        build_report(_dataset(tmp_path / "merged.hdf5", num_episodes=2, entry=entry), metrics={})

    dataset = _dataset(tmp_path / "d.hdf5", entry=entry)
    result_file = _run(tmp_path, "run-a")
    _log(tmp_path, "run-a", [True, True])
    with pytest.raises(ValueError, match="attempt log holds 2 attempts"):
        build_report(dataset, result_file=result_file, metrics={})

    result_file = _run(tmp_path, "run-a", source_sha256="0" * 64)
    with pytest.raises(ValueError, match="not the one the run started from"):
        build_report(dataset, result_file=result_file, source=_dataset(tmp_path / "source.hdf5"), metrics={})


def test_gate_flags_violations_and_refuses_a_non_finite_bound(tmp_path):
    row = {"status": "ok", "value": 0.4, "primary": None, "version": 2}
    missing = {"status": "ok", "value": {"p50": 1.0}, "primary": "p95", "version": 1}
    report = {"metrics": {"low": row, "old": row, "fine": row, "absent": {"status": "not_applicable"}, "gap": missing}}
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        "low: {min: 0.5}\nold: {version: 1}\nfine: {min: 0.3, max: 0.5, version: 2}\nabsent: {min: 0}\ngap: {max: 2}\n"
    )

    violations = gate_violations(report, str(gate))

    assert [violation.split(":")[0] for violation in violations] == ["low", "old", "absent", "gap"]
    for bad_bound in (".nan", "false", "{p95: 1}"):
        gate.write_text(f"absent: {{min: {bad_bound}}}\n")
        with pytest.raises(ValueError, match="not a finite number"):
            gate_violations(report, str(gate))


def test_non_finite_numbers_never_reach_the_report(tmp_path):
    metrics = {
        "m": _spec("m", lambda ev: Value(float("nan"), n=1)),
        "d": _spec("d", lambda ev: Value(1.0, n=1, details={"interval_95": [0.5, float("inf")]})),
    }
    report = build_report(_dataset(tmp_path / "d.hdf5"), metrics=metrics)
    gate = tmp_path / "gate.yaml"
    gate.write_text("m: {min: 0}\n")

    assert report["metrics"]["m"]["status"] == "error"
    assert gate_violations(report, str(gate)) == ["m: error"]
    assert report["metrics"]["d"]["details"] == {"interval_95": [0.5, None]}


def test_metrics_import_without_torch_or_isaaclab():
    code = "import sys, autodata_metrics; print(sorted({'torch', 'isaaclab', 'autodata_core'} & set(sys.modules)))"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    assert output.strip() == "[]"
