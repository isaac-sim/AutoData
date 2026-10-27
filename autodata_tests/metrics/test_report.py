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


def _run(tmp_path: Path, run_id: str, num_success: int = 1) -> str:
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"num_success": num_success, "run": {"run_id": run_id}}))
    (tmp_path / "result.attempts.jsonl").write_text(json.dumps({"run_id": run_id, "success": True}) + "\n")
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


def test_report_refuses_data_from_another_run_or_merged_data(tmp_path):
    entry = {"run_id": "run-a", "exported_episodes": 1}
    with pytest.raises(AssertionError, match="different runs"):
        build_report(_dataset(tmp_path / "d.hdf5", entry=entry), result_file=_run(tmp_path, "run-b"), metrics={})
    with pytest.raises(AssertionError, match="merged or edited"):
        build_report(_dataset(tmp_path / "merged.hdf5", num_episodes=2, entry=entry), metrics={})


def test_gate_fails_on_bound_inapplicable_metric_and_outdated_version(tmp_path):
    row = {"status": "ok", "value": 0.4, "primary": None, "version": 2}
    report = {"metrics": {"low": row, "old": row, "fine": row, "absent": {"status": "not_applicable"}}}
    gate = tmp_path / "gate.yaml"
    gate.write_text("low: {min: 0.5}\nold: {version: 1}\nfine: {min: 0.3, max: 0.5, version: 2}\nabsent: {min: 0}\n")

    violations = gate_violations(report, str(gate))

    assert [violation.split(":")[0] for violation in violations] == ["low", "old", "absent"]


def test_metrics_import_without_torch_or_isaaclab():
    code = "import sys, autodata_metrics; print(sorted({'torch', 'isaaclab', 'autodata_core'} & set(sys.modules)))"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    assert output.strip() == "[]"
