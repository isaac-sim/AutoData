# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated robustness tests: every malformed record or gate is refused or reported, and no check raises."""

import h5py
import json
import numpy as np
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from autodata_metrics import build_report, format_report, gate_results, load_gate
from autodata_metrics import report as report_module
from autodata_metrics.evidence import Evidence
from autodata_metrics.registry import REGISTERED_CHECKS
from autodata_utils.generation_result import input_record

ABSENT = object()
SOURCE_RECORD = "source record"
"""Stands for the source's path and SHA-256, filled in once the source file is written."""
WRONG_VALUES = (ABSENT, None, "", "x", -1, 1.5, 10**400, True, float("nan"), [], {}, {"x": 1})


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr(report_module, "code_revision", lambda: None)


def _write_dataset(path: Path, env_args: dict, num_episodes: int = 2) -> str:
    pose = np.tile(np.eye(4), (4, 1, 1))
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        for index in range(num_episodes):
            episode = data.create_group(f"demo_{index}")
            episode.create_dataset("actions", data=np.zeros((4, 7)))
            episode.create_dataset("step_success", data=[False, False, True, True])
            episode.create_dataset("obs/datagen_info/eef_pose/arm", data=pose)
            episode.create_dataset("obs/datagen_info/target_eef_pose/arm", data=pose)
            episode.create_dataset("states/articulation/robot/joint_velocity", data=np.zeros((4, 2)))
    return str(path)


def _samples(tmp_path: Path) -> dict[str, dict]:
    """Records as AutoData writes them, for a run that exported two episodes from a source of two."""
    annotation = {
        "mode": "auto",
        "input_episodes": 2,
        "processed_episodes": 2,
        "annotated_episodes": 2,
        "setup_time_s": 10.0,
        "time_s": 20.0,
    }
    sim_args = {"dt": 0.01, "decimation": 5}
    source_entry = {"version": 1, "run_id": "annotation", "exported_episodes": 2, "annotation": annotation}
    run = {
        "run_id": "run-a",
        "seed": 1,
        "revision": None,
        "inputs": {"source": SOURCE_RECORD, "env_profile": None},
        "device": "cuda:0",
        "gpu": "GPU",
        "cpu": "CPU",
        "num_gpus": 1,
        "num_envs": 1,
        "setup_time_s": 5.0,
        "generation_time_s": 30.0,
        "attempts_file": "result.run-a.attempts.jsonl",
    }
    attempt = {
        "run_id": "run-a",
        "env_id": 0,
        "success": True,
        "start_time_s": 0.0,
        "end_time_s": 1.0,
        "initial_object_positions": {"cube": [0.1, 0.2]},
        "extras": {"planning_time_s": 0.5, "planning_failed": False},
    }
    entry = {
        "version": 1,
        "run_id": "run-a",
        "exported_episodes": 2,
        "pose_action": {"arm": {"action_slice": [0, 6], "bound": 1.0, "absolute": True}},
        "action_noise": 0.03,
        "joints": {"asset": "robot", "names": ["j1", "j2"]},
        "hardware_limits": {"velocity": {"j[12]": 2.0}, "acceleration": {"j1": 10.0}},
    }
    return {
        "env_args": {"env_name": "Test-v0", "sim_args": sim_args, "autodata": entry},
        "source_env_args": {"env_name": "Test-v0", "sim_args": sim_args, "autodata": source_entry},
        "result": {"algorithm": "mimicgen", "num_success": 2, "num_failures": 1, "num_attempts": 3, "run": run},
        "attempt": attempt,
    }


def _paths(record: dict, prefix: tuple = ()) -> Iterator[tuple]:
    """Every field path of a record, nested objects included."""
    for key, value in record.items():
        yield (*prefix, key)
        if isinstance(value, dict):
            yield from _paths(value, (*prefix, key))


def _replaced(record: dict, path: tuple, value: object) -> dict:
    copy = json.loads(json.dumps(record))
    parent = copy
    for key in path[:-1]:
        parent = parent[key]
    if value is ABSENT:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    return copy


def _report_or_refusal(tmp_path: Path, records: dict) -> dict | None:
    """Build, summarize and serialize the report of these records; None when the report refuses them."""
    dataset = _write_dataset(tmp_path / "data.hdf5", records["env_args"])
    source = _write_dataset(tmp_path / "source.hdf5", records["source_env_args"])
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(records["result"]).replace(json.dumps(SOURCE_RECORD), json.dumps(input_record(source)))
    )
    first = records["attempt"]
    lines = [first, {**first, "start_time_s": 1.0, "end_time_s": 2.0}, {**first, "success": False, "start_time_s": 2.0}]
    (tmp_path / "result.run-a.attempts.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    gate = tmp_path / "gate.yaml"
    gate.write_text("generation_success_rate: {min: 0.1}\naction_rotation_error: {max: 0.001}\n")
    try:
        report = build_report(dataset, result_file=str(result_file), source=source)
    except ValueError:
        return None
    json.dumps(report, allow_nan=False)
    format_report(report, gate_results(report, load_gate(str(gate))))
    return report


def test_every_malformed_record_field_is_refused_or_reported(tmp_path):
    samples = _samples(tmp_path)
    report = _report_or_refusal(tmp_path, samples)
    assert report is not None and report["metrics"]["time_to_dataset"]["status"] == "ok"
    unread = {**samples["attempt"], "initial_object_positions": {"cube": [float("nan"), 0.0]}}
    assert _report_or_refusal(tmp_path, {**samples, "attempt": unread}) is not None
    negative = {**samples["attempt"], "extras": {"planning_time_s": -5.0, "planning_failed": False}}
    assert _report_or_refusal(tmp_path, {**samples, "attempt": negative}) is None
    crashes = []
    for kind in ("env_args", "source_env_args", "result", "attempt"):
        for path in _paths(samples[kind]):
            for wrong in WRONG_VALUES:
                records = {**samples, kind: _replaced(samples[kind], path, wrong)}
                try:
                    _report_or_refusal(tmp_path, records)
                except Exception as exc:  # Anything but a refusal is a crash this test exists to find.
                    crashes.append(f"{kind}.{'.'.join(path)} = {wrong!r}: {type(exc).__name__}: {exc}")
    assert not crashes, "\n".join(crashes[:20])


def test_unreadable_run_files_are_refused(tmp_path):
    samples = _samples(tmp_path)
    crashes = []
    for name in ("result.json", "result.run-a.attempts.jsonl"):
        for raw in (b"", b"{", b"\xff\xfe", b"[" * 10000 + b"]" * 10000):
            _report_or_refusal(tmp_path, samples)
            (tmp_path / name).write_bytes(raw)
            try:
                build_report(
                    str(tmp_path / "data.hdf5"),
                    result_file=str(tmp_path / "result.json"),
                    source=str(tmp_path / "source.hdf5"),
                )
            except ValueError:
                continue
            except Exception as exc:  # Anything but a refusal is a crash this test exists to find.
                crashes.append(f"{name} = {raw[:8]!r}: {type(exc).__name__}: {exc}")
    assert not crashes, "\n".join(crashes)


def test_every_malformed_gate_field_is_refused_or_evaluated(tmp_path):
    samples = _samples(tmp_path)
    _report_or_refusal(tmp_path, samples)
    report = build_report(
        str(tmp_path / "data.hdf5"), result_file=str(tmp_path / "result.json"), source=str(tmp_path / "source.hdf5")
    )
    gate = {
        "generation_success_rate": {"min": 0.1, "version": 1},
        "action_rotation_error": {"max": 0.001, "key": "p95"},
    }
    crashes = []
    for path in _paths(gate):
        for wrong in WRONG_VALUES:
            gate_file = tmp_path / "gate.yaml"
            gate_file.write_text(json.dumps(_replaced(gate, path, wrong)))
            try:
                loaded = load_gate(str(gate_file))
            except ValueError:
                continue
            try:
                format_report(report, gate_results(report, loaded))
            except Exception as exc:  # Evaluating a loaded gate must never raise.
                crashes.append(f"{'.'.join(path)} = {wrong!r}: {type(exc).__name__}: {exc}")
    assert not crashes, "\n".join(crashes)


def test_checks_never_fail_an_episode_by_raising(tmp_path):
    steps = 3
    degenerate = {
        "scalar": {"obs/scalar": np.float32(1.0), "obs/robot/root_pose": np.float32(1.0)},
        "strings": {"obs/names": np.array([b"a", b"b", b"c"])},
        "empty_camera": {"obs/camera": np.zeros((0, 4, 4, 3), np.uint8)},
        "float_camera": {"obs/depth": np.zeros((steps, 4, 4, 1), np.float32)},
        "gray_camera": {"obs/gray": np.zeros((steps, 4, 4), np.uint8)},
        "zero_pose": {"obs/datagen_info/eef_pose/arm": np.zeros((steps, 4, 4))},
        "short_success": {"step_success": np.zeros(1, bool)},
        "no_actions": {},
        "empty_actions": {"actions": np.zeros((0, 7))},
    }
    path = tmp_path / "d.hdf5"
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test-v0"})
        for name, arrays in degenerate.items():
            episode = data.create_group(name)
            if name != "no_actions":
                episode.create_dataset("actions", data=arrays.pop("actions", np.zeros((steps, 7))))
            episode.create_group("obs/datagen_info/target_eef_pose")
            for key, array in arrays.items():
                episode.create_dataset(key, data=array)

    with h5py.File(path, "r") as h5_file:
        evidence = Evidence(h5_file)
        report_module.run_checks(evidence, REGISTERED_CHECKS)
        raised = {
            f"{check} on {episode}: {reason}"
            for check, failed in evidence.failures.items()
            for episode, reason in failed.items()
            if re.match(r"^[A-Z]\w*(Error|Exception):", reason)
        }
    assert not raised, "\n".join(sorted(raised))
    report = build_report(str(path))
    json.dumps(report, allow_nan=False)
    format_report(report)
