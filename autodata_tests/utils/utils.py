# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assertion helpers for the AutoData test suite."""

import h5py
import json
import os

from autodata_metrics.records import ATTEMPT, RESULT, parse, parse_env_args
from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess


def assert_valid_dataset(output_file: str, min_num_demos: int) -> None:
    """Assert the run wrote a dataset containing at least min_num_demos entries."""

    assert os.path.exists(output_file), f"Expected output dataset at {output_file}, but it was not created."
    with h5py.File(output_file, "r") as f:
        assert "data" in f, f"Output dataset {output_file} has no top-level 'data' group."
        data_group = f["data"]
        assert isinstance(data_group, h5py.Group), f"'data' in {output_file} is not an HDF5 group."
        num_demos = len(data_group.keys())
    assert num_demos >= min_num_demos, f"Expected at least {min_num_demos} demos, found {num_demos}."


def read_generation_result(result_file: str) -> tuple[int, int]:
    """Read successful-demo and attempt counts from a completed generation result file.

    Args:
        result_file: JSON result file passed to the data-generation CLI.

    Returns:
        Tuple of (num_success, num_attempts).
    """
    assert os.path.exists(result_file), f"Expected generation result file at {result_file}, but it was not created."
    with open(result_file, encoding="utf-8") as result_handle:
        result = json.load(result_handle)

    num_success = result["num_success"]
    num_failures = result["num_failures"]
    num_attempts = result["num_attempts"]
    assert num_success >= 0, f"Generation result has invalid num_success: {num_success}."
    assert num_failures >= 0, f"Generation result has invalid num_failures: {num_failures}."
    assert num_attempts > 0, f"Generation result has invalid num_attempts: {num_attempts}."
    assert num_success + num_failures == num_attempts, "Generation result counts do not add up."
    return num_success, num_attempts


def assert_recorded_episodes(dataset_file: str) -> dict:
    """Assert the per-step recording and the ``autodata`` entry agree with the dataset's episodes.

    The entry has the shape the metrics read, every episode has one ``step_success`` value per action, its success
    attribute equals whether ``step_success`` ever held, and the entry counts every exported episode.

    Returns:
        The dataset's ``autodata`` entry.
    """

    with h5py.File(dataset_file, "r") as h5_file:
        data = h5_file["data"]
        entry = parse_env_args(json.loads(data.attrs["env_args"]), dataset_file)["autodata"]
        assert entry["exported_episodes"] == len(data), f"The entry counts {entry['exported_episodes']} episodes."
        for name, episode in data.items():
            step_success = episode["step_success"][:]
            assert len(step_success) == len(episode["actions"]), f"{name}: step_success and actions differ in length."
            assert bool(step_success.any()) == bool(episode.attrs["success"]), f"{name}: step_success disagrees."
    return entry


def assert_recorded_run(dataset_file: str, result_file: str) -> list[dict]:
    """Assert a generation run's dataset, result JSON and attempt log have the shapes the metrics read and describe
    the same run.

    Returns:
        The attempt-log lines.
    """

    entry = assert_recorded_episodes(dataset_file)
    with open(result_file, encoding="utf-8") as result_handle:
        result = parse(json.load(result_handle), RESULT, result_file)
    attempts_file = os.path.join(os.path.dirname(result_file), result["run"]["attempts_file"])
    with open(attempts_file, encoding="utf-8") as attempts_handle:
        attempts = [parse(json.loads(line), ATTEMPT, attempts_file) for line in attempts_handle if line.strip()]
    assert entry["exported_episodes"] == result["num_success"], "The dataset and result count different successes."
    assert len(attempts) == result["num_attempts"], "The attempt log and result count different attempts."
    assert (
        sum(attempt["success"] for attempt in attempts) == result["num_success"]
    ), "The attempt log and result count different successes."
    run_ids = {entry["run_id"], result["run"]["run_id"], *(attempt["run_id"] for attempt in attempts)}
    assert len(run_ids) == 1, f"The dataset, result and attempt log carry different run ids: {run_ids}"
    return attempts


PAIR_METRICS = frozenset({"workspace_coverage", "start_spread", "approach_spread", "behavior_diversity"})
"""Diversity metrics averaged over pairs of episodes; they apply once a run exports two episodes."""


def assert_ok_metrics(
    dataset_file: str, result_file: str, expected_ok: set[str], pair_metrics: frozenset[str] = frozenset()
) -> None:
    """Report a generation run's metrics with ``validate_dataset.py``; assert every episode passes the checks and
    exactly ``expected_ok`` computed a value, with ``pair_metrics`` too when the run exported two episodes or more.

    A renamed recorded key or an adapter that stops recording a stream turns a metric not applicable, which
    this catches.
    """

    report_file = os.path.join(os.path.dirname(dataset_file), "report.json")
    run_subprocess([
        TestPaths.python_path,
        os.path.join(TestPaths.scripts_dir, "validate_dataset.py"),
        dataset_file,
        "--report",
        "--result_file",
        result_file,
        "--report_file",
        report_file,
    ])
    with open(report_file, encoding="utf-8") as report_handle:
        report = json.load(report_handle)
    assert (
        report["num_valid_episodes"] == report["num_episodes"]
    ), f"{report['num_episodes'] - report['num_valid_episodes']} episodes failed checks: {report['check_failures']}"
    rows = report["metrics"]
    ok = {name for name, row in rows.items() if row["status"] == "ok"}
    expected = expected_ok | (pair_metrics if report["num_valid_episodes"] >= 2 else set())
    assert ok == expected, f"Computed {sorted(ok)}, expected {sorted(expected)}: {rows}"


def assert_no_metric_inputs(dataset_file: str) -> None:
    """Assert a dataset recorded without ``--record_metric_inputs`` has no metric inputs, as before they existed."""

    with h5py.File(dataset_file, "r") as h5_file:
        data = h5_file["data"]
        assert "autodata" not in json.loads(data.attrs["env_args"]), "The dataset has an autodata entry."
        for name, episode in data.items():
            assert "step_success" not in episode, f"{name} records step_success."
            assert "datagen_info" not in episode["obs"], f"{name} records datagen_info."
