# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assertion helpers for the AutoData test suite."""

import contextlib
import h5py
import json
import os
import shutil
import tempfile
from collections.abc import Iterator

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


def report_metrics(dataset_file: str, result_file: str) -> dict:
    """Report a generation run's metrics with ``validate_dataset.py`` into ``report.json`` beside the dataset.

    Returns:
        The report.
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
        return json.load(report_handle)


def assert_ok_metrics(dataset_file: str, result_file: str, expected_ok: set[str]) -> None:
    """Report a generation run's metrics; assert every episode passes the checks and the metrics in ``expected_ok``
    computed a value, as :func:`assert_computed_metrics` describes."""

    report = report_metrics(dataset_file, result_file)
    assert (
        report["num_valid_episodes"] == report["num_episodes"]
    ), f"{report['num_episodes'] - report['num_valid_episodes']} episodes failed checks: {report['check_failures']}"
    assert_computed_metrics(report, expected_ok)


def assert_computed_metrics(report: dict, expected_ok: set[str], optional: frozenset[str] = frozenset()) -> None:
    """Assert exactly ``expected_ok`` computed a value, with ``start_layout_spread`` and ``workspace_spread`` too when
    two episodes or more are valid; the metrics in ``optional`` may compute or not.

    A renamed recorded key or an adapter that stops recording a stream turns a metric not applicable, which
    this catches.
    """

    rows = report["metrics"]
    ok = {name for name, row in rows.items() if row["status"] == "ok"} - optional
    spreads = {"start_layout_spread", "workspace_spread"} if report["num_valid_episodes"] >= 2 else set()
    expected = (expected_ok | spreads) - optional
    assert ok == expected, f"Computed {sorted(ok)}, expected {sorted(expected)}: {rows}"


@contextlib.contextmanager
def benchmark_temp_dir(name: str) -> Iterator[str]:
    """Yield a temporary directory for the benchmark ``name``. On exit, also after a failure, copy its files other than
    HDF5 datasets (the generation result, its attempt log and the metrics report) to ``$AUTODATA_BENCHMARK_DIR/<name>``
    when that variable is set. A failed copy fails a passing benchmark, but does not replace a failing one's error.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            yield temp_dir
        except BaseException:
            with contextlib.suppress(OSError):
                _export_benchmark_files(temp_dir, name)
            raise
        _export_benchmark_files(temp_dir, name)


def _export_benchmark_files(source_dir: str, name: str) -> None:
    """Copy the files of ``source_dir`` other than HDF5 datasets to ``$AUTODATA_BENCHMARK_DIR/<name>``, if set."""

    if not os.environ.get("AUTODATA_BENCHMARK_DIR"):
        return
    export_dir = os.path.join(os.environ["AUTODATA_BENCHMARK_DIR"], name)
    os.makedirs(export_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
        path = os.path.join(source_dir, file_name)
        if os.path.isfile(path) and not file_name.endswith(".hdf5"):
            # copyfile, not copy2: the result file is written private (0600), and the copy must be readable by
            # whatever uploads it.
            shutil.copyfile(path, os.path.join(export_dir, file_name))


def assert_no_metric_inputs(dataset_file: str) -> None:
    """Assert a dataset recorded without ``--record_metric_inputs`` has no metric inputs, as before they existed."""

    with h5py.File(dataset_file, "r") as h5_file:
        data = h5_file["data"]
        assert "autodata" not in json.loads(data.attrs["env_args"]), "The dataset has an autodata entry."
        for name, episode in data.items():
            assert "step_success" not in episode, f"{name} records step_success."
            assert "datagen_info" not in episode["obs"], f"{name} records datagen_info."
