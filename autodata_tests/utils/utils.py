# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assertion helpers for the AutoData test suite."""

import h5py
import json
import os


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

    Every episode has one ``step_success`` value per action, its success attribute equals whether
    ``step_success`` ever held, and the entry counts every exported episode.

    Returns:
        The dataset's ``autodata`` entry.
    """

    with h5py.File(dataset_file, "r") as h5_file:
        data = h5_file["data"]
        entry = json.loads(data.attrs["env_args"])["autodata"]
        assert entry["exported_episodes"] == len(data), f"The entry counts {entry['exported_episodes']} episodes."
        for name, episode in data.items():
            step_success = episode["step_success"][:]
            assert len(step_success) == len(episode["actions"]), f"{name}: step_success and actions differ in length."
            assert bool(step_success.any()) == bool(episode.attrs["success"]), f"{name}: step_success disagrees."
    return entry


def assert_recorded_run(dataset_file: str, result_file: str) -> list[dict]:
    """Assert a generation run's dataset, result JSON and attempt log describe the same run.

    Returns:
        The attempt-log lines.
    """

    entry = assert_recorded_episodes(dataset_file)
    with open(result_file, encoding="utf-8") as result_handle:
        result = json.load(result_handle)
    attempts_file = os.path.join(os.path.dirname(result_file), result["run"]["attempts_file"])
    with open(attempts_file, encoding="utf-8") as attempts_handle:
        attempts = [json.loads(line) for line in attempts_handle if line.strip()]
    assert entry["exported_episodes"] == result["num_success"], "The dataset and result count different successes."
    assert len(attempts) == result["num_attempts"], "The attempt log and result count different attempts."
    assert (
        sum(attempt["success"] for attempt in attempts) == result["num_success"]
    ), "The attempt log and result count different successes."
    run_ids = {entry["run_id"], result["run"]["run_id"], *(attempt["run_id"] for attempt in attempts)}
    assert len(run_ids) == 1, f"The dataset, result and attempt log carry different run ids: {run_ids}"
    return attempts


def assert_no_metric_inputs(dataset_file: str) -> None:
    """Assert a dataset recorded without ``--record_metric_inputs`` has no metric inputs, as before they existed."""

    with h5py.File(dataset_file, "r") as h5_file:
        data = h5_file["data"]
        assert "autodata" not in json.loads(data.attrs["env_args"]), "The dataset has an autodata entry."
        for name, episode in data.items():
            assert "step_success" not in episode, f"{name} records step_success."
            assert "datagen_info" not in episode["obs"], f"{name} records datagen_info."
