# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
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

    assert result["status"] == "completed", f"Generation did not complete: {result['status']}."
    num_success = result["num_success"]
    num_failures = result["num_failures"]
    num_attempts = result["num_attempts"]
    assert num_success >= 0 and num_failures >= 0 and num_attempts > 0, "Generation result has invalid counts."
    assert num_success + num_failures == num_attempts, "Generation result counts do not add up."
    return num_success, num_attempts
