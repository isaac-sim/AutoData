# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared assertion helpers for the AutoData test suite."""

import h5py
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
