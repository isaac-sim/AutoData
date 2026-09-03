# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HDF5 dataset validator."""

import h5py
import json
from pathlib import Path

from scripts import validate_dataset


def _create_dataset(path: Path, *, include_actions: bool = True) -> None:
    """Create the smallest dataset accepted by the structural validator."""

    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test-Env-v0", "sim_args": {"device": "cpu"}})
        demo = data.create_group("demo_0")
        if include_actions:
            demo.create_dataset("actions", data=[[0.0]])
        demo.create_group("initial_state")
        demo.create_group("obs")


def test_main_returns_zero_for_valid_dataset(tmp_path: Path) -> None:
    """A structurally valid dataset should produce a successful exit status."""

    dataset_path = tmp_path / "valid.hdf5"
    _create_dataset(dataset_path)

    assert validate_dataset.main([str(dataset_path)]) == 0


def test_main_returns_nonzero_for_invalid_dataset_by_default(tmp_path: Path) -> None:
    """An invalid dataset should fail without requiring an opt-in CLI flag."""

    dataset_path = tmp_path / "invalid.hdf5"
    _create_dataset(dataset_path, include_actions=False)

    assert validate_dataset.main([str(dataset_path)]) == 1


def test_main_returns_nonzero_for_missing_file(tmp_path: Path) -> None:
    """An unreadable input path should produce a failing exit status."""

    assert validate_dataset.main([str(tmp_path / "missing.hdf5")]) == 1
