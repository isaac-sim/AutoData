# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that shipped datasets store the current registered task IDs directly."""

import gymnasium as gym
import h5py
import json
from pathlib import Path

import pytest

from autodata_interfaces.env import get_env_name_from_dataset
from autodata_tests.utils.constants import TestPaths


@pytest.mark.parametrize(
    ("dataset_path", "expected_task"),
    [
        (
            "autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5",
            "IsaacContrib-Stack-Cube-Franka-IK-Rel",
        ),
        (
            "autodata_tests/test_data/annotated_dataset_franka_stack_skillgen.hdf5",
            "IsaacContrib-Stack-Cube-Franka-IK-Rel",
        ),
        (
            "autodata_tests/test_data/annotated_dataset_gr1_pick_place_dexmimicgen.hdf5",
            "IsaacContrib-PickPlace-GR1T2-Abs",
        ),
        (
            "autodata_tests/test_data/annotated_dataset_g1_pick_place_dexmimicgen.hdf5",
            "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        ),
        ("datasets/annotated_datasets/dataset_franka_annotated.hdf5", "IsaacContrib-Stack-Cube-Franka-IK-Rel"),
        ("datasets/annotated_datasets/dataset_franka_skillgen_annotated.hdf5", "IsaacContrib-Stack-Cube-Franka-IK-Rel"),
        ("datasets/annotated_datasets/dataset_gr1_annotated.hdf5", "IsaacContrib-PickPlace-GR1T2-Abs"),
        ("datasets/annotated_datasets/dataset_g1_annotated.hdf5", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs"),
        ("datasets/teleop_datasets/dataset_franka.hdf5", "IsaacContrib-Stack-Cube-Franka-IK-Rel"),
        ("datasets/teleop_datasets/dataset_gr1.hdf5", "IsaacContrib-PickPlace-GR1T2-Abs"),
        ("datasets/teleop_datasets/dataset_g1.hdf5", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs"),
    ],
)
def test_source_dataset_stores_registered_task(dataset_path, expected_task):
    """Read raw HDF5 metadata so a runtime alias cannot mask an outdated dataset."""
    path = Path(TestPaths.repo_root) / dataset_path
    with h5py.File(path, "r") as dataset:
        task_name = json.loads(dataset["data"].attrs["env_args"])["env_name"]
    assert task_name == expected_task
    assert get_env_name_from_dataset(str(path)) == task_name
    assert gym.spec(task_name).id == task_name
