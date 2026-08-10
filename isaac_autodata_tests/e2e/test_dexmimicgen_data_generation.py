# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest

from isaac_autodata_tests.utils.constants import TestPaths
from isaac_autodata_tests.utils.subprocess import run_subprocess
from isaac_autodata_tests.utils.utils import assert_valid_dataset

HEADLESS = True
GENERATION_NUM_TRIALS = 1


def _run_gr1_pick_place_dexmimicgen(num_envs: int, device: str) -> None:
    """Run DexMimicGen data generation for the GR1T2 pick-place task."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "Isaac-PickPlace-GR1T2-Abs-v0",
            "--alg",
            "dexmimicgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "gr1_pick_place.yaml"),
            "--embodiment",
            os.path.join(TestPaths.embodiments_dir, "gr1_ik_abs.yaml"),
            "--input_file",
            os.path.join(TestPaths.test_data_dir, "annotated_dataset_gr1_pick_place_dexmimicgen.hdf5"),
            "--output_file",
            output_file,
            "--generation_num_trials",
            str(GENERATION_NUM_TRIALS),
            "--num_envs",
            str(num_envs),
            "--device",
            device,
            "--viz",
            "none" if HEADLESS else "kit",
        ]
        run_subprocess(args)

        assert_valid_dataset(output_file, min_num_demos=GENERATION_NUM_TRIALS)


def _run_g1_pick_place_dexmimicgen(num_envs: int, device: str) -> None:
    """Run DexMimicGen data generation for the G1 loco-manipulation pick-place task."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
            "--alg",
            "dexmimicgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "g1_pick_place.yaml"),
            "--embodiment",
            os.path.join(TestPaths.embodiments_dir, "g1_ik_abs.yaml"),
            "--input_file",
            os.path.join(TestPaths.test_data_dir, "annotated_dataset_g1_pick_place_dexmimicgen.hdf5"),
            "--output_file",
            output_file,
            "--generation_num_trials",
            str(GENERATION_NUM_TRIALS),
            "--num_envs",
            str(num_envs),
            "--device",
            device,
            "--viz",
            "none" if HEADLESS else "kit",
        ]
        run_subprocess(args)

        assert_valid_dataset(output_file, min_num_demos=GENERATION_NUM_TRIALS)


@pytest.mark.with_subprocess
def test_gr1_pick_place_dexmimicgen_data_generation_single_env_cpu():
    """DexMimicGen generation for the GR1T2 pick-place task on a single env on CPU."""
    _run_gr1_pick_place_dexmimicgen(num_envs=1, device="cpu")


@pytest.mark.with_subprocess
def test_gr1_pick_place_dexmimicgen_data_generation_multi_env_cpu():
    """DexMimicGen generation for the GR1T2 pick-place task on multiple parallel envs on CPU."""
    _run_gr1_pick_place_dexmimicgen(num_envs=10, device="cpu")


@pytest.mark.with_subprocess
def test_gr1_pick_place_dexmimicgen_data_generation_single_env_cuda():
    """DexMimicGen generation for the GR1T2 pick-place task on a single env on GPU."""
    _run_gr1_pick_place_dexmimicgen(num_envs=1, device="cuda")


@pytest.mark.with_subprocess
def test_gr1_pick_place_dexmimicgen_data_generation_multi_env_cuda():
    """DexMimicGen generation for the GR1T2 pick-place task on multiple parallel envs on GPU."""
    _run_gr1_pick_place_dexmimicgen(num_envs=10, device="cuda")


@pytest.mark.with_subprocess
def test_g1_pick_place_dexmimicgen_data_generation_single_env_cpu():
    """DexMimicGen generation for the G1 pick-place task on a single env on CPU."""
    _run_g1_pick_place_dexmimicgen(num_envs=1, device="cpu")


@pytest.mark.with_subprocess
def test_g1_pick_place_dexmimicgen_data_generation_multi_env_cpu():
    """DexMimicGen generation for the G1 pick-place task on multiple parallel envs on CPU."""
    _run_g1_pick_place_dexmimicgen(num_envs=10, device="cpu")


@pytest.mark.with_subprocess
def test_g1_pick_place_dexmimicgen_data_generation_single_env_cuda():
    """DexMimicGen generation for the G1 pick-place task on a single env on GPU."""
    _run_g1_pick_place_dexmimicgen(num_envs=1, device="cuda")


@pytest.mark.with_subprocess
def test_g1_pick_place_dexmimicgen_data_generation_multi_env_cuda():
    """DexMimicGen generation for the G1 pick-place task on multiple parallel envs on GPU."""
    _run_g1_pick_place_dexmimicgen(num_envs=10, device="cuda")


if __name__ == "__main__":
    test_gr1_pick_place_dexmimicgen_data_generation_single_env_cpu()
    test_gr1_pick_place_dexmimicgen_data_generation_multi_env_cpu()
    test_gr1_pick_place_dexmimicgen_data_generation_single_env_cuda()
    test_gr1_pick_place_dexmimicgen_data_generation_multi_env_cuda()
    test_g1_pick_place_dexmimicgen_data_generation_single_env_cpu()
    test_g1_pick_place_dexmimicgen_data_generation_multi_env_cpu()
    test_g1_pick_place_dexmimicgen_data_generation_single_env_cuda()
    test_g1_pick_place_dexmimicgen_data_generation_multi_env_cuda()
