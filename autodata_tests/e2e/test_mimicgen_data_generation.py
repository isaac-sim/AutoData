# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_valid_dataset

HEADLESS = True
GENERATION_NUM_TRIALS = 1


def _run_franka_cube_stack_mimicgen(num_envs: int, device: str) -> None:
    """Run MimicGen data generation for the Franka cube-stack task"""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "Isaac-Stack-Cube-Franka-IK-Rel-v0",
            "--alg",
            "mimicgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "franka_cube_stack.yaml"),
            "--embodiment",
            os.path.join(TestPaths.embodiments_dir, "franka_ik_rel.yaml"),
            "--input_file",
            os.path.join(TestPaths.test_data_dir, "annotated_dataset_franka_stack_mimicgen.hdf5"),
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
def test_franka_cube_stack_mimicgen_data_generation_single_env_cpu():
    """MimicGen generation for the Franka cube-stack task on a single env on CPU."""
    _run_franka_cube_stack_mimicgen(num_envs=1, device="cpu")


@pytest.mark.with_subprocess
def test_franka_cube_stack_mimicgen_data_generation_multi_env_cpu():
    """MimicGen generation for the Franka cube-stack task on multiple parallel envs on CPU."""
    _run_franka_cube_stack_mimicgen(num_envs=10, device="cpu")


@pytest.mark.with_subprocess
def test_franka_cube_stack_mimicgen_data_generation_single_env_cuda():
    """MimicGen generation for the Franka cube-stack task on a single env on GPU."""
    _run_franka_cube_stack_mimicgen(num_envs=1, device="cuda")


@pytest.mark.with_subprocess
def test_franka_cube_stack_mimicgen_data_generation_multi_env_cuda():
    """MimicGen generation for the Franka cube-stack task on multiple parallel envs on GPU."""
    _run_franka_cube_stack_mimicgen(num_envs=10, device="cuda")


if __name__ == "__main__":
    test_franka_cube_stack_mimicgen_data_generation_single_env_cpu()
    test_franka_cube_stack_mimicgen_data_generation_multi_env_cpu()
    test_franka_cube_stack_mimicgen_data_generation_single_env_cuda()
    test_franka_cube_stack_mimicgen_data_generation_multi_env_cuda()
