# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data-generation success-rate perf test for the Franka cube-stack (MimicGen) task."""

import os
import tempfile

import pytest

from isaac_autodata_tests.utils.constants import TestPaths
from isaac_autodata_tests.utils.subprocess import run_subprocess_capture
from isaac_autodata_tests.utils.utils import assert_valid_dataset, parse_datagen_success_rate

# --- Tunables (change these to adjust the perf test) ---------------------------------------------
SUCCESS_RATE_THRESHOLD = 0.30  # minimum acceptable data-gen success rate
NUM_ENVS = 100
NUM_TRIALS = 500
DEVICE = "cuda"
HEADLESS = True
TIMEOUT_SEC = 7200  # generous wall-clock budget for generating NUM_TRIALS demos across NUM_ENVS envs
# -------------------------------------------------------------------------------------------------


@pytest.mark.with_subprocess
def test_mimicgen_franka_data_generation_success_rate():
    """Generate NUM_TRIALS demos on NUM_ENVS envs and assert the success rate exceeds the threshold."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--task",
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
            str(NUM_TRIALS),
            "--num_envs",
            str(NUM_ENVS),
            "--device",
            DEVICE,
            "--viz",
            "none" if HEADLESS else "kit",
        ]
        output = run_subprocess_capture(args, timeout_sec=TIMEOUT_SEC)
        assert_valid_dataset(output_file, min_num_demos=1)

    num_success, num_attempts = parse_datagen_success_rate(output)
    assert num_attempts > 0, "generation reported zero attempts"
    success_rate = num_success / num_attempts
    assert success_rate >= SUCCESS_RATE_THRESHOLD, (
        f"data-gen success rate {success_rate:.1%} ({num_success}/{num_attempts}) is below the "
        f"required threshold of {SUCCESS_RATE_THRESHOLD:.0%}"
    )


if __name__ == "__main__":
    test_mimicgen_franka_data_generation_success_rate()
