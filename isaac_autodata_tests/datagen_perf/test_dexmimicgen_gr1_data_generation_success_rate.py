# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-generation success-rate perf test for the GR1T2 pick-place (DexMimicGen) task."""

import os
import tempfile

import pytest

from isaac_autodata_tests.utils.constants import TestPaths
from isaac_autodata_tests.utils.subprocess import run_subprocess
from isaac_autodata_tests.utils.utils import assert_valid_dataset, read_generation_result

# --- Tunables (change these to adjust the perf test) ---------------------------------------------
SUCCESS_RATE_THRESHOLD = 0.70  # minimum acceptable data-gen success rate
NUM_ENVS = 100
NUM_TRIALS = 500
DEVICE = "cpu"
HEADLESS = True
TIMEOUT_SEC = 7200  # generous wall-clock budget for generating NUM_TRIALS demos across NUM_ENVS envs
# -------------------------------------------------------------------------------------------------


@pytest.mark.with_subprocess
def test_dexmimicgen_gr1_data_generation_success_rate():
    """Generate NUM_TRIALS demos on NUM_ENVS envs and assert the success rate exceeds the threshold."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")
        result_file = os.path.join(temp_dir, "generation_result.json")

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
            "--result_file",
            result_file,
            "--generation_num_trials",
            str(NUM_TRIALS),
            "--num_envs",
            str(NUM_ENVS),
            "--device",
            DEVICE,
            "--viz",
            "none" if HEADLESS else "kit",
        ]
        run_subprocess(args, timeout_sec=TIMEOUT_SEC)
        assert_valid_dataset(output_file, min_num_demos=1)
        num_success, num_attempts = read_generation_result(result_file)

    assert num_attempts > 0, "generation reported zero attempts"
    success_rate = num_success / num_attempts
    print(f"\nDexMimicGen GR1 data-gen success rate: {success_rate:.1%} ({num_success}/{num_attempts})")
    assert success_rate >= SUCCESS_RATE_THRESHOLD, (
        f"data-gen success rate {success_rate:.1%} ({num_success}/{num_attempts}) is below the "
        f"required threshold of {SUCCESS_RATE_THRESHOLD:.0%}"
    )


if __name__ == "__main__":
    test_dexmimicgen_gr1_data_generation_success_rate()
