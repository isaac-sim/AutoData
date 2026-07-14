# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SkillGen data-generation test for the Franka bin-stack variant.

The bin variant runs on the generic ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` task with the
``franka_bin_stack`` environment profile overlaying the sorting-bin scene and its reset
distributions, and reuses the cube-stack annotated SkillGen source dataset — no specialized
bin env id is involved.
"""

import json
import os
import tempfile

import pytest

from isaac_autodata_tests.utils.constants import TestPaths
from isaac_autodata_tests.utils.subprocess import run_subprocess
from isaac_autodata_tests.utils.utils import assert_valid_dataset

HEADLESS = True
GENERATION_NUM_TRIALS = 1


def _run_franka_bin_stack_skillgen(num_envs: int, device: str) -> None:
    """Run SkillGen data generation for the Franka bin-stack variant."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")
        result_file = os.path.join(temp_dir, "generation_result.json")
        env_profile = os.path.join(TestPaths.env_profiles_dir, "franka_bin_stack.yaml")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--task",
            "Isaac-Stack-Cube-Franka-IK-Rel-v0",
            "--alg",
            "skillgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "franka_bin_stack_skillgen.yaml"),
            "--env_profile",
            env_profile,
            "--embodiment",
            os.path.join(TestPaths.embodiments_dir, "franka_ik_rel_skillgen.yaml"),
            "--input_file",
            os.path.join(TestPaths.test_data_dir, "annotated_dataset_franka_stack_skillgen.hdf5"),
            "--output_file",
            output_file,
            "--result_file",
            result_file,
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

        with open(result_file, encoding="utf-8") as result_handle:
            result = json.load(result_handle)
        assert result["env_profile"] == {
            "name": "franka_bin_stack",
            "path": env_profile,
            "planner": "franka_stack_cube_bin",
        }


@pytest.mark.with_subprocess
def test_franka_bin_stack_skillgen_data_generation_single_env_cuda():
    """SkillGen generation for the Franka bin-stack variant on a single env on GPU."""
    _run_franka_bin_stack_skillgen(num_envs=1, device="cuda")


@pytest.mark.with_subprocess
def test_franka_bin_stack_skillgen_data_generation_multi_env_cuda():
    """SkillGen generation for the Franka bin-stack variant on multiple parallel envs on GPU."""
    _run_franka_bin_stack_skillgen(num_envs=3, device="cuda")


if __name__ == "__main__":
    test_franka_bin_stack_skillgen_data_generation_single_env_cuda()
    test_franka_bin_stack_skillgen_data_generation_multi_env_cuda()
