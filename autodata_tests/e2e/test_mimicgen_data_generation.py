# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import h5py
import json
import numpy as np
import os
import tempfile

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_kpi_report, assert_valid_dataset

HEADLESS = True
GENERATION_NUM_TRIALS = 1
# KPIs that a one-demo MimicGen run with a source dataset and embodiment must produce.
APPLICABLE_KPIS = {
    "success_rate",
    "demos_per_hour",
    "demos_per_human_demo",
    "tracking_error",
    "action_saturation",
    "motion_smoothness",
    "failure_share",
}


def _run_franka_cube_stack_mimicgen(num_envs: int, device: str) -> None:
    """Run MimicGen data generation for the Franka cube-stack task, then report its KPIs."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")
        result_file = os.path.join(temp_dir, "generation_result.json")
        report_file = os.path.join(temp_dir, "kpis.json")
        source_file = os.path.join(TestPaths.test_data_dir, "annotated_dataset_franka_stack_mimicgen.hdf5")
        embodiment_file = os.path.join(TestPaths.embodiments_dir, "franka_ik_rel.yaml")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "IsaacContrib-Stack-Cube-Franka-IK-Rel",
            "--alg",
            "mimicgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "franka_cube_stack.yaml"),
            "--embodiment",
            embodiment_file,
            "--input_file",
            source_file,
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
        with h5py.File(output_file, "r") as dataset:
            for episode in dataset["data"].values():
                assert len(episode["obs/datagen_info/target_eef_pose/franka"]) == len(episode["actions"])
                # franka_ik_rel.yaml has no EEF offset, so the recorded pose must equal the pre-step observation.
                recorded = episode["obs/datagen_info/eef_pose/franka"][:, :3, 3]
                np.testing.assert_allclose(recorded, episode["obs/eef_pos"][:], atol=1e-6)
        with open(result_file, encoding="utf-8") as result_handle:
            result = json.load(result_handle)

        run_subprocess([
            TestPaths.python_path,
            os.path.join(TestPaths.scripts_dir, "validate_dataset.py"),
            output_file,
            "--report",
            "--result_file",
            result_file,
            "--source",
            source_file,
            "--embodiment",
            embodiment_file,
            "--report_file",
            report_file,
        ])
        kpis = assert_kpi_report(report_file, APPLICABLE_KPIS)
        hardware = result["run"]["gpu"] if device == "cuda" else result["run"]["cpu"]
        assert hardware and hardware in kpis["demos_per_hour"]["note"]


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
