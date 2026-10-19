# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SkillGen data-generation test for the Franka bin-stack variant.

The bin variant runs on the generic ``IsaacContrib-Stack-Cube-Franka-IK-Rel`` task with the
``franka_bin_stack`` environment profile overlaying the sorting-bin scene and its reset
distributions, and reuses the cube-stack annotated SkillGen source dataset — no specialized
bin env id is involved.
"""

import json
import os
import tempfile

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_kpi_report, assert_valid_dataset, read_attempt_log

HEADLESS = True
GENERATION_NUM_TRIALS = 1
# KPIs that a one-demo SkillGen run with a result file must produce.
APPLICABLE_KPIS = {"success_rate", "failure_share", "planner_failure_types", "planning_time"}


def _run_franka_bin_stack_skillgen(num_envs: int, device: str) -> None:
    """Run SkillGen data generation for the Franka bin-stack variant."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")
        result_file = os.path.join(temp_dir, "generation_result.json")
        report_file = os.path.join(temp_dir, "kpis.json")
        env_profile = os.path.join(TestPaths.env_profiles_dir, "franka_bin_stack.yaml")

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "IsaacContrib-Stack-Cube-Franka-IK-Rel",
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
        attempts = read_attempt_log(result_file)
        # Every SkillGen attempt plans at least its first transit.
        assert all(attempt["planner_calls"] for attempt in attempts)
        for attempt in attempts:
            last_call = attempt["planner_calls"][-1]
            planner_failed = attempt["failure_cause"] == "planner"
            assert last_call["success"] != planner_failed and (last_call["status"] == "Success") != planner_failed

        run_subprocess([
            TestPaths.python_path,
            os.path.join(TestPaths.scripts_dir, "validate_dataset.py"),
            output_file,
            "--report",
            "--result_file",
            result_file,
            "--report_file",
            report_file,
        ])
        assert_kpi_report(report_file, APPLICABLE_KPIS)


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
