# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structured data-generation result files."""

import json
from pathlib import Path

from autodata_utils.generation_result import AttemptLog, attempts_file_for, write_generation_result


def test_write_generation_result_writes_completed_stats_atomically(tmp_path):
    result_file = tmp_path / "results" / "generation_result.json"

    write_generation_result(
        result_file=str(result_file),
        algorithm="mimicgen",
        requested_trials=10,
        stats={"num_success": 8, "num_failures": 2, "num_attempts": 10},
    )

    with result_file.open(encoding="utf-8") as result_handle:
        result = json.load(result_handle)
    assert result == {
        "algorithm": "mimicgen",
        "num_attempts": 10,
        "num_failures": 2,
        "num_success": 8,
        "requested_trials": 10,
    }
    assert not list(result_file.parent.glob(f".{result_file.name}.*.tmp"))


def test_write_generation_result_records_env_profile_when_given(tmp_path):
    result_file = tmp_path / "generation_result.json"

    write_generation_result(
        result_file=str(result_file),
        algorithm="skillgen",
        requested_trials=1,
        stats={"num_success": 1, "num_failures": 0, "num_attempts": 1},
        env_profile={
            "name": "franka_bin_stack",
            "path": "autodata_examples/env_profiles/franka_bin_stack.yaml",
            "planner": "franka_stack_cube_bin",
        },
    )

    with result_file.open(encoding="utf-8") as result_handle:
        result = json.load(result_handle)
    assert result["env_profile"] == {
        "name": "franka_bin_stack",
        "path": "autodata_examples/env_profiles/franka_bin_stack.yaml",
        "planner": "franka_stack_cube_bin",
    }


def test_rerun_with_the_same_result_file_keeps_the_earlier_attempt_log(tmp_path: Path) -> None:
    result_file = str(tmp_path / "result.json")
    AttemptLog(attempts_file_for(result_file, "run-a"), "run-a").write({"success": True})

    AttemptLog(attempts_file_for(result_file, "run-b"), "run-b")

    earlier = Path(attempts_file_for(result_file, "run-a")).read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in earlier] == ["run-a"]
