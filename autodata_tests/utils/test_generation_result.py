# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structured data-generation result files."""

import json
from pathlib import Path
from types import SimpleNamespace

from autodata_utils.generation_result import (
    AttemptLog,
    attempts_file_for,
    record_generation_attempt,
    write_generation_result,
)


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
        environment={"name": "example", "physics_manager": "example:Physics", "control_dt_s": 0.05},
        execution={"mode": "buffered", "num_envs": 2, "timings_s": {"replay_s": 1.5}},
    )

    with result_file.open(encoding="utf-8") as result_handle:
        result = json.load(result_handle)
    assert result["env_profile"] == {
        "name": "franka_bin_stack",
        "path": "autodata_examples/env_profiles/franka_bin_stack.yaml",
        "planner": "franka_stack_cube_bin",
    }
    assert result["environment"] == {
        "name": "example",
        "physics_manager": "example:Physics",
        "control_dt_s": 0.05,
    }
    assert result["execution"] == {"mode": "buffered", "num_envs": 2, "timings_s": {"replay_s": 1.5}}


def test_record_generation_attempt_counts_pre_replay_failures():
    stats = {"num_success": 0, "num_failures": 0, "num_attempts": 0}
    attempts = []
    for env_id, success in ((0, False), (1, True)):
        record_generation_attempt(
            SimpleNamespace(success=success, diagnostics=(("stage", "complete" if success else "planning"),)),
            env_id,
            stats,
            attempts,
        )
    assert stats == {"num_success": 1, "num_failures": 1, "num_attempts": 2}
    assert [attempt["attempt"] for attempt in attempts] == [1, 2]
    assert [attempt["env_id"] for attempt in attempts] == [0, 1]
    assert attempts[0]["diagnostics"] == {"stage": "planning"}


def test_rerun_with_the_same_result_file_keeps_the_earlier_attempt_log(tmp_path: Path) -> None:
    result_file = str(tmp_path / "result.json")
    AttemptLog(attempts_file_for(result_file, "run-a"), "run-a").write({"success": True})

    AttemptLog(attempts_file_for(result_file, "run-b"), "run-b")

    earlier = Path(attempts_file_for(result_file, "run-a")).read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in earlier] == ["run-a"]


def test_attempt_log_accepts_scoped_and_batched_initial_states(tmp_path):
    """Metrics select row zero for a scoped snapshot and the environment row for a batch."""
    import torch

    for count in (1, 3):
        path = tmp_path / f"{count}.jsonl"
        poses = torch.arange(count * 7).reshape(count, 7)
        result = SimpleNamespace(
            success=True,
            diagnostics=(),
            extras={"planning_failed": False},
            initial_state={"rigid_object": {"bowl": {"root_pose": poses}}},
        )
        record_generation_attempt(
            result,
            2,
            {"num_success": 0, "num_failures": 0, "num_attempts": 0},
            [],
            attempt_log=AttemptLog(str(path), "test"),
            start_time_s=0.0,
        )
        record = json.loads(path.read_text())
        assert record["initial_object_positions"]["bowl"] == poses[0 if count == 1 else 2, :2].tolist()
        assert record["extras"] == {"planning_failed": False}
