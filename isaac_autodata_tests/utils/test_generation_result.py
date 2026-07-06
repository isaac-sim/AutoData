# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for structured data-generation result files."""

import json

from isaac_autodata_utils.generation_result import GENERATION_RESULT_SCHEMA_VERSION, write_generation_result


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
        "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
        "status": "completed",
    }
    assert not list(result_file.parent.glob(f".{result_file.name}.*.tmp"))
