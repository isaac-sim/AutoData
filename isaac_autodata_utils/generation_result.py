# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Structured result files for completed data-generation runs."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def write_generation_result(
    result_file: str,
    algorithm: str,
    requested_trials: int,
    stats: Mapping[str, int],
    env_profile: Mapping[str, str | None] | None = None,
) -> None:
    """Atomically write the final outcome of a completed data-generation run.

    Args:
        result_file: Destination JSON path.
        algorithm: Name of the generation algorithm that ran.
        requested_trials: Number of trials or successful demos requested by the generation policy.
        stats: Final ``num_success``, ``num_failures``, and ``num_attempts`` counters.
        env_profile: The ``name``, ``path``, and ``planner`` of the environment profile applied
            during generation, or None when the run used the unmodified base task. Recorded here
            because the generated dataset stores only the base env id and would otherwise not
            reveal that the scene was modified.
    """
    required_stats = ("num_success", "num_failures", "num_attempts")
    for stat_name in required_stats:
        assert stat_name in stats, f"Generation stats are missing {stat_name}."
        assert stats[stat_name] >= 0, f"Generation stat {stat_name} must be non-negative."
    assert (
        stats["num_success"] + stats["num_failures"] == stats["num_attempts"]
    ), "Generation success and failure counts must sum to the number of attempts."

    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "algorithm": algorithm,
        "requested_trials": requested_trials,
        "num_success": stats["num_success"],
        "num_failures": stats["num_failures"],
        "num_attempts": stats["num_attempts"],
    }
    if env_profile is not None:
        result["env_profile"] = dict(env_profile)

    file_descriptor, temporary_file_name = tempfile.mkstemp(
        prefix=f".{result_path.name}.", suffix=".tmp", dir=result_path.parent, text=True
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as result_handle:
            json.dump(result, result_handle, indent=2, sort_keys=True)
            result_handle.write("\n")
            result_handle.flush()
            os.fsync(result_handle.fileno())
        os.replace(temporary_file_name, result_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_file_name)
