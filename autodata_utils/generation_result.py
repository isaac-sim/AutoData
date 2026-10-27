# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured result files for completed data-generation runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_generation_result(
    result_file: str,
    algorithm: str,
    requested_trials: int,
    stats: Mapping[str, int],
    env_profile: Mapping[str, str | None] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
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
        run_metadata: How the run was configured and how long it took, stored under ``run``.
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
    if run_metadata is not None:
        result["run"] = dict(run_metadata)

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


def attempts_file_for(result_file: str) -> str:
    """Path of the attempt log written next to ``result_file``."""
    return str(Path(result_file).with_suffix(".attempts.jsonl"))


class AttemptLog:
    """Appends one JSON line per completed generation attempt, stamped with the run id."""

    def __init__(self, path: str, run_id: str) -> None:
        self._path = Path(path)
        self._run_id = run_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": self._run_id, **record}, sort_keys=True) + "\n")


def input_record(path: str | None) -> dict[str, str] | None:
    """Path and SHA-256 of a run input, or None when the input was not given."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": path, "sha256": digest.hexdigest()}


def code_revision() -> str | None:
    """Git commit of the AutoData checkout, or None without git or outside a checkout."""
    if shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None
