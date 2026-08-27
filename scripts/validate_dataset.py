# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structurally validate Isaac AutoData HDF5 datasets.

Prints a summary table (episode count, env id, sim args) per file followed by any issues.

Usage::

    python scripts/validate_dataset.py <file.hdf5> [<file2.hdf5> ...]
    python scripts/validate_dataset.py directory/*.hdf5

The command exits nonzero if any supplied file is invalid.
"""

from __future__ import annotations

import argparse
import h5py
import json
import os
import sys

REQUIRED_DEMO_FIELDS = {
    "actions": h5py.Dataset,
    "initial_state": h5py.Group,
    "obs": h5py.Group,
}


class ValidationResult:
    """Outcome of validating a single HDF5 file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.num_episodes: int | None = None
        self.env_name: str | None = None
        self.sim_args: dict | None = None
        self.env_args: object | None = None
        self.issues: list[str] = []

    @property
    def valid(self) -> bool:
        return not self.issues


def _decode(value: object) -> str:
    """Decode an HDF5 attribute value to str."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value)


def validate_file(path: str) -> ValidationResult:
    """Structurally validate one HDF5 dataset and collect its metadata."""

    result = ValidationResult(path)

    if not os.path.exists(path):
        result.issues.append("file does not exist")
        return result

    try:
        h5_file = h5py.File(path, "r")
    except Exception as exc:  # report any failures (not HDF5, corrupt, permissions, etc.)
        result.issues.append(f"could not open as HDF5: {exc}")
        return result

    with h5_file:
        if "data" not in h5_file:
            result.issues.append("missing top-level 'data' group")
            return result
        data = h5_file["data"]
        if not isinstance(data, h5py.Group):
            result.issues.append("top-level 'data' is not a group")
            return result

        _read_env_metadata(data, result)

        episode_names = list(data.keys())
        result.num_episodes = len(episode_names)
        if result.num_episodes == 0:
            result.issues.append("dataset is empty (no episodes under 'data')")
            return result

        for name in episode_names:
            demo = data[name]
            if not isinstance(demo, h5py.Group):
                result.issues.append(f"{name}: not a group")
                continue
            missing = [field for field in REQUIRED_DEMO_FIELDS if field not in demo]
            if missing:
                result.issues.append(f"{name}: missing required field(s): {', '.join(missing)}")
            # Present fields must be the right kind: actions is a dataset, initial_state/obs are groups.
            for field, expected_kind in REQUIRED_DEMO_FIELDS.items():
                if field in demo and not isinstance(demo[field], expected_kind):
                    kind = "group" if expected_kind is h5py.Group else "dataset"
                    result.issues.append(f"{name}: '{field}' must be a {kind}")

    return result


def _read_env_metadata(data: h5py.Group, result: ValidationResult) -> None:
    """Parse the env_args attribute of the dataset (env id + sim args)."""

    env_args_raw = data.attrs.get("env_args")
    if env_args_raw is None:
        return
    try:
        env_args = json.loads(_decode(env_args_raw))
    except (json.JSONDecodeError, TypeError):
        result.env_args = _decode(env_args_raw)  # keep the raw value if it is not valid JSON
        return
    result.env_args = env_args
    if isinstance(env_args, dict):
        result.env_name = env_args.get("env_name")
        sim_args = env_args.get("sim_args")
        result.sim_args = sim_args if isinstance(sim_args, dict) else None


def _format_sim_args(sim_args: dict | None) -> str:
    if not sim_args:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sim_args.items())


def _render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a table given headers and rows."""

    columns = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(cell)) for cell in col) for col in columns]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths))

    return "\n".join([fmt(headers), fmt(tuple("-" * w for w in widths)), *(fmt(row) for row in rows)])


def print_results(results: list[ValidationResult]) -> None:
    """Print the summary table followed by per-file issues."""
    rows = [
        (
            r.path,
            "-" if r.num_episodes is None else str(r.num_episodes),
            r.env_name or "-",
            _format_sim_args(r.sim_args),
            "VALID" if r.valid else "INVALID",
        )
        for r in results
    ]
    print(_render_table(("File", "Episodes", "Env ID", "sim_args", "Status"), rows))

    for r in results:
        if not r.valid:
            print(f"\n{r.path}: INVALID")
            for issue in r.issues:
                print(f"  - {issue}")


def main(argv: list[str] | None = None) -> int:
    """Validate the requested files and return nonzero if any are invalid."""

    parser = argparse.ArgumentParser(description="Validate Isaac AutoData demonstration HDF5 datasets.")
    parser.add_argument("files", nargs="+", help="HDF5 dataset file(s) to validate.")
    args = parser.parse_args(argv)

    results = [validate_file(path) for path in args.files]
    print_results(results)

    num_invalid = sum(not r.valid for r in results)
    total = len(results)
    print()
    print(f"All {total} file(s) valid." if num_invalid == 0 else f"{num_invalid} of {total} file(s) invalid.")

    return 1 if num_invalid else 0


if __name__ == "__main__":
    sys.exit(main())
