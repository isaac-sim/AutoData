# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``autodata`` entry of a dataset's ``env_args``: what the dataset metrics need to know about its making."""

from __future__ import annotations

import h5py
import json
import os
from collections.abc import Mapping
from typing import Any

AUTODATA_ENTRY_VERSION = 1


def exported_dataset_files(output_dir: str, output_file_name: str, kept_failed: bool) -> list[str]:
    """The HDF5 files this run's recorder exported.

    A ``_failed`` file counts only when this run kept failed episodes; otherwise one left by an earlier run would be
    labelled with this run's id.

    Args:
        output_dir: Export directory.
        output_file_name: Dataset file name without the extension.
        kept_failed: Whether the run exported failed episodes to a separate ``_failed`` file.
    """
    suffixes = ("", "_failed") if kept_failed else ("",)
    candidates = [os.path.join(output_dir, f"{output_file_name}{suffix}.hdf5") for suffix in suffixes]
    return [path for path in candidates if os.path.exists(path)]


def merge_autodata_entry(dataset_path: str, entry: Mapping[str, Any]) -> None:
    """Store ``entry`` under ``autodata`` in the dataset's ``env_args``, with the number of exported episodes.

    Call it after the env has closed the file. The episode count lets a reader detect a dataset merged from
    several runs, since merging keeps the first file's ``env_args``.

    Args:
        dataset_path: Exported HDF5 dataset.
        entry: The entry, as built by :func:`~autodata_interfaces.env.collect_autodata_entry`.
    """
    with h5py.File(dataset_path, "r+") as h5_file:
        data = h5_file["data"]
        raw_env_args = data.attrs.get("env_args", "{}")
        env_args = json.loads(raw_env_args.decode() if isinstance(raw_env_args, bytes) else raw_env_args)
        env_args["autodata"] = {**entry, "exported_episodes": len(data)}
        data.attrs["env_args"] = json.dumps(env_args)
