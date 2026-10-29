# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a metric reads: a dataset's episodes and ``env_args``, and its generation run's record and attempt log."""

from __future__ import annotations

import h5py
import json
import numpy as np
from collections.abc import Mapping
from functools import cached_property
from typing import Any


def _has_path(record: Any, dotted_key: str) -> bool:
    """Whether ``record[a][b]`` exists for ``dotted_key`` ``"a.b"``."""
    for part in dotted_key.split("."):
        if not isinstance(record, Mapping) or part not in record:
            return False
        record = record[part]
    return True


def read_env_args(h5_file: h5py.File) -> dict[str, Any]:
    """The dataset's ``env_args``, or empty when it records none.

    Raises:
        ValueError: If ``env_args`` is not a JSON object, or its ``autodata`` entry lacks the run id and exported
            episode count; a run cannot be verified against corrupt metadata.
    """
    raw_env_args = h5_file["data"].attrs.get("env_args", "{}")
    try:
        env_args = json.loads(raw_env_args.decode() if isinstance(raw_env_args, bytes) else raw_env_args)
    except (TypeError, ValueError):
        env_args = None
    if not isinstance(env_args, dict):
        raise ValueError(f"{h5_file.filename}: env_args are not a JSON object.")
    entry = env_args.get("autodata")
    if "autodata" in env_args and not (
        isinstance(entry, dict)
        and isinstance(entry.get("run_id"), str)
        and isinstance(entry.get("exported_episodes"), int)
    ):
        raise ValueError(f"{h5_file.filename}: the autodata entry lacks its run_id or exported_episodes.")
    return env_args


class Episode:
    """One episode of a dataset; arrays are read when accessed."""

    def __init__(self, group: h5py.Group) -> None:
        self.name = group.name.rsplit("/", 1)[-1]
        self._group = group

    def __contains__(self, key: str) -> bool:
        return key in self._group

    def __getitem__(self, key: str) -> np.ndarray:
        return self._group[key][()]

    def children(self, key: str) -> list[str]:
        """Names under the group ``key``, such as the end effectors under ``obs/datagen_info/eef_pose``."""
        return list(self._group[key])

    def datasets(self) -> list[tuple[str, h5py.Dataset]]:
        """Every dataset of the episode with its path, unread until accessed."""
        found: list[tuple[str, h5py.Dataset]] = []
        self._group.visititems(
            lambda path, item: found.append((path, item)) if isinstance(item, h5py.Dataset) else None
        )
        return found


class Evidence:
    """A dataset, and optionally the generation run that produced it, as metrics read them.

    Attributes:
        all_episodes: The dataset's episodes.
        episodes: The episodes that passed every episode check; dataset metrics read these.
        failures: Per episode check that applied to some episode, the failing episodes' names and reasons.
        check_groups: Per episode check considered, its group; a check absent from ``failures`` did not apply.
        env_args: The dataset's ``env_args``; its ``autodata`` entry was written by the producer.
        result: Generation result, or None for a dataset without a run record or an interrupted run.
        run: The result's ``run`` block, or empty.
        attempts: Attempt-log lines, or None.
        source: Evidence of the source dataset the run started from, or None.
    """

    def __init__(
        self,
        h5_file: h5py.File,
        result: Mapping[str, Any] | None = None,
        attempts: list[dict[str, Any]] | None = None,
        source: Evidence | None = None,
    ) -> None:
        self.env_args = read_env_args(h5_file)
        self.all_episodes = [Episode(group) for group in h5_file["data"].values()]
        self.episodes = list(self.all_episodes)
        self.failures: dict[str, dict[str, str]] = {}
        self.check_groups: dict[str, str] = {}
        self.result = result
        self.run: Mapping[str, Any] = (result or {}).get("run", {})
        self.attempts = attempts
        self.source = source

    def has(self, key: str) -> bool:
        """Whether the evidence key ``key`` is present.

        A key is one of:

        - an HDF5 path present in every episode, such as ``step_success``;
        - ``env_args.<path>`` or ``run.<path>``, a dotted path into ``env_args`` or the run block;
        - ``attempts``, the attempt log, or ``attempts.<path>``, a dotted path present in every attempt line;
        - ``source``, the source dataset, or ``source.<key>``, a key of the source dataset.
        """
        head, _, rest = key.partition(".")
        if head == "source":
            return self.source is not None and (not rest or self.source.has(rest))
        if head == "attempts":
            return bool(self.attempts) and (not rest or all(_has_path(line, rest) for line in self.attempts))
        if head == "env_args":
            return _has_path(self.env_args, rest)
        if head == "run":
            return _has_path(self.run, rest)
        return bool(self.episodes) and all(key in episode for episode in self.episodes)

    @cached_property
    def camera_paths(self) -> frozenset[str]:
        """Paths of the camera streams, 4-D ``uint8`` arrays, that any episode records."""
        return frozenset(
            path
            for episode in self.all_episodes
            for path, dataset in episode.datasets()
            if dataset.dtype == np.uint8 and dataset.ndim == 4
        )
