# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a metric reads: a dataset's episodes and ``env_args``, and its generation run's record and attempt log."""

from __future__ import annotations

import copy
import h5py
import json
import numpy as np
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from autodata_metrics.records import InvalidRecord, parse_env_args


def _has_path(record: Any, dotted_key: str) -> bool:
    """Whether ``record[a][b]`` exists for ``dotted_key`` ``"a.b"``."""
    for part in dotted_key.split("."):
        if not isinstance(record, Mapping) or part not in record:
            return False
        record = record[part]
    return True


def read_env_args(h5_file: h5py.File) -> dict[str, Any]:
    """The dataset's ``env_args``, with its ``autodata`` entry parsed; empty when it records none.

    Raises:
        InvalidRecord: If ``env_args`` is not JSON, or it or its entry lacks the shape the metrics read.
    """
    where = f"{h5_file.filename} env_args"
    raw_env_args = h5_file["data"].attrs.get("env_args", "{}")
    try:
        env_args = json.loads(raw_env_args.decode() if isinstance(raw_env_args, bytes) else raw_env_args)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InvalidRecord(f"{where} must be JSON: {exc}") from exc
    return parse_env_args(env_args, where)


class Episode:
    """One episode of a dataset; arrays are read when accessed."""

    def __init__(self, group: h5py.Group) -> None:
        self.name = group.name.rsplit("/", 1)[-1]
        self._group = group
        self._listing: list[tuple[str, h5py.Dataset]] | None = None

    def __contains__(self, key: str) -> bool:
        return key in self._group

    def __getitem__(self, key: str) -> np.ndarray:
        return self._group[key][()]

    def children(self, key: str) -> list[str]:
        """Names under the group ``key``, such as the end effectors under ``obs/datagen_info/eef_pose``."""
        return list(self._group[key])

    @contextmanager
    def listed(self) -> Iterator[None]:
        """Within the block, :meth:`datasets` walks the episode once, so several checks share one listing."""
        try:
            self._listing = self.datasets()
        except Exception:  # An episode that cannot be walked is walked by each check, which fails it with the error.
            self._listing = None
        try:
            yield
        finally:
            self._listing = None

    def datasets(self) -> list[tuple[str, h5py.Dataset]]:
        """Every dataset of the episode with its path, unread until accessed."""
        if self._listing is not None:
            return self._listing
        found: list[tuple[str, h5py.Dataset]] = []
        self._group.visititems(
            lambda path, item: found.append((path, item)) if isinstance(item, h5py.Dataset) else None
        )
        return found


def _episode(data: h5py.Group, name: str) -> Episode:
    """The episode ``name`` of the ``data`` group.

    Raises:
        InvalidRecord: If the episode's group cannot be opened.
    """
    try:
        return Episode(data[name])
    except (KeyError, OSError, RuntimeError) as exc:
        raise InvalidRecord(f"{data.file.filename} episode {name} cannot be read: {exc}") from exc


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
        self.all_episodes = [_episode(h5_file["data"], name) for name in h5_file["data"]]
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

    @property
    def entry(self) -> Mapping[str, Any]:
        """The ``autodata`` entry of ``env_args``, or empty."""
        return self.env_args.get("autodata", {})

    def as_baseline_for(self, run: Evidence) -> Evidence:
        """This source dataset, with the run's ``autodata`` fields where its own entry lacks them, so a baseline
        reads the robot layout and limits; datasheet limits fill in kind by kind."""
        baseline = copy.copy(self)
        entry = {**run.entry, **self.entry}
        limits = {**run.entry.get("hardware_limits", {}), **self.entry.get("hardware_limits", {})}
        if limits:
            entry["hardware_limits"] = limits
        baseline.env_args = {**self.env_args, "autodata": entry}
        return baseline

    @cached_property
    def baseline(self) -> Evidence | None:
        """The source dataset as a baseline for this run, built once on first use, after the episode checks."""
        return self.source.as_baseline_for(self) if self.source is not None else None

    @cached_property
    def camera_paths(self) -> frozenset[str]:
        """Paths of the camera streams, 4-D ``uint8`` arrays, that any readable episode records."""
        paths = set()
        for episode in self.all_episodes:
            try:
                paths.update(
                    path for path, dataset in episode.datasets() if dataset.dtype == np.uint8 and dataset.ndim == 4
                )
            except Exception:  # An episode that cannot be read fails its own checks.
                continue
        return frozenset(paths)
