# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a metric reads: a dataset's episodes and ``env_args``, and its generation run's record and attempt log."""

from __future__ import annotations

import h5py
import json
import numpy as np
from collections.abc import Mapping
from typing import Any

_MISSING = object()


def lookup(record: Any, dotted_key: str) -> Any:
    """``record[a][b]`` for ``dotted_key`` ``"a.b"``, or a sentinel that :func:`is_present` rejects."""
    for part in dotted_key.split("."):
        if not isinstance(record, Mapping) or part not in record:
            return _MISSING
        record = record[part]
    return record


def is_present(value: Any) -> bool:
    """Whether a :func:`lookup` found its key."""
    return value is not _MISSING


class Episode:
    """One episode of a dataset; arrays are read when accessed."""

    def __init__(self, group: h5py.Group) -> None:
        self.name = group.name.rsplit("/", 1)[-1]
        self._group = group

    def __contains__(self, key: str) -> bool:
        return key in self._group

    def __getitem__(self, key: str) -> np.ndarray:
        return self._group[key][()]


class Evidence:
    """A dataset, and optionally the generation run that produced it, as metrics read them.

    Attributes:
        episodes: The dataset's episodes.
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
        data = h5_file["data"]
        raw_env_args = data.attrs.get("env_args", "{}")
        self.env_args: dict[str, Any] = json.loads(
            raw_env_args.decode() if isinstance(raw_env_args, bytes) else raw_env_args
        )
        self.episodes = [Episode(group) for group in data.values()]
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
            return bool(self.attempts) and (not rest or all(is_present(lookup(line, rest)) for line in self.attempts))
        if head == "env_args":
            return is_present(lookup(self.env_args, rest))
        if head == "run":
            return is_present(lookup(self.run, rest))
        return bool(self.episodes) and all(key in episode for episode in self.episodes)
