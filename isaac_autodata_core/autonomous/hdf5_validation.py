# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded structural and numeric validation for recorded autonomous HDF5 datasets."""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Iterator
from numbers import Integral
from typing import Any

_HDF5_VALIDATION_CHUNK_BYTES = 8 << 20


def hdf5_episode_count(descriptor: int, *, kind: str) -> int:
    """Validate a held HDF5 dataset and return its episode count.

    Args:
        descriptor: Readable descriptor for the exact staged dataset inode.
        kind: Dataset role: ``successful``, ``failed``, or ``filesystem probe``.

    Returns:
        Number of validated episode groups.
    """

    signature = os.pread(descriptor, 8, 0)
    if signature != b"\x89HDF\r\n\x1a\n":
        raise ValueError(f"staged {kind} dataset does not have an HDF5 signature")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to validate staged autonomous datasets") from exc

    duplicate = os.dup(descriptor)
    with os.fdopen(duplicate, "rb", closefd=True) as stream:
        with h5py.File(stream, "r") as dataset:
            data = dataset.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError(f"staged {kind} dataset has no HDF5 'data' group")
            if kind == "filesystem probe":
                return len(data)
            format_version = dataset.attrs.get("format_version")
            if isinstance(format_version, bool) or not isinstance(format_version, Integral) or format_version != 1:
                raise ValueError(f"staged {kind} dataset has unsupported or missing format_version")
            episode_names = list(data)
            episode_indices: list[int] = []
            expected_success = kind == "successful"
            for episode_name in episode_names:
                match = re.fullmatch(r"demo_(\d+)", episode_name)
                if match is None:
                    raise ValueError(f"staged {kind} dataset has invalid episode name {episode_name!r}")
                episode_indices.append(int(match.group(1)))
                episode = data[episode_name]
                if not isinstance(episode, h5py.Group):
                    raise ValueError(f"staged {kind} dataset episode {episode_name!r} is not a group")
                _validate_hdf5_episode(episode, episode_name, expected_success=expected_success, h5py=h5py)
            if sorted(episode_indices) != list(range(len(episode_indices))):
                raise ValueError(f"staged {kind} dataset episode indices must be contiguous from zero")
            return len(episode_names)


def _validate_hdf5_episode(episode: Any, episode_name: str, *, expected_success: bool, h5py: Any) -> None:
    import numpy as np

    initial_state = episode.get("initial_state")
    if not isinstance(initial_state, h5py.Group):
        raise ValueError(f"staged dataset episode {episode_name!r} has no 'initial_state' group")
    num_samples = episode.attrs.get("num_samples")
    minimum_samples = 1 if expected_success else 0
    if isinstance(num_samples, bool) or not isinstance(num_samples, Integral) or num_samples < minimum_samples:
        raise ValueError(f"staged dataset episode {episode_name!r} has invalid num_samples")
    success = episode.attrs.get("success")
    if not isinstance(success, (bool, np.bool_)) or bool(success) is not expected_success:
        raise ValueError(f"staged dataset episode {episode_name!r} has inconsistent success metadata")

    # A failed attempt can be retained before its first task step. Such an episode has a
    # reset-time initial state and num_samples=0, but no task-stream groups or action datasets.
    require_task_stream = expected_success or num_samples > 0
    task_groups: dict[str, Any] = {}
    for name in ("obs", "states"):
        value = episode.get(name)
        if value is None and not require_task_stream:
            continue
        if not isinstance(value, h5py.Group):
            raise ValueError(f"staged dataset episode {episode_name!r} has no {name!r} group")
        task_groups[name] = value
    task_datasets: dict[str, Any] = {}
    for name in ("actions", "processed_actions"):
        value = episode.get(name)
        if value is None and not require_task_stream:
            continue
        if not isinstance(value, h5py.Dataset) or len(value.shape) != 2 or value.shape[0] != num_samples:
            raise ValueError(f"staged dataset episode {episode_name!r} {name!r} must be rank-2 with num_samples rows")
        if value.shape[1] < 1:
            raise ValueError(f"staged dataset episode {episode_name!r} {name!r} must have a nonempty feature axis")
        task_datasets[name] = value
    for name, value in task_datasets.items():
        _require_finite_hdf5_dataset(value, episode_name, name)
    _validate_hdf5_dataset_group(
        initial_state,
        episode_name,
        "initial_state",
        h5py=h5py,
        require_datasets=expected_success,
        require_nonempty_datasets=expected_success,
    )
    for group_name, group in task_groups.items():
        _validate_hdf5_dataset_group(
            group,
            episode_name,
            group_name,
            h5py=h5py,
            expected_rows=num_samples,
            require_datasets=require_task_stream,
        )


def _validate_hdf5_dataset_group(
    group: Any,
    episode_name: str,
    group_name: str,
    *,
    h5py: Any,
    expected_rows: int | None = None,
    require_datasets: bool,
    require_nonempty_datasets: bool = False,
) -> None:
    datasets: list[tuple[str, Any]] = []

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            datasets.append((name, value))

    group.visititems(collect)
    if require_datasets and not datasets:
        raise ValueError(f"staged dataset episode {episode_name!r} {group_name!r} group has no datasets")
    for dataset_name, value in datasets:
        qualified_name = f"{group_name}/{dataset_name}"
        if expected_rows is not None and (len(value.shape) < 1 or value.shape[0] != expected_rows):
            raise ValueError(
                f"staged dataset episode {episode_name!r} {qualified_name!r} must have num_samples leading rows"
            )
        _require_finite_hdf5_dataset(
            value,
            episode_name,
            qualified_name,
            require_nonempty=require_nonempty_datasets,
        )


def _require_finite_hdf5_dataset(
    dataset: Any,
    episode_name: str,
    dataset_name: str,
    *,
    require_nonempty: bool = False,
) -> None:
    import numpy as np

    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} must be numeric")
    if require_nonempty and dataset.size < 1:
        raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} must not be empty")
    for selection in _bounded_hdf5_selections(dataset.shape, dataset.dtype.itemsize):
        values = dataset[()] if not selection else dataset[selection]
        if not np.isfinite(values).all():
            raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} contains non-finite values")


def _bounded_hdf5_selections(shape: tuple[int, ...], itemsize: int) -> Iterator[tuple[slice, ...]]:
    """Yield numeric hyperslabs no larger than the validation byte bound."""

    if not shape:
        yield ()
        return
    if any(size == 0 for size in shape):
        return
    max_items = max(1, _HDF5_VALIDATION_CHUNK_BYTES // max(1, itemsize))
    block_shape = [1] * len(shape)
    remaining_items = max_items
    for axis in range(len(shape) - 1, -1, -1):
        block_shape[axis] = min(shape[axis], remaining_items)
        remaining_items = max(1, remaining_items // block_shape[axis])
    starts = (range(0, size, block_size) for size, block_size in zip(shape, block_shape))
    for offsets in itertools.product(*starts):
        yield tuple(
            slice(offset, min(offset + block_size, size))
            for offset, block_size, size in zip(offsets, block_shape, shape)
        )


__all__ = ["hdf5_episode_count"]
