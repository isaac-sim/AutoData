# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Append-only, bounded JSON Lines run log for autonomous generation."""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import stat
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isaac_autodata_core.autonomous.task_motion import ExecutionEvent, JsonValue, TaskMotionPlan

DEFAULT_MAX_RECORD_BYTES = 4_000_000
MAX_RUN_RECORD_COLLECTION_ITEMS = 100_000
MAX_RUN_RECORD_JSON_NODES = 200_000


class RunLogWriteUncertainError(RuntimeError):
    """A run-log write may be partial or durable, so no later append is safe."""


def safe_exception_record(exc: BaseException, *, include_traceback: bool = False) -> dict[str, str]:
    """Return a bounded exception summary without process environment or object repr data.

    Args:
        exc: Exception to summarize.
        include_traceback: Include a bounded formatted traceback for explicit debug runs.
    """

    message = str(exc).replace("\x00", "").strip()
    record = {
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}"[:512],
        "message": message[:2048],
    }
    if include_traceback:
        record["traceback"] = "".join(traceback.format_exception(exc))[-16_384:]
    return record


class RunLogWriter:
    """Synchronously append validated records to a JSON Lines run log.

    The writer is safe for multiple generation tasks in one process. It holds one append-only file
    descriptor for its lifetime, writes exactly one line under a process-local lock, and calls
    :func:`os.fsync` so completed records survive a later process crash. A descriptor reserved by a
    descriptor-safe output transaction can be supplied directly, avoiding a path re-open.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fd: int | None = None,
        close_fd: bool = True,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.suffix.lower() != ".jsonl":
            raise ValueError("run log path must end in .jsonl")
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int) or max_record_bytes < 1024:
            raise ValueError("max_record_bytes must be an integer of at least 1024")
        if type(close_fd) is not bool:
            raise TypeError("close_fd must be a boolean")
        self.max_record_bytes = max_record_bytes
        self._lock = threading.Lock()
        self._close_fd = close_fd
        self._closed = False
        self._poisoned = False
        if fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        else:
            if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
                raise ValueError("fd must be a non-negative file descriptor")
            descriptor_stat = os.fstat(fd)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise ValueError("run log descriptor must refer to a regular file")
            descriptor_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            if not descriptor_flags & os.O_APPEND:
                raise ValueError("run log descriptor must be opened with O_APPEND")
            if descriptor_flags & os.O_ACCMODE == os.O_RDONLY:
                raise ValueError("run log descriptor must be writable")
            self._fd = fd

    def append(self, record: Mapping[str, Any] | ExecutionEvent | TaskMotionPlan) -> None:
        """Validate and durably append one run-log record.

        Args:
            record: JSON-compatible mapping or a supported typed contract.
        """

        if isinstance(record, (ExecutionEvent, TaskMotionPlan)):
            payload = record.to_dict()
        elif isinstance(record, Mapping):
            payload = dict(record)
        else:
            raise TypeError(f"unsupported run-log record {type(record).__name__}")
        normalized = normalize_run_record(payload)
        assert isinstance(normalized, dict)
        serialized = _serialize_normalized_json(normalized)
        encoded = (serialized + "\n").encode("utf-8")
        if len(encoded) > self.max_record_bytes:
            raise ValueError(f"run-log record is {len(encoded)} bytes; maximum is {self.max_record_bytes}")

        with self._lock:
            if self._closed:
                raise RuntimeError("run-log writer is closed")
            if self._poisoned:
                raise RunLogWriteUncertainError(
                    "run log is poisoned by an earlier append/fsync failure; inspect and repair its tail"
                )
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(self._fd, remaining)
                    if written <= 0:
                        raise OSError("run-log append made no progress")
                    remaining = remaining[written:]
                os.fsync(self._fd)
            except BaseException as exc:
                self._poisoned = True
                raise RunLogWriteUncertainError(
                    "run-log append/fsync durability is unknown; no later log write or output commit is safe"
                ) from exc

    def close(self) -> None:
        """Close the held descriptor when this writer owns it."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._close_fd:
                os.close(self._fd)

    def __enter__(self) -> RunLogWriter:
        """Return this writer as a context manager."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Close the writer when leaving a context manager."""

        self.close()


def normalize_run_record(value: Any, *, max_serialized_bytes: int | None = None) -> JsonValue:
    """Normalize one value under the run log's strict JSON and optional byte bounds.

    Args:
        value: Candidate JSON-compatible value.
        max_serialized_bytes: Optional maximum canonical UTF-8 payload size.

    Returns:
        A detached JSON-compatible value using sorted string mapping keys and lists for sequences.

    Raises:
        ValueError: If the value is malformed, too deep, too large, or contains too many items.
    """

    if max_serialized_bytes is not None and (
        isinstance(max_serialized_bytes, bool) or not isinstance(max_serialized_bytes, int) or max_serialized_bytes <= 0
    ):
        raise ValueError("max_serialized_bytes must be null or a positive integer")
    normalized = _normalize_json(value)
    if max_serialized_bytes is not None:
        serialized_bytes = len(_serialize_normalized_json(normalized).encode("utf-8"))
        if serialized_bytes > max_serialized_bytes:
            raise ValueError(f"run-record value is {serialized_bytes} bytes; maximum is {max_serialized_bytes}")
    return normalized


def _serialize_normalized_json(value: JsonValue) -> str:
    """Serialize an already-normalized value exactly as the append-only run log does."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalize_json(value: Any, depth: int = 0, node_budget: list[int] | None = None) -> JsonValue:
    if node_budget is None:
        node_budget = [MAX_RUN_RECORD_JSON_NODES]
    node_budget[0] -= 1
    if node_budget[0] < 0:
        raise ValueError("run-log record exceeds maximum JSON node count")
    if depth > 12:
        raise ValueError("run-log record exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("run-log numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_RUN_RECORD_COLLECTION_ITEMS:
            raise ValueError("run-log mapping exceeds maximum item count")
        keys = list(itertools.islice(iter(value), MAX_RUN_RECORD_COLLECTION_ITEMS + 1))
        if len(keys) > MAX_RUN_RECORD_COLLECTION_ITEMS or len(keys) != len(value):
            raise ValueError("run-log mapping exceeds or misreports maximum item count")
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("run-log mapping keys must be strings")
        result: dict[str, JsonValue] = {}
        for key in sorted(keys):
            result[key] = _normalize_json(value[key], depth + 1, node_budget)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_RUN_RECORD_COLLECTION_ITEMS:
            raise ValueError("run-log sequence exceeds maximum item count")
        return [_normalize_json(item, depth + 1, node_budget) for item in value]
    raise ValueError(f"run log contains unsupported value type {type(value).__name__}")
