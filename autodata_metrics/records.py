# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shapes of the records a report reads, checked once when a record is read.

A record that does not match its shape is refused with an error naming the file and the field, so no metric reads a
malformed value. Fields a shape does not list are left unchecked, so producers can add fields that older readers
ignore.
"""

from __future__ import annotations

import numbers
import sys
from dataclasses import dataclass
from typing import Any

from autodata_utils.dataset_metadata import AUTODATA_ENTRY_VERSION


class InvalidRecord(ValueError):
    """A record without the shape the report reads."""


@dataclass(frozen=True)
class Maybe:
    """A field that may be absent or null."""

    shape: Any


@dataclass(frozen=True)
class ListOf:
    """A list of values of one shape, of a fixed length when ``length`` is given."""

    shape: Any
    length: int | None = None


@dataclass(frozen=True)
class MapOf:
    """An object whose values all have one shape."""

    shape: Any


NUMBER = "a finite number"
COUNT = "a non-negative integer"
DURATION = "a non-negative finite number"
POSITIVE = "a positive finite number"

INPUT = {"path": str, "sha256": str}
RUN = {
    "run_id": str,
    "seed": Maybe(int),
    "revision": Maybe(str),
    "inputs": Maybe(MapOf(Maybe(INPUT))),
    "device": Maybe(str),
    "gpu": Maybe(str),
    "cpu": Maybe(str),
    "num_gpus": Maybe(COUNT),
    "num_envs": Maybe(COUNT),
    "setup_time_s": Maybe(DURATION),
    "generation_time_s": Maybe(DURATION),
    "attempts_file": Maybe(str),
}
RESULT = {"num_success": COUNT, "num_failures": Maybe(COUNT), "num_attempts": COUNT, "run": Maybe(RUN)}
ATTEMPT = {
    "run_id": str,
    "env_id": COUNT,
    "success": bool,
    "start_time_s": NUMBER,
    "end_time_s": NUMBER,
    "extras": Maybe({"planning_time_s": Maybe(DURATION), "planning_failed": Maybe(bool)}),
}
ANNOTATION = {
    "mode": Maybe(("auto", "manual")),
    "input_episodes": COUNT,
    "processed_episodes": COUNT,
    "annotated_episodes": COUNT,
    "setup_time_s": Maybe(DURATION),
    "time_s": DURATION,
}
ENTRY = {
    "version": COUNT,
    "run_id": str,
    "exported_episodes": COUNT,
    "pose_action": Maybe(MapOf({"action_slice": ListOf(int, 2), "bound": Maybe(NUMBER)})),
    "annotation": Maybe(ANNOTATION),
}
ENV_ARGS = {"autodata": Maybe(ENTRY), "sim_args": Maybe({"dt": Maybe(POSITIVE), "decimation": Maybe(POSITIVE)})}


def parse(value: Any, shape: Any, where: str) -> Any:
    """``value`` checked against ``shape``, with its absent or null optional fields left out.

    A shape is a type, a tuple of allowed values, :data:`NUMBER`, :data:`COUNT`, :data:`DURATION`, :data:`POSITIVE`,
    a :class:`Maybe`, :class:`ListOf` or :class:`MapOf`, or a dict of field shapes; a dict's fields are required
    unless their shape is a :class:`Maybe`.

    Raises:
        InvalidRecord: Naming ``where`` and the field, if ``value`` does not have ``shape``.
    """
    if isinstance(shape, Maybe):
        return None if value is None else parse(value, shape.shape, where)
    if isinstance(shape, dict):
        _require(isinstance(value, dict), value, "an object", where)
        parsed = dict(value)
        for name, field_shape in shape.items():
            _require(name in value or isinstance(field_shape, Maybe), value, f"an object with {name!r}", where)
            field = parse(value.get(name), field_shape, f"{where}.{name}")
            if field is None:
                parsed.pop(name, None)
            else:
                parsed[name] = field
        return parsed
    if isinstance(shape, ListOf):
        _require(isinstance(value, list) and shape.length in (None, len(value)), value, "a list", where)
        return [parse(item, shape.shape, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(shape, MapOf):
        _require(isinstance(value, dict), value, "an object", where)
        parsed = {key: parse(item, shape.shape, f"{where}.{key}") for key, item in value.items()}
        return {key: item for key, item in parsed.items() if item is not None}
    if isinstance(shape, tuple):
        _require(value in shape, value, f"one of {list(shape)}", where)
    elif shape in (NUMBER, DURATION, POSITIVE):
        in_range = is_finite_number(value) and (shape is NUMBER or value > 0 or (shape is DURATION and value == 0))
        _require(in_range, value, shape, where)
    elif shape is COUNT:
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, value, shape, where)
    else:
        exact = isinstance(value, shape) and not (shape is int and isinstance(value, bool))
        _require(exact, value, shape.__name__, where)
    return value


def parse_env_args(env_args: Any, where: str) -> dict[str, Any]:
    """A dataset's ``env_args`` with its ``autodata`` entry parsed, refusing an entry from a newer AutoData."""
    entry = env_args.get("autodata") if isinstance(env_args, dict) else None
    version = entry.get("version") if isinstance(entry, dict) else None
    if isinstance(version, int) and version > AUTODATA_ENTRY_VERSION:
        raise InvalidRecord(f"{where}.autodata was written by a newer AutoData (entry version {version}).")
    parsed = parse(env_args, ENV_ARGS, where)
    annotation = parsed.get("autodata", {}).get("annotation")
    if annotation is not None:
        counts = [annotation[name] for name in ("annotated_episodes", "processed_episodes", "input_episodes")]
        _require(counts == sorted(counts), counts, "annotated <= processed <= input", f"{where}.autodata.annotation")
    return parsed


def is_finite_number(value: Any) -> bool:
    """Whether ``value`` is a real number other than a bool, of a magnitude a float can hold."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and abs(value) <= sys.float_info.max


def _require(condition: bool, value: Any, expected: str, where: str) -> None:
    if not condition:
        raise InvalidRecord(f"{where} must be {expected}, not {value!r}.")
