# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict parser for the AutoData v1 envelope around Arena environment intent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypeVar

from isaac_autodata_interfaces.autonomous._validation import (
    IssueCollector,
    require_bool,
    require_finite_number,
    require_int,
    require_mapping,
    require_string,
)
from isaac_autodata_interfaces.autonomous._yaml import load_yaml_document
from isaac_autodata_interfaces.autonomous.task_request_types import (
    GenerationConfig,
    MotionBackend,
    OutputConfig,
    PlannerBackend,
    PlannerConfig,
    TaskRequest,
    canonical_json,
)

TASK_REQUEST_SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 128
MAX_PATH_LENGTH = 4096
MAX_SUCCESSFUL_EPISODES = 1_000_000
MAX_ATTEMPTS = 10_000_000
MAX_NUM_ENVS = 4096
MAX_SEED = 2**63 - 1
MAX_PLANNER_TIME_S = 86_400.0
MAX_BATCH_SIZE = 65_536
MAX_INTERPOLATION_DT_S = 10.0
MAX_OPAQUE_DEPTH = 64
MAX_OPAQUE_NODES = 100_000

_EnumT = TypeVar("_EnumT", PlannerBackend, MotionBackend)


def load_task_request(path: str | Path) -> TaskRequest:
    """Load and validate an AutoData envelope without importing Arena, Isaac, or planner code."""

    source_path = Path(path).expanduser().resolve(strict=False)
    return task_request_from_dict(load_yaml_document(source_path), source_path=source_path)


def task_request_from_dict(data: Any, *, source_path: str | Path) -> TaskRequest:
    """Validate parsed YAML-compatible data against the exact AutoData v1 envelope."""

    issues = IssueCollector()
    root = require_mapping(data, (), issues)
    if root is None:
        issues.raise_if_any()
        raise AssertionError("unreachable")
    issues.check_keys(
        root,
        (),
        required=("schema_version", "name", "environment", "planner", "generation", "output"),
    )

    schema_version = _parse_required_int(root, "schema_version", (), issues, minimum=1, maximum=1)
    if schema_version is not None and schema_version != TASK_REQUEST_SCHEMA_VERSION:
        issues.add(
            ("schema_version",),
            "unsupported_schema_version",
            f"expected schema_version {TASK_REQUEST_SCHEMA_VERSION}, got {schema_version}",
        )
    name = _parse_required_string(root, "name", (), issues)
    if name is not None:
        if name != name.strip():
            issues.add(("name",), "invalid_name", "name must not have leading or trailing whitespace")
        if len(name) > MAX_NAME_LENGTH:
            issues.add(("name",), "value_too_long", f"name must contain at most {MAX_NAME_LENGTH} characters")

    environment_intent_json = (
        _parse_environment(root["environment"], ("environment",), issues) if "environment" in root else None
    )
    planner = _parse_planner(root["planner"], ("planner",), issues) if "planner" in root else None
    generation = _parse_generation(root["generation"], ("generation",), issues) if "generation" in root else None
    output = _parse_output(root["output"], ("output",), Path(source_path), issues) if "output" in root else None

    issues.raise_if_any()
    assert schema_version is not None
    assert name is not None
    assert environment_intent_json is not None
    assert planner is not None
    assert generation is not None
    assert output is not None
    return TaskRequest(
        schema_version=schema_version,
        name=name,
        environment_intent_json=environment_intent_json,
        planner=planner,
        generation=generation,
        output=output,
        source_path=Path(source_path).expanduser().resolve(strict=False),
    )


def _parse_environment(value: Any, path: tuple[str | int, ...], issues: IssueCollector) -> str | None:
    environment = require_mapping(value, path, issues)
    if environment is None:
        return None
    issues.check_keys(environment, path, required=("intent",))
    if "intent" not in environment:
        return None
    intent = require_mapping(environment["intent"], path + ("intent",), issues)
    if intent is None:
        return None
    _validate_opaque_json(intent, path + ("intent",), issues)
    if issues.issues:
        return None
    return canonical_json(intent)


def _validate_opaque_json(value: Any, path: tuple[str | int, ...], issues: IssueCollector) -> None:
    """Enforce only the serialization boundary; Arena owns all nested intent semantics."""

    stack: list[tuple[Any, tuple[str | int, ...], int]] = [(value, path, 0)]
    visited = 0
    while stack:
        current, current_path, depth = stack.pop()
        visited += 1
        if visited > MAX_OPAQUE_NODES:
            issues.add(path, "opaque_intent_too_large", f"intent exceeds {MAX_OPAQUE_NODES} values")
            return
        if depth > MAX_OPAQUE_DEPTH:
            issues.add(current_path, "opaque_intent_too_deep", f"intent nesting exceeds {MAX_OPAQUE_DEPTH}")
            continue
        if current is None or type(current) in (str, bool, int):
            continue
        if type(current) is float:
            if not math.isfinite(current):
                issues.add(current_path, "non_finite", "opaque intent numbers must be finite")
            continue
        if type(current) is list:
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], current_path + (index,), depth + 1))
            continue
        if type(current) is dict:
            invalid_keys = [key for key in current if type(key) is not str]
            if invalid_keys:
                issues.add(current_path, "invalid_mapping_key", "opaque intent mapping keys must be strings")
                continue
            for key in reversed(list(current)):
                stack.append((current[key], current_path + (key,), depth + 1))
            continue
        issues.add(
            current_path,
            "non_json_intent_value",
            f"opaque Arena intent must contain JSON-compatible values, got {type(current).__name__}",
        )


def _parse_planner(value: Any, path: tuple[str | int, ...], issues: IssueCollector) -> PlannerConfig | None:
    planner = require_mapping(value, path, issues)
    if planner is None:
        return None
    fields = (
        "backend",
        "motion_backend",
        "collisions",
        "max_time_s",
        "batch_size",
        "interpolation_dt_s",
        "profile",
        "animate",
    )
    issues.check_keys(planner, path, required=fields)
    backend = _parse_enum(planner, "backend", path, PlannerBackend, issues)
    motion_backend = _parse_enum(planner, "motion_backend", path, MotionBackend, issues)
    collisions = _parse_required_bool(planner, "collisions", path, issues)
    max_time_s = _parse_required_number(
        planner,
        "max_time_s",
        path,
        issues,
        minimum_exclusive=0.0,
        maximum=MAX_PLANNER_TIME_S,
    )
    batch_size = _parse_required_int(
        planner,
        "batch_size",
        path,
        issues,
        minimum=1,
        maximum=MAX_BATCH_SIZE,
    )
    interpolation_dt_s = _parse_required_number(
        planner,
        "interpolation_dt_s",
        path,
        issues,
        minimum_exclusive=0.0,
        maximum=MAX_INTERPOLATION_DT_S,
    )
    profile = _parse_required_bool(planner, "profile", path, issues)
    animate = _parse_required_bool(planner, "animate", path, issues)
    if None in (
        backend,
        motion_backend,
        collisions,
        max_time_s,
        batch_size,
        interpolation_dt_s,
        profile,
        animate,
    ):
        return None
    assert isinstance(backend, PlannerBackend)
    assert isinstance(motion_backend, MotionBackend)
    assert isinstance(collisions, bool)
    assert isinstance(max_time_s, float)
    assert isinstance(batch_size, int)
    assert isinstance(interpolation_dt_s, float)
    assert isinstance(profile, bool)
    assert isinstance(animate, bool)
    return PlannerConfig(
        backend=backend,
        motion_backend=motion_backend,
        collisions=collisions,
        max_time_s=max_time_s,
        batch_size=batch_size,
        interpolation_dt_s=interpolation_dt_s,
        profile=profile,
        animate=animate,
    )


def _parse_generation(
    value: Any,
    path: tuple[str | int, ...],
    issues: IssueCollector,
) -> GenerationConfig | None:
    generation = require_mapping(value, path, issues)
    if generation is None:
        return None
    fields = ("successful_episodes", "seed", "num_envs", "max_attempts")
    issues.check_keys(generation, path, required=fields)
    successful_episodes = _parse_required_int(
        generation,
        "successful_episodes",
        path,
        issues,
        minimum=1,
        maximum=MAX_SUCCESSFUL_EPISODES,
    )
    seed = _parse_required_int(generation, "seed", path, issues, minimum=0, maximum=MAX_SEED)
    num_envs = _parse_required_int(generation, "num_envs", path, issues, minimum=1, maximum=MAX_NUM_ENVS)
    max_attempts = _parse_required_int(generation, "max_attempts", path, issues, minimum=1, maximum=MAX_ATTEMPTS)
    if successful_episodes is not None and max_attempts is not None and max_attempts < successful_episodes:
        issues.add(
            path + ("max_attempts",),
            "attempt_budget_too_small",
            f"max_attempts ({max_attempts}) must be at least successful_episodes ({successful_episodes})",
        )
    if None in (successful_episodes, seed, num_envs, max_attempts):
        return None
    assert isinstance(successful_episodes, int)
    assert isinstance(seed, int)
    assert isinstance(num_envs, int)
    assert isinstance(max_attempts, int)
    return GenerationConfig(
        successful_episodes=successful_episodes,
        seed=seed,
        num_envs=num_envs,
        max_attempts=max_attempts,
    )


def _parse_output(
    value: Any,
    path: tuple[str | int, ...],
    source_path: Path,
    issues: IssueCollector,
) -> OutputConfig | None:
    output = require_mapping(value, path, issues)
    if output is None:
        return None
    issues.check_keys(output, path, required=("dataset", "keep_failed"), optional=("run_log",))
    dataset = (
        _parse_bounded_path(output["dataset"], path + ("dataset",), source_path, ".hdf5", issues)
        if "dataset" in output
        else None
    )
    keep_failed = _parse_required_bool(output, "keep_failed", path, issues)
    run_log = None
    if "run_log" in output and output["run_log"] is not None:
        run_log = _parse_bounded_path(output["run_log"], path + ("run_log",), source_path, ".jsonl", issues)
    if dataset is None or keep_failed is None:
        return None
    return OutputConfig(dataset=dataset, keep_failed=keep_failed, run_log=run_log)


def _parse_bounded_path(
    value: Any,
    path: tuple[str | int, ...],
    source_path: Path,
    suffix: str,
    issues: IssueCollector,
) -> str | None:
    text = require_string(value, path, issues)
    if text is None:
        return None
    if len(text) > MAX_PATH_LENGTH:
        issues.add(path, "path_too_long", f"path must contain at most {MAX_PATH_LENGTH} characters")
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        issues.add(path, "absolute_path", "path must be relative to the request file")
        return None
    if candidate.parts and candidate.parts[0].startswith("~"):
        issues.add(path, "home_path", "home-directory expansion is not allowed")
        return None
    if ".." in candidate.parts:
        issues.add(path, "path_traversal", "parent-directory traversal is not allowed")
        return None
    if candidate.suffix.lower() != suffix:
        issues.add(path, "invalid_path_suffix", f"path must end in {suffix}")
        return None
    request_dir = source_path.expanduser().resolve(strict=False).parent
    resolved = (request_dir / candidate).resolve(strict=False)
    try:
        resolved.relative_to(request_dir)
    except ValueError:
        issues.add(path, "path_escape", "resolved path escapes the request directory")
        return None
    return candidate.as_posix()


def _parse_enum(
    mapping: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
    enum_type: type[_EnumT],
    issues: IssueCollector,
) -> _EnumT | None:
    value = _parse_required_string(mapping, key, path, issues)
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError:
        choices = [item.value for item in enum_type]
        issues.add(path + (key,), "unsupported_value", f"expected one of {choices}, got {value!r}")
        return None


def _parse_required_string(
    mapping: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
    issues: IssueCollector,
) -> str | None:
    if key not in mapping:
        return None
    return require_string(mapping[key], path + (key,), issues)


def _parse_required_bool(
    mapping: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
    issues: IssueCollector,
) -> bool | None:
    if key not in mapping:
        return None
    return require_bool(mapping[key], path + (key,), issues)


def _parse_required_int(
    mapping: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
    issues: IssueCollector,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if key not in mapping:
        return None
    return require_int(mapping[key], path + (key,), issues, minimum=minimum, maximum=maximum)


def _parse_required_number(
    mapping: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
    issues: IssueCollector,
    *,
    minimum_exclusive: float,
    maximum: float,
) -> float | None:
    if key not in mapping:
        return None
    return require_finite_number(
        mapping[key],
        path + (key,),
        issues,
        minimum_exclusive=minimum_exclusive,
        maximum=maximum,
    )
