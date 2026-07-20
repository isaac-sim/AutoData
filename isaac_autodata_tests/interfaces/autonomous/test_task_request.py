# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from isaac_autodata_interfaces.autonomous import (
    AutonomousValidationError,
    MotionBackend,
    PlannerBackend,
    load_task_request,
    task_request_from_dict,
)
from isaac_autodata_interfaces.autonomous._yaml import MAX_YAML_BYTES


def _request_dict() -> dict:
    return {
        "schema_version": 1,
        "name": "franka_pick_cube_into_bowl",
        "environment": {
            "intent": {
                "reasoning": "Arena owns this schema",
                "background": "maple_table_robolab",
                "embodiment": "franka_ik",
                "items": [{"query": "cube", "category_tags": ["object"]}],
                "initial_state_graph": [],
                "tasks": [],
                "future_arena_field": {"AutoData": "must not interpret this"},
            }
        },
        "planner": {
            "backend": "auto",
            "motion_backend": "auto",
            "collisions": True,
            "max_time_s": 60.0,
            "batch_size": 128,
            "interpolation_dt_s": 0.05,
            "profile": False,
            "animate": False,
        },
        "generation": {
            "successful_episodes": 3,
            "seed": 7,
            "num_envs": 1,
            "max_attempts": 9,
        },
        "output": {
            "dataset": "outputs/data.hdf5",
            "keep_failed": False,
            "run_log": "outputs/data.jsonl",
        },
    }


def _parse(data: dict, tmp_path: Path):
    return task_request_from_dict(data, source_path=tmp_path / "request.yaml")


def _issue_codes(exc: pytest.ExceptionInfo[AutonomousValidationError]) -> set[tuple[str, str]]:
    return {(issue.field_path, issue.code) for issue in exc.value.issues}


def test_parse_valid_envelope_preserves_opaque_arena_intent(tmp_path: Path):
    request = _parse(_request_dict(), tmp_path)

    assert request.schema_version == 1
    assert request.planner.backend is PlannerBackend.AUTO
    assert request.planner.motion_backend is MotionBackend.AUTO
    assert request.generation.max_attempts == 9
    assert request.output.dataset == "outputs/data.hdf5"
    assert request.environment_intent["future_arena_field"] == {"AutoData": "must not interpret this"}
    assert len(request.digest) == 64


def test_canonical_request_and_digest_ignore_mapping_insertion_order(tmp_path: Path):
    first = _request_dict()
    second = dict(reversed(list(copy.deepcopy(first).items())))
    second["environment"]["intent"] = dict(reversed(list(second["environment"]["intent"].items())))

    request_a = _parse(first, tmp_path)
    request_b = _parse(second, tmp_path)

    assert request_a.canonical_json() == request_b.canonical_json()
    assert request_a.digest == request_b.digest


@pytest.mark.parametrize(
    ("container_path", "unknown_key", "expected_path"),
    [
        ((), "unexpected", "$.unexpected"),
        (("environment",), "robot", "$.environment.robot"),
        (("planner",), "timeout", "$.planner.timeout"),
        (("generation",), "retry_forever", "$.generation.retry_forever"),
        (("output",), "directory", "$.output.directory"),
    ],
)
def test_unknown_autodata_keys_are_rejected(
    tmp_path: Path,
    container_path: tuple[str, ...],
    unknown_key: str,
    expected_path: str,
):
    data = _request_dict()
    container = data
    for key in container_path:
        container = container[key]
    container[unknown_key] = "invalid"

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert (expected_path, "unknown_field") in _issue_codes(exc)


@pytest.mark.parametrize(
    "field_path",
    [
        ("planner", "collisions"),
        ("planner", "profile"),
        ("planner", "animate"),
        ("output", "keep_failed"),
    ],
)
@pytest.mark.parametrize("invalid", [0, 1, "false", "true"])
def test_booleans_require_exact_boolean_type(tmp_path: Path, field_path: tuple[str, str], invalid):
    data = _request_dict()
    data[field_path[0]][field_path[1]] = invalid

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert (f"$.{field_path[0]}.{field_path[1]}", "invalid_type") in _issue_codes(exc)


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        ("planner", "max_time_s", 0.0),
        ("planner", "max_time_s", float("nan")),
        ("planner", "batch_size", 0),
        ("planner", "batch_size", 1.5),
        ("planner", "interpolation_dt_s", float("inf")),
        ("generation", "successful_episodes", 0),
        ("generation", "seed", -1),
        ("generation", "num_envs", 0),
        ("generation", "max_attempts", 0),
    ],
)
def test_numeric_types_and_ranges_are_strict(tmp_path: Path, section: str, field: str, invalid):
    data = _request_dict()
    data[section][field] = invalid

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert any(issue.field_path == f"$.{section}.{field}" for issue in exc.value.issues)


@pytest.mark.parametrize(
    ("field", "value", "choices"),
    [
        ("backend", "mimicgen", {"auto", "schedulestream"}),
        ("motion_backend", "rrt", {"auto", "curobo_v1", "curobo_v2"}),
    ],
)
def test_planner_enums_are_closed(tmp_path: Path, field: str, value: str, choices: set[str]):
    data = _request_dict()
    data["planner"][field] = value

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    issue = next(issue for issue in exc.value.issues if issue.field_path == f"$.planner.{field}")
    assert issue.code == "unsupported_value"
    assert all(choice in issue.message for choice in choices)


def test_attempt_budget_cannot_be_smaller_than_episode_target(tmp_path: Path):
    data = _request_dict()
    data["generation"]["successful_episodes"] = 10
    data["generation"]["max_attempts"] = 9

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert ("$.generation.max_attempts", "attempt_budget_too_small") in _issue_codes(exc)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("dataset", "/tmp/data.hdf5", "absolute_path"),
        ("dataset", "../data.hdf5", "path_traversal"),
        ("dataset", "~/data.hdf5", "home_path"),
        ("dataset", "outputs/data.json", "invalid_path_suffix"),
        ("run_log", "/tmp/data.jsonl", "absolute_path"),
        ("run_log", "../data.jsonl", "path_traversal"),
        ("run_log", "outputs/data.log", "invalid_path_suffix"),
    ],
)
def test_output_paths_are_relative_bounded_and_typed(tmp_path: Path, field: str, value: str, code: str):
    data = _request_dict()
    data["output"][field] = value

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert (f"$.output.{field}", code) in _issue_codes(exc)


def test_existing_symlink_cannot_escape_request_directory(tmp_path: Path):
    request_dir = tmp_path / "request"
    outside = tmp_path / "outside"
    request_dir.mkdir()
    outside.mkdir()
    (request_dir / "escape").symlink_to(outside, target_is_directory=True)
    data = _request_dict()
    data["output"]["dataset"] = "escape/data.hdf5"

    with pytest.raises(AutonomousValidationError) as exc:
        task_request_from_dict(data, source_path=request_dir / "request.yaml")

    assert ("$.output.dataset", "path_escape") in _issue_codes(exc)


def test_run_log_may_be_omitted_or_null(tmp_path: Path):
    omitted = _request_dict()
    omitted["output"].pop("run_log")
    explicit_null = _request_dict()
    explicit_null["output"]["run_log"] = None

    assert _parse(omitted, tmp_path).output.run_log is None
    assert _parse(explicit_null, tmp_path).output.run_log is None


def test_opaque_intent_rejects_only_non_json_serialization_values(tmp_path: Path):
    data = _request_dict()
    data["environment"]["intent"]["when"] = Path("not-json")

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert ("$.environment.intent.when", "non_json_intent_value") in _issue_codes(exc)


def test_opaque_intent_rejects_non_finite_numbers(tmp_path: Path):
    data = _request_dict()
    data["environment"]["intent"]["score"] = float("nan")

    with pytest.raises(AutonomousValidationError) as exc:
        _parse(data, tmp_path)

    assert ("$.environment.intent.score", "non_finite") in _issue_codes(exc)


def test_yaml_duplicate_key_reports_precise_field_path(tmp_path: Path):
    example = Path(__file__).parents[3] / "isaac_autodata_examples" / "autonomous" / "franka_pick_cube_into_bowl.yaml"
    text = example.read_text(encoding="utf-8")
    text = text.replace("    background: maple_table_robolab", "    background: first\n    background: second")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(AutonomousValidationError) as exc:
        load_task_request(path)

    assert ("$.environment.intent.background", "duplicate_key") in _issue_codes(exc)


def test_yaml_timestamp_is_rejected_as_non_json_opaque_value(tmp_path: Path):
    example = Path(__file__).parents[3] / "isaac_autodata_examples" / "autonomous" / "franka_pick_cube_into_bowl.yaml"
    text = example.read_text(encoding="utf-8").replace(
        "    reasoning: >-", "    generated_at: 2026-07-18\n    reasoning: >-"
    )
    path = tmp_path / "timestamp.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(AutonomousValidationError) as exc:
        load_task_request(path)

    assert ("$.environment.intent.generated_at", "non_json_intent_value") in _issue_codes(exc)


def test_yaml_12_relation_words_remain_strings(tmp_path: Path):
    example = Path(__file__).parents[3] / "isaac_autodata_examples" / "autonomous" / "franka_pick_cube_into_bowl.yaml"
    request = load_task_request(example)

    relations = request.environment_intent["initial_state_graph"]
    assert [relation["kind"] for relation in relations] == ["is_anchor", "on", "on"]


def test_yaml_aliases_are_rejected_before_materialization(tmp_path: Path):
    path = tmp_path / "aliases.yaml"
    path.write_text(
        "schema_version: 1\n"
        "name: alias_request\n"
        "environment: &environment\n"
        "  intent: {}\n"
        "environment_copy: *environment\n",
        encoding="utf-8",
    )

    with pytest.raises(AutonomousValidationError) as exc:
        load_task_request(path)

    assert ("$.environment_copy", "yaml_alias_not_allowed") in _issue_codes(exc)


def test_yaml_document_size_is_bounded(tmp_path: Path):
    path = tmp_path / "large.yaml"
    path.write_bytes(b"x" * (MAX_YAML_BYTES + 1))

    with pytest.raises(AutonomousValidationError) as exc:
        load_task_request(path)

    assert ("$", "document_too_large") in _issue_codes(exc)


def test_parser_import_does_not_import_arena_isaac_or_torch():
    code = """
import sys
from isaac_autodata_interfaces.autonomous.task_request import task_request_from_dict
heavy = ('isaaclab_arena', 'isaaclab', 'isaacsim', 'schedulestream', 'curobo', 'torch')
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in heavy)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_real_example_parses_without_importing_arena():
    example = Path(__file__).parents[3] / "isaac_autodata_examples" / "autonomous" / "franka_pick_cube_into_bowl.yaml"

    request = load_task_request(example)

    assert request.name == "franka_pick_cube_into_bowl"
    assert request.environment_intent["tasks"][0]["kind"] == "PickAndPlaceTask"
    assert request.planner.backend is PlannerBackend.SCHEDULESTREAM
    assert request.planner.motion_backend is MotionBackend.AUTO
