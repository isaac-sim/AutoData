# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy bridge to Arena's authoritative environment-intent API."""

from __future__ import annotations

import importlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.task_request_types import (
    ArenaCompilationResult,
    CompilerTraceEvent,
    GoalStage,
    SpatialGoalConstraint,
    canonical_json,
    sha256_json,
)

REQUIRED_ARENA_COMMIT = "8a74e794b621b0f8d3627d096a1bae9ce11e7b56"
REQUIRED_ARENA_CAPABILITY = (
    "EnvironmentIntentSpec.model_validate + IntentCompiler.compile/resolution_errors + "
    "ArenaEnvInitialGraphSpec.link/to_dict"
)
MAX_SAFE_TEXT_LENGTH = 2048


class ArenaIntentBridge(Protocol):
    """Interface accepted by the pure AutoData task compiler and fake-backed tests."""

    def compile_and_link(self, intent: dict[str, Any], *, seed: int) -> ArenaCompilationResult:
        """Validate, compile, and link one opaque Arena environment intent."""


class LazyArenaIntentBridge:
    """Import and invoke Arena's intent API only when resolution is explicitly requested."""

    def __init__(self, module_loader: Callable[[str], Any] = importlib.import_module) -> None:
        self._module_loader = module_loader

    def compile_and_link(self, intent: dict[str, Any], *, seed: int) -> ArenaCompilationResult:
        """Validate, compile, and link an Arena intent with isolated deterministic randomness."""

        random_state = random.getstate()
        try:
            random.seed(seed)
            environment_intent_cls, compiler_cls = self._load_api()
            try:
                intent_spec = environment_intent_cls.model_validate(intent)
            except Exception as exc:
                raise AutonomousValidationError([
                    ValidationIssue(
                        ("environment", "intent"),
                        "arena_intent_invalid",
                        f"Arena EnvironmentIntentSpec rejected the intent: {_safe_exception(exc)}",
                    )
                ]) from None

            try:
                compiler = compiler_cls()
                initial_graph_model = compiler.compile(intent_spec)
            except AutonomousValidationError:
                raise
            except Exception as exc:
                raise AutonomousValidationError([
                    ValidationIssue(
                        ("environment", "intent"),
                        "arena_compile_failed",
                        f"Arena IntentCompiler failed: {_safe_exception(exc)}",
                    )
                ]) from None

            if not hasattr(compiler, "resolution_errors") or not hasattr(compiler, "trace"):
                raise _arena_api_error("IntentCompiler lacks resolution_errors or trace after compile")
            resolution_errors = list(compiler.resolution_errors)
            if resolution_errors:
                issues = []
                for event in resolution_errors:
                    trace_event = _trace_event_to_plain(event)
                    issues.append(
                        ValidationIssue(
                            ("environment", "intent"),
                            "arena_resolution_error",
                            f"Arena resolution stage {trace_event.stage!r} could not resolve "
                            f"{trace_event.query!r}; chosen={trace_event.chosen!r}; note={trace_event.note!r}",
                        )
                    )
                raise AutonomousValidationError(issues)

            if not hasattr(initial_graph_model, "link"):
                raise _arena_api_error("compiled initial graph lacks link()")
            try:
                linked_graph_model = initial_graph_model.link()
            except Exception as exc:
                raise AutonomousValidationError([
                    ValidationIssue(
                        ("environment", "intent"),
                        "arena_link_failed",
                        f"Arena initial graph linking failed: {_safe_exception(exc)}",
                    )
                ]) from None

            initial_graph = _graph_model_to_dict(initial_graph_model, "initial")
            linked_graph = _graph_model_to_dict(linked_graph_model, "linked")
            trace = tuple(_trace_event_to_plain(event) for event in compiler.trace)
            return build_arena_compilation_result(initial_graph, linked_graph, trace)
        finally:
            random.setstate(random_state)

    def is_available(self) -> bool:
        """Return whether the required Arena intent API can be imported in this process."""

        try:
            self._load_api()
        except AutonomousValidationError:
            return False
        return True

    def _load_api(self) -> tuple[type, type]:
        try:
            intent_module = self._module_loader("isaaclab_arena.agentic_environment_generation.environment_intent_spec")
            compiler_module = self._module_loader("isaaclab_arena.agentic_environment_generation.intent_compiler")
            environment_intent_cls = getattr(intent_module, "EnvironmentIntentSpec")
            compiler_cls = getattr(compiler_module, "IntentCompiler")
        except Exception as exc:
            raise _arena_api_error(_safe_exception(exc)) from None
        if not callable(getattr(environment_intent_cls, "model_validate", None)):
            raise _arena_api_error("EnvironmentIntentSpec lacks model_validate()")
        if not callable(getattr(compiler_cls, "compile", None)):
            raise _arena_api_error("IntentCompiler lacks compile()")
        return environment_intent_cls, compiler_cls


def build_arena_compilation_result(
    initial_graph: Mapping[str, Any],
    linked_graph: Mapping[str, Any],
    compiler_trace: Sequence[CompilerTraceEvent | Mapping[str, Any]] = (),
) -> ArenaCompilationResult:
    """Validate plain Arena artifacts and derive their digest and ordered spatial goal stages.

    This helper is also the supported construction path for fake bridges in pure tests.
    """

    initial_plain = _plain_mapping(initial_graph, ("arena", "initial_graph"))
    linked_plain = _plain_mapping(linked_graph, ("arena", "linked_graph"))
    trace = tuple(_trace_event_to_plain(event) for event in compiler_trace)
    goal_stages = extract_goal_stages(linked_plain)
    return ArenaCompilationResult(
        initial_graph_json=canonical_json(initial_plain),
        linked_graph_json=canonical_json(linked_plain),
        compiler_trace=trace,
        graph_digest=sha256_json(linked_plain),
        goal_stages=goal_stages,
    )


def extract_goal_stages(linked_graph: Mapping[str, Any]) -> tuple[GoalStage, ...]:
    """Extract each linked task's success-state spatial constraints in task order."""

    graph = _plain_mapping(linked_graph, ("arena", "linked_graph"))
    tasks = graph.get("tasks")
    state_specs = graph.get("state_specs")
    if type(tasks) is not list:
        raise _graph_issue(("arena", "linked_graph", "tasks"), "expected a task list")
    if type(state_specs) is not list:
        raise _graph_issue(("arena", "linked_graph", "state_specs"), "expected a state-spec list")

    states_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(state_specs):
        path = ("arena", "linked_graph", "state_specs", index)
        state = _require_plain_dict(value, path)
        state_id = _require_plain_string(state.get("id"), path + ("id",))
        if state_id in states_by_id:
            raise _graph_issue(path + ("id",), f"duplicate state id {state_id!r}")
        states_by_id[state_id] = state

    stages: list[GoalStage] = []
    for index, value in enumerate(tasks):
        task_path = ("arena", "linked_graph", "tasks", index)
        task = _require_plain_dict(value, task_path)
        task_id = _require_plain_string(task.get("id"), task_path + ("id",))
        task_kind = _require_plain_string(task.get("kind"), task_path + ("kind",))
        success_id = _require_plain_string(task.get("success_state_spec_id"), task_path + ("success_state_spec_id",))
        if success_id not in states_by_id:
            raise _graph_issue(
                task_path + ("success_state_spec_id",),
                f"task references missing success state {success_id!r}",
            )
        state = states_by_id[success_id]
        constraints = state.get("spatial_constraints", [])
        if type(constraints) is not list:
            raise _graph_issue(
                ("arena", "linked_graph", "state_specs", success_id, "spatial_constraints"),
                "expected a spatial-constraint list",
            )
        typed_constraints = tuple(
            _parse_spatial_constraint(constraint, task_path + ("success_state", "spatial_constraints", offset))
            for offset, constraint in enumerate(constraints)
        )
        stages.append(
            GoalStage(
                index=index,
                task_id=task_id,
                task_kind=task_kind,
                success_state_spec_id=success_id,
                spatial_constraints=typed_constraints,
            )
        )
    return tuple(stages)


def _parse_spatial_constraint(value: Any, path: tuple[str | int, ...]) -> SpatialGoalConstraint:
    constraint = _require_plain_dict(value, path)
    constraint_id = _require_plain_string(constraint.get("id"), path + ("id",))
    kind = _require_plain_string(constraint.get("kind"), path + ("kind",))
    subject = _require_plain_string(constraint.get("subject"), path + ("subject",))
    reference_value = constraint.get("reference")
    if reference_value is not None and type(reference_value) is not str:
        raise _graph_issue(path + ("reference",), "expected string or null")
    params = constraint.get("params", {})
    params_plain = _require_plain_dict(params, path + ("params",))
    return SpatialGoalConstraint(
        id=constraint_id,
        kind=kind,
        subject=subject,
        reference=reference_value,
        params_json=canonical_json(params_plain),
    )


def _graph_model_to_dict(model: Any, graph_kind: str) -> dict[str, Any]:
    try:
        if callable(getattr(model, "to_dict", None)):
            value = model.to_dict()
        elif callable(getattr(model, "model_dump", None)):
            value = model.model_dump(mode="json", exclude_none=True)
        else:
            raise TypeError(f"{type(model).__name__} lacks to_dict() and model_dump()")
        return _plain_mapping(value, ("arena", f"{graph_kind}_graph"))
    except AutonomousValidationError:
        raise
    except Exception as exc:
        raise AutonomousValidationError([
            ValidationIssue(
                ("arena", f"{graph_kind}_graph"),
                "arena_graph_serialization_failed",
                f"Arena {graph_kind} graph could not be serialized: {_safe_exception(exc)}",
            )
        ]) from None


def _plain_mapping(value: Mapping[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _graph_issue(path, f"expected mapping, got {type(value).__name__}")
    plain = dict(value)
    _validate_plain_json(plain, path)
    return json.loads(canonical_json(plain))


def _validate_plain_json(value: Any, path: tuple[str | int, ...]) -> None:
    stack: list[tuple[Any, tuple[str | int, ...]]] = [(value, path)]
    while stack:
        current, current_path = stack.pop()
        if current is None or type(current) in (str, bool, int):
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise _graph_issue(current_path, "number must be finite")
            continue
        if type(current) is list:
            stack.extend((item, current_path + (index,)) for index, item in enumerate(current))
            continue
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                raise _graph_issue(current_path, "mapping keys must be strings")
            stack.extend((item, current_path + (key,)) for key, item in current.items())
            continue
        raise _graph_issue(current_path, f"expected JSON-compatible value, got {type(current).__name__}")


def _trace_event_to_plain(value: CompilerTraceEvent | Mapping[str, Any] | Any) -> CompilerTraceEvent:
    if isinstance(value, CompilerTraceEvent):
        return value
    if isinstance(value, Mapping):
        stage = value.get("stage")
        query = value.get("query")
        chosen = value.get("chosen")
        note = value.get("note", "")
    else:
        stage = getattr(value, "stage", None)
        query = getattr(value, "query", None)
        chosen = getattr(value, "chosen", None)
        note = getattr(value, "note", "")
    if type(stage) is not str or type(query) is not str or type(note) is not str:
        raise _graph_issue(("arena", "compiler_trace"), "trace stage, query, and note must be strings")
    if chosen is not None and type(chosen) is not str:
        raise _graph_issue(("arena", "compiler_trace"), "trace chosen must be a string or null")
    return CompilerTraceEvent(
        stage=_bounded_text(stage),
        query=_bounded_text(query),
        chosen=None if chosen is None else _bounded_text(chosen),
        note=_bounded_text(note),
    )


def _require_plain_dict(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    if type(value) is not dict:
        raise _graph_issue(path, f"expected mapping, got {type(value).__name__}")
    return value


def _require_plain_string(value: Any, path: tuple[str | int, ...]) -> str:
    if type(value) is not str or not value:
        raise _graph_issue(path, "expected non-empty string")
    return value


def _graph_issue(path: tuple[str | int, ...], message: str) -> AutonomousValidationError:
    return AutonomousValidationError([ValidationIssue(path, "arena_graph_contract_error", message)])


def _arena_api_error(detail: str) -> AutonomousValidationError:
    message = (
        f"Arena agentic intent API is unavailable or incompatible ({detail}). Required capability:"
        f" {REQUIRED_ARENA_CAPABILITY}. Use IsaacLab-Arena commit {REQUIRED_ARENA_COMMIT} or a reviewed compatible"
        " commit."
    )
    return AutonomousValidationError(
        [ValidationIssue(("environment", "intent"), "arena_intent_api_unavailable", message)]
    )


def _safe_exception(exc: Exception) -> str:
    return _bounded_text(f"{type(exc).__name__}: {exc}")


def _bounded_text(value: str) -> str:
    if len(value) <= MAX_SAFE_TEXT_LENGTH:
        return value
    return value[: MAX_SAFE_TEXT_LENGTH - 3] + "..."
