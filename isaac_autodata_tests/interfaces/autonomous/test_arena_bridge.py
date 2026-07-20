# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaac_autodata_interfaces.autonomous import (
    REQUIRED_ARENA_CAPABILITY,
    REQUIRED_ARENA_COMMIT,
    AutonomousValidationError,
    CompilerTraceEvent,
    LazyArenaIntentBridge,
    build_arena_compilation_result,
    compile_loaded_task_request,
    load_task_request,
    task_request_from_dict,
)


def _initial_graph() -> dict:
    return {
        "env_name": "llm_gen_maple_table_PickAndPlaceTask",
        "nodes": [],
        "tasks": [{"kind": "PickAndPlaceTask", "params": {}, "description": "place"}],
        "initial_state_spec": {
            "id": "state_initial",
            "is_delta": False,
            "spatial_constraints": [],
            "task_constraints": [],
        },
    }


def _linked_graph(selected: str = "pick_cube") -> dict:
    return {
        "env_name": "llm_gen_maple_table_PickAndPlaceTask",
        "nodes": [],
        "tasks": [
            {
                "id": "task_0_PickAndPlaceTask",
                "kind": "PickAndPlaceTask",
                "params": {"pick_up_object": selected},
                "description": "place",
                "initial_state_spec_id": "state_initial",
                "success_state_spec_id": "state_spec_1",
            },
            {
                "id": "task_1_OpenDoorTask",
                "kind": "OpenDoorTask",
                "params": {},
                "description": "open",
                "initial_state_spec_id": "state_spec_1",
                "success_state_spec_id": "state_spec_2",
            },
        ],
        "state_specs": [
            {
                "id": "state_initial",
                "is_delta": False,
                "spatial_constraints": [],
                "task_constraints": [],
            },
            {
                "id": "state_spec_1",
                "is_delta": True,
                "spatial_constraints": [{
                    "id": "state_spec_1_pick_cube_on_destination_bowl",
                    "kind": "on",
                    "subject": selected,
                    "reference": "destination_bowl",
                    "params": {"margin": 0.01},
                }],
                "task_constraints": [],
            },
            {
                "id": "state_spec_2",
                "is_delta": True,
                "spatial_constraints": [],
                "task_constraints": [],
            },
        ],
        "cli_override_specs": [],
    }


def _envelope(tmp_path: Path):
    return task_request_from_dict(
        {
            "schema_version": 1,
            "name": "bridge_test",
            "environment": {"intent": {"Arena": {"owns": ["all", "of", "this"]}}},
            "planner": {
                "backend": "schedulestream",
                "motion_backend": "curobo_v2",
                "collisions": True,
                "max_time_s": 30.0,
                "batch_size": 64,
                "interpolation_dt_s": 0.1,
                "profile": False,
                "animate": False,
            },
            "generation": {
                "successful_episodes": 2,
                "seed": 11,
                "num_envs": 1,
                "max_attempts": 4,
            },
            "output": {
                "dataset": "outputs/data.hdf5",
                "keep_failed": False,
                "run_log": "outputs/data.jsonl",
            },
        },
        source_path=tmp_path / "request.yaml",
    )


class _FakeBridge:
    def __init__(self):
        self.calls: list[tuple[dict, int]] = []

    def compile_and_link(self, intent: dict, *, seed: int):
        self.calls.append((copy.deepcopy(intent), seed))
        return build_arena_compilation_result(
            _initial_graph(),
            _linked_graph(),
            [CompilerTraceEvent(stage="item.exact", query="cube", chosen="cube", note="")],
        )


def test_fake_bridge_resolves_source_free_request_and_absolute_outputs(tmp_path: Path):
    envelope = _envelope(tmp_path)
    bridge = _FakeBridge()

    resolved = compile_loaded_task_request(envelope, bridge=bridge)

    assert bridge.calls == [({"Arena": {"owns": ["all", "of", "this"]}}, 11)]
    assert resolved.source_dataset_path is None
    assert resolved.output.dataset == (tmp_path / "outputs" / "data.hdf5").resolve()
    assert resolved.output.run_log == (tmp_path / "outputs" / "data.jsonl").resolve()
    assert resolved.environment_name == "llm_gen_maple_table_PickAndPlaceTask"
    assert len(resolved.request_digest) == 64
    assert len(resolved.graph_digest) == 64
    assert len(resolved.digest) == 64


def test_goal_stages_follow_linked_task_and_constraint_order():
    result = build_arena_compilation_result(_initial_graph(), _linked_graph())

    assert [stage.task_kind for stage in result.goal_stages] == ["PickAndPlaceTask", "OpenDoorTask"]
    first, second = result.goal_stages
    assert first.index == 0
    assert first.success_state_spec_id == "state_spec_1"
    assert len(first.spatial_constraints) == 1
    constraint = first.spatial_constraints[0]
    assert constraint.kind == "on"
    assert constraint.subject == "pick_cube"
    assert constraint.reference == "destination_bowl"
    assert constraint.params == {"margin": 0.01}
    assert second.spatial_constraints == ()


def test_graph_digest_is_independent_of_mapping_key_order():
    first = _linked_graph()
    second = dict(reversed(list(copy.deepcopy(first).items())))

    result_a = build_arena_compilation_result(_initial_graph(), first)
    result_b = build_arena_compilation_result(_initial_graph(), second)

    assert result_a.graph_digest == result_b.graph_digest
    assert result_a.linked_graph_json == result_b.linked_graph_json


def test_resolved_request_is_deterministic_for_same_fake_bridge(tmp_path: Path):
    envelope = _envelope(tmp_path)

    first = compile_loaded_task_request(envelope, bridge=_FakeBridge())
    second = compile_loaded_task_request(envelope, bridge=_FakeBridge())

    assert first.canonical_json() == second.canonical_json()
    assert first.digest == second.digest


def test_missing_success_state_is_a_structured_graph_contract_error():
    linked = _linked_graph()
    linked["tasks"][0]["success_state_spec_id"] = "missing"

    with pytest.raises(AutonomousValidationError) as exc:
        build_arena_compilation_result(_initial_graph(), linked)

    issue = exc.value.issues[0]
    assert issue.code == "arena_graph_contract_error"
    assert issue.field_path == "$.arena.linked_graph.tasks[0].success_state_spec_id"


def test_unexpected_fake_bridge_failure_is_structured(tmp_path: Path):
    class BrokenBridge:
        def compile_and_link(self, intent: dict, *, seed: int):
            raise RuntimeError("backend exploded\nsecret second line")

    with pytest.raises(AutonomousValidationError) as exc:
        compile_loaded_task_request(_envelope(tmp_path), bridge=BrokenBridge())

    issue = exc.value.issues[0]
    assert issue.code == "arena_bridge_failed"
    assert issue.field_path == "$.environment.intent"
    assert "Traceback" not in issue.message
    assert "\n" not in issue.message


def test_missing_arena_api_error_names_required_capability_and_commit():
    def missing(_name: str):
        raise ModuleNotFoundError("old Arena checkout")

    bridge = LazyArenaIntentBridge(module_loader=missing)
    with pytest.raises(AutonomousValidationError) as exc:
        bridge.compile_and_link({}, seed=1)

    issue = exc.value.issues[0]
    assert issue.code == "arena_intent_api_unavailable"
    assert REQUIRED_ARENA_CAPABILITY in issue.message
    assert REQUIRED_ARENA_COMMIT in issue.message
    assert "Traceback" not in issue.message


class _FakeIntentSpec:
    @classmethod
    def model_validate(cls, value: dict):
        random.random()
        return copy.deepcopy(value)


class _FakeInitialGraphModel:
    def __init__(self, selected: str):
        self.selected = selected

    def to_dict(self):
        value = _initial_graph()
        value["selected"] = self.selected
        return value

    def link(self):
        return _FakeLinkedGraphModel(self.selected)


class _FakeLinkedGraphModel:
    def __init__(self, selected: str):
        self.selected = selected

    def to_dict(self):
        return _linked_graph(self.selected)


class _FakeIntentCompiler:
    def __init__(self):
        self.trace = []
        self.resolution_errors = []

    def compile(self, _spec):
        selected = random.choice(["cube_1", "cube_2", "cube_3"])
        self.trace.append(SimpleNamespace(stage="task.resolved_param", query="cube", chosen=selected, note=""))
        return _FakeInitialGraphModel(selected)


def _fake_module_loader(name: str):
    if name.endswith("environment_intent_spec"):
        return SimpleNamespace(EnvironmentIntentSpec=_FakeIntentSpec)
    if name.endswith("intent_compiler"):
        return SimpleNamespace(IntentCompiler=_FakeIntentCompiler)
    raise ModuleNotFoundError(name)


def test_lazy_bridge_seeds_and_restores_global_random_state():
    bridge = LazyArenaIntentBridge(module_loader=_fake_module_loader)
    random.seed(123456)
    state_before = random.getstate()

    first = bridge.compile_and_link({"opaque": True}, seed=42)
    state_after = random.getstate()
    second = bridge.compile_and_link({"opaque": True}, seed=42)

    assert state_after == state_before
    assert random.getstate() == state_before
    assert first.linked_graph == second.linked_graph
    assert first.compiler_trace == second.compiler_trace


def test_resolution_error_fails_and_restores_random_state():
    class ErrorCompiler(_FakeIntentCompiler):
        def compile(self, spec):
            model = super().compile(spec)
            self.resolution_errors = [
                SimpleNamespace(stage="item.required_tags.miss", query="cube", chosen=None, note="not found")
            ]
            return model

    def loader(name: str):
        if name.endswith("environment_intent_spec"):
            return SimpleNamespace(EnvironmentIntentSpec=_FakeIntentSpec)
        return SimpleNamespace(IntentCompiler=ErrorCompiler)

    bridge = LazyArenaIntentBridge(module_loader=loader)
    random.seed(9876)
    state_before = random.getstate()

    with pytest.raises(AutonomousValidationError) as exc:
        bridge.compile_and_link({"opaque": True}, seed=42)

    assert exc.value.issues[0].code == "arena_resolution_error"
    assert "item.required_tags.miss" in exc.value.issues[0].message
    assert random.getstate() == state_before


def test_default_bridge_live_example_when_current_arena_api_is_available():
    bridge = LazyArenaIntentBridge()
    if not bridge.is_available():
        pytest.skip("host Arena install predates the required agentic intent API")
    example = Path(__file__).parents[3] / "isaac_autodata_examples" / "autonomous" / "franka_pick_cube_into_bowl.yaml"
    request = load_task_request(example)

    result = bridge.compile_and_link(request.environment_intent, seed=request.generation.seed)

    assert result.linked_graph["tasks"]
    assert result.goal_stages[0].task_kind == "PickAndPlaceTask"
