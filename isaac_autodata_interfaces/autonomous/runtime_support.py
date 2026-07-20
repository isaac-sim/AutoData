# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import-free product capability gate for the live autonomous generation lane.

Arena remains the authority that validates and links user-authored semantics. This module applies a
second, deliberately narrower check: whether a resolved graph can be executed by the concrete
AutoData runtime that is available today. It consumes only resolved plain data and selected runtime
names, so callers can run it before launching Isaac or importing a planner implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.task_request_types import CompiledTaskRequest, canonical_json

RUNTIME_SUPPORT_SCHEMA_VERSION = 1
CURRENT_RUNTIME_SUPPORT = "franka_pick_and_place_custream_v1"

_SUPPORTED_MOTION_BACKEND = "curobo_v1"
_SUPPORTED_SCHEDULESTREAM_APPLICATION = "custream"
_SUPPORTED_EMBODIMENT = "franka_ik"
_SUPPORTED_TASK_KIND = "PickAndPlaceTask"
_SUPPORTED_RELATION = "on"
_SUPPORTED_BACKGROUND_ASSET = "maple_table_robolab"
_SUPPORTED_PICK_UP_ASSET = "rubiks_cube_hot3d_robolab"
_SUPPORTED_DESTINATION_ASSET = "bowl_ycb_robolab"
_REQUIRED_TASK_PARAMS = frozenset({"background_scene", "destination_location", "pick_up_object"})
_MAX_GRAPH_ID_LENGTH = 128
_LIVE_GRAPH_ID_PATTERN = re.compile(
    rf"[A-Za-z_][A-Za-z0-9_]{{0,{_MAX_GRAPH_ID_LENGTH - 1}}}",
    flags=re.ASCII,
)
_MAX_LIVE_BATCH_SIZE = 1024
_MAX_LIVE_SUCCESSFUL_EPISODES = 10
_MAX_LIVE_ATTEMPTS_PER_SUCCESS = 5
_MAX_LIVE_PLANNER_TIME_S = 60.0
_LIVE_INTERPOLATION_DT_S = 0.02


class RuntimeSupportError(AutonomousValidationError):
    """A compiled task request is outside the currently executable autonomous product slice."""


@dataclass(frozen=True)
class RuntimeSupportProfile:
    """Attestable identity of one request admitted by the pure product capability gate."""

    profile: str
    request_digest: str
    graph_digest: str
    motion_backend: str
    schedulestream_application: str
    embodiment_node_id: str
    embodiment_name: str
    task_id: str
    task_kind: str
    success_state_spec_id: str
    success_constraint_id: str
    pick_up_object_id: str
    destination_location_id: str
    background_scene_id: str
    schema_version: int = RUNTIME_SUPPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the complete deterministic profile attestation payload."""

        return {
            "embodiment": {
                "name": self.embodiment_name,
                "node_id": self.embodiment_node_id,
            },
            "graph_digest": self.graph_digest,
            "motion_backend": self.motion_backend,
            "profile": self.profile,
            "request_digest": self.request_digest,
            "schedulestream_application": self.schedulestream_application,
            "schema_version": self.schema_version,
            "task": {
                "background_scene_id": self.background_scene_id,
                "destination_location_id": self.destination_location_id,
                "id": self.task_id,
                "kind": self.task_kind,
                "pick_up_object_id": self.pick_up_object_id,
                "success_constraint_id": self.success_constraint_id,
                "success_state_spec_id": self.success_state_spec_id,
            },
        }

    def canonical_json(self) -> str:
        """Return canonical JSON suitable for parent/child process attestation."""

        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the complete runtime-support profile."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_runtime_support(
    request: CompiledTaskRequest,
    *,
    motion_backend: str,
    schedulestream_application: str,
) -> RuntimeSupportProfile:
    """Validate and attest the currently executable source-free runtime slice.

    The first live product profile is intentionally narrow: one ``franka_ik`` embodiment, one
    linked ``PickAndPlaceTask``, one parameter-free ``on`` success relation, one environment, and
    the reviewed ``curobo_v1``/``custream`` pairing. Schema support for cuRobo v2 remains a design
    target; it is not a claim that a live custream2 provider exists.

    Args:
        request: Fully resolved, linked, source-demo-free request.
        motion_backend: Motion backend selected by the import-free runtime probe.
        schedulestream_application: ScheduleStream application selected by the compatibility gate.

    Returns:
        Deterministic profile data that can be digest-attested across a process boundary.

    Raises:
        RuntimeSupportError: If any selected runtime or linked-graph property is outside the
            reviewed executable profile.
    """

    issues: list[ValidationIssue] = []
    _validate_selected_runtime(motion_backend, schedulestream_application, issues)
    _validate_live_operational_limits(request, issues)

    num_envs = getattr(getattr(request, "generation", None), "num_envs", None)
    if type(num_envs) is not int or num_envs != 1:
        issues.append(
            ValidationIssue(
                ("generation", "num_envs"),
                "live_num_envs_unsupported",
                f"the current autonomous runtime requires exactly one environment; got {num_envs!r}",
            )
        )

    try:
        linked_graph = request.linked_graph
    except Exception:
        linked_graph = None
    if type(linked_graph) is not dict:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph"),
                "linked_graph_invalid",
                "the resolved linked graph must be a plain mapping",
            )
        )
        raise RuntimeSupportError(issues)

    nodes = _plain_mapping_list(linked_graph.get("nodes"), ("arena", "linked_graph", "nodes"), issues)
    if len(nodes) != 4:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "nodes"),
                "live_graph_node_count_unsupported",
                f"the reviewed live profile requires exactly four scene nodes; got {len(nodes)}",
            )
        )
    nodes_by_id = _validate_graph_nodes(nodes, issues)
    embodiment = _select_embodiment(nodes, issues)

    tasks = _plain_mapping_list(linked_graph.get("tasks"), ("arena", "linked_graph", "tasks"), issues)
    task = _select_task(tasks, issues)

    state_specs = _plain_mapping_list(
        linked_graph.get("state_specs"),
        ("arena", "linked_graph", "state_specs"),
        issues,
    )
    if len(state_specs) != 2:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "state_specs"),
                "live_state_spec_count_unsupported",
                f"the reviewed live profile requires exactly initial and success state specs; got {len(state_specs)}",
            )
        )
    states_by_id = _validate_state_specs(state_specs, issues)
    _validate_task_state_bindings(task, set(states_by_id), issues)

    try:
        goal_stages = request.goal_stages
    except Exception:
        goal_stages = None
    stage = _select_goal_stage(goal_stages, issues)
    constraint = _select_goal_constraint(stage, issues)

    task_params = _validate_task(task, nodes_by_id, issues)
    _validate_initial_state(task, task_params, states_by_id, issues)
    _validate_stage_binding(task, stage, issues)
    _validate_constraint(constraint, task_params, nodes_by_id, issues)
    _validate_success_state_projection(task, constraint, states_by_id, issues)

    if issues:
        raise RuntimeSupportError(issues)

    assert embodiment is not None
    assert task is not None
    assert stage is not None
    assert constraint is not None
    assert task_params is not None
    return RuntimeSupportProfile(
        profile=CURRENT_RUNTIME_SUPPORT,
        request_digest=request.request_digest,
        graph_digest=request.graph_digest,
        motion_backend=motion_backend,
        schedulestream_application=schedulestream_application,
        embodiment_node_id=embodiment["id"],
        embodiment_name=embodiment["name"],
        task_id=task["id"],
        task_kind=task["kind"],
        success_state_spec_id=stage.success_state_spec_id,
        success_constraint_id=constraint.id,
        pick_up_object_id=task_params["pick_up_object"],
        destination_location_id=task_params["destination_location"],
        background_scene_id=task_params["background_scene"],
    )


def _validate_selected_runtime(
    motion_backend: Any,
    application: Any,
    issues: list[ValidationIssue],
) -> None:
    if motion_backend == "curobo_v2" and application == "custream2":
        issues.append(
            ValidationIssue(
                ("runtime", "selected_motion_backend"),
                "live_curobo_v2_unsupported",
                "curobo_v2/custream2 is schema-compatible but has no reviewed live IsaacLab provider yet",
            )
        )
        return
    if motion_backend != _SUPPORTED_MOTION_BACKEND:
        issues.append(
            ValidationIssue(
                ("runtime", "selected_motion_backend"),
                "live_motion_backend_unsupported",
                f"the current live profile requires {_SUPPORTED_MOTION_BACKEND!r}; got {motion_backend!r}",
            )
        )
    if application != _SUPPORTED_SCHEDULESTREAM_APPLICATION:
        issues.append(
            ValidationIssue(
                ("runtime", "schedulestream_application"),
                "live_schedulestream_application_unsupported",
                f"the current live profile requires {_SUPPORTED_SCHEDULESTREAM_APPLICATION!r}; got {application!r}",
            )
        )


def _validate_live_operational_limits(
    request: CompiledTaskRequest,
    issues: list[ValidationIssue],
) -> None:
    planner = getattr(request, "planner", None)
    generation = getattr(request, "generation", None)
    output = getattr(request, "output", None)
    checks = (
        (
            getattr(planner, "collisions", None) is True,
            ("planner", "collisions"),
            "live_collisions_required",
            "the reviewed live profile requires collision checking",
        ),
        (
            type(getattr(planner, "max_time_s", None)) in (int, float)
            and 0 < float(planner.max_time_s) <= _MAX_LIVE_PLANNER_TIME_S,
            ("planner", "max_time_s"),
            "live_planner_time_limit",
            f"live planner max_time_s must be in (0, {_MAX_LIVE_PLANNER_TIME_S:g}]",
        ),
        (
            type(getattr(planner, "batch_size", None)) is int and 1 <= planner.batch_size <= _MAX_LIVE_BATCH_SIZE,
            ("planner", "batch_size"),
            "live_batch_size_limit",
            f"live planner batch_size must be in [1, {_MAX_LIVE_BATCH_SIZE}]",
        ),
        (
            getattr(planner, "profile", None) is False,
            ("planner", "profile"),
            "live_profile_mode_unsupported",
            "planner profiling is disabled in the reviewed live profile",
        ),
        (
            getattr(planner, "animate", None) is False,
            ("planner", "animate"),
            "live_animation_unsupported",
            "planner animation is disabled in the reviewed live profile",
        ),
        (
            type(getattr(planner, "interpolation_dt_s", None)) in (int, float)
            and abs(float(planner.interpolation_dt_s) - _LIVE_INTERPOLATION_DT_S) <= 1e-9,
            ("planner", "interpolation_dt_s"),
            "live_interpolation_dt_unsupported",
            f"the reviewed live profile requires interpolation_dt_s={_LIVE_INTERPOLATION_DT_S:g}",
        ),
        (
            type(getattr(generation, "successful_episodes", None)) is int
            and 1 <= generation.successful_episodes <= _MAX_LIVE_SUCCESSFUL_EPISODES,
            ("generation", "successful_episodes"),
            "live_success_target_limit",
            f"live successful_episodes must be in [1, {_MAX_LIVE_SUCCESSFUL_EPISODES}]",
        ),
        (
            type(getattr(generation, "max_attempts", None)) is int
            and type(getattr(generation, "successful_episodes", None)) is int
            and generation.successful_episodes >= 1
            and generation.successful_episodes <= generation.max_attempts
            and generation.max_attempts <= generation.successful_episodes * _MAX_LIVE_ATTEMPTS_PER_SUCCESS,
            ("generation", "max_attempts"),
            "live_attempt_limit",
            "live max_attempts must be at least successful_episodes and at most five attempts per requested success",
        ),
        (
            getattr(output, "run_log", None) is not None,
            ("output", "run_log"),
            "live_run_log_required",
            "the reviewed live profile requires a durable run_log JSONL ledger",
        ),
    )
    for accepted, path, code, message in checks:
        if not accepted:
            issues.append(ValidationIssue(path, code, message))


def _plain_mapping_list(
    value: Any,
    path: tuple[str | int, ...],
    issues: list[ValidationIssue],
) -> list[dict[str, Any]]:
    if type(value) is not list:
        issues.append(ValidationIssue(path, "linked_graph_shape_invalid", "expected a plain list"))
        return []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            issues.append(
                ValidationIssue(
                    path + (index,),
                    "linked_graph_shape_invalid",
                    "expected a plain mapping",
                )
            )
            # Preserve the source index so any additional count or identity failures continue to
            # point at the original linked-graph location.
            result.append({})
            continue
        result.append(item)
    return result


def _validate_graph_nodes(
    nodes: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        path = ("arena", "linked_graph", "nodes", index)
        node_id = node.get("id")
        if not _is_graph_id(node_id):
            issues.append(ValidationIssue(path + ("id",), "graph_id_invalid", _graph_id_message(node_id)))
            continue
        if node_id in by_id:
            issues.append(
                ValidationIssue(
                    path + ("id",),
                    "graph_id_duplicate",
                    f"graph node id {node_id!r} is duplicated",
                )
            )
            continue
        by_id[node_id] = node
    return by_id


def _select_embodiment(
    nodes: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, Any] | None:
    indexed = [(index, node) for index, node in enumerate(nodes) if node.get("type") == "embodiment"]
    if len(indexed) != 1:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "nodes"),
                "live_embodiment_count_unsupported",
                f"the current live profile requires exactly one embodiment node; got {len(indexed)}",
            )
        )
        return None
    index, embodiment = indexed[0]
    name = embodiment.get("name")
    if name != _SUPPORTED_EMBODIMENT:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "nodes", index, "name"),
                "live_embodiment_unsupported",
                f"the current live profile requires {_SUPPORTED_EMBODIMENT!r}; got {name!r}",
            )
        )
    if embodiment.get("params") != {}:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "nodes", index, "params"),
                "live_embodiment_params_unsupported",
                "the reviewed Franka DIK frame/controller profile requires empty embodiment params",
            )
        )
    return embodiment


def _select_task(
    tasks: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, Any] | None:
    if len(tasks) != 1:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "tasks"),
                "live_task_count_unsupported",
                f"the current live profile requires exactly one linked task; got {len(tasks)}",
            )
        )
        return None
    task = tasks[0]
    if task.get("kind") != _SUPPORTED_TASK_KIND:
        issues.append(
            ValidationIssue(
                ("arena", "linked_graph", "tasks", 0, "kind"),
                "live_task_kind_unsupported",
                f"the current live profile requires {_SUPPORTED_TASK_KIND!r}; got {task.get('kind')!r}",
            )
        )
    return task


def _validate_state_specs(
    states: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, tuple[int, dict[str, Any]]]:
    by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, state in enumerate(states):
        path = ("arena", "linked_graph", "state_specs", index, "id")
        state_id = state.get("id")
        if not _is_graph_id(state_id):
            issues.append(ValidationIssue(path, "graph_id_invalid", _graph_id_message(state_id)))
        elif state_id in by_id:
            issues.append(ValidationIssue(path, "graph_id_duplicate", f"state id {state_id!r} is duplicated"))
        else:
            by_id[state_id] = (index, state)
    return by_id


def _validate_task_state_bindings(
    task: dict[str, Any] | None,
    state_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    if task is None:
        return
    for field_name in ("initial_state_spec_id", "success_state_spec_id"):
        value = task.get(field_name)
        path = ("arena", "linked_graph", "tasks", 0, field_name)
        if not _is_graph_id(value):
            issues.append(ValidationIssue(path, "graph_id_invalid", _graph_id_message(value)))
        elif value not in state_ids:
            issues.append(
                ValidationIssue(
                    path,
                    "state_spec_reference_missing",
                    f"task references unknown state spec {value!r}",
                )
            )


def _select_goal_stage(value: Any, issues: list[ValidationIssue]) -> Any | None:
    if type(value) is not tuple or len(value) != 1:
        count = len(value) if isinstance(value, (list, tuple)) else "invalid"
        issues.append(
            ValidationIssue(
                ("arena", "goal_stages"),
                "live_goal_stage_count_unsupported",
                f"the current live profile requires exactly one ordered goal stage; got {count}",
            )
        )
        return None
    stage = value[0]
    if getattr(stage, "index", None) != 0:
        issues.append(
            ValidationIssue(
                ("arena", "goal_stages", 0, "index"),
                "goal_stage_index_invalid",
                "the sole goal stage must have index 0",
            )
        )
    return stage


def _select_goal_constraint(stage: Any | None, issues: list[ValidationIssue]) -> Any | None:
    if stage is None:
        return None
    constraints = getattr(stage, "spatial_constraints", None)
    if type(constraints) is not tuple or len(constraints) != 1:
        count = len(constraints) if isinstance(constraints, (list, tuple)) else "invalid"
        issues.append(
            ValidationIssue(
                ("arena", "goal_stages", 0, "spatial_constraints"),
                "live_goal_constraint_count_unsupported",
                f"the current live profile requires exactly one spatial success constraint; got {count}",
            )
        )
        return None
    return constraints[0]


def _validate_task(
    task: dict[str, Any] | None,
    nodes_by_id: dict[str, dict[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, str] | None:
    if task is None:
        return None
    task_path = ("arena", "linked_graph", "tasks", 0)
    task_id = task.get("id")
    if not _is_graph_id(task_id):
        issues.append(ValidationIssue(task_path + ("id",), "graph_id_invalid", _graph_id_message(task_id)))

    params = task.get("params")
    if type(params) is not dict or any(type(key) is not str for key in params):
        issues.append(
            ValidationIssue(
                task_path + ("params",),
                "task_params_invalid",
                "PickAndPlaceTask params must be a plain mapping with string keys",
            )
        )
        return None
    actual_keys = frozenset(params)
    for missing in sorted(_REQUIRED_TASK_PARAMS - actual_keys):
        issues.append(
            ValidationIssue(
                task_path + ("params", missing),
                "task_param_missing",
                f"the current PickAndPlaceTask profile requires param {missing!r}",
            )
        )
    for unsupported in sorted(actual_keys - _REQUIRED_TASK_PARAMS):
        issues.append(
            ValidationIssue(
                task_path + ("params", unsupported),
                "live_task_param_unsupported",
                f"task param {unsupported!r} is outside the reviewed live profile",
            )
        )

    typed: dict[str, str] = {}
    expected_types = {
        "background_scene": "background",
        "destination_location": "object",
        "pick_up_object": "object",
    }
    expected_assets = {
        "background_scene": _SUPPORTED_BACKGROUND_ASSET,
        "destination_location": _SUPPORTED_DESTINATION_ASSET,
        "pick_up_object": _SUPPORTED_PICK_UP_ASSET,
    }
    asset_issue_codes = {
        "background_scene": "live_background_asset_unsupported",
        "destination_location": "live_destination_asset_unsupported",
        "pick_up_object": "live_pick_up_geometry_unsupported",
    }
    for name in sorted(_REQUIRED_TASK_PARAMS & actual_keys):
        value = params[name]
        path = task_path + ("params", name)
        if not _is_graph_id(value):
            issues.append(ValidationIssue(path, "graph_id_invalid", _graph_id_message(value)))
            continue
        typed[name] = value
        node = nodes_by_id.get(value)
        if node is None:
            issues.append(
                ValidationIssue(path, "task_param_node_missing", f"task param references unknown graph node {value!r}")
            )
        elif node.get("type") != expected_types[name]:
            issues.append(
                ValidationIssue(
                    path,
                    "task_param_node_type_unsupported",
                    f"task param {name!r} requires a {expected_types[name]!r} node; {value!r} is {node.get('type')!r}",
                )
            )
        elif node.get("name") != expected_assets[name] or node.get("params") != {}:
            geometry_label = "cuboid top-grasp strategy" if name == "pick_up_object" else "reviewed live scene"
            issues.append(
                ValidationIssue(
                    path,
                    asset_issue_codes[name],
                    f"the {geometry_label} requires the unmodified {expected_assets[name]!r} asset; "
                    f"got name={node.get('name')!r}, "
                    f"params={node.get('params')!r}",
                )
            )
    if typed.get("pick_up_object") == typed.get("destination_location") and "pick_up_object" in typed:
        issues.append(
            ValidationIssue(
                task_path + ("params", "destination_location"),
                "task_endpoints_not_distinct",
                "pick_up_object and destination_location must reference distinct graph nodes",
            )
        )
    return typed if set(typed) == _REQUIRED_TASK_PARAMS else None


def _validate_initial_state(
    task: dict[str, Any] | None,
    task_params: dict[str, str] | None,
    states_by_id: dict[str, tuple[int, dict[str, Any]]],
    issues: list[ValidationIssue],
) -> None:
    """Require the exact reviewed, non-successful pick/place initial semantics."""

    if task is None or task_params is None:
        return
    state_entry = states_by_id.get(task.get("initial_state_spec_id"))
    if state_entry is None:
        return
    state_index, state = state_entry
    path = ("arena", "linked_graph", "state_specs", state_index)
    if state.get("is_delta") is not False:
        issues.append(
            ValidationIssue(
                path + ("is_delta",),
                "live_initial_state_delta_unsupported",
                "the reviewed initial state must be a complete non-delta state",
            )
        )
    if state.get("task_constraints") != []:
        issues.append(
            ValidationIssue(
                path + ("task_constraints",),
                "live_initial_task_constraints_unsupported",
                "the reviewed initial state requires an empty task_constraints list",
            )
        )
    constraints = state.get("spatial_constraints")
    if type(constraints) is not list:
        issues.append(
            ValidationIssue(
                path + ("spatial_constraints",),
                "linked_graph_shape_invalid",
                "the initial spatial constraints must be a plain list",
            )
        )
        return
    expected = {
        ("is_anchor", task_params["background_scene"], None),
        ("on", task_params["pick_up_object"], task_params["background_scene"]),
        ("on", task_params["destination_location"], task_params["background_scene"]),
    }
    actual: list[tuple[str, str, str | None]] = []
    valid_shape = True
    for index, constraint in enumerate(constraints):
        constraint_path = path + ("spatial_constraints", index)
        if type(constraint) is not dict:
            issues.append(
                ValidationIssue(
                    constraint_path,
                    "linked_graph_shape_invalid",
                    "each initial spatial constraint must be a plain mapping",
                )
            )
            valid_shape = False
            continue
        if not _is_graph_id(constraint.get("id")):
            issues.append(
                ValidationIssue(
                    constraint_path + ("id",),
                    "graph_id_invalid",
                    _graph_id_message(constraint.get("id")),
                )
            )
            valid_shape = False
        if constraint.get("params") != {}:
            issues.append(
                ValidationIssue(
                    constraint_path + ("params",),
                    "live_initial_constraint_params_unsupported",
                    "reviewed initial spatial constraints require empty params",
                )
            )
            valid_shape = False
        kind = constraint.get("kind")
        subject = constraint.get("subject")
        reference = constraint.get("reference")
        if (
            not isinstance(kind, str)
            or not _is_graph_id(subject)
            or (reference is not None and not _is_graph_id(reference))
        ):
            valid_shape = False
            continue
        actual.append((kind, subject, reference))
    if not valid_shape or len(actual) != 3 or set(actual) != expected or len(set(actual)) != len(actual):
        issues.append(
            ValidationIssue(
                path + ("spatial_constraints",),
                "live_initial_state_semantics_unsupported",
                "the reviewed initial state must anchor the Maple table and place the distinct cube and bowl on it",
            )
        )


def _validate_stage_binding(
    task: dict[str, Any] | None,
    stage: Any | None,
    issues: list[ValidationIssue],
) -> None:
    if task is None or stage is None:
        return
    comparisons = (
        ("task_id", task.get("id")),
        ("task_kind", task.get("kind")),
        ("success_state_spec_id", task.get("success_state_spec_id")),
    )
    for field_name, expected in comparisons:
        actual = getattr(stage, field_name, None)
        if actual != expected:
            issues.append(
                ValidationIssue(
                    ("arena", "goal_stages", 0, field_name),
                    "goal_stage_task_mismatch",
                    f"goal stage {field_name} {actual!r} does not match linked task value {expected!r}",
                )
            )


def _validate_constraint(
    constraint: Any | None,
    task_params: dict[str, str] | None,
    nodes_by_id: dict[str, dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    if constraint is None:
        return
    path = ("arena", "goal_stages", 0, "spatial_constraints", 0)
    constraint_id = getattr(constraint, "id", None)
    if not _is_graph_id(constraint_id):
        issues.append(ValidationIssue(path + ("id",), "graph_id_invalid", _graph_id_message(constraint_id)))
    relation = getattr(constraint, "kind", None)
    if relation != _SUPPORTED_RELATION:
        issues.append(
            ValidationIssue(
                path + ("kind",),
                "live_goal_relation_unsupported",
                f"the current live profile requires relation {_SUPPORTED_RELATION!r}; got {relation!r}",
            )
        )
    subject = getattr(constraint, "subject", None)
    reference = getattr(constraint, "reference", None)
    for field_name, value in (("subject", subject), ("reference", reference)):
        if not _is_graph_id(value):
            issues.append(ValidationIssue(path + (field_name,), "graph_id_invalid", _graph_id_message(value)))
        elif value not in nodes_by_id:
            issues.append(
                ValidationIssue(
                    path + (field_name,),
                    "goal_constraint_node_missing",
                    f"goal constraint references unknown graph node {value!r}",
                )
            )
    try:
        params = constraint.params
    except Exception:
        params = None
    if type(params) is not dict:
        issues.append(
            ValidationIssue(path + ("params",), "goal_constraint_params_invalid", "constraint params are invalid")
        )
    elif params:
        issues.append(
            ValidationIssue(
                path + ("params",),
                "live_goal_params_unsupported",
                "the current live 'on' relation requires empty params",
            )
        )
    if task_params is None:
        return
    expected = {
        "subject": task_params["pick_up_object"],
        "reference": task_params["destination_location"],
    }
    for field_name, actual in (("subject", subject), ("reference", reference)):
        if actual != expected[field_name]:
            issues.append(
                ValidationIssue(
                    path + (field_name,),
                    "goal_constraint_task_mismatch",
                    f"goal {field_name} {actual!r} does not match task endpoint {expected[field_name]!r}",
                )
            )


def _validate_success_state_projection(
    task: dict[str, Any] | None,
    constraint: Any | None,
    states_by_id: dict[str, tuple[int, dict[str, Any]]],
    issues: list[ValidationIssue],
) -> None:
    """Prove the typed goal constraint is exactly the linked success-state constraint."""

    if task is None or constraint is None:
        return
    state_entry = states_by_id.get(task.get("success_state_spec_id"))
    if state_entry is None:
        return
    state_index, state = state_entry
    state_path = ("arena", "linked_graph", "state_specs", state_index)
    if state.get("is_delta") is not True:
        issues.append(
            ValidationIssue(
                state_path + ("is_delta",),
                "live_success_state_delta_required",
                "the reviewed success state must be a delta state",
            )
        )
    if state.get("task_constraints") != []:
        issues.append(
            ValidationIssue(
                state_path + ("task_constraints",),
                "live_success_task_constraints_unsupported",
                "the reviewed success state requires an empty task_constraints list",
            )
        )
    constraints_path = ("arena", "linked_graph", "state_specs", state_index, "spatial_constraints")
    raw_constraints = state.get("spatial_constraints")
    if type(raw_constraints) is not list or len(raw_constraints) != 1:
        count = len(raw_constraints) if type(raw_constraints) is list else "invalid"
        issues.append(
            ValidationIssue(
                constraints_path,
                "live_goal_constraint_count_unsupported",
                f"the linked success state must contain exactly one spatial constraint; got {count}",
            )
        )
        return
    raw = raw_constraints[0]
    if type(raw) is not dict:
        issues.append(
            ValidationIssue(
                constraints_path + (0,),
                "linked_graph_shape_invalid",
                "the linked success constraint must be a plain mapping",
            )
        )
        return
    try:
        typed_params = constraint.params
    except Exception:
        typed_params = None
    expected = {
        "id": getattr(constraint, "id", None),
        "kind": getattr(constraint, "kind", None),
        "params": typed_params,
        "reference": getattr(constraint, "reference", None),
        "subject": getattr(constraint, "subject", None),
    }
    actual = {
        "id": raw.get("id"),
        "kind": raw.get("kind"),
        "params": raw.get("params", {}),
        "reference": raw.get("reference"),
        "subject": raw.get("subject"),
    }
    if actual != expected:
        issues.append(
            ValidationIssue(
                ("arena", "goal_stages", 0, "spatial_constraints", 0),
                "goal_projection_mismatch",
                "the typed goal constraint does not exactly match the linked success-state constraint",
            )
        )


def _is_graph_id(value: Any) -> bool:
    return type(value) is str and _LIVE_GRAPH_ID_PATTERN.fullmatch(value) is not None


def _graph_id_message(value: Any) -> str:
    return (
        "live graph identity must be an ASCII USD-safe identifier matching "
        f"[A-Za-z_][A-Za-z0-9_]{{0,{_MAX_GRAPH_ID_LENGTH - 1}}}; got {value!r}"
    )
