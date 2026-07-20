# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from isaac_autodata_interfaces.autonomous import (
    CURRENT_RUNTIME_SUPPORT,
    ArenaCompilationResult,
    CompiledTaskRequest,
    GenerationConfig,
    GoalStage,
    MotionBackend,
    PlannerBackend,
    PlannerConfig,
    ResolvedOutputConfig,
    RuntimeSupportError,
    SpatialGoalConstraint,
    canonical_json,
    sha256_json,
    validate_runtime_support,
)


def _linked_graph() -> dict:
    return {
        "env_name": "llm_gen_maple_table_PickAndPlaceTask",
        "nodes": [
            {"id": "robot", "name": "franka_ik", "type": "embodiment", "params": {}},
            {"id": "table", "name": "maple_table_robolab", "type": "background", "params": {}},
            {"id": "pick_cube", "name": "rubiks_cube_hot3d_robolab", "type": "object", "params": {}},
            {"id": "destination_bowl", "name": "bowl_ycb_robolab", "type": "object", "params": {}},
        ],
        "tasks": [{
            "id": "task_0_PickAndPlaceTask",
            "kind": "PickAndPlaceTask",
            "params": {
                "background_scene": "table",
                "destination_location": "destination_bowl",
                "pick_up_object": "pick_cube",
            },
            "description": "place",
            "initial_state_spec_id": "state_initial",
            "success_state_spec_id": "state_success",
        }],
        "state_specs": [
            {
                "id": "state_initial",
                "is_delta": False,
                "spatial_constraints": [
                    {
                        "id": "state_initial_0_is_anchor_table",
                        "kind": "is_anchor",
                        "subject": "table",
                        "params": {},
                    },
                    {
                        "id": "state_initial_1_pick_cube_on_table",
                        "kind": "on",
                        "subject": "pick_cube",
                        "reference": "table",
                        "params": {},
                    },
                    {
                        "id": "state_initial_2_destination_bowl_on_table",
                        "kind": "on",
                        "subject": "destination_bowl",
                        "reference": "table",
                        "params": {},
                    },
                ],
                "task_constraints": [],
            },
            {
                "id": "state_success",
                "is_delta": True,
                "spatial_constraints": [{
                    "id": "state_success_pick_cube_on_destination_bowl",
                    "kind": "on",
                    "subject": "pick_cube",
                    "reference": "destination_bowl",
                    "params": {},
                }],
                "task_constraints": [],
            },
        ],
        "cli_override_specs": [],
    }


def _goal_stage(
    *,
    kind: str = "on",
    subject: str = "pick_cube",
    reference: str | None = "destination_bowl",
    params: dict | None = None,
) -> GoalStage:
    return GoalStage(
        index=0,
        task_id="task_0_PickAndPlaceTask",
        task_kind="PickAndPlaceTask",
        success_state_spec_id="state_success",
        spatial_constraints=(
            SpatialGoalConstraint(
                id="state_success_pick_cube_on_destination_bowl",
                kind=kind,
                subject=subject,
                reference=reference,
                params_json=canonical_json({} if params is None else params),
            ),
        ),
    )


def _resolved_request(
    tmp_path: Path,
    *,
    linked_graph: dict | None = None,
    goal_stages: tuple[GoalStage, ...] | None = None,
    num_envs: int = 1,
) -> CompiledTaskRequest:
    linked = copy.deepcopy(_linked_graph() if linked_graph is None else linked_graph)
    stages = (_goal_stage(),) if goal_stages is None else goal_stages
    arena = ArenaCompilationResult(
        initial_graph_json=canonical_json({"env_name": linked["env_name"]}),
        linked_graph_json=canonical_json(linked),
        compiler_trace=(),
        graph_digest=sha256_json(linked),
        goal_stages=stages,
    )
    return CompiledTaskRequest(
        schema_version=1,
        compiler_version="test",
        name="capability_test",
        canonical_request_json=canonical_json({"name": "capability_test"}),
        request_digest="a" * 64,
        planner=PlannerConfig(
            backend=PlannerBackend.SCHEDULESTREAM,
            motion_backend=MotionBackend.CUROBO_V1,
            collisions=True,
            max_time_s=10.0,
            batch_size=4,
            interpolation_dt_s=0.02,
            profile=False,
            animate=False,
        ),
        generation=GenerationConfig(
            successful_episodes=1,
            seed=7,
            num_envs=num_envs,
            max_attempts=2,
        ),
        output=ResolvedOutputConfig(
            dataset=tmp_path / "data.hdf5",
            keep_failed=False,
            run_log=tmp_path / "data.jsonl",
        ),
        arena=arena,
    )


def _codes(exc: pytest.ExceptionInfo[RuntimeSupportError]) -> set[tuple[str, str]]:
    return {(issue.field_path, issue.code) for issue in exc.value.issues}


def test_valid_profile_is_deterministic_and_attestable(tmp_path: Path) -> None:
    request = _resolved_request(tmp_path)

    first = validate_runtime_support(
        request,
        motion_backend="curobo_v1",
        schedulestream_application="custream",
    )
    second = validate_runtime_support(
        request,
        motion_backend="curobo_v1",
        schedulestream_application="custream",
    )

    assert first == second
    assert first.profile == CURRENT_RUNTIME_SUPPORT
    assert first.pick_up_object_id == "pick_cube"
    assert first.destination_location_id == "destination_bowl"
    assert first.background_scene_id == "table"
    assert first.to_dict()["task"]["success_state_spec_id"] == "state_success"
    assert len(first.digest) == 64
    assert first.canonical_json() == canonical_json(first.to_dict())


def test_capability_module_is_import_free() -> None:
    script = """
import json
import sys
import isaac_autodata_interfaces.autonomous.runtime_support
blocked = ('isaaclab', 'isaaclab_arena', 'torch', 'curobo', 'schedulestream', 'omni')
loaded = sorted(name for name in sys.modules if name.split('.')[0] in blocked)
print(json.dumps(loaded))
"""

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_v2_is_explicitly_schema_only_not_live(tmp_path: Path) -> None:
    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path),
            motion_backend="curobo_v2",
            schedulestream_application="custream2",
        )

    assert _codes(exc) == {("$.runtime.selected_motion_backend", "live_curobo_v2_unsupported")}
    assert "no reviewed live IsaacLab provider" in exc.value.issues[0].message


@pytest.mark.parametrize(
    ("motion_backend", "application", "expected"),
    [
        ("unknown", "custream", ("$.runtime.selected_motion_backend", "live_motion_backend_unsupported")),
        (
            "curobo_v1",
            "custream2",
            ("$.runtime.schedulestream_application", "live_schedulestream_application_unsupported"),
        ),
    ],
)
def test_mixed_or_unknown_runtime_is_rejected(
    tmp_path: Path,
    motion_backend: str,
    application: str,
    expected: tuple[str, str],
) -> None:
    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path),
            motion_backend=motion_backend,
            schedulestream_application=application,
        )

    assert expected in _codes(exc)


def test_parallel_generation_is_rejected_before_launch(tmp_path: Path) -> None:
    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, num_envs=2),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert _codes(exc) == {("$.generation.num_envs", "live_num_envs_unsupported")}


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        ("planner", "collisions", False, ("$.planner.collisions", "live_collisions_required")),
        ("planner", "max_time_s", 60.1, ("$.planner.max_time_s", "live_planner_time_limit")),
        ("planner", "batch_size", 1025, ("$.planner.batch_size", "live_batch_size_limit")),
        ("planner", "profile", True, ("$.planner.profile", "live_profile_mode_unsupported")),
        ("planner", "animate", True, ("$.planner.animate", "live_animation_unsupported")),
        (
            "planner",
            "interpolation_dt_s",
            0.01,
            ("$.planner.interpolation_dt_s", "live_interpolation_dt_unsupported"),
        ),
        (
            "generation",
            "successful_episodes",
            11,
            ("$.generation.successful_episodes", "live_success_target_limit"),
        ),
        ("generation", "max_attempts", 6, ("$.generation.max_attempts", "live_attempt_limit")),
    ],
)
def test_live_profile_rejects_resource_amplification_before_launch(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    expected: tuple[str, str],
) -> None:
    request = _resolved_request(tmp_path)
    config = getattr(request, section)
    request = replace(request, **{section: replace(config, **{field: value})})

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            request,
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert expected in _codes(exc)


def test_live_profile_requires_durable_run_log_before_launch(tmp_path: Path) -> None:
    request = _resolved_request(tmp_path)
    request = replace(request, output=replace(request.output, run_log=None))

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            request,
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert _codes(exc) == {("$.output.run_log", "live_run_log_required")}


@pytest.mark.parametrize("embodiment_name", ["droid_abs_joint_pos", "g1_ik"])
def test_only_franka_ik_embodiment_is_live(tmp_path: Path, embodiment_name: str) -> None:
    linked = _linked_graph()
    linked["nodes"][0]["name"] = embodiment_name

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (
        "$.arena.linked_graph.nodes[0].name",
        "live_embodiment_unsupported",
    ) in _codes(exc)


def test_live_franka_profile_rejects_embodiment_overrides(tmp_path: Path) -> None:
    linked = _linked_graph()
    linked["nodes"][0]["params"] = {"initial_joint_pose": [0.0] * 9}

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (
        "$.arena.linked_graph.nodes[0].params",
        "live_embodiment_params_unsupported",
    ) in _codes(exc)


def test_live_scene_rejects_unreviewed_distractor_nodes(tmp_path: Path) -> None:
    linked = _linked_graph()
    linked["nodes"].append({"id": "distractor", "name": "bowl_ycb_robolab", "type": "object", "params": {}})

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert ("$.arena.linked_graph.nodes", "live_graph_node_count_unsupported") in _codes(exc)


def test_multiple_embodiments_are_rejected(tmp_path: Path) -> None:
    linked = _linked_graph()
    linked["nodes"].append({"id": "robot_2", "name": "franka_ik", "type": "embodiment", "params": {}})

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert ("$.arena.linked_graph.nodes", "live_embodiment_count_unsupported") in _codes(exc)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("second_task", ("$.arena.linked_graph.tasks", "live_task_count_unsupported")),
        ("wrong_kind", ("$.arena.linked_graph.tasks[0].kind", "live_task_kind_unsupported")),
    ],
)
def test_only_one_pick_and_place_task_is_live(
    tmp_path: Path,
    mutation: str,
    expected: tuple[str, str],
) -> None:
    linked = _linked_graph()
    if mutation == "second_task":
        linked["tasks"].append(copy.deepcopy(linked["tasks"][0]))
        linked["tasks"][1]["id"] = "task_1"
    else:
        linked["tasks"][0]["kind"] = "OpenDoorTask"

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert expected in _codes(exc)


def test_task_params_are_exact_and_resolve_to_expected_node_types(tmp_path: Path) -> None:
    linked = _linked_graph()
    params = linked["tasks"][0]["params"]
    params.pop("background_scene")
    params["episode_length_s"] = 20.0
    params["destination_location"] = "table"

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    codes = _codes(exc)
    assert ("$.arena.linked_graph.tasks[0].params.background_scene", "task_param_missing") in codes
    assert ("$.arena.linked_graph.tasks[0].params.episode_length_s", "live_task_param_unsupported") in codes
    assert (
        "$.arena.linked_graph.tasks[0].params.destination_location",
        "task_param_node_type_unsupported",
    ) in codes


@pytest.mark.parametrize(
    ("asset_name", "asset_params"),
    [
        ("bowl_ycb_robolab", {}),
        ("rubiks_cube_hot3d_robolab", {"scale": 2.0}),
    ],
)
def test_live_pickup_requires_reviewed_unmodified_cube_geometry(
    tmp_path: Path,
    asset_name: str,
    asset_params: dict,
) -> None:
    linked = _linked_graph()
    linked["nodes"][2]["name"] = asset_name
    linked["nodes"][2]["params"] = asset_params

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (
        "$.arena.linked_graph.tasks[0].params.pick_up_object",
        "live_pick_up_geometry_unsupported",
    ) in _codes(exc)


@pytest.mark.parametrize(
    ("node_index", "field", "asset_name", "asset_params", "expected_code"),
    [
        (1, "background_scene", "other_table", {}, "live_background_asset_unsupported"),
        (1, "background_scene", "maple_table_robolab", {"scale": 2.0}, "live_background_asset_unsupported"),
        (3, "destination_location", "other_bowl", {}, "live_destination_asset_unsupported"),
        (3, "destination_location", "bowl_ycb_robolab", {"scale": 2.0}, "live_destination_asset_unsupported"),
    ],
)
def test_live_scene_requires_reviewed_unmodified_table_and_bowl(
    tmp_path: Path,
    node_index: int,
    field: str,
    asset_name: str,
    asset_params: dict,
    expected_code: str,
) -> None:
    linked = _linked_graph()
    linked["nodes"][node_index]["name"] = asset_name
    linked["nodes"][node_index]["params"] = asset_params

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (f"$.arena.linked_graph.tasks[0].params.{field}", expected_code) in _codes(exc)


def test_task_and_state_graph_ids_must_be_unique_and_resolved(tmp_path: Path) -> None:
    linked = _linked_graph()
    linked["nodes"][3]["id"] = "pick_cube"
    linked["tasks"][0]["success_state_spec_id"] = "missing_state"

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    codes = _codes(exc)
    assert ("$.arena.linked_graph.nodes[3].id", "graph_id_duplicate") in codes
    assert ("$.arena.linked_graph.tasks[0].success_state_spec_id", "state_spec_reference_missing") in codes


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "../Robot",
        "cube/name",
        "{ENV_REGEX_NS}",
        "cube.*",
        "cube[0]",
        "cube\nother",
        "-cube",
        "éclair",
        "a" * 129,
    ],
)
def test_live_object_ids_must_be_ascii_usd_safe(tmp_path: Path, unsafe_id: str) -> None:
    linked = _linked_graph()
    linked["nodes"][2]["id"] = unsafe_id
    linked["tasks"][0]["params"]["pick_up_object"] = unsafe_id
    linked["state_specs"][0]["spatial_constraints"][1]["subject"] = unsafe_id
    linked["state_specs"][1]["spatial_constraints"][0]["subject"] = unsafe_id

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(
                tmp_path,
                linked_graph=linked,
                goal_stages=(_goal_stage(subject=unsafe_id),),
            ),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert ("$.arena.linked_graph.nodes[2].id", "graph_id_invalid") in _codes(exc)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("already_successful", "live_initial_state_semantics_unsupported"),
        ("missing_constraints", "live_initial_state_semantics_unsupported"),
        ("delta", "live_initial_state_delta_unsupported"),
        ("task_constraint", "live_initial_task_constraints_unsupported"),
        ("constraint_params", "live_initial_constraint_params_unsupported"),
    ],
)
def test_live_initial_state_must_be_exact_and_nontrivial(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    linked = _linked_graph()
    initial = linked["state_specs"][0]
    if mutation == "already_successful":
        initial["spatial_constraints"][1]["reference"] = "destination_bowl"
    elif mutation == "missing_constraints":
        initial["spatial_constraints"] = []
    elif mutation == "delta":
        initial["is_delta"] = True
    elif mutation == "task_constraint":
        initial["task_constraints"] = [{"kind": "already_done"}]
    else:
        initial["spatial_constraints"][1]["params"] = {"margin": 0.01}

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert any(code == expected_code for _, code in _codes(exc))


def test_goal_stage_must_match_the_single_linked_task(tmp_path: Path) -> None:
    stage = replace(_goal_stage(), task_id="different_task", success_state_spec_id="different_state")

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, goal_stages=(stage,)),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    codes = _codes(exc)
    assert ("$.arena.goal_stages[0].task_id", "goal_stage_task_mismatch") in codes
    assert ("$.arena.goal_stages[0].success_state_spec_id", "goal_stage_task_mismatch") in codes


def test_goal_projection_must_exactly_match_linked_success_state(tmp_path: Path) -> None:
    linked = _linked_graph()
    linked["state_specs"][1]["spatial_constraints"][0]["reference"] = "table"

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, linked_graph=linked),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (
        "$.arena.goal_stages[0].spatial_constraints[0]",
        "goal_projection_mismatch",
    ) in _codes(exc)


def test_multiple_goal_stages_are_not_flattened(tmp_path: Path) -> None:
    stages = (_goal_stage(), replace(_goal_stage(), index=1, task_id="task_1"))

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, goal_stages=stages),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert ("$.arena.goal_stages", "live_goal_stage_count_unsupported") in _codes(exc)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (
            _goal_stage(kind="in"),
            ("$.arena.goal_stages[0].spatial_constraints[0].kind", "live_goal_relation_unsupported"),
        ),
        (
            _goal_stage(params={"margin": 0.01}),
            ("$.arena.goal_stages[0].spatial_constraints[0].params", "live_goal_params_unsupported"),
        ),
        (
            _goal_stage(subject="destination_bowl", reference="pick_cube"),
            ("$.arena.goal_stages[0].spatial_constraints[0].subject", "goal_constraint_task_mismatch"),
        ),
    ],
)
def test_goal_relation_params_and_task_endpoints_are_not_reinterpreted(
    tmp_path: Path,
    stage: GoalStage,
    expected: tuple[str, str],
) -> None:
    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, goal_stages=(stage,)),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert expected in _codes(exc)


def test_exactly_one_success_constraint_is_required(tmp_path: Path) -> None:
    stage = replace(_goal_stage(), spatial_constraints=())

    with pytest.raises(RuntimeSupportError) as exc:
        validate_runtime_support(
            _resolved_request(tmp_path, goal_stages=(stage,)),
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    assert (
        "$.arena.goal_stages[0].spatial_constraints",
        "live_goal_constraint_count_unsupported",
    ) in _codes(exc)
