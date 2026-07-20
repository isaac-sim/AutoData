# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaac_autodata_interfaces.autonomous.schedulestream import ScheduleStreamProviderError
from isaac_autodata_interfaces.autonomous.schedulestream.episode_planner import create_schedulestream_episode_planner

_PINNED_PANDA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
    "Robots/FrankaEmika/panda_instanceable.usd"
)
_PINNED_PANDA_USD_BYTES = 8_038
_PINNED_PANDA_USD_SHA256 = "7f5a0c0aa6760cfbd348e08bc464d4b94341f027f51c2d9e42406ceefcc7787f"
_REGISTRY_PANDA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
    "Arena/assets/robot_library/franka_panda_hand_on_stand.usd"
)


class _ActionManager:
    active_terms = ("arm",)

    def get_term(self, name: str):
        assert name == "arm"
        return SimpleNamespace(cfg=SimpleNamespace(body_name="panda_hand"))


def _request(
    *,
    task_pick: str = "pick_cube",
    task_destination: str = "destination_bowl",
    goal_subject: str = "pick_cube",
    goal_destination: str = "destination_bowl",
):
    planner = SimpleNamespace(
        animate=False,
        batch_size=8,
        collisions=True,
        interpolation_dt_s=0.02,
        max_time_s=10.0,
        profile=False,
    )
    planner.to_dict = lambda: {"batch_size": 8, "collisions": True}
    return SimpleNamespace(
        generation=SimpleNamespace(num_envs=1),
        goal_stages=(
            SimpleNamespace(
                spatial_constraints=(SimpleNamespace(kind="on", subject=goal_subject, reference=goal_destination),)
            ),
        ),
        graph_digest="a" * 64,
        linked_graph={
            "nodes": [
                {
                    "id": "pick_cube",
                    "name": "rubiks_cube_hot3d_robolab",
                    "params": {},
                    "type": "object",
                },
                {
                    "id": "destination_bowl",
                    "name": "bowl_ycb_robolab",
                    "params": {},
                    "type": "object",
                },
            ],
            "tasks": [{
                "id": "task",
                "params": {
                    "background_scene": "table",
                    "destination_location": task_destination,
                    "pick_up_object": task_pick,
                },
            }],
        },
        planner=planner,
    )


def _bundle():
    base_env = SimpleNamespace(
        action_manager=_ActionManager(),
        sim=SimpleNamespace(get_physics_dt=lambda: 0.005),
    )
    adapter = SimpleNamespace(get_eef_names=lambda: ("franka",))
    return SimpleNamespace(
        embodiment_adapter=adapter,
        env=SimpleNamespace(unwrapped=base_env),
        runtime_asset_evidence=_runtime_asset_evidence(),
        success_contract_evidence={"attested": True, "predicate": "object_on_destination"},
        step_dt_s=0.02,
    )


def _runtime_asset_evidence():
    return {
        "attested": True,
        "attestation_scope": "runtime_usd_root_layer_identity_only",
        "kinematic_frame_attestation": "separate_live_provider_attestation_required",
        "motion_backend": "curobo_v1",
        "override": "composed_scene.robot.spawn.usd_path_only",
        "profile": "franka_ik_custream_v1_official_root_usd",
        "reason": "pinned_official_isaac_5_1_root_layer_content_identity",
        "referenced_usd_dependencies_attested": False,
        "registry_usd_basename": "franka_panda_hand_on_stand.usd",
        "registry_usd_path": _REGISTRY_PANDA_USD,
        "root_layer": {
            "attestation_method": "https_exact_url_sha256_v1",
            "attested": True,
            "bytes": _PINNED_PANDA_USD_BYTES,
            "content_encoding": "identity",
            "expected_bytes": _PINNED_PANDA_USD_BYTES,
            "expected_sha256": _PINNED_PANDA_USD_SHA256,
            "final_url": _PINNED_PANDA_USD,
            "http_status": 200,
            "max_bytes": 1 << 20,
            "redirects_allowed": False,
            "scope": "root_layer_bytes_only",
            "sha256": _PINNED_PANDA_USD_SHA256,
            "url": _PINNED_PANDA_USD,
        },
        "runtime_uri_policy": "pinned_exact_https_no_redirect",
        "runtime_usd_basename": "panda_instanceable.usd",
        "runtime_usd_path": _PINNED_PANDA_USD,
        "runtime_usd_release": "Isaac 5.1",
        "schedulestream_application": "custream",
        "schema_version": 2,
        "semantic_embodiment": "franka_ik",
    }


def _compatibility():
    identity = SimpleNamespace(source_commit="b" * 40, version="test")
    return SimpleNamespace(
        capabilities=SimpleNamespace(schedulestream=identity),
        motion_backend="curobo_v1",
        schedulestream_application="custream",
    )


@pytest.mark.parametrize(
    "registry_path",
    [
        _REGISTRY_PANDA_USD,
        "omniverse://custom-nucleus/Isaac/IsaacLab/Arena/assets/robot_library/franka_panda_hand_on_stand.usd",
    ],
)
def test_episode_planner_passes_linked_task_pick_id_to_v1_factory(registry_path: str) -> None:
    captured = {}
    native = SimpleNamespace(close=lambda: None)
    bundle = _bundle()
    bundle.runtime_asset_evidence["registry_usd_path"] = registry_path

    def factory(env, predicates, **kwargs):
        captured.update(env=env, predicates=predicates, kwargs=kwargs)
        return native

    planner = create_schedulestream_episode_planner(
        bundle,
        _request(),
        _compatibility(),
        v1_factory=factory,
    )

    assert captured["kwargs"]["graspable_object"] == "pick_cube"
    assert captured["kwargs"]["destination_object"] == "destination_bowl"
    assert captured["kwargs"]["graspable_asset_name"] == "rubiks_cube_hot3d_robolab"
    assert captured["kwargs"]["destination_asset_name"] == "bowl_ycb_robolab"
    assert captured["predicates"][0].subject == "pick_cube"
    assert planner._command_planner is native
    assert planner._runtime_asset_evidence["runtime_usd_basename"] == "panda_instanceable.usd"


def test_episode_planner_rejects_missing_runtime_asset_attestation() -> None:
    bundle = _bundle()
    bundle.runtime_asset_evidence = {}
    factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match="runtime asset schema-v2"):
        create_schedulestream_episode_planner(
            bundle,
            _request(),
            _compatibility(),
            v1_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
        )

    assert factory_calls == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda evidence: evidence.update(referenced_usd_dependencies_attested=True), "official-root"),
        (
            lambda evidence: evidence.update(
                registry_usd_path=_REGISTRY_PANDA_USD.replace(
                    "franka_panda_hand_on_stand.usd", "panda_instanceable.usd"
                )
            ),
            "registry USD path",
        ),
        (lambda evidence: evidence.update(registry_usd_basename="panda_instanceable.usd"), "official-root"),
        (lambda evidence: evidence.update(registry_usd_path=_REGISTRY_PANDA_USD + "?version=1"), "registry USD path"),
        (
            lambda evidence: evidence.update(
                registry_usd_path="x" * 4_097 + "/Arena/assets/robot_library/franka_panda_hand_on_stand.usd"
            ),
            "registry USD path",
        ),
        (lambda evidence: evidence["root_layer"].update(sha256="0" * 64), "root layer"),
        (lambda evidence: evidence["root_layer"].update(unreviewed=True), "root layer"),
    ],
)
def test_episode_planner_rejects_inexact_runtime_asset_schema_v2(mutate, message: str) -> None:
    bundle = _bundle()
    mutate(bundle.runtime_asset_evidence)
    factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match=message):
        create_schedulestream_episode_planner(
            bundle,
            _request(),
            _compatibility(),
            v1_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
        )

    assert factory_calls == []


def test_episode_planner_rejects_task_pick_and_goal_subject_mismatch() -> None:
    factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match="does not exactly match"):
        create_schedulestream_episode_planner(
            _bundle(),
            _request(task_pick="pick_cube", goal_subject="different_object"),
            _compatibility(),
            v1_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
        )

    assert factory_calls == []


def test_episode_planner_rejects_task_destination_and_goal_target_mismatch() -> None:
    factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match="does not exactly match"):
        create_schedulestream_episode_planner(
            _bundle(),
            _request(goal_destination="different_bowl"),
            _compatibility(),
            v1_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
        )

    assert factory_calls == []


def test_episode_planner_rejects_duplicate_linked_node_ids() -> None:
    request = _request()
    request.linked_graph["nodes"].append(dict(request.linked_graph["nodes"][1]))
    factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match="duplicated"):
        create_schedulestream_episode_planner(
            _bundle(),
            request,
            _compatibility(),
            v1_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
        )

    assert factory_calls == []
