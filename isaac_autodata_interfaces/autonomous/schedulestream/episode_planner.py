# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Narrow runtime factory exposing ScheduleStream as an autonomous ``EpisodePlanner``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from isaac_autodata_core.autonomous.attempt_generation import AttemptGenerationError, AttemptRequest, FailureStage
from isaac_autodata_core.autonomous.task_motion import SceneSnapshot, TaskMotionPlan
from isaac_autodata_interfaces.autonomous.schedulestream.command_types import (
    ScheduleStreamClosedError,
    ScheduleStreamLoweringContext,
    ScheduleStreamProviderError,
)
from isaac_autodata_interfaces.autonomous.schedulestream.custream_v1 import (
    V1_REVIEWED_DESTINATION_ASSET,
    V1_REVIEWED_GRASPABLE_ASSET,
    V1IsaacLabCommandPlanner,
    V1IsaacLabPlannerConfig,
    create_v1_isaaclab_command_planner,
)

_V1Factory = Callable[..., V1IsaacLabCommandPlanner]

_PINNED_PANDA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
    "Robots/FrankaEmika/panda_instanceable.usd"
)
_PINNED_PANDA_USD_BYTES = 8_038
_PINNED_PANDA_USD_SHA256 = "7f5a0c0aa6760cfbd348e08bc464d4b94341f027f51c2d9e42406ceefcc7787f"
_MAX_RUNTIME_USD_BYTES = 1 << 20
_REVIEWED_REGISTRY_USD_BASENAME = "franka_panda_hand_on_stand.usd"
_REVIEWED_REGISTRY_USD_SUFFIX = f"/Arena/assets/robot_library/{_REVIEWED_REGISTRY_USD_BASENAME}"
_MAX_REGISTRY_USD_PATH_LENGTH = 4_096


class V1ScheduleStreamEpisodePlanner:
    """IK-executable v1 planner using calibrated upstream ``PathController`` traces."""

    def __init__(
        self,
        command_planner: V1IsaacLabCommandPlanner,
        *,
        eef_name: str,
        link_name: str,
        step_dt_s: float,
        backend_version: str,
        graph_digest: str,
        planner_metadata: dict[str, Any],
        runtime_asset_evidence: Mapping[str, Any],
        success_contract_evidence: Mapping[str, Any],
    ) -> None:
        self._command_planner = command_planner
        self._eef_name = eef_name
        self._link_name = link_name
        self._step_dt_s = step_dt_s
        self._backend_version = backend_version
        self._graph_digest = graph_digest
        self._planner_metadata = planner_metadata
        self._runtime_asset_evidence = dict(runtime_asset_evidence)
        self._success_contract_evidence = dict(success_contract_evidence)
        self._closed = False

    def plan(self, request: AttemptRequest, snapshot: SceneSnapshot) -> TaskMotionPlan:
        """Plan one attempt and return only segments supported by the current IK executor."""

        if self._closed:
            raise ScheduleStreamClosedError("ScheduleStream episode planner is closed")
        context = ScheduleStreamLoweringContext(
            request_digest=request.request_digest,
            snapshot_digest=snapshot.digest,
            seed=request.seed,
            goal=request.goal,
            eef_name=self._eef_name,
            frame="world",
            step_dt_s=self._step_dt_s,
            eef_by_link={self._link_name: self._eef_name},
            backend_version=self._backend_version,
            metadata={
                "attempt_index": request.attempt_index,
                "arena_success_contract": self._success_contract_evidence,
                "graph_digest": self._graph_digest,
                "planner": self._planner_metadata,
                "runtime_asset": self._runtime_asset_evidence,
            },
        )
        result = self._command_planner.plan_task_motion_plan(
            context,
            request.env_id,
            link_name=self._link_name,
        )
        if result is None:
            diagnostics = self._command_planner.planning_diagnostics
            diagnostic_suffix = ""
            if diagnostics:
                diagnostic_suffix = f"; diagnostics={json.dumps(diagnostics, sort_keys=True, separators=(',', ':'))}"
            raise AttemptGenerationError(
                FailureStage.PLANNING,
                "schedulestream_no_plan",
                f"ScheduleStream returned no plan within the configured limits{diagnostic_suffix}",
            )
        return result

    def close(self) -> None:
        """Close the owned native planner idempotently."""

        if self._closed:
            return
        self._closed = True
        self._command_planner.close()


def create_schedulestream_episode_planner(
    bundle: Any,
    resolved_request: Any,
    compatibility: Any,
    *,
    attachment_state: Any | None = None,
    v1_factory: _V1Factory | None = None,
) -> V1ScheduleStreamEpisodePlanner:
    """Create the selected runtime planner after Isaac/Arena application launch.

    Args:
        bundle: Live ``ArenaRuntimeBundle`` with environment and embodiment adapter.
        resolved_request: Validated ``CompiledTaskRequest``.
        compatibility: Result from ``select_schedulestream_backend``.
        attachment_state: Shared executor state accepted for the stable integration seam. Dense v1
            lowering currently relies on gripper commands and final physical verification instead.
        v1_factory: Pure-test injection hook for the concrete v1 planner factory.
    """

    del attachment_state
    application = getattr(compatibility, "schedulestream_application", None)
    motion_backend = getattr(compatibility, "motion_backend", None)
    if application != "custream" or motion_backend != "curobo_v1":
        if application == "custream2" and motion_backend == "curobo_v2":
            raise ScheduleStreamProviderError(
                "custream2 command lowering is available, but no reviewed live v2 IsaacLab world/planner "
                "factory is wired into the current image"
            )
        raise ScheduleStreamProviderError(
            f"unsupported selected ScheduleStream runtime application={application!r}, "
            f"motion_backend={motion_backend!r}"
        )
    if int(getattr(resolved_request.generation, "num_envs", 0)) != 1:
        raise ScheduleStreamProviderError("the concrete ScheduleStream episode planner requires num_envs=1")
    base_env = getattr(bundle.env, "unwrapped", bundle.env)
    adapter = bundle.embodiment_adapter
    eef_names = tuple(adapter.get_eef_names())
    if len(eef_names) != 1:
        raise ScheduleStreamProviderError(f"v1 single-arm episode planner requires exactly one EEF, got {eef_names}")
    eef_name = eef_names[0]
    link_name = _single_action_body_name(base_env)
    step_dt_s = _positive_finite(bundle.step_dt_s, "bundle.step_dt_s")
    physics_dt_s = _positive_finite(base_env.sim.get_physics_dt(), "IsaacLab physics dt")
    scale_dt = step_dt_s / physics_dt_s
    requested_dt_s = _positive_finite(
        resolved_request.planner.interpolation_dt_s,
        "planner.interpolation_dt_s",
    )
    if not math.isclose(requested_dt_s, step_dt_s, rel_tol=1e-5, abs_tol=1e-6):
        raise ScheduleStreamProviderError(
            f"planner dt {requested_dt_s:g}s does not match live environment dt {step_dt_s:g}s"
        )
    config = V1IsaacLabPlannerConfig(
        batch_size=resolved_request.planner.batch_size,
        scale_dt=scale_dt,
        collisions=resolved_request.planner.collisions,
        max_time_s=resolved_request.planner.max_time_s,
        profile=resolved_request.planner.profile,
        animate=resolved_request.planner.animate,
    )

    from isaac_autodata_interfaces.autonomous.arena_environment import goal_predicates_from_request

    predicates = goal_predicates_from_request(resolved_request)
    graspable_object, destination_object, graspable_asset_name, destination_asset_name = _task_pick_place_binding(
        resolved_request,
        predicates,
    )
    success_contract_evidence = getattr(bundle, "success_contract_evidence", None)
    if not isinstance(success_contract_evidence, Mapping) or success_contract_evidence.get("attested") is not True:
        raise ScheduleStreamProviderError("live Arena success contract was not attested before planner construction")
    runtime_asset_evidence = getattr(bundle, "runtime_asset_evidence", None)
    _require_v1_runtime_asset_evidence(runtime_asset_evidence)
    v1_factory = v1_factory or create_v1_isaaclab_command_planner
    command_planner = v1_factory(
        base_env,
        predicates,
        graspable_object=graspable_object,
        destination_object=destination_object,
        graspable_asset_name=graspable_asset_name,
        destination_asset_name=destination_asset_name,
        config=config,
        eef_pose_reader=_world_eef_pose_reader(base_env, adapter),
    )
    return V1ScheduleStreamEpisodePlanner(
        command_planner,
        eef_name=eef_name,
        link_name=link_name,
        step_dt_s=step_dt_s,
        backend_version=_backend_version(compatibility),
        graph_digest=resolved_request.graph_digest,
        planner_metadata=resolved_request.planner.to_dict(),
        runtime_asset_evidence=runtime_asset_evidence,
        success_contract_evidence=success_contract_evidence,
    )


def _task_pick_place_binding(resolved_request: Any, predicates: tuple[Any, ...]) -> tuple[str, str, str, str]:
    """Attest the linked cube-to-bowl task endpoints and exact semantic goal."""

    linked_graph = getattr(resolved_request, "linked_graph", None)
    tasks = linked_graph.get("tasks") if isinstance(linked_graph, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ScheduleStreamProviderError("resolved graph must contain exactly one linked task")
    params = tasks[0].get("params")
    pick_up_object = params.get("pick_up_object") if isinstance(params, dict) else None
    destination_object = params.get("destination_location") if isinstance(params, dict) else None
    if not _bounded_graph_id(pick_up_object):
        raise ScheduleStreamProviderError("linked task has no valid pick_up_object scene ID")
    if not _bounded_graph_id(destination_object):
        raise ScheduleStreamProviderError("linked task has no valid destination_location scene ID")
    if pick_up_object == destination_object:
        raise ScheduleStreamProviderError("linked task pickup and destination IDs must be distinct")
    observed_goals = tuple(
        (getattr(predicate, "relation", None), getattr(predicate, "subject", None), getattr(predicate, "target", None))
        for predicate in predicates
    )
    if observed_goals != (("on", pick_up_object, destination_object),):
        raise ScheduleStreamProviderError("linked task pickup/destination does not exactly match one semantic on goal")

    nodes = linked_graph.get("nodes")
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, dict) for node in nodes):
        raise ScheduleStreamProviderError("resolved graph nodes must be a non-empty plain mapping list")
    node_ids = [node.get("id") for node in nodes]
    if any(not _bounded_graph_id(node_id) for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ScheduleStreamProviderError("resolved graph node IDs are invalid or duplicated")
    nodes_by_id = {node["id"]: node for node in nodes}
    expected = (
        (pick_up_object, V1_REVIEWED_GRASPABLE_ASSET, "pickup"),
        (destination_object, V1_REVIEWED_DESTINATION_ASSET, "destination"),
    )
    for object_id, asset_name, role in expected:
        node = nodes_by_id.get(object_id)
        if (
            not isinstance(node, dict)
            or node.get("type") != "object"
            or node.get("name") != asset_name
            or node.get("params") != {}
        ):
            raise ScheduleStreamProviderError(
                f"linked {role} {object_id!r} is not the unmodified reviewed asset {asset_name!r}"
            )
    return pick_up_object, destination_object, V1_REVIEWED_GRASPABLE_ASSET, V1_REVIEWED_DESTINATION_ASSET


def _bounded_graph_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 512 and "\x00" not in value


def _require_v1_runtime_asset_evidence(evidence: Any) -> None:
    """Require the complete schema-v2 root-layer identity attestation without widening its scope."""

    expected_root_layer = {
        "attestation_method": "https_exact_url_sha256_v1",
        "attested": True,
        "bytes": _PINNED_PANDA_USD_BYTES,
        "content_encoding": "identity",
        "expected_bytes": _PINNED_PANDA_USD_BYTES,
        "expected_sha256": _PINNED_PANDA_USD_SHA256,
        "final_url": _PINNED_PANDA_USD,
        "http_status": 200,
        "max_bytes": _MAX_RUNTIME_USD_BYTES,
        "redirects_allowed": False,
        "scope": "root_layer_bytes_only",
        "sha256": _PINNED_PANDA_USD_SHA256,
        "url": _PINNED_PANDA_USD,
    }
    expected_top_level = {
        "attested": True,
        "attestation_scope": "runtime_usd_root_layer_identity_only",
        "kinematic_frame_attestation": "separate_live_provider_attestation_required",
        "motion_backend": "curobo_v1",
        "override": "composed_scene.robot.spawn.usd_path_only",
        "profile": "franka_ik_custream_v1_official_root_usd",
        "reason": "pinned_official_isaac_5_1_root_layer_content_identity",
        "referenced_usd_dependencies_attested": False,
        "registry_usd_basename": _REVIEWED_REGISTRY_USD_BASENAME,
        "runtime_usd_basename": "panda_instanceable.usd",
        "runtime_usd_path": _PINNED_PANDA_USD,
        "runtime_usd_release": "Isaac 5.1",
        "runtime_uri_policy": "pinned_exact_https_no_redirect",
        "schedulestream_application": "custream",
        "schema_version": 2,
        "semantic_embodiment": "franka_ik",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != set(expected_top_level) | {
        "registry_usd_path",
        "root_layer",
    }:
        raise ScheduleStreamProviderError("live custream v1 runtime asset schema-v2 keys were not exactly attested")
    if any(evidence.get(key) != value for key, value in expected_top_level.items()):
        raise ScheduleStreamProviderError("live custream v1 official-root runtime profile was not exactly attested")
    registry_path = evidence.get("registry_usd_path")
    if (
        not isinstance(registry_path, str)
        or not registry_path
        or registry_path != registry_path.strip()
        or len(registry_path) > _MAX_REGISTRY_USD_PATH_LENGTH
        or "\x00" in registry_path
        or not registry_path.endswith(_REVIEWED_REGISTRY_USD_SUFFIX)
        or registry_path.rsplit("/", 1)[-1] != evidence["registry_usd_basename"]
    ):
        raise ScheduleStreamProviderError(
            "live Arena registry USD path is not a bounded path to the reviewed robot asset"
        )
    root_layer = evidence.get("root_layer")
    if not isinstance(root_layer, Mapping) or dict(root_layer) != expected_root_layer:
        raise ScheduleStreamProviderError("live custream v1 runtime USD root layer was not exactly attested")


def _single_action_body_name(env: Any) -> str:
    action_manager = getattr(env, "action_manager", None)
    active_terms = () if action_manager is None else getattr(action_manager, "active_terms", ())
    body_names = []
    for term_name in active_terms:
        try:
            term = action_manager.get_term(term_name)
            body_name = getattr(term.cfg, "body_name", None)
        except Exception as exc:
            raise ScheduleStreamProviderError(f"failed to inspect IsaacLab action term {term_name!r}") from exc
        if isinstance(body_name, str) and body_name:
            body_names.append(body_name)
    body_names = list(dict.fromkeys(body_names))
    if len(body_names) != 1:
        raise ScheduleStreamProviderError(f"expected exactly one Cartesian action body, got {body_names}")
    return body_names[0]


def _world_eef_pose_reader(env: Any, adapter: Any) -> Callable[[int, str], Any]:
    def read(env_id: int, eef_name: str) -> Any:
        poses = adapter.get_eef_poses(env_ids=[env_id])
        if eef_name not in poses:
            raise ScheduleStreamProviderError(f"embodiment adapter has no observed EEF {eef_name!r}")
        pose = poses[eef_name][0]
        clone = getattr(pose, "clone", None)
        copy = getattr(pose, "copy", None)
        result = clone() if callable(clone) else copy() if callable(copy) else pose
        scene = getattr(env, "scene", None)
        origins = None if scene is None else getattr(scene, "env_origins", None)
        if origins is not None:
            try:
                result[:3, 3] += origins[env_id]
            except Exception as exc:
                raise ScheduleStreamProviderError(
                    "failed to convert observed EEF pose from environment origin to world frame"
                ) from exc
        return result

    return read


def _backend_version(compatibility: Any) -> str:
    capabilities = getattr(compatibility, "capabilities", None)
    identity = getattr(capabilities, "schedulestream", None)
    version = getattr(identity, "version", None)
    commit = getattr(identity, "source_commit", None)
    if isinstance(version, str) and version:
        if isinstance(commit, str) and commit:
            return f"{version}+{commit[:12]}"
        return version
    return "unknown"


def _positive_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ScheduleStreamProviderError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleStreamProviderError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ScheduleStreamProviderError(f"{field_name} must be positive and finite")
    return result
