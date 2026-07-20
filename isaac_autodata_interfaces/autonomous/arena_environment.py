# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Materialize a resolved Arena intent as a recordable autonomous environment."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaac_autodata_core.autonomous.output_transaction import RecordingTargets
from isaac_autodata_core.autonomous.task_motion import GoalPredicate
from isaac_autodata_interfaces.autonomous.task_request_types import CompiledTaskRequest

_FRANKA_IK_COMMAND_TO_OBS_OFFSET_M = (0.0, 0.0, -0.0036)
_CUSTREAM_V1_RUNTIME_ASSET_PROFILE = "franka_ik_custream_v1_official_root_usd"
_CUSTREAM_V1_REGISTRY_USD_BASENAME = "franka_panda_hand_on_stand.usd"
_CUSTREAM_V1_RUNTIME_USD_BASENAME = "panda_instanceable.usd"
_CUSTREAM_V1_RUNTIME_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
    "Robots/FrankaEmika/panda_instanceable.usd"
)
_CUSTREAM_V1_RUNTIME_USD_BYTES = 8_038
_CUSTREAM_V1_RUNTIME_USD_SHA256 = "7f5a0c0aa6760cfbd348e08bc464d4b94341f027f51c2d9e42406ceefcc7787f"
_RUNTIME_USD_MAX_BYTES = 1 << 20
_RUNTIME_USD_READ_CHUNK_BYTES = 64 << 10
_RUNTIME_USD_TIMEOUT_S = 10.0

RuntimeUsdAttestor = Callable[[str], Mapping[str, Any]]


@dataclass
class ArenaRuntimeBundle:
    """Live resources created from a compiled task request after SimulationApp launch."""

    env: Any
    embodiment_adapter: Any
    success_term: Any
    graph_spec: Any
    step_dt_s: float
    success_contract_evidence: dict[str, Any] = field(default_factory=dict)
    runtime_asset_evidence: dict[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close the live environment."""

        if self._closed:
            return
        self._closed = True
        self.env.close()


def goal_predicates_from_request(request: CompiledTaskRequest) -> tuple[GoalPredicate, ...]:
    """Project Arena's linked task-stage constraints into planner-neutral goal predicates."""

    predicates: list[GoalPredicate] = []
    for stage in request.goal_stages:
        for constraint in stage.spatial_constraints:
            predicates.append(
                GoalPredicate(
                    relation=constraint.kind,
                    subject=constraint.subject,
                    target=constraint.reference,
                )
            )
    if not predicates:
        raise ValueError("resolved Arena task graph contains no spatial success constraints")
    return tuple(predicates)


def make_embodiment_adapter(request: CompiledTaskRequest) -> Any:
    """Create the reviewed AutoData controller binding for the resolved Arena embodiment.

    Arena remains authoritative for semantic asset selection. This binding only describes the
    live action/observation layout needed to execute planner poses. The prototype intentionally
    supports the registered ``franka_ik`` profile and fails explicitly for all others.
    """

    nodes = request.linked_graph.get("nodes", [])
    embodiment_nodes = [node for node in nodes if node.get("type") == "embodiment"]
    if len(embodiment_nodes) != 1:
        raise ValueError(f"autonomous generation requires exactly one embodiment node, got {len(embodiment_nodes)}")
    embodiment_name = embodiment_nodes[0].get("name")
    if embodiment_name != "franka_ik":
        raise NotImplementedError(
            f"autonomous runtime support for Arena embodiment {embodiment_name!r} is not available; "
            "the prototype currently supports 'franka_ik'"
        )

    from isaac_autodata_interfaces.embodiments.embodiment_types import PoseObsKeys
    from isaac_autodata_interfaces.embodiments.single_arm_embodiment_adapter import DeltaPoseIKSingleArmAdapter

    return DeltaPoseIKSingleArmAdapter(
        name="arena_franka_ik",
        description="Arena Franka IK relative-pose runtime binding",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
        obs_group="policy",
        # Arena observes panda_hand + 0.1034 m while its DIK command frame is panda_hand +
        # 0.107 m. ``eef_offset`` is command-to-observation, so -3.6 mm makes the adapter
        # report the exact command frame. The live provider independently attests this against
        # the instantiated action term and fails closed if Arena changes either frame.
        eef_offset=_FRANKA_IK_COMMAND_TO_OBS_OFFSET_M,
    )


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable urllib's default redirect following for pinned asset requests."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl


def _open_runtime_usd_request(request: urllib.request.Request, timeout_s: float) -> Any:
    """Open one HTTPS request with redirects disabled and platform TLS verification enabled."""

    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout_s)


def _validate_content_identity_inputs(
    expected_sha256: str,
    expected_bytes: int,
    max_bytes: int,
    timeout_s: float,
) -> None:
    """Validate bounded exact-content attestation inputs."""

    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or expected_bytes > max_bytes
    ):
        raise ValueError("runtime USD expected byte count must be positive and within the download bound")
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or max_bytes > _RUNTIME_USD_MAX_BYTES
    ):
        raise ValueError(f"runtime USD download bound must be in [1, {_RUNTIME_USD_MAX_BYTES}]")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("runtime USD expected SHA-256 must contain 64 hexadecimal characters")
    try:
        bytes.fromhex(expected_sha256)
    except ValueError as exc:
        raise ValueError("runtime USD expected SHA-256 must contain 64 hexadecimal characters") from exc
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
        raise ValueError("runtime USD timeout must be finite and in (0, 30] seconds")


def _request_exact_url_content(
    url: str,
    timeout_s: float,
    opener: Callable[[urllib.request.Request, float], Any] | None,
) -> Any:
    """Issue one identity-encoded request and normalize transport failures."""

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": "Isaac-AutoData-root-layer-attestor/1",
        },
        method="GET",
    )
    open_request = _open_runtime_usd_request if opener is None else opener
    try:
        return open_request(request, timeout_s)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise ValueError(f"runtime USD redirects are forbidden (HTTP {exc.code})") from exc
        raise ValueError(f"runtime USD root layer request failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"runtime USD root layer request failed: {type(exc).__name__}: {str(exc)[:256]}") from exc
    except Exception as exc:
        raise ValueError(f"runtime USD root layer request failed: {type(exc).__name__}: {str(exc)[:256]}") from exc


def _validate_content_response_headers(headers: Any, max_bytes: int) -> None:
    """Reject encoded, malformed, or declared-oversize content."""

    if headers is None or not callable(getattr(headers, "get", None)):
        raise ValueError("runtime USD root layer response has no inspectable headers")
    content_encoding = headers.get("Content-Encoding")
    if content_encoding is not None and str(content_encoding).strip().lower() not in ("", "identity"):
        raise ValueError("runtime USD root layer response must use identity content encoding")
    content_length = headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_bytes = int(content_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime USD root layer Content-Length is invalid") from exc
    if declared_bytes < 0:
        raise ValueError("runtime USD root layer Content-Length is invalid")
    if declared_bytes > max_bytes:
        raise ValueError(f"runtime USD root layer exceeds the {max_bytes}-byte download bound")


def _validate_content_response_identity(response: Any, expected_url: str, max_bytes: int) -> str:
    """Attest response status, final URL, encoding, and declared size."""

    status = getattr(response, "status", None)
    if status is None and callable(getattr(response, "getcode", None)):
        status = response.getcode()
    if type(status) is not int or status != 200:
        raise ValueError(f"runtime USD root layer request returned non-success status {status!r}")
    final_url = response.geturl() if callable(getattr(response, "geturl", None)) else None
    if final_url != expected_url:
        raise ValueError("runtime USD redirects or final-URL changes are forbidden")
    _validate_content_response_headers(getattr(response, "headers", None), max_bytes)
    return final_url


def _read_bounded_content(response: Any, max_bytes: int) -> bytes:
    """Read at most one byte beyond the configured ceiling to detect streamed overflow."""

    payload = bytearray()
    while True:
        chunk = response.read(min(_RUNTIME_USD_READ_CHUNK_BYTES, max_bytes + 1 - len(payload)))
        if not chunk:
            return bytes(payload)
        if not isinstance(chunk, (bytes, bytearray)):
            raise ValueError("runtime USD root layer response returned non-byte content")
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise ValueError(f"runtime USD root layer exceeds the {max_bytes}-byte download bound")


def _consume_exact_url_response(response: Any, expected_url: str, max_bytes: int) -> tuple[bytes, str]:
    """Validate and consume one response while normalizing bounded read failures."""

    try:
        with response as opened_response:
            final_url = _validate_content_response_identity(opened_response, expected_url, max_bytes)
            return _read_bounded_content(opened_response, max_bytes), final_url
    except ValueError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"runtime USD root layer read failed: {type(exc).__name__}: {str(exc)[:256]}") from exc
    except Exception as exc:
        raise ValueError(f"runtime USD root layer read failed: {type(exc).__name__}: {str(exc)[:256]}") from exc


def _attest_exact_url_content(
    url: str,
    *,
    expected_url: str,
    expected_sha256: str,
    expected_bytes: int,
    max_bytes: int,
    timeout_s: float,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> dict[str, Any]:
    """Fetch and hash one exact HTTPS object under strict redirect and byte bounds."""

    if url != expected_url or not expected_url.startswith("https://"):
        raise ValueError("runtime USD root layer URL must equal the pinned HTTPS production URL")
    _validate_content_identity_inputs(expected_sha256, expected_bytes, max_bytes, timeout_s)
    response = _request_exact_url_content(url, timeout_s, opener)
    payload, final_url = _consume_exact_url_response(response, expected_url, max_bytes)

    actual_bytes = len(payload)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"runtime USD root layer byte count mismatch: expected {expected_bytes}, observed {actual_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise ValueError("runtime USD root layer SHA-256 does not match the reviewed Isaac 5.1 object")
    return {
        "attestation_method": "https_exact_url_sha256_v1",
        "attested": True,
        "bytes": actual_bytes,
        "content_encoding": "identity",
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "final_url": final_url,
        "http_status": 200,
        "max_bytes": max_bytes,
        "redirects_allowed": False,
        "scope": "root_layer_bytes_only",
        "sha256": actual_sha256,
        "url": url,
    }


def attest_custream_v1_runtime_usd_root(
    runtime_usd_path: str,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> dict[str, Any]:
    """Attest the exact official Isaac 5.1 Panda root-layer bytes.

    This attestation covers only the bytes returned for ``panda_instanceable.usd``. Referenced USD
    dependencies are not fetched or attested here. Robot kinematics and command/observation frame
    compatibility remain a separate fail-closed live-provider attestation.

    Args:
        runtime_usd_path: Exact pinned official-production HTTPS URL.
        opener: Optional injected request opener used by bounded offline unit tests.

    Returns:
        JSON-compatible root-layer content identity evidence.

    Raises:
        ValueError: If the URL, HTTP response, byte bound, size, or digest does not match.
    """

    return _attest_exact_url_content(
        runtime_usd_path,
        expected_url=_CUSTREAM_V1_RUNTIME_USD_PATH,
        expected_sha256=_CUSTREAM_V1_RUNTIME_USD_SHA256,
        expected_bytes=_CUSTREAM_V1_RUNTIME_USD_BYTES,
        max_bytes=_RUNTIME_USD_MAX_BYTES,
        timeout_s=_RUNTIME_USD_TIMEOUT_S,
        opener=opener,
    )


def _validated_runtime_usd_root_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and bound evidence returned by an injected or production root-layer attestor."""

    expected = {
        "attestation_method": "https_exact_url_sha256_v1",
        "attested": True,
        "bytes": _CUSTREAM_V1_RUNTIME_USD_BYTES,
        "content_encoding": "identity",
        "expected_bytes": _CUSTREAM_V1_RUNTIME_USD_BYTES,
        "expected_sha256": _CUSTREAM_V1_RUNTIME_USD_SHA256,
        "final_url": _CUSTREAM_V1_RUNTIME_USD_PATH,
        "http_status": 200,
        "max_bytes": _RUNTIME_USD_MAX_BYTES,
        "redirects_allowed": False,
        "scope": "root_layer_bytes_only",
        "sha256": _CUSTREAM_V1_RUNTIME_USD_SHA256,
        "url": _CUSTREAM_V1_RUNTIME_USD_PATH,
    }
    if not isinstance(value, Mapping) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("runtime USD root-layer attestor returned incomplete or mismatched identity evidence")
    return expected


def apply_custream_v1_runtime_asset_profile(
    request: CompiledTaskRequest,
    scene_cfg: Any,
    *,
    runtime_usd_path: str,
    root_layer_attestor: RuntimeUsdAttestor | None = None,
) -> dict[str, Any]:
    """Select the reviewed runtime USD and attest only its official root-layer identity.

    Arena's registered ``franka_ik`` embodiment currently uses a combined robot-and-stand USD.
    This profile changes only the composed articulation's ``spawn.usd_path`` to the pinned official
    Isaac 5.1 Panda root layer. The content attestation does not cover referenced USD dependencies
    and makes no claim of URDF/USD kinematic equivalence. Live kinematic and action-frame agreement
    must be attested independently by the motion-provider boundary.

    Args:
        request: Fully linked semantic request admitted by the live v1 runtime-support profile.
        scene_cfg: Composed Arena scene configuration, before environment construction.
        runtime_usd_path: Exact official-production Panda USD selected by this reviewed profile.
        root_layer_attestor: Optional injected exact-content attestor for bounded unit tests.

    Returns:
        JSON-compatible evidence describing the exact reviewed runtime substitution.

    Raises:
        ValueError: If the semantic embodiment, asset path, or root-layer identity has drifted.
    """

    linked_graph = getattr(request, "linked_graph", None)
    nodes = linked_graph.get("nodes") if isinstance(linked_graph, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("custream v1 runtime asset profile requires a linked Arena node list")
    embodiment_names = [
        node.get("name") for node in nodes if isinstance(node, dict) and node.get("type") == "embodiment"
    ]
    if embodiment_names != ["franka_ik"]:
        raise ValueError(
            "custream v1 runtime asset profile requires exactly the Arena 'franka_ik' embodiment; "
            f"got {embodiment_names}"
        )

    robot_cfg = getattr(scene_cfg, "robot", None)
    spawn_cfg = None if robot_cfg is None else getattr(robot_cfg, "spawn", None)
    registry_usd_path = None if spawn_cfg is None else getattr(spawn_cfg, "usd_path", None)
    if not isinstance(registry_usd_path, str) or not registry_usd_path:
        raise ValueError("composed Arena franka_ik scene has no robot spawn USD path")
    registry_usd_basename = os.path.basename(registry_usd_path)
    if registry_usd_basename != _CUSTREAM_V1_REGISTRY_USD_BASENAME:
        raise ValueError(
            "Arena franka_ik registry asset changed from the reviewed custream v1 source "
            f"{_CUSTREAM_V1_REGISTRY_USD_BASENAME!r} to {registry_usd_basename!r}"
        )
    if not isinstance(runtime_usd_path, str) or not runtime_usd_path:
        raise ValueError("live FRANKA_PANDA_HIGH_PD_CFG has no runtime USD path")
    runtime_usd_basename = os.path.basename(runtime_usd_path)
    if runtime_usd_basename != _CUSTREAM_V1_RUNTIME_USD_BASENAME:
        raise ValueError(
            "IsaacLab Franka runtime asset changed from the reviewed custream v1 target "
            f"{_CUSTREAM_V1_RUNTIME_USD_BASENAME!r} to {runtime_usd_basename!r}"
        )
    if runtime_usd_path != _CUSTREAM_V1_RUNTIME_USD_PATH:
        raise ValueError("custream v1 runtime asset must use the reviewed pinned official production URI")
    if registry_usd_path == runtime_usd_path:
        raise ValueError("custream v1 runtime asset substitution unexpectedly resolves to the registry asset")

    attest_root_layer = attest_custream_v1_runtime_usd_root if root_layer_attestor is None else root_layer_attestor
    try:
        raw_root_layer_evidence = attest_root_layer(runtime_usd_path)
        root_layer_evidence = _validated_runtime_usd_root_evidence(raw_root_layer_evidence)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise ValueError(f"custream v1 runtime USD root-layer attestation failed: {exc}") from exc
        raise ValueError(
            f"custream v1 runtime USD root-layer attestation failed: {type(exc).__name__}: {str(exc)[:256]}"
        ) from exc

    spawn_cfg.usd_path = runtime_usd_path
    return {
        "attested": True,
        "attestation_scope": "runtime_usd_root_layer_identity_only",
        "kinematic_frame_attestation": "separate_live_provider_attestation_required",
        "motion_backend": "curobo_v1",
        "override": "composed_scene.robot.spawn.usd_path_only",
        "profile": _CUSTREAM_V1_RUNTIME_ASSET_PROFILE,
        "reason": "pinned_official_isaac_5_1_root_layer_content_identity",
        "referenced_usd_dependencies_attested": False,
        "registry_usd_basename": registry_usd_basename,
        "registry_usd_path": registry_usd_path,
        "root_layer": root_layer_evidence,
        "runtime_usd_basename": runtime_usd_basename,
        "runtime_usd_path": runtime_usd_path,
        "runtime_usd_release": "Isaac 5.1",
        "runtime_uri_policy": "pinned_exact_https_no_redirect",
        "schedulestream_application": "custream",
        "schema_version": 2,
        "semantic_embodiment": "franka_ik",
    }


def validate_planner_timing(requested_dt_s: float, environment_dt_s: float, *, tolerance_s: float = 1e-6) -> None:
    """Reject planner/environment sample-rate drift until an explicit resampler is selected."""

    for name, value in (("requested_dt_s", requested_dt_s), ("environment_dt_s", environment_dt_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isclose(requested_dt_s, environment_dt_s, rel_tol=1e-5, abs_tol=tolerance_s):
        raise ValueError(
            "planner interpolation_dt_s must match the Isaac environment step_dt until a reviewed "
            f"resampler is enabled (planner={requested_dt_s:.9g}s, environment={environment_dt_s:.9g}s)"
        )


def attest_pick_and_place_success_contract(
    success_term: Any,
    scene_cfg: Any,
    *,
    pick_up_object: str,
    destination_location: str,
) -> dict[str, Any]:
    """Fail closed unless Arena exposes the exact reviewed contact-and-velocity success term."""

    success_func = getattr(success_term, "func", None)
    success_params = getattr(success_term, "params", None)
    if (
        not callable(success_func)
        or getattr(success_func, "__name__", None) != "check_success"
        or getattr(success_func, "__module__", None) != "isaaclab_arena.tasks.terminations"
        or not isinstance(success_params, dict)
    ):
        raise ValueError("Arena success term must be the reviewed check_success composition")
    if set(success_params) != {"mode", "predicates"}:
        raise ValueError("Arena check_success parameters changed from the reviewed contract")
    mode = getattr(success_params["mode"], "value", success_params["mode"])
    predicates = success_params["predicates"]
    if mode != "ALL" or not isinstance(predicates, list) or len(predicates) != 1:
        raise ValueError("Arena success must combine exactly one predicate in ALL mode")
    predicate = predicates[0]
    predicate_func = getattr(predicate, "func", None)
    predicate_params = getattr(predicate, "params", None)
    if (
        not callable(predicate_func)
        or getattr(predicate_func, "__name__", None) != "object_on_destination"
        or getattr(predicate_func, "__module__", None) != "isaaclab_arena.tasks.terminations"
        or not isinstance(predicate_params, dict)
    ):
        raise ValueError("Arena success predicate must be object_on_destination")
    expected_keys = {"contact_sensor_cfg", "force_threshold", "object_cfg", "velocity_threshold"}
    if set(predicate_params) != expected_keys:
        raise ValueError("Arena object_on_destination parameters changed from the reviewed contract")
    object_name = getattr(predicate_params["object_cfg"], "name", None)
    contact_sensor_name = getattr(predicate_params["contact_sensor_cfg"], "name", None)
    if object_name != pick_up_object or contact_sensor_name != "pick_up_object_contact_sensor":
        raise ValueError("Arena success predicate does not bind the task-selected pickup object and contact sensor")
    force_threshold = predicate_params["force_threshold"]
    velocity_threshold = predicate_params["velocity_threshold"]
    if type(force_threshold) not in (int, float) or not math.isclose(float(force_threshold), 0.1, abs_tol=1e-12):
        raise ValueError("Arena success force threshold must remain exactly 0.1 N")
    if type(velocity_threshold) not in (int, float) or not math.isclose(float(velocity_threshold), 0.1, abs_tol=1e-12):
        raise ValueError("Arena success velocity threshold must remain exactly 0.1 m/s")

    sensor_cfg = getattr(scene_cfg, "pick_up_object_contact_sensor", None)
    sensor_prim_path = getattr(sensor_cfg, "prim_path", None)
    filter_paths = getattr(sensor_cfg, "filter_prim_paths_expr", None)
    expected_sensor_prim_path = f"{{ENV_REGEX_NS}}/{pick_up_object}"
    expected_filter_prim_path = f"{{ENV_REGEX_NS}}/{destination_location}"
    if sensor_prim_path != expected_sensor_prim_path:
        raise ValueError("Arena pickup contact sensor prim path does not bind the task-selected object")
    if not isinstance(filter_paths, list) or filter_paths != [expected_filter_prim_path]:
        raise ValueError("Arena pickup contact sensor does not filter exactly against the destination object")
    return {
        "attested": True,
        "contact_sensor": contact_sensor_name,
        "contact_sensor_filter": filter_paths[0],
        "force_threshold_n": float(force_threshold),
        "mode": mode,
        "predicate": "object_on_destination",
        "subject": object_name,
        "velocity_threshold_m_s": float(velocity_threshold),
    }


def build_arena_runtime(
    request: CompiledTaskRequest,
    args_cli: Any,
    *,
    recording_targets: RecordingTargets | None = None,
    allow_output_overwrite: bool = False,
) -> ArenaRuntimeBundle:
    """Build Arena graph, configure source-free HDF5 recording, and create the live environment.

    This function must be called only after Isaac SimulationApp has launched. All heavy Isaac/Arena
    imports are intentionally local.
    """

    if request.generation.num_envs != 1:
        raise NotImplementedError("ScheduleStream autonomous generation currently requires generation.num_envs: 1")
    dataset_path = Path(request.output.dataset)
    if dataset_path.suffix.lower() != ".hdf5":
        raise ValueError(f"output dataset must end in .hdf5: {dataset_path}")
    if recording_targets is None:
        dataset_targets = [dataset_path]
        if request.output.keep_failed:
            dataset_targets.append(dataset_path.with_name(f"{dataset_path.stem}_failed{dataset_path.suffix}"))
        if not allow_output_overwrite:
            occupied = [path for path in dataset_targets if path.exists() or path.is_symlink()]
            if occupied:
                targets = ", ".join(str(path) for path in occupied)
                raise FileExistsError(f"refusing to overwrite existing output dataset target(s): {targets}")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_export_dir_path = str(dataset_path.parent)
        dataset_filename = dataset_path.stem
    else:
        if allow_output_overwrite:
            raise ValueError("recording_targets cannot be combined with allow_output_overwrite")
        if recording_targets.dataset_filename != dataset_path.stem:
            raise ValueError("recording target filename must preserve the resolved dataset stem")
        descriptor_prefix = "/proc/self/fd/"
        if not recording_targets.dataset_export_dir_path.startswith(descriptor_prefix):
            raise ValueError("recording target directory must be backed by a process file descriptor")
        descriptor_text = recording_targets.dataset_export_dir_path[len(descriptor_prefix) :]
        if not descriptor_text.isdigit():
            raise ValueError("recording target directory has an invalid file descriptor")
        descriptor_stat = os.fstat(int(descriptor_text))
        if not stat.S_ISDIR(descriptor_stat.st_mode):
            raise ValueError("recording target descriptor must refer to a directory")
        dataset_export_dir_path = recording_targets.dataset_export_dir_path
        dataset_filename = recording_targets.dataset_filename

    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    args_cli.num_envs = request.generation.num_envs
    args_cli.seed = request.generation.seed
    args_cli.placement_seed = request.generation.seed
    args_cli.mimic = False

    graph_spec = ArenaEnvGraphSpec.from_dict(request.linked_graph)
    arena_env = graph_spec.to_arena_env(enable_cameras=bool(getattr(args_cli, "enable_cameras", False)))
    builder = ArenaEnvBuilder(arena_env, args_cli)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    runtime_asset_evidence = apply_custream_v1_runtime_asset_profile(
        request,
        env_cfg.scene,
        runtime_usd_path=_CUSTREAM_V1_RUNTIME_USD_PATH,
    )

    terminations = getattr(env_cfg, "terminations", None)
    success_term = None if terminations is None else getattr(terminations, "success", None)
    if success_term is None:
        raise ValueError("resolved Arena environment does not expose a 'success' termination term")
    tasks = request.linked_graph.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError("resolved Arena graph must contain exactly one task for success-term attestation")
    task_params = tasks[0].get("params")
    if not isinstance(task_params, dict):
        raise ValueError("resolved Arena task has no linked parameters for success-term attestation")
    pick_up_object = task_params.get("pick_up_object")
    destination_location = task_params.get("destination_location")
    if not isinstance(pick_up_object, str) or not isinstance(destination_location, str):
        raise ValueError("resolved Arena task has invalid pickup/destination bindings")
    success_contract_evidence = attest_pick_and_place_success_contract(
        success_term,
        env_cfg.scene,
        pick_up_object=pick_up_object,
        destination_location=destination_location,
    )
    # AttemptGenerator owns resets and evaluates the saved term after final settling. Leaving
    # terminations active would auto-reset the environment and destroy final-state evidence.
    env_cfg.terminations = None
    env_cfg.env_name = request.environment_name
    if getattr(env_cfg, "observations", None) is not None and getattr(env_cfg.observations, "policy", None) is not None:
        env_cfg.observations.policy.concatenate_terms = False

    recorder_cfg = ActionStateRecorderManagerCfg()
    recorder_cfg.dataset_export_dir_path = dataset_export_dir_path
    recorder_cfg.dataset_filename = dataset_filename
    recorder_cfg.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        if request.output.keep_failed
        else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    env_cfg.recorders = recorder_cfg

    env = builder.make_registered(env_cfg=env_cfg, env_kwargs=env_kwargs)
    try:
        base_env = getattr(env, "unwrapped", env)
        step_dt_s = float(base_env.step_dt)
        validate_planner_timing(request.planner.interpolation_dt_s, step_dt_s)
        adapter = make_embodiment_adapter(request)
        adapter.bind_env(base_env)
    except Exception:
        env.close()
        raise
    return ArenaRuntimeBundle(
        env=env,
        embodiment_adapter=adapter,
        success_term=success_term,
        graph_spec=graph_spec,
        step_dt_s=step_dt_s,
        success_contract_evidence=success_contract_evidence,
        runtime_asset_evidence=runtime_asset_evidence,
    )
