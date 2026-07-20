# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import os
import urllib.error
from types import SimpleNamespace

import pytest

from isaac_autodata_core.autonomous.output_transaction import RecordingTargets
from isaac_autodata_interfaces.autonomous.arena_environment import (
    ArenaRuntimeBundle,
    _attest_exact_url_content,
    apply_custream_v1_runtime_asset_profile,
    attest_custream_v1_runtime_usd_root,
    attest_pick_and_place_success_contract,
    build_arena_runtime,
    goal_predicates_from_request,
    make_embodiment_adapter,
    validate_planner_timing,
)

_PINNED_PANDA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
    "Robots/FrankaEmika/panda_instanceable.usd"
)
_PINNED_PANDA_USD_BYTES = 8_038
_PINNED_PANDA_USD_SHA256 = "7f5a0c0aa6760cfbd348e08bc464d4b94341f027f51c2d9e42406ceefcc7787f"
_MAX_RUNTIME_USD_BYTES = 1 << 20


def _root_layer_evidence(**overrides):
    evidence = {
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
    evidence.update(overrides)
    return evidence


class _Response:
    def __init__(self, payload, *, url, status=200, headers=None):
        self._payload = io.BytesIO(payload)
        self._url = url
        self.status = status
        self.headers = {"Content-Length": str(len(payload))} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._payload.close()

    def geturl(self):
        return self._url

    def read(self, size):
        return self._payload.read(size)


class _CloseCountingEnvironment:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _request(*, embodiment_name="franka_ik"):
    constraint = SimpleNamespace(kind="on", subject="cube", reference="bowl")
    stage = SimpleNamespace(spatial_constraints=(constraint,))
    return SimpleNamespace(
        goal_stages=(stage,),
        linked_graph={"nodes": [{"id": "robot", "name": embodiment_name, "type": "embodiment"}]},
    )


def test_goal_projection_preserves_linked_scene_ids():
    predicates = goal_predicates_from_request(_request())

    assert [predicate.to_dict() for predicate in predicates] == [
        {"relation": "on", "subject": "cube", "target": "bowl"}
    ]


def test_empty_arena_goal_is_rejected():
    with pytest.raises(ValueError, match="no spatial success constraints"):
        goal_predicates_from_request(SimpleNamespace(goal_stages=()))


def test_franka_runtime_binding_is_explicit():
    adapter = make_embodiment_adapter(_request())

    assert adapter.name == "arena_franka_ik"
    assert adapter.get_eef_names() == ("franka",)
    assert adapter.gripper_action_dim == 1
    assert adapter.eef_offset == pytest.approx((0.0, 0.0, -0.0036))


def test_unknown_embodiment_does_not_silently_reuse_franka_profile():
    with pytest.raises(NotImplementedError, match="droid"):
        make_embodiment_adapter(_request(embodiment_name="droid"))


def _scene_robot_cfg(usd_basename: str = "franka_panda_hand_on_stand.usd"):
    return SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=f"omniverse://arena/{usd_basename}")))


def test_custream_v1_runtime_asset_profile_substitutes_only_reviewed_usd():
    scene_cfg = _scene_robot_cfg()
    runtime_usd_path = _PINNED_PANDA_USD
    attested_urls = []

    def attest_root_layer(url):
        attested_urls.append(url)
        return _root_layer_evidence()

    evidence = apply_custream_v1_runtime_asset_profile(
        _request(),
        scene_cfg,
        runtime_usd_path=runtime_usd_path,
        root_layer_attestor=attest_root_layer,
    )

    assert attested_urls == [_PINNED_PANDA_USD]
    assert scene_cfg.robot.spawn.usd_path == runtime_usd_path
    assert evidence == {
        "attested": True,
        "attestation_scope": "runtime_usd_root_layer_identity_only",
        "kinematic_frame_attestation": "separate_live_provider_attestation_required",
        "motion_backend": "curobo_v1",
        "override": "composed_scene.robot.spawn.usd_path_only",
        "profile": "franka_ik_custream_v1_official_root_usd",
        "reason": "pinned_official_isaac_5_1_root_layer_content_identity",
        "referenced_usd_dependencies_attested": False,
        "registry_usd_basename": "franka_panda_hand_on_stand.usd",
        "registry_usd_path": "omniverse://arena/franka_panda_hand_on_stand.usd",
        "root_layer": _root_layer_evidence(),
        "runtime_usd_basename": "panda_instanceable.usd",
        "runtime_usd_path": runtime_usd_path,
        "runtime_usd_release": "Isaac 5.1",
        "runtime_uri_policy": "pinned_exact_https_no_redirect",
        "schedulestream_application": "custream",
        "schema_version": 2,
        "semantic_embodiment": "franka_ik",
    }


def test_runtime_asset_profile_fails_closed_before_mutation_when_root_attestation_fails():
    scene_cfg = _scene_robot_cfg()
    original_path = scene_cfg.robot.spawn.usd_path

    def fail_attestation(_url):
        raise ValueError("offline or mismatched")

    with pytest.raises(ValueError, match="root-layer attestation failed"):
        apply_custream_v1_runtime_asset_profile(
            _request(),
            scene_cfg,
            runtime_usd_path=_PINNED_PANDA_USD,
            root_layer_attestor=fail_attestation,
        )

    assert scene_cfg.robot.spawn.usd_path == original_path


def test_runtime_asset_profile_rejects_incomplete_injected_root_evidence():
    scene_cfg = _scene_robot_cfg()
    original_path = scene_cfg.robot.spawn.usd_path

    with pytest.raises(ValueError, match="incomplete or mismatched"):
        apply_custream_v1_runtime_asset_profile(
            _request(),
            scene_cfg,
            runtime_usd_path=_PINNED_PANDA_USD,
            root_layer_attestor=lambda _url: _root_layer_evidence(sha256="0" * 64),
        )

    assert scene_cfg.robot.spawn.usd_path == original_path


def test_exact_url_content_attestation_hashes_bounded_identity_response_without_network():
    url = "https://assets.example.test/root.usd"
    payload = b"#usda 1.0\n"
    opened = []

    def opener(request, timeout_s):
        opened.append((request, timeout_s))
        return _Response(payload, url=url)

    evidence = _attest_exact_url_content(
        url,
        expected_url=url,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
        max_bytes=64,
        timeout_s=2.0,
        opener=opener,
    )

    assert len(opened) == 1
    assert opened[0][0].full_url == url
    assert opened[0][0].get_header("Accept-encoding") == "identity"
    assert opened[0][1] == pytest.approx(2.0)
    assert evidence["attested"] is True
    assert evidence["scope"] == "root_layer_bytes_only"
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()


def test_exact_url_content_attestation_rejects_redirect_or_final_url_change_without_network():
    url = "https://assets.example.test/root.usd"
    payload = b"root"
    digest = hashlib.sha256(payload).hexdigest()

    def redirect_error(request, _timeout_s):
        raise urllib.error.HTTPError(request.full_url, 302, "redirect", {"Location": url + "?mirror=1"}, None)

    with pytest.raises(ValueError, match="redirects are forbidden"):
        _attest_exact_url_content(
            url,
            expected_url=url,
            expected_sha256=digest,
            expected_bytes=len(payload),
            max_bytes=64,
            timeout_s=2.0,
            opener=redirect_error,
        )

    with pytest.raises(ValueError, match="final-URL changes are forbidden"):
        _attest_exact_url_content(
            url,
            expected_url=url,
            expected_sha256=digest,
            expected_bytes=len(payload),
            max_bytes=64,
            timeout_s=2.0,
            opener=lambda *_args: _Response(payload, url=url + "?mirror=1"),
        )


@pytest.mark.parametrize(
    ("headers", "payload"),
    [
        ({"Content-Length": "65"}, b"root"),
        ({}, b"x" * 65),
    ],
)
def test_exact_url_content_attestation_rejects_declared_or_streamed_oversize(headers, payload):
    url = "https://assets.example.test/root.usd"

    with pytest.raises(ValueError, match="exceeds the 64-byte download bound"):
        _attest_exact_url_content(
            url,
            expected_url=url,
            expected_sha256=hashlib.sha256(b"root").hexdigest(),
            expected_bytes=4,
            max_bytes=64,
            timeout_s=2.0,
            opener=lambda *_args: _Response(payload, url=url, headers=headers),
        )


def test_exact_url_content_attestation_rejects_size_or_sha256_mismatch():
    url = "https://assets.example.test/root.usd"
    payload = b"root"

    with pytest.raises(ValueError, match="byte count mismatch"):
        _attest_exact_url_content(
            url,
            expected_url=url,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_bytes=5,
            max_bytes=64,
            timeout_s=2.0,
            opener=lambda *_args: _Response(payload, url=url),
        )

    with pytest.raises(ValueError, match="SHA-256"):
        _attest_exact_url_content(
            url,
            expected_url=url,
            expected_sha256="0" * 64,
            expected_bytes=len(payload),
            max_bytes=64,
            timeout_s=2.0,
            opener=lambda *_args: _Response(payload, url=url),
        )


def test_official_root_attestor_uses_fixed_size_and_digest_without_network():
    payload = b"x" * _PINNED_PANDA_USD_BYTES

    with pytest.raises(ValueError, match="SHA-256"):
        attest_custream_v1_runtime_usd_root(
            _PINNED_PANDA_USD,
            opener=lambda *_args: _Response(payload, url=_PINNED_PANDA_USD),
        )


@pytest.mark.parametrize(
    ("resolved_request", "scene_cfg", "runtime_usd_path", "message"),
    [
        (
            _request(embodiment_name="droid"),
            _scene_robot_cfg(),
            _PINNED_PANDA_USD,
            "franka_ik",
        ),
        (
            _request(),
            _scene_robot_cfg("unreviewed_robot.usd"),
            _PINNED_PANDA_USD,
            "registry asset changed",
        ),
        (
            _request(),
            _scene_robot_cfg(),
            "omniverse://isaaclab/Robots/FrankaEmika/unreviewed_robot.usd",
            "runtime asset changed",
        ),
        (
            _request(),
            _scene_robot_cfg(),
            "https://unreviewed.invalid/Robots/FrankaEmika/panda_instanceable.usd",
            "pinned official production URI",
        ),
    ],
)
def test_custream_v1_runtime_asset_profile_rejects_semantic_or_asset_drift(
    resolved_request,
    scene_cfg,
    runtime_usd_path,
    message,
):
    original_path = scene_cfg.robot.spawn.usd_path

    with pytest.raises(ValueError, match=message):
        apply_custream_v1_runtime_asset_profile(
            resolved_request,
            scene_cfg,
            runtime_usd_path=runtime_usd_path,
        )

    assert scene_cfg.robot.spawn.usd_path == original_path


def test_planner_timing_requires_exact_environment_cadence():
    validate_planner_timing(0.05, 0.050000001)

    with pytest.raises(ValueError, match="must match"):
        validate_planner_timing(0.05, 0.02)


def check_success():
    pass


def object_on_destination():
    pass


check_success.__module__ = "isaaclab_arena.tasks.terminations"
object_on_destination.__module__ = "isaaclab_arena.tasks.terminations"


def _success_contract(
    *,
    force_threshold=0.1,
    sensor_path="{ENV_REGEX_NS}/cube",
    filter_path="{ENV_REGEX_NS}/bowl",
):
    predicate = SimpleNamespace(
        func=object_on_destination,
        params={
            "contact_sensor_cfg": SimpleNamespace(name="pick_up_object_contact_sensor"),
            "force_threshold": force_threshold,
            "object_cfg": SimpleNamespace(name="cube"),
            "velocity_threshold": 0.1,
        },
    )
    success = SimpleNamespace(func=check_success, params={"mode": "ALL", "predicates": [predicate]})
    scene = SimpleNamespace(
        pick_up_object_contact_sensor=SimpleNamespace(
            prim_path=sensor_path,
            filter_prim_paths_expr=[filter_path],
        )
    )
    return success, scene


def test_success_contract_attests_exact_task_sensor_and_thresholds():
    success, scene = _success_contract()

    evidence = attest_pick_and_place_success_contract(
        success,
        scene,
        pick_up_object="cube",
        destination_location="bowl",
    )

    assert evidence["attested"] is True
    assert evidence["subject"] == "cube"
    assert evidence["force_threshold_n"] == pytest.approx(0.1)


def test_success_contract_rejects_threshold_or_contact_filter_drift():
    success, scene = _success_contract(force_threshold=0.2)
    with pytest.raises(ValueError, match="force threshold"):
        attest_pick_and_place_success_contract(
            success,
            scene,
            pick_up_object="cube",
            destination_location="bowl",
        )

    success, scene = _success_contract(filter_path="{ENV_REGEX_NS}/table")
    with pytest.raises(ValueError, match="destination object"):
        attest_pick_and_place_success_contract(
            success,
            scene,
            pick_up_object="cube",
            destination_location="bowl",
        )


@pytest.mark.parametrize(
    ("sensor_path", "filter_path", "message"),
    [
        ("{ENV_REGEX_NS}/not_cube", "{ENV_REGEX_NS}/bowl", "prim path"),
        ("{ENV_REGEX_NS}/cube", "{ENV_REGEX_NS}/not_bowl", "destination object"),
        ("/World/envs/env_.*/cube", "{ENV_REGEX_NS}/bowl", "prim path"),
    ],
)
def test_success_contract_requires_exact_safe_object_paths(sensor_path, filter_path, message):
    success, scene = _success_contract(sensor_path=sensor_path, filter_path=filter_path)

    with pytest.raises(ValueError, match=message):
        attest_pick_and_place_success_contract(
            success,
            scene,
            pick_up_object="cube",
            destination_location="bowl",
        )


def _runtime_request(dataset, *, keep_failed=False):
    return SimpleNamespace(
        generation=SimpleNamespace(num_envs=1),
        output=SimpleNamespace(dataset=dataset, keep_failed=keep_failed),
    )


def test_runtime_rejects_existing_dataset_before_heavy_imports(tmp_path):
    dataset = tmp_path / "output.hdf5"
    dataset.write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_arena_runtime(_runtime_request(dataset), SimpleNamespace())


def test_runtime_rejects_existing_failed_dataset_before_heavy_imports(tmp_path):
    dataset = tmp_path / "output.hdf5"
    (tmp_path / "output_failed.hdf5").write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="failed"):
        build_arena_runtime(_runtime_request(dataset, keep_failed=True), SimpleNamespace())


def test_runtime_rejects_broken_dataset_symlink_before_heavy_imports(tmp_path):
    dataset = tmp_path / "output.hdf5"
    dataset.symlink_to(tmp_path / "missing.hdf5")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_arena_runtime(_runtime_request(dataset), SimpleNamespace())


def test_runtime_requires_recording_override_to_preserve_dataset_stem(tmp_path):
    dataset = tmp_path / "output.hdf5"
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        targets = RecordingTargets(
            dataset_export_dir_path=f"/proc/self/fd/{descriptor}",
            dataset_filename="redirected",
        )
        with pytest.raises(ValueError, match="preserve"):
            build_arena_runtime(
                _runtime_request(dataset),
                SimpleNamespace(),
                recording_targets=targets,
            )
    finally:
        os.close(descriptor)


def test_runtime_bundle_close_is_idempotent():
    env = _CloseCountingEnvironment()
    bundle = ArenaRuntimeBundle(env, object(), object(), object(), 0.02)

    bundle.close()
    bundle.close()

    assert env.close_calls == 1
