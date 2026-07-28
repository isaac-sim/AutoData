# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from isaac_autodata_examples.compile_task_request import ExitCode, run_cli
from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.task_request_types import (
    ArenaCompilationResult,
    CompiledTaskRequest,
    GenerationConfig,
    GoalStage,
    MotionBackend,
    PlannerBackend,
    PlannerConfig,
    ResolvedOutputConfig,
    SpatialGoalConstraint,
    canonical_json,
    sha256_json,
)
from isaac_autodata_interfaces.motion_planners.curobo.backend_selection import (
    CuroboApiGeneration,
    CuroboRuntimeCapabilities,
    DistributionIdentity,
    select_schedulestream_backend,
)


def _resolved_request(request_directory: Path) -> CompiledTaskRequest:
    canonical_request = {
        "environment": {"intent": {"task": "pick cube into bowl"}},
        "generation": {
            "successful_episodes": 3,
            "max_attempts": 9,
            "num_envs": 2,
            "seed": 17,
        },
        "name": "franka-pick-cube-into-bowl",
        "output": {
            "dataset": "outputs/dataset.hdf5",
            "keep_failed": False,
            "run_log": "outputs/run_log.jsonl",
        },
        "planner": {
            "animate": False,
            "backend": "schedulestream",
            "batch_size": 16,
            "collisions": True,
            "interpolation_dt_s": 0.04,
            "max_time_s": 10.0,
            "motion_backend": "auto",
            "profile": False,
        },
        "schema_version": 1,
    }
    linked_graph = {
        "env_name": "Agentic-Franka-Pick-Cube-Into-Bowl-v0",
        "state_specs": [{
            "id": "success",
            "spatial_constraints": [{
                "id": "cube-in-bowl",
                "kind": "inside",
                "params": {},
                "reference": "bowl",
                "subject": "cube",
            }],
        }],
        "tasks": [{"id": "pick-place", "kind": "pick_place", "success_state_spec_id": "success"}],
    }
    constraint = SpatialGoalConstraint(
        id="cube-in-bowl",
        kind="inside",
        subject="cube",
        reference="bowl",
        params_json=canonical_json({}),
    )
    arena = ArenaCompilationResult(
        initial_graph_json=canonical_json(linked_graph),
        linked_graph_json=canonical_json(linked_graph),
        compiler_trace=(),
        graph_digest=sha256_json(linked_graph),
        goal_stages=(
            GoalStage(
                index=0,
                task_id="pick-place",
                task_kind="pick_place",
                success_state_spec_id="success",
                spatial_constraints=(constraint,),
            ),
        ),
    )
    planner = PlannerConfig(
        backend=PlannerBackend.SCHEDULESTREAM,
        motion_backend=MotionBackend.AUTO,
        collisions=True,
        max_time_s=10.0,
        batch_size=16,
        interpolation_dt_s=0.04,
        profile=False,
        animate=False,
    )
    generation = GenerationConfig(
        successful_episodes=3,
        seed=17,
        num_envs=2,
        max_attempts=9,
    )
    return CompiledTaskRequest(
        schema_version=1,
        compiler_version="test",
        name="franka-pick-cube-into-bowl",
        canonical_request_json=canonical_json(canonical_request),
        request_digest=sha256_json(canonical_request),
        planner=planner,
        generation=generation,
        output=ResolvedOutputConfig(
            dataset=request_directory / "outputs/dataset.hdf5",
            keep_failed=False,
            run_log=request_directory / "outputs/run_log.jsonl",
        ),
        arena=arena,
    )


def _v2_capabilities() -> CuroboRuntimeCapabilities:
    return CuroboRuntimeCapabilities(
        api_generation=CuroboApiGeneration.V2,
        curobo=DistributionIdentity("nvidia-curobo", "2.0.0", source_commit="curobo-test-commit"),
        schedulestream=DistributionIdentity(
            "schedulestream",
            "0.1.0",
            source_commit="schedulestream-test-commit",
        ),
        has_schedulestream_v1=False,
        has_schedulestream_v2=True,
        missing_v1_markers=("curobo.wrap.reacher.motion_gen",),
        missing_v2_markers=(),
        missing_schedulestream_v1_markers=("schedulestream.applications.custream.example",),
        missing_schedulestream_v2_markers=(),
    )


def _run(
    request_path: Path,
    *arguments: str,
    compiler: Callable[[Path], Any] | None = None,
    capability_detector: Callable[[], Any] | None = None,
) -> tuple[int, str, str]:
    resolved = _resolved_request(request_path.parent)
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = run_cli(
        [str(request_path), *arguments],
        compiler=compiler or (lambda _path: resolved),
        capability_detector=capability_detector,
        backend_selector=select_schedulestream_backend,
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


def test_module_import_does_not_load_simulator_or_planner_stacks() -> None:
    forbidden = ("isaaclab", "isaaclab_arena", "isaacsim", "schedulestream", "curobo", "torch")
    program = (
        "import json, sys; "
        "import isaac_autodata_examples.compile_task_request; "
        f"print(json.dumps([name for name in {forbidden!r} if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_dry_run_does_not_import_runtime_probe_or_heavy_stacks() -> None:
    forbidden = (
        "isaac_autodata_interfaces.motion_planners.curobo.backend_selection",
        "isaaclab",
        "isaaclab_arena",
        "isaacsim",
        "schedulestream",
        "curobo",
        "torch",
    )
    program = f"""
import io
import json
import sys
from isaac_autodata_examples.compile_task_request import run_cli

class Resolved:
    digest = "resolved-digest"

    def to_dict(self):
        return {{"name": "test"}}

stdout = io.StringIO()
stderr = io.StringIO()
code = run_cli(
    ["unused.yaml", "--dry-run", "--json"],
    compiler=lambda _path: Resolved(),
    capability_detector=lambda: (_ for _ in ()).throw(AssertionError("probe called")),
    stdout=stdout,
    stderr=stderr,
)
print(json.dumps({{"code": code, "forbidden": [name for name in {forbidden!r} if name in sys.modules]}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {"code": 0, "forbidden": []}


def test_direct_script_invocation_bootstraps_repository_imports(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "isaac_autodata_examples/compile_task_request.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.yaml"), "--dry-run", "--json"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == ExitCode.REQUEST_COMPILATION_FAILED
    assert completed.stdout == ""
    error = json.loads(completed.stderr)["error"]
    assert error["details"]["issues"][0]["code"] == "file_not_found"


def test_dry_run_emits_resolved_json_and_skips_capability_probe(tmp_path: Path) -> None:
    def fail_if_called() -> Any:
        raise AssertionError("dry-run must not probe the runtime")

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--dry-run",
        "--json",
        capability_detector=fail_if_called,
    )

    assert exit_code == ExitCode.SUCCESS
    assert stderr == ""
    result = json.loads(stdout)
    assert result["mode"] == "dry_run"
    assert result["compiled_task"]["planner"]["motion_backend"] == "auto"
    assert result["runtime_preflight"] == {"reason": "dry_run_requested", "status": "skipped"}
    assert result["execution"] == {"dataset_generated": False, "simulation_launched": False}


def test_runtime_preflight_occurs_after_compilation_and_selects_custream2(tmp_path: Path) -> None:
    events: list[str] = []
    resolved = _resolved_request(tmp_path)

    def compiler(_path: Path) -> CompiledTaskRequest:
        events.append("compile")
        return resolved

    def detector() -> CuroboRuntimeCapabilities:
        events.append("detect")
        return _v2_capabilities()

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run_cli(
        [str(tmp_path / "request.yaml"), "--json"],
        compiler=compiler,
        capability_detector=detector,
        backend_selector=select_schedulestream_backend,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.SUCCESS
    assert events == ["compile", "detect"]
    assert stderr.getvalue() == ""
    report = json.loads(stdout.getvalue())["runtime_preflight"]
    assert report["status"] == "passed"
    assert report["requested_motion_backend"] == "auto"
    assert report["selected_motion_backend"] == "curobo_v2"
    assert report["schedulestream_application"] == "custream2"
    assert report["capabilities"]["curobo"]["source_commit"] == "curobo-test-commit"


def test_human_output_makes_non_execution_boundary_explicit(tmp_path: Path) -> None:
    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        capability_detector=_v2_capabilities,
    )

    assert exit_code == ExitCode.SUCCESS
    assert stderr == ""
    assert "Runtime preflight: passed" in stdout
    assert "Selected motion backend: curobo_v2" in stdout
    assert "ScheduleStream application: custream2" in stdout
    assert "Simulation launched: no." in stdout
    assert "Dataset generated: no." in stdout


def test_validation_error_is_structured_and_returns_exit_three(tmp_path: Path) -> None:
    issue = ValidationIssue(("planner", "motion_backend"), "invalid_enum", "unsupported backend")

    def compiler(_path: Path) -> Any:
        raise AutonomousValidationError([issue])

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--json",
        compiler=compiler,
        capability_detector=_v2_capabilities,
    )

    assert exit_code == ExitCode.REQUEST_COMPILATION_FAILED
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["category"] == "request_compilation"
    assert error["exit_code"] == 3
    assert error["details"]["issues"] == [
        {"code": "invalid_enum", "message": "unsupported backend", "path": "$.planner.motion_backend"}
    ]


def test_incompatible_runtime_is_actionable_and_returns_exit_four(tmp_path: Path) -> None:
    unavailable = CuroboRuntimeCapabilities(
        api_generation=CuroboApiGeneration.UNAVAILABLE,
        curobo=None,
        schedulestream=None,
        has_schedulestream_v1=False,
        has_schedulestream_v2=False,
        missing_v1_markers=("curobo.wrap.reacher.motion_gen",),
        missing_v2_markers=("curobo.motion_planner",),
        missing_schedulestream_v1_markers=("schedulestream.applications.custream.example",),
        missing_schedulestream_v2_markers=("schedulestream.applications.custream2.policy",),
    )

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--json",
        "--write-compiled",
        "must-not-exist.json",
        capability_detector=lambda: unavailable,
    )

    assert exit_code == ExitCode.RUNTIME_PREFLIGHT_FAILED
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["category"] == "runtime_preflight"
    assert error["code"] == "backend_incompatible"
    assert error["exit_code"] == 4
    assert error["details"]["requested_motion_backend"] == "auto"
    assert "pinned runtime" in error["details"]["remediation"]
    assert error["details"]["capabilities"]["api_generation"] == "unavailable"
    assert not (tmp_path / "must-not-exist.json").exists()


def test_writes_exact_canonical_artifact_beneath_request_directory(tmp_path: Path) -> None:
    request_path = tmp_path / "request.yaml"
    resolved = _resolved_request(tmp_path)
    exit_code, stdout, stderr = _run(
        request_path,
        "--dry-run",
        "--json",
        "--write-compiled",
        "resolved/request.json",
    )

    artifact = tmp_path / "resolved/request.json"
    assert exit_code == ExitCode.SUCCESS
    assert stderr == ""
    assert artifact.read_text(encoding="utf-8") == resolved.canonical_json() + "\n"
    assert json.loads(stdout)["compiled_task_path"] == str(artifact)
    assert stat.S_IMODE(artifact.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(artifact.parent.stat().st_mode) & 0o077 == 0
    assert list(artifact.parent.glob(".autodata-compiled-*.tmp")) == []


@pytest.mark.parametrize(
    ("artifact_name", "expected_code"),
    [
        ("../escaped.json", "artifact_path_traversal"),
        ("/tmp/absolute.json", "artifact_path_absolute"),
        ("resolved/request.yaml", "artifact_extension_invalid"),
        (".git/request.json", "artifact_path_forbidden"),
        ("resolved//request.json", "artifact_path_traversal"),
        (r"resolved\request.json", "artifact_path_invalid"),
    ],
)
def test_rejects_unsafe_artifact_names(tmp_path: Path, artifact_name: str, expected_code: str) -> None:
    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--dry-run",
        "--json",
        "--write-compiled",
        artifact_name,
    )

    assert exit_code == ExitCode.ARTIFACT_WRITE_FAILED
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["code"] == expected_code
    assert error["exit_code"] == 5


def test_refuses_artifact_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--dry-run",
        "--json",
        "--write-compiled",
        "escape/request.json",
    )

    assert exit_code == ExitCode.ARTIFACT_WRITE_FAILED
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "artifact_path_unsafe"
    assert not (outside / "request.json").exists()


def test_refuses_to_replace_final_artifact_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("external\n", encoding="utf-8")
    target = tmp_path / "resolved.json"
    target.symlink_to(outside)

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--dry-run",
        "--json",
        "--write-compiled",
        target.name,
    )

    assert exit_code == ExitCode.ARTIFACT_WRITE_FAILED
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "artifact_exists"
    assert target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "external\n"


def test_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    target = tmp_path / "resolved.json"
    target.write_text("human-owned\n", encoding="utf-8")

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--dry-run",
        "--json",
        "--write-compiled",
        target.name,
    )

    assert exit_code == ExitCode.ARTIFACT_WRITE_FAILED
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "artifact_exists"
    assert target.read_text(encoding="utf-8") == "human-owned\n"


def test_unexpected_compiler_failure_is_bounded_without_traceback(tmp_path: Path) -> None:
    def compiler(_path: Path) -> Any:
        raise RuntimeError("first line\nsecond line")

    exit_code, stdout, stderr = _run(
        tmp_path / "request.yaml",
        "--json",
        compiler=compiler,
        capability_detector=_v2_capabilities,
    )

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    assert "Traceback" not in stderr
    error = json.loads(stderr)["error"]
    assert error["exit_code"] == 6
    assert error["message"] == "Unexpected RuntimeError: first line second line"
