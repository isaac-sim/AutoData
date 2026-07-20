# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from isaac_autodata_examples.generate_task_dataset import (
    ExitCode,
    RuntimeStack,
    _run_gui_generation_inline,
    _run_runtime_child_process,
    _RuntimeChildProcessResult,
    build_argument_parser,
    run_cli,
)
from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.runtime_support import validate_runtime_support


class _Capabilities:
    def to_dict(self) -> dict[str, Any]:
        return {
            "api_generation": "v1",
            "curobo": {"distribution": "nvidia-curobo", "version": "0.7.7"},
            "schedulestream": {"distribution": "schedulestream", "version": "0.1.0"},
        }


class _ResolvedRequest:
    def __init__(self, root: Path) -> None:
        self.name = "franka-pick-cube-into-bowl"
        self.request_digest = "b" * 64
        self.digest = "a" * 64
        self.graph_digest = "c" * 64
        self.environment_name = "Agentic-Franka-Pick-Cube-Into-Bowl-v0"
        self.planner = SimpleNamespace(
            animate=False,
            backend=SimpleNamespace(value="schedulestream"),
            batch_size=32,
            collisions=True,
            interpolation_dt_s=0.02,
            max_time_s=10.0,
            motion_backend=SimpleNamespace(value="auto"),
            profile=False,
        )
        self.generation = SimpleNamespace(
            successful_episodes=3,
            max_attempts=7,
            num_envs=1,
            seed=41,
        )
        self.output = SimpleNamespace(
            dataset=root / "outputs" / "dataset.hdf5",
            keep_failed=False,
            run_log=root / "outputs" / "run_log.jsonl",
        )
        self.linked_graph = {
            "env_name": self.environment_name,
            "nodes": [
                {"id": "table", "name": "maple_table_robolab", "params": {}, "type": "background"},
                {"id": "robot", "name": "franka_ik", "params": {}, "type": "embodiment"},
                {
                    "id": "cube",
                    "name": "rubiks_cube_hot3d_robolab",
                    "params": {},
                    "type": "object",
                },
                {"id": "bowl", "name": "bowl_ycb_robolab", "params": {}, "type": "object"},
            ],
            "state_specs": [
                {
                    "id": "initial",
                    "is_delta": False,
                    "spatial_constraints": [
                        {
                            "id": "table_anchor",
                            "kind": "is_anchor",
                            "params": {},
                            "subject": "table",
                        },
                        {
                            "id": "cube_on_table",
                            "kind": "on",
                            "params": {},
                            "reference": "table",
                            "subject": "cube",
                        },
                        {
                            "id": "bowl_on_table",
                            "kind": "on",
                            "params": {},
                            "reference": "table",
                            "subject": "bowl",
                        },
                    ],
                    "task_constraints": [],
                },
                {
                    "id": "success",
                    "is_delta": True,
                    "spatial_constraints": [{
                        "id": "cube_on_bowl",
                        "kind": "on",
                        "params": {},
                        "reference": "bowl",
                        "subject": "cube",
                    }],
                    "task_constraints": [],
                },
            ],
            "tasks": [{
                "id": "pick_and_place",
                "initial_state_spec_id": "initial",
                "kind": "PickAndPlaceTask",
                "params": {
                    "background_scene": "table",
                    "destination_location": "bowl",
                    "pick_up_object": "cube",
                },
                "success_state_spec_id": "success",
            }],
        }
        self.goal_stages = (
            SimpleNamespace(
                index=0,
                task_id="pick_and_place",
                task_kind="PickAndPlaceTask",
                success_state_spec_id="success",
                spatial_constraints=(
                    SimpleNamespace(
                        id="cube_on_bowl",
                        kind="on",
                        params={},
                        reference="bowl",
                        subject="cube",
                    ),
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arena": {
                "graph_digest": self.graph_digest,
                "linked_graph": self.linked_graph,
            },
            "generation": {
                "successful_episodes": self.generation.successful_episodes,
                "max_attempts": self.generation.max_attempts,
                "num_envs": self.generation.num_envs,
                "seed": self.generation.seed,
            },
            "name": self.name,
            "output": {
                "dataset": str(self.output.dataset),
                "keep_failed": self.output.keep_failed,
                "run_log": str(self.output.run_log),
            },
            "request_digest": self.request_digest,
        }


class _Summary:
    def __init__(self, *, target_reached: bool = True) -> None:
        self.target_reached = target_reached

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": 4,
            "failures": 1,
            "last_attempt_id": "attempt-000003",
            "request_digest": "b" * 64,
            "requested_successful_episodes": 3,
            "stop_reason": "requested_successes" if self.target_reached else "max_attempts",
            "successes": 3 if self.target_reached else 2,
            "target_reached": self.target_reached,
        }


class _Artifact:
    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": 3,
            "kind": "successful",
            "path": "/attested/dataset.hdf5",
            "sha256": "e" * 64,
            "size_bytes": 4096,
        }


class _OutputTransaction:
    def __init__(self, harness: _RuntimeHarness) -> None:
        self._harness = harness
        self.recording_targets = SimpleNamespace(
            dataset_export_dir_path="/proc/self/fd/123",
            dataset_filename=harness.resolved.output.dataset.stem,
        )

    def open_run_log_writer(self, writer_factory: Any) -> Any:
        return writer_factory(self._harness.resolved.output.run_log)

    def publish(self, **kwargs: Any) -> tuple[_Artifact, ...]:
        self._harness.events.append("publish:outputs")
        self._harness.publish_kwargs = kwargs
        if self._harness.publish_error is not None:
            raise self._harness.publish_error
        commit_record = kwargs["commit_record"]
        if commit_record is not None:
            kwargs["run_log_writer"].append({**commit_record, "artifacts": [_Artifact().to_dict()]})
        return (_Artifact(),)

    def close(self) -> None:
        return


class _Resource:
    def __init__(self, label: str, events: list[str], *, close_error: Exception | None = None) -> None:
        self._label = label
        self._events = events
        self._close_error = close_error

    def close(self) -> None:
        self._events.append(f"close:{self._label}")
        if self._close_error is not None:
            raise self._close_error


class _EventStream(io.StringIO):
    def __init__(self, label: str, events: list[str]) -> None:
        super().__init__()
        self._label = label
        self._events = events

    def write(self, value: str) -> int:
        self._events.append(f"emit:{self._label}")
        return super().write(value)

    def flush(self) -> None:
        self._events.append(f"flush:{self._label}")
        super().flush()


class _RuntimeHarness:
    def __init__(self, resolved: _ResolvedRequest) -> None:
        self.resolved = resolved
        self.events: list[str] = []
        self.app_options: dict[str, Any] | None = None
        self.arena_args: Any | None = None
        self.generation_kwargs: dict[str, Any] | None = None
        self.run_close: bool | None = None
        self.run_log_records: list[dict[str, Any]] = []
        self.run_log_append_error: Exception | None = None
        self.run_log_append_error_record_type: str | None = None
        self.publish_kwargs: dict[str, Any] | None = None
        self.publish_error: Exception | None = None
        self.attachment_state = object()
        self.runtime_attachment_state: Any | None = None
        self.executor_attachment_state: Any | None = None
        self.planner_attachment_state: Any | None = None
        self.summary = _Summary()
        self.runtime_error: Exception | None = None
        self.run_error: BaseException | None = None
        self.close_errors: dict[str, Exception] = {}

    def compile(self, path: Path) -> _ResolvedRequest:
        self.events.append("compile")
        assert path.name == "request.yaml"
        return self.resolved

    def detect(self) -> _Capabilities:
        self.events.append("detect")
        return _Capabilities()

    def select(self, requested: str, capabilities: _Capabilities) -> Any:
        self.events.append("select")
        assert requested == "auto"
        assert isinstance(capabilities, _Capabilities)
        return SimpleNamespace(
            capabilities=capabilities,
            motion_backend="curobo_v1",
            schedulestream_application="custream",
        )

    def load_app_launcher(self) -> Any:
        self.events.append("load:app_launcher")

        def launch(options: dict[str, Any]) -> Any:
            self.events.append("launch:app")
            self.app_options = options
            app = _Resource("app", self.events, close_error=self.close_errors.get("app"))
            return SimpleNamespace(app=app)

        return launch

    def load_runtime_stack(self) -> RuntimeStack:
        self.events.append("load:runtime_stack")
        return RuntimeStack(
            output_transaction_factory=self._reserve_outputs,
            arena_runtime_builder=self._build_arena_runtime,
            goal_projector=self._project_goal,
            attachment_state_factory=self._make_attachment_state,
            runtime_factory=self._make_runtime,
            success_verifier_factory=self._make_success_verifier,
            executor_factory=self._make_executor,
            planner_factory=self._make_planner,
            run_log_writer_factory=self._make_run_log_writer,
            generator_factory=self._make_generator,
            generation_request_factory=self._make_generation_request,
            run_loop=self._run_loop,
        )

    def _reserve_outputs(self, **kwargs: Any) -> _OutputTransaction:
        self.events.append("reserve:outputs")
        assert kwargs == {
            "dataset_path": self.resolved.output.dataset,
            "keep_failed": self.resolved.output.keep_failed,
            "run_log_path": self.resolved.output.run_log,
            "request_directory": self.resolved.output.dataset.parent.parent,
        }
        return _OutputTransaction(self)

    def _build_arena_runtime(
        self,
        resolved: _ResolvedRequest,
        args: Any,
        *,
        recording_targets: Any,
        allow_output_overwrite: bool,
    ) -> Any:
        self.events.append("build:arena_runtime")
        assert resolved is self.resolved
        assert allow_output_overwrite is False
        assert recording_targets.dataset_export_dir_path == "/proc/self/fd/123"
        assert recording_targets.dataset_filename == self.resolved.output.dataset.stem
        self.arena_args = args
        resource = _Resource("arena", self.events, close_error=self.close_errors.get("arena"))
        return SimpleNamespace(
            close=resource.close,
            embodiment_adapter=object(),
            env=object(),
            success_term=object(),
        )

    def _project_goal(self, resolved: _ResolvedRequest) -> tuple[str, ...]:
        self.events.append("project:goal")
        assert resolved is self.resolved
        return ("cube-inside-bowl",)

    def _make_attachment_state(self) -> object:
        self.events.append("create:attachment_state")
        return self.attachment_state

    def _make_runtime(self, env: Any, adapter: Any, **kwargs: Any) -> Any:
        del env, adapter
        self.events.append("create:runtime")
        self.runtime_attachment_state = kwargs["attachment_state"]
        assert kwargs["graph_nodes"] == tuple(self.resolved.linked_graph["nodes"])
        if self.runtime_error is not None:
            raise self.runtime_error
        return object()

    def _make_success_verifier(self, success_term: Any) -> Any:
        del success_term
        self.events.append("create:success_verifier")
        return object()

    def _make_executor(self, env: Any, adapter: Any, verifier: Any, **kwargs: Any) -> Any:
        del env, adapter, verifier
        self.events.append("create:executor")
        self.executor_attachment_state = kwargs["attachment_state"]
        return object()

    def _make_planner(
        self,
        bundle: Any,
        resolved: _ResolvedRequest,
        compatibility: Any,
        *,
        attachment_state: Any | None = None,
    ) -> _Resource:
        del bundle
        self.events.append("create:planner")
        assert resolved is self.resolved
        assert compatibility.motion_backend == "curobo_v1"
        self.planner_attachment_state = attachment_state
        return _Resource("planner", self.events, close_error=self.close_errors.get("planner"))

    def _make_run_log_writer(self, path: Path, **_kwargs: Any) -> Any:
        self.events.append("create:run_log_writer")
        assert path == self.resolved.output.run_log

        def append(record: dict[str, Any]) -> None:
            self.events.append(f"append:{record['record_type']}")
            if self.run_log_append_error is not None and record["record_type"] == self.run_log_append_error_record_type:
                raise self.run_log_append_error
            self.run_log_records.append(record)

        return SimpleNamespace(append=append, close=lambda: None)

    def _make_generator(self, runtime: Any, planner: Any, executor: Any, **kwargs: Any) -> Any:
        del runtime, planner, executor
        self.events.append("create:generator")
        assert kwargs["run_log_writer"] is not None
        return object()

    def _make_generation_request(self, **kwargs: Any) -> Any:
        self.events.append("create:generation_request")
        self.generation_kwargs = kwargs
        return SimpleNamespace(**kwargs)

    async def _run_loop(self, generator: Any, request: Any, *, close: bool) -> _Summary:
        del generator, request
        self.events.append("run:generation")
        self.run_close = close
        if self.run_error is not None:
            raise self.run_error
        return self.summary


def _run_child(
    harness: _RuntimeHarness,
    *arguments: str,
    expected_digest: str | None = None,
    terminal_status_writer: Any | None = None,
    stdout: io.StringIO | None = None,
    stderr: io.StringIO | None = None,
) -> tuple[int, str, str]:
    stdout = stdout or io.StringIO()
    stderr = stderr or io.StringIO()
    exit_code = run_cli(
        [
            str(harness.resolved.output.dataset.parent.parent / "request.yaml"),
            "--runtime-child",
            "--expected-compiled-task-digest",
            expected_digest or harness.resolved.digest,
            "--expected-motion-backend",
            "curobo_v1",
            "--expected-schedulestream-application",
            "custream",
            "--expected-runtime-support-digest",
            _runtime_support_digest(harness.resolved),
            "--json",
            *arguments,
        ],
        compiler=harness.compile,
        capability_detector=harness.detect,
        backend_selector=harness.select,
        app_launcher_factory_loader=harness.load_app_launcher,
        runtime_stack_loader=harness.load_runtime_stack,
        terminal_status_writer=terminal_status_writer or (lambda _status_fd, _exit_code: None),
        stdout=stdout,
        stderr=stderr,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _runtime_support_digest(resolved: _ResolvedRequest) -> str:
    return validate_runtime_support(
        resolved,
        motion_backend="curobo_v1",
        schedulestream_application="custream",
    ).digest


def _run_parent(
    harness: _RuntimeHarness,
    child_process_runner: Any,
    *arguments: str,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run_cli(
        [str(harness.resolved.output.dataset.parent.parent / "request.yaml"), "--json", *arguments],
        compiler=harness.compile,
        capability_detector=harness.detect,
        backend_selector=harness.select,
        child_process_runner=child_process_runner,
        stdout=stdout,
        stderr=stderr,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_module_import_does_not_load_runtime_or_simulator_stacks() -> None:
    forbidden_prefixes = (
        "isaac_autodata_core.autonomous",
        "isaac_autodata_interfaces.autonomous.arena_environment",
        "isaac_autodata_interfaces.autonomous.isaaclab_runtime",
        "isaac_autodata_interfaces.autonomous.schedulestream",
        "isaaclab",
        "isaaclab_arena",
        "isaacsim",
        "schedulestream",
        "curobo",
        "torch",
    )
    program = f"""
import json
import sys
import isaac_autodata_examples.generate_task_dataset
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in {forbidden_prefixes!r})
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_display_mode_defaults_headless_and_exposes_gui() -> None:
    parser = build_argument_parser()

    assert parser.parse_args(["request.yaml"]).headless is True
    assert parser.parse_args(["request.yaml", "--headless"]).headless is True
    assert parser.parse_args(["request.yaml", "--gui"]).headless is False
    assert parser.parse_args(["request.yaml", "--no-headless"]).headless is False

    help_text = parser.format_help()
    assert "--gui" in help_text
    assert "--headless" in help_text
    assert "--no-headless" not in help_text


@pytest.mark.parametrize(
    "display_options",
    [
        ("--gui", "--headless"),
        ("--gui", "--no-headless"),
        ("--headless", "--no-headless"),
    ],
)
def test_display_modes_are_mutually_exclusive(display_options: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_argument_parser().parse_args(["request.yaml", *display_options])

    assert exc_info.value.code == 2


def test_gui_selects_kit_and_does_not_claim_the_main_thread_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))

    def reject_asyncio_run(awaitable: Any) -> None:
        with suppress(BaseException):
            awaitable.close()
        raise AssertionError("GUI generation must not call asyncio.run")

    monkeypatch.setattr(asyncio, "run", reject_asyncio_run)
    exit_code, _stdout, stderr = _run_child(harness, "--gui")

    assert exit_code == ExitCode.SUCCESS
    assert stderr == ""
    assert harness.app_options == {
        "device": "cuda:0",
        "enable_cameras": False,
        "headless": False,
        "visualizer": ["kit"],
    }
    assert "run:generation" in harness.events


def test_gui_inline_generation_returns_without_suspending() -> None:
    expected = object()

    async def complete_inline() -> object:
        return expected

    assert _run_gui_generation_inline(complete_inline()) is expected


def test_gui_inline_generation_rejects_suspension_and_closes_coroutine() -> None:
    finalized = False

    async def suspend() -> None:
        nonlocal finalized
        try:
            await asyncio.sleep(0)
        finally:
            finalized = True

    operation = suspend()
    with pytest.raises(RuntimeError, match="GUI generation unexpectedly suspended"):
        _run_gui_generation_inline(operation)

    assert finalized
    assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED


def test_parent_preflights_then_launches_an_attested_fresh_child(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    commands: list[tuple[str, ...]] = []

    def launch_child(command: Any) -> int:
        harness.events.append("spawn:runtime_child")
        commands.append(tuple(command))
        return int(ExitCode.GENERATION_INCOMPLETE)

    exit_code, stdout, stderr = _run_parent(
        harness,
        launch_child,
        "--device",
        "cuda:3",
        "--gui",
        "--enable-cameras",
    )

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    assert stdout == ""
    assert stderr == ""
    assert harness.events == ["compile", "detect", "select", "spawn:runtime_child"]
    assert "load:app_launcher" not in harness.events
    command = commands[0]
    assert command[:2] == (sys.executable, "-u")
    assert Path(command[2]).name == "generate_task_dataset.py"
    assert Path(command[3]) == (tmp_path / "request.yaml").resolve()
    assert command[command.index("--expected-compiled-task-digest") + 1] == harness.resolved.digest
    assert command[command.index("--expected-motion-backend") + 1] == "curobo_v1"
    assert command[command.index("--expected-schedulestream-application") + 1] == "custream"
    assert command[command.index("--expected-runtime-support-digest") + 1] == _runtime_support_digest(harness.resolved)
    assert "--runtime-child" in command
    assert "--status-fd" not in command
    assert "--no-headless" in command
    assert "--enable-cameras" in command


def test_parent_uses_terminal_channel_instead_of_false_process_success(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    false_success = _RuntimeChildProcessResult(
        process_return_code=0,
        status_payload=b'{"exit_code":8,"protocol_version":1}\n',
    )

    exit_code, stdout, stderr = _run_parent(harness, lambda _command: false_success)

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    assert stdout == ""
    assert stderr == ""


def test_parent_rejects_success_status_followed_by_abnormal_process_exit(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    false_success = _RuntimeChildProcessResult(
        process_return_code=-9,
        status_payload=b'{"exit_code":0,"protocol_version":1}\n',
    )

    exit_code, stdout, stderr = _run_parent(harness, lambda _command: false_success)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["code"] == "child_success_process_failed"
    assert error["details"] == {"process_return_code": -9}


def test_parent_accepts_failure_status_despite_abnormal_process_exit(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    child_failure = _RuntimeChildProcessResult(
        process_return_code=-9,
        status_payload=b'{"exit_code":8,"protocol_version":1}\n',
    )

    exit_code, stdout, stderr = _run_parent(harness, lambda _command: child_failure)

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    assert stdout == ""
    assert stderr == ""


def test_real_subprocess_runner_passes_private_status_descriptor() -> None:
    program = (
        'import os,sys; fd=int(sys.argv[-1]); os.write(fd,b\'{"exit_code":8,"protocol_version":1}\\n\'); os.close(fd)'
    )

    result = _run_runtime_child_process((sys.executable, "-c", program))

    assert result.process_return_code == 0
    assert result.status_payload == b'{"exit_code":8,"protocol_version":1}\n'


def test_subprocess_runner_timeout_escalates_process_group_and_reaps(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []
    popen_options: dict[str, Any] = {}
    group_alive = True

    class HungProcess:
        pid = 424_201
        returncode: int | None = None
        reaped = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if signal.SIGKILL not in signals:
                raise subprocess.TimeoutExpired(cmd="runtime-child", timeout=timeout)
            self.returncode = -signal.SIGKILL
            self.reaped = True
            return self.returncode

    process = HungProcess()

    def create_process(_command: Any, **options: Any) -> HungProcess:
        popen_options.update(options)
        return process

    def signal_group(_pid: int, signum: int) -> None:
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        if signum == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(subprocess, "Popen", create_process)
    monkeypatch.setattr(os, "killpg", signal_group)

    result = _run_runtime_child_process(
        (sys.executable, "-c", "pass"),
        wall_timeout_s=1e-9,
        terminate_grace_s=1e-9,
    )

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.reaped is True
    assert result == _RuntimeChildProcessResult(process_return_code=-signal.SIGKILL, status_payload=b"")
    assert popen_options["start_new_session"] is True
    assert len(popen_options["pass_fds"]) == 1


def test_subprocess_runner_keyboard_interrupt_kills_group_and_prevents_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    inherited_status_fd: int | None = None
    group_alive = True

    class InterruptedProcess:
        pid = 424_202
        returncode: int | None = None
        reaped = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if not signals:
                raise KeyboardInterrupt
            self.returncode = -signal.SIGINT
            self.reaped = True
            return self.returncode

    process = InterruptedProcess()

    def create_process(command: Any, **_options: Any) -> InterruptedProcess:
        nonlocal inherited_status_fd
        inherited_status_fd = int(command[-1])
        return process

    def signal_group(_pid: int, signum: int) -> None:
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        group_alive = False

    monkeypatch.setattr(subprocess, "Popen", create_process)
    monkeypatch.setattr(os, "killpg", signal_group)

    with pytest.raises(KeyboardInterrupt):
        _run_runtime_child_process(
            (sys.executable, "-c", "pass"),
            wall_timeout_s=10.0,
            terminate_grace_s=0.1,
        )

    assert signals == [signal.SIGINT]
    assert process.reaped is True
    assert inherited_status_fd is not None
    with pytest.raises(OSError):
        os.fstat(inherited_status_fd)


def test_subprocess_runner_forwards_sigterm_to_child_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    installed_handlers: dict[int, Any] = {}
    signals: list[int] = []
    group_alive = True

    class TerminatedProcess:
        pid = 424_203
        returncode: int | None = None
        reaped = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.returncode = -signal.SIGTERM
            self.reaped = True
            return self.returncode

    process = TerminatedProcess()

    def install_handler(signum: int, handler: Any) -> None:
        installed_handlers[signum] = handler

    def signal_group(_pid: int, signum: int) -> None:
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        group_alive = False

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(os, "killpg", signal_group)
    monkeypatch.setattr(signal, "signal", install_handler)

    with pytest.raises(KeyboardInterrupt):
        _run_runtime_child_process(
            (sys.executable, "-c", "pass"),
            wall_timeout_s=10.0,
            terminate_grace_s=0.1,
        )

    assert signals == [signal.SIGTERM]
    assert process.reaped is True


def test_subprocess_runner_kills_descendant_that_outlives_successful_direct_child(tmp_path: Path) -> None:
    descendant_record = tmp_path / "descendant.txt"
    descendant_program = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.write(1,b'R'); time.sleep(60)"
    )
    direct_program = (
        "import os,pathlib,subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{descendant_program!r}],stdout=subprocess.PIPE); "
        "assert child.stdout.read(1)==b'R'; "
        "pathlib.Path(sys.argv[1]).write_text(f'{child.pid} {os.getpgid(child.pid)}',encoding='utf-8'); "
        "fd=int(sys.argv[-1]); "
        'os.write(fd,b\'{"exit_code":0,"protocol_version":1}\\n\'); '
        "os.close(fd)"
    )

    try:
        with pytest.raises(RuntimeError, match="descendants remained"):
            _run_runtime_child_process(
                (sys.executable, "-c", direct_program, str(descendant_record)),
                wall_timeout_s=5.0,
                terminate_grace_s=0.25,
            )
    finally:
        if descendant_record.exists():
            descendant_pid = int(descendant_record.read_text(encoding="utf-8").split()[0])
            with suppress(ProcessLookupError):
                os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"", "child_status_missing"),
        (b"not-json\n", "child_status_malformed"),
        (b'{"exit_code":true,"protocol_version":1}\n', "child_status_malformed"),
        (b'{"exit_code":0,"protocol_version":99}\n', "child_status_malformed"),
        (b'{"exit_code":0,"exit_code":8,"protocol_version":1}\n', "child_status_malformed"),
        (b"x" * 513, "child_status_malformed"),
    ],
)
def test_parent_refuses_missing_or_malformed_terminal_status(
    tmp_path: Path,
    payload: bytes,
    expected_code: str,
) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    untrusted_success = _RuntimeChildProcessResult(
        process_return_code=0,
        status_payload=payload,
    )

    exit_code, stdout, stderr = _run_parent(harness, lambda _command: untrusted_success)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["category"] == "runtime_handoff"
    assert error["code"] == expected_code
    assert error["details"] == {
        "process_return_code": 0,
        "status_bytes": len(payload),
    }
    assert "Traceback" not in stderr


def test_runtime_child_launches_app_before_compile_and_refuses_digest_mismatch(
    tmp_path: Path,
) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))

    exit_code, stdout, stderr = _run_child(harness, expected_digest="d" * 64)

    assert exit_code == ExitCode.RUNTIME_SETUP_FAILED
    assert stdout == ""
    assert harness.events == ["load:app_launcher", "launch:app", "compile", "close:app"]
    error = json.loads(stderr)["error"]
    assert error["category"] == "runtime_handoff"
    assert error["code"] == "compiled_task_digest_mismatch"
    assert error["details"] == {
        "actual_compiled_task_digest": harness.resolved.digest,
        "expected_compiled_task_digest": "d" * 64,
    }


def test_happy_path_wires_runtime_in_order_and_closes_owned_resources(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))

    exit_code, stdout, stderr = _run_child(
        harness,
        "--device",
        "cuda:2",
        "--enable-cameras",
    )

    assert exit_code == ExitCode.SUCCESS
    assert stderr == ""
    result = json.loads(stdout)
    assert result["status"] == "completed"
    assert result["generation"]["target_reached"] is True
    assert result["backend"]["selected_motion_backend"] == "curobo_v1"
    assert harness.events == [
        "load:app_launcher",
        "launch:app",
        "compile",
        "detect",
        "select",
        "load:runtime_stack",
        "reserve:outputs",
        "create:run_log_writer",
        "append:run_started",
        "build:arena_runtime",
        "project:goal",
        "create:attachment_state",
        "create:runtime",
        "create:success_verifier",
        "create:executor",
        "create:planner",
        "create:generator",
        "create:generation_request",
        "run:generation",
        "append:run_summary",
        "close:planner",
        "close:arena",
        "publish:outputs",
        "append:run_committed",
        "close:app",
    ]
    assert harness.app_options == {"device": "cuda:2", "enable_cameras": True, "headless": True}
    assert harness.arena_args.device == "cuda:2"
    assert harness.arena_args.placement_seed == 41
    assert harness.arena_args.seed == 41
    assert harness.arena_args.solve_relations is True
    assert harness.runtime_attachment_state is harness.attachment_state
    assert harness.executor_attachment_state is harness.attachment_state
    assert harness.planner_attachment_state is harness.attachment_state
    assert harness.generation_kwargs == {
        "base_seed": 41,
        "successful_episodes": 3,
        "goal": ("cube-inside-bowl",),
        "keep_failed": False,
        "max_attempts": 7,
        "num_envs": 1,
        "request_digest": "b" * 64,
        "expected_plan_backend": "schedulestream_custream",
    }
    assert harness.run_close is False
    assert [record["record_type"] for record in harness.run_log_records] == [
        "run_started",
        "run_summary",
        "run_committed",
    ]
    assert harness.run_log_records[0]["preflight"]["status"] == "passed"


def test_child_emits_flushes_and_publishes_status_before_app_close(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    stdout = _EventStream("stdout", harness.events)
    stderr = _EventStream("stderr", harness.events)

    def publish_status(status_fd: int | None, exit_code: int) -> None:
        assert status_fd is None
        harness.events.append(f"publish:status:{exit_code}")

    exit_code, _, error_output = _run_child(
        harness,
        terminal_status_writer=publish_status,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.SUCCESS
    assert error_output == ""
    emit_index = harness.events.index("emit:stdout")
    status_index = harness.events.index("publish:status:0")
    app_close_index = harness.events.index("close:app")
    assert harness.events.index("close:planner") < emit_index
    assert harness.events.index("close:arena") < emit_index
    assert harness.events.index("publish:outputs") < emit_index
    assert harness.events.index("flush:stdout", emit_index) < status_index
    assert harness.events.index("flush:stderr", emit_index) < status_index
    assert status_index < app_close_index
    assert app_close_index == len(harness.events) - 1


def test_child_marks_status_descriptor_non_inheritable_before_app_launch(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)

    def publish_status(status_fd: int | None, _exit_code: int) -> None:
        assert status_fd == write_fd
        assert os.get_inheritable(write_fd) is False

    try:
        exit_code, _, _ = _run_child(
            harness,
            "--status-fd",
            str(write_fd),
            terminal_status_writer=publish_status,
        )
    finally:
        os.close(write_fd)
        os.close(read_fd)

    assert exit_code == ExitCode.SUCCESS


def test_child_flushes_failure_status_before_app_close(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.run_error = RuntimeError("execution failed")
    stdout = _EventStream("stdout", harness.events)
    stderr = _EventStream("stderr", harness.events)

    def publish_status(_status_fd: int | None, exit_code: int) -> None:
        harness.events.append(f"publish:status:{exit_code}")

    exit_code, _, _ = _run_child(
        harness,
        terminal_status_writer=publish_status,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    emit_index = harness.events.index("emit:stderr")
    status_index = harness.events.index(f"publish:status:{int(ExitCode.GENERATION_INCOMPLETE)}")
    app_close_index = harness.events.index("close:app")
    assert harness.events.index("flush:stderr", emit_index) < status_index
    assert status_index < app_close_index
    assert app_close_index == len(harness.events) - 1


def test_incomplete_summary_is_reported_after_cleanup(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.summary = _Summary(target_reached=False)

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    assert stderr == ""
    result = json.loads(stdout)
    assert result["status"] == "incomplete"
    assert result["generation"]["stop_reason"] == "max_attempts"
    assert harness.events[-5:] == [
        "close:planner",
        "close:arena",
        "publish:outputs",
        "append:run_committed",
        "close:app",
    ]


def test_semantic_validation_failure_prevents_probe_and_app_import(tmp_path: Path) -> None:
    events: list[str] = []
    issue = ValidationIssue(("generation", "successful_episodes"), "invalid_integer", "must be positive")

    def compiler(_path: Path) -> Any:
        events.append("compile")
        raise AutonomousValidationError([issue])

    def forbidden() -> Any:
        raise AssertionError("post-compilation dependency must not be called")

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run_cli(
        [str(tmp_path / "request.yaml"), "--json"],
        compiler=compiler,
        capability_detector=forbidden,
        backend_selector=lambda _requested, _capabilities: forbidden(),
        app_launcher_factory_loader=forbidden,
        runtime_stack_loader=forbidden,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.REQUEST_COMPILATION_FAILED
    assert stdout.getvalue() == ""
    assert events == ["compile"]
    error = json.loads(stderr.getvalue())["error"]
    assert error["details"]["issues"] == [issue.to_dict()]


def test_runtime_preflight_failure_prevents_app_import(tmp_path: Path) -> None:
    resolved = _ResolvedRequest(tmp_path)
    events: list[str] = []

    def detector() -> Any:
        events.append("detect")
        raise RuntimeError("probe unavailable\nwithout mutating the host")

    def forbidden() -> Any:
        raise AssertionError("AppLauncher must not be imported after failed preflight")

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run_cli(
        [str(tmp_path / "request.yaml"), "--json"],
        compiler=lambda _path: resolved,
        capability_detector=detector,
        backend_selector=lambda _requested, _capabilities: forbidden(),
        app_launcher_factory_loader=forbidden,
        runtime_stack_loader=forbidden,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.RUNTIME_PREFLIGHT_FAILED
    assert stdout.getvalue() == ""
    assert events == ["detect"]
    error = json.loads(stderr.getvalue())["error"]
    assert error["details"]["requested_motion_backend"] == "auto"
    assert error["details"]["capabilities"] == {"status": "probe_failed_before_result"}
    assert error["message"] == "probe unavailable without mutating the host"
    assert "Traceback" not in stderr.getvalue()


def test_malformed_backend_selection_is_structured_and_never_reaches_child(tmp_path: Path) -> None:
    resolved = _ResolvedRequest(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_cli(
        [str(tmp_path / "request.yaml"), "--json"],
        compiler=lambda _path: resolved,
        capability_detector=_Capabilities,
        backend_selector=lambda _requested, capabilities: SimpleNamespace(
            capabilities=capabilities,
            motion_backend="unreviewed_backend",
            schedulestream_application="custream",
        ),
        child_process_runner=lambda _command: (_ for _ in ()).throw(AssertionError("child launched")),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.RUNTIME_PREFLIGHT_FAILED
    assert stdout.getvalue() == ""
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "backend_incompatible"
    assert error["message"] == "backend selector returned no supported motion backend"


@pytest.mark.parametrize("occupied_target", ("dataset", "failed_dataset", "run_log"))
def test_existing_output_prevents_app_launch(tmp_path: Path, occupied_target: str) -> None:
    resolved = _ResolvedRequest(tmp_path)
    if occupied_target == "failed_dataset":
        resolved.output.keep_failed = True
        dataset_path = resolved.output.dataset
        target = dataset_path.with_name(f"{dataset_path.stem}_failed{dataset_path.suffix}")
    else:
        target = getattr(resolved.output, occupied_target)
    target.parent.mkdir(parents=True)
    target.write_text("human-owned\n", encoding="utf-8")
    harness = _RuntimeHarness(resolved)

    exit_code, stdout, stderr = _run_parent(
        harness,
        lambda _command: (_ for _ in ()).throw(AssertionError("child launched")),
    )

    assert exit_code == ExitCode.OUTPUT_CONFLICT
    assert stdout == ""
    assert harness.events == ["compile", "detect", "select"]
    assert target.read_text(encoding="utf-8") == "human-owned\n"
    assert json.loads(stderr)["error"]["code"] == "output_conflict"


def test_runtime_child_rechecks_output_after_digest_attestation(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.resolved.output.dataset.parent.mkdir(parents=True)
    harness.resolved.output.dataset.write_text("appeared-after-parent-preflight\n", encoding="utf-8")

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.OUTPUT_CONFLICT
    assert stdout == ""
    assert harness.events == [
        "load:app_launcher",
        "launch:app",
        "compile",
        "detect",
        "select",
        "close:app",
    ]
    assert json.loads(stderr)["error"]["code"] == "output_conflict"


def test_app_launch_failure_does_not_load_heavy_runtime_stack(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))

    def fail_launch() -> Any:
        harness.events.append("load:app_launcher")

        def launcher(_options: dict[str, Any]) -> Any:
            harness.events.append("launch:app")
            raise RuntimeError("Kit startup failed")

        return launcher

    harness.load_app_launcher = fail_launch  # type: ignore[method-assign]

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.APP_LAUNCH_FAILED
    assert stdout == ""
    assert harness.events[-2:] == ["load:app_launcher", "launch:app"]
    assert "load:runtime_stack" not in harness.events
    assert "close:" not in " ".join(harness.events)
    assert "Traceback" not in stderr


def test_runtime_setup_failure_closes_arena_and_app(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.runtime_error = RuntimeError("runtime binding rejected")

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.RUNTIME_SETUP_FAILED
    assert stdout == ""
    assert harness.events[-3:] == ["close:arena", "append:run_aborted", "close:app"]
    assert harness.run_log_records[-1]["record_type"] == "run_aborted"
    assert "create:planner" not in harness.events
    assert json.loads(stderr)["error"]["details"]["exception_type"] == "RuntimeError"


def test_generation_failure_closes_planner_arena_and_app(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.run_error = RuntimeError("attempt execution failed\nwith bounded diagnostics")

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.GENERATION_INCOMPLETE
    assert stdout == ""
    assert harness.events[-4:] == [
        "close:planner",
        "close:arena",
        "append:run_aborted",
        "close:app",
    ]
    assert harness.run_log_records[-1]["record_type"] == "run_aborted"
    error = json.loads(stderr)["error"]
    assert error["category"] == "generation"
    assert error["message"].endswith("attempt execution failed with bounded diagnostics")
    assert "Traceback" not in stderr


def test_publication_failure_is_terminally_recorded_before_status(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.publish_error = ValueError("staged dataset schema rejected")

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    assert harness.events[-4:] == [
        "close:arena",
        "publish:outputs",
        "append:run_aborted",
        "close:app",
    ]
    assert harness.run_log_records[-1]["record_type"] == "run_aborted"
    error = json.loads(stderr)["error"]
    assert error["category"] == "output_publication"
    assert error["code"] == "dataset_publication_failed"


def test_ambiguous_commit_does_not_write_the_ledger_again(tmp_path: Path) -> None:
    class DatasetCommitUncertainError(RuntimeError):
        pass

    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.publish_error = DatasetCommitUncertainError("ledger append durability is unknown")

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    publish_index = harness.events.index("publish:outputs")
    assert not any(event.startswith("append:") for event in harness.events[publish_index + 1 :])
    error = json.loads(stderr)["error"]
    assert error["code"] == "dataset_commit_uncertain"
    assert "inspect" in error["details"]["recovery"]


def test_ambiguous_run_log_commit_does_not_write_the_ledger_again(tmp_path: Path) -> None:
    class RunLogWriteUncertainError(RuntimeError):
        pass

    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.run_log_append_error = RunLogWriteUncertainError("commit record fsync outcome is unknown")
    harness.run_log_append_error_record_type = "run_committed"

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    ambiguous_append_index = harness.events.index("append:run_committed")
    assert not any(event.startswith("append:") for event in harness.events[ambiguous_append_index + 1 :])
    error = json.loads(stderr)["error"]
    assert error["category"] == "run_log"
    assert error["code"] == "run_log_write_uncertain"
    assert error["details"]["phase"] == "output_publication"
    assert "Inspect its existing JSONL tail" in error["details"]["recovery"]


def test_ambiguous_terminal_failure_append_replaces_prior_failure_without_retry(tmp_path: Path) -> None:
    class RunLogWriteUncertainError(RuntimeError):
        pass

    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.run_error = RuntimeError("generation stopped")
    harness.run_log_append_error = RunLogWriteUncertainError("abort record may already be durable")
    harness.run_log_append_error_record_type = "run_aborted"

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.INTERNAL_ERROR
    assert stdout == ""
    ambiguous_append_index = harness.events.index("append:run_aborted")
    assert not any(event.startswith("append:") for event in harness.events[ambiguous_append_index + 1 :])
    error = json.loads(stderr)["error"]
    assert error["code"] == "run_log_write_uncertain"
    assert error["details"]["phase"] == "terminal_run_log"
    assert error["details"]["preceding_failure"]["code"] == "generation_failed"


def test_cleanup_failure_attempts_all_callbacks_and_returns_exit_nine(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.close_errors = {
        "arena": RuntimeError("arena close failed"),
        "planner": RuntimeError("planner close failed"),
    }

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.CLEANUP_FAILED
    assert stdout == ""
    assert harness.events[-4:] == [
        "close:planner",
        "close:arena",
        "append:run_cleanup_failed",
        "close:app",
    ]
    assert harness.run_log_records[-1]["record_type"] == "run_cleanup_failed"
    error = json.loads(stderr)["error"]
    assert [item["resource"] for item in error["details"]["cleanup_failures"]] == [
        "episode_planner",
        "arena_runtime",
    ]
    assert error["details"]["generation_summary"]["target_reached"] is True


def test_operator_interrupt_returns_130_after_cleanup(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    harness.run_error = KeyboardInterrupt()

    exit_code, stdout, stderr = _run_child(harness)

    assert exit_code == ExitCode.INTERRUPTED
    assert stdout == ""
    assert harness.events[-4:] == [
        "close:planner",
        "close:arena",
        "append:run_interrupted",
        "close:app",
    ]
    assert harness.run_log_records[-1]["record_type"] == "run_interrupted"
    error = json.loads(stderr)["error"]
    assert error["category"] == "interrupted"
    assert error["details"]["phase"] == "generation"


def test_human_summary_contains_terminal_counts_and_output_paths(tmp_path: Path) -> None:
    harness = _RuntimeHarness(_ResolvedRequest(tmp_path))
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_cli(
        [
            str(tmp_path / "request.yaml"),
            "--runtime-child",
            "--expected-compiled-task-digest",
            harness.resolved.digest,
            "--expected-motion-backend",
            "curobo_v1",
            "--expected-schedulestream-application",
            "custream",
            "--expected-runtime-support-digest",
            _runtime_support_digest(harness.resolved),
        ],
        compiler=harness.compile,
        capability_detector=harness.detect,
        backend_selector=harness.select,
        app_launcher_factory_loader=harness.load_app_launcher,
        runtime_stack_loader=harness.load_runtime_stack,
        terminal_status_writer=lambda _status_fd, _exit_code: None,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == ExitCode.SUCCESS
    assert stderr.getvalue() == ""
    assert "Dataset generation completed." in stdout.getvalue()
    assert "attempts=4, successes=3, failures=1, requested_successes=3" in stdout.getvalue()
    assert str(harness.resolved.output.dataset) in stdout.getvalue()
    assert "Cleanup completed: true" in stdout.getvalue()


def test_direct_script_compilation_failure_is_structured_without_launching_isaac(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "isaac_autodata_examples/generate_task_dataset.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.yaml"), "--json"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == ExitCode.REQUEST_COMPILATION_FAILED
    assert completed.stdout == ""
    error = json.loads(completed.stderr)["error"]
    assert error["category"] == "request_compilation"
    assert error["details"]["issues"][0]["code"] == "file_not_found"
