#!/usr/bin/env python
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run bounded source-free dataset generation from one AutoData task request.

Semantic compilation and the import-free cuRobo/ScheduleStream capability preflight finish in a
parent process before Isaac AppLauncher is imported in a fresh child. The child launches Isaac
first, recompiles and digest-attests the request, then constructs the linked Arena environment,
creates the selected ScheduleStream planner, and executes bounded attempts. A private pipe carries
terminal status before SimulationApp shutdown, whose process exit code is not trusted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, TextIO

_RUNTIME_STATUS_PROTOCOL_VERSION = 1
_MAX_RUNTIME_STATUS_BYTES = 512

# The admitted live profile permits up to 50 attempts, with a planner budget of up to 60 seconds
# per solve and several solves per attempt. Six hours leaves conservative simulator/controller
# headroom while still giving the parent a hard upper bound. Shutdown gets a separate short grace
# period before the complete child process group is killed.
_RUNTIME_CHILD_WALL_TIMEOUT_S = 6 * 60 * 60
_RUNTIME_CHILD_TERMINATE_GRACE_S = 30.0
_RUNTIME_CHILD_WAIT_POLL_S = 0.25

# Direct script execution puts only the examples directory on sys.path. Keep the documented
# repository-local invocation working without requiring an editable installation.
if __package__ in (None, ""):
    _REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, _REPOSITORY_ROOT)


class ExitCode(IntEnum):
    """Stable process exits exposed by the dataset generation boundary."""

    SUCCESS = 0
    USAGE = 2
    REQUEST_COMPILATION_FAILED = 3
    RUNTIME_PREFLIGHT_FAILED = 4
    OUTPUT_CONFLICT = 5
    APP_LAUNCH_FAILED = 6
    RUNTIME_SETUP_FAILED = 7
    GENERATION_INCOMPLETE = 8
    CLEANUP_FAILED = 9
    INTERNAL_ERROR = 10
    INTERRUPTED = 130


@dataclass(frozen=True)
class RuntimeStack:
    """Heavy runtime callables loaded only after SimulationApp starts."""

    output_transaction_factory: Callable[..., Any]
    arena_runtime_builder: Callable[..., Any]
    goal_projector: Callable[[Any], tuple[Any, ...]]
    attachment_state_factory: Callable[[], Any]
    runtime_factory: Callable[..., Any]
    success_verifier_factory: Callable[[Any], Any]
    executor_factory: Callable[..., Any]
    planner_factory: Callable[..., Any]
    run_log_writer_factory: Callable[..., Any]
    generator_factory: Callable[..., Any]
    generation_request_factory: Callable[..., Any]
    run_loop: Callable[..., Any]


@dataclass(frozen=True)
class _RuntimeChildProcessResult:
    """Raw status-channel result returned by the real subprocess runner."""

    process_return_code: int
    status_payload: bytes


@dataclass(frozen=True)
class CleanupIssue:
    """One bounded failure from an owned resource closer."""

    resource: str
    exception_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a safe JSON-compatible cleanup issue."""

        return {
            "exception_type": self.exception_type,
            "message": self.message,
            "resource": self.resource,
        }


class _CleanupStack:
    """Close acquired resources in reverse order while attempting every callback."""

    def __init__(self) -> None:
        self._callbacks: list[tuple[str, Callable[[], None]]] = []

    def push(self, resource: str, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError(f"{resource} close callback must be callable")
        self._callbacks.append((resource, callback))

    def close(self) -> tuple[CleanupIssue, ...]:
        issues: list[CleanupIssue] = []
        while self._callbacks:
            resource, callback = self._callbacks.pop()
            try:
                callback()
            except Exception as exc:
                issues.append(
                    CleanupIssue(
                        resource=resource,
                        exception_type=type(exc).__name__,
                        message=_safe_exception_message(exc),
                    )
                )
        return tuple(issues)


@dataclass(frozen=True)
class CliFailure:
    """Structured terminal failure emitted by the process boundary."""

    category: str
    code: str
    exit_code: ExitCode
    message: str
    details: Mapping[str, Any]

    def with_cleanup(self, issues: tuple[CleanupIssue, ...]) -> CliFailure:
        details = dict(self.details)
        details["cleanup_failures"] = [issue.to_dict() for issue in issues]
        return CliFailure(
            category=self.category,
            code=self.code,
            exit_code=self.exit_code,
            message=self.message,
            details=details,
        )


class _CliAbort(Exception):
    """Carry one already-classified child failure through the cleanup boundary."""

    def __init__(self, failure: CliFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class _RuntimeChildStatusError(ValueError):
    """A missing, malformed, or untrusted child terminal-status record."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _run_gui_generation_inline(awaitable: Awaitable[Any]) -> Any:
    """Run the current synchronous Isaac generation stack without claiming Kit's event loop.

    Kit advances its own asyncio loop while GUI simulation steps update the application. The live
    AutoData runtime uses async interfaces for orchestration and cancellation, but its Isaac calls
    do not suspend. Driving that coroutine inline lets Kit retain main-thread loop ownership. A
    future genuinely asynchronous runtime must provide a different integration instead of silently
    moving simulator work to another thread.

    Args:
        awaitable: Generation operation that must complete without suspending.
    """

    iterator = awaitable.__await__()
    try:
        next(iterator)
    except StopIteration as completion:
        return completion.value
    except BaseException:
        with suppress(BaseException):
            iterator.close()
        raise

    with suppress(BaseException):
        iterator.close()
    raise RuntimeError(
        "GUI generation unexpectedly suspended; Kit owns the main-thread event loop and the "
        "current Isaac generation runtime must complete inline"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the complete lightweight CLI parser without importing Isaac or planner packages."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0   requested successful episode target reached\n"
            "  2   command-line usage error\n"
            "  3   request validation or Arena compilation failed\n"
            "  4   cuRobo/ScheduleStream runtime preflight failed\n"
            "  5   an output target already exists or is unsafe\n"
            "  6   Isaac AppLauncher failed\n"
            "  7   Arena runtime, planner, or executor setup failed\n"
            "  8   generation failed, exhausted its bound, or stopped unrecoverably\n"
            "  9   one or more owned resources failed to close\n"
            "  10  internal execution or process-handoff error\n"
            "  130 interrupted by the operator"
        ),
    )
    parser.add_argument("request", type=Path, help="Path to an AutoData v1 task request YAML.")
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        dest="output_format",
        help="Summary and error format (default: human).",
    )
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output_format",
        help="Shorthand for --format json.",
    )
    parser.add_argument(
        "--device",
        type=_device_argument,
        default="cuda:0",
        help="Isaac simulation device: cpu, cuda, or cuda:N (default: cuda:0).",
    )
    display_mode = parser.add_mutually_exclusive_group()
    display_mode.add_argument(
        "--gui",
        action="store_false",
        dest="headless",
        help="Launch the Kit GUI instead of running headless.",
    )
    display_mode.add_argument(
        "--headless",
        action="store_true",
        dest="headless",
        help="Run without the Kit GUI (default).",
    )
    display_mode.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--enable-cameras",
        action="store_true",
        help="Enable Arena camera sensors in headless or GUI execution.",
    )
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-compiled-task-digest",
        type=_sha256_argument,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-motion-backend",
        choices=("curobo_v1", "curobo_v2"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-schedulestream-application",
        choices=("custream", "custream2"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-runtime-support-digest",
        type=_sha256_argument,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--status-fd",
        type=_status_fd_argument,
        help=argparse.SUPPRESS,
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    compiler: Callable[[Path], Any] | None = None,
    capability_detector: Callable[[], Any] | None = None,
    backend_selector: Callable[[str, Any], Any] | None = None,
    child_process_runner: (
        Callable[
            [Sequence[str]],
            int | _RuntimeChildProcessResult,
        ]
        | None
    ) = None,
    app_launcher_factory_loader: Callable[[], Callable[[Mapping[str, Any]], Any]] | None = None,
    runtime_stack_loader: Callable[[], RuntimeStack] | None = None,
    terminal_status_writer: Callable[[int | None, int], None] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Validate in the parent, then execute in an App-first fresh child process."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    if args.runtime_child:
        missing = [
            option
            for option, value in (
                ("--expected-compiled-task-digest", args.expected_compiled_task_digest),
                ("--expected-motion-backend", args.expected_motion_backend),
                (
                    "--expected-schedulestream-application",
                    args.expected_schedulestream_application,
                ),
                (
                    "--expected-runtime-support-digest",
                    args.expected_runtime_support_digest,
                ),
            )
            if value is None
        ]
        if args.status_fd is None and terminal_status_writer is None:
            missing.append("--status-fd")
        if missing:
            parser.error(f"--runtime-child requires {', '.join(missing)}")
        return _run_runtime_child(
            args,
            compiler=compiler,
            capability_detector=capability_detector,
            backend_selector=backend_selector,
            app_launcher_factory_loader=app_launcher_factory_loader,
            runtime_stack_loader=runtime_stack_loader,
            terminal_status_writer=terminal_status_writer,
            stdout=stdout,
            stderr=stderr,
        )

    unexpected_internal_options = [
        option
        for option, value in (
            ("--expected-compiled-task-digest", args.expected_compiled_task_digest),
            ("--expected-motion-backend", args.expected_motion_backend),
            (
                "--expected-schedulestream-application",
                args.expected_schedulestream_application,
            ),
            (
                "--expected-runtime-support-digest",
                args.expected_runtime_support_digest,
            ),
            ("--status-fd", args.status_fd),
        )
        if value is not None
    ]
    if unexpected_internal_options:
        parser.error(f"internal options require --runtime-child: {', '.join(unexpected_internal_options)}")

    return _run_parent(
        args,
        compiler=compiler,
        capability_detector=capability_detector,
        backend_selector=backend_selector,
        child_process_runner=child_process_runner,
        stdout=stdout,
        stderr=stderr,
    )


def _run_parent(
    args: argparse.Namespace,
    *,
    compiler: Callable[[Path], Any] | None,
    capability_detector: Callable[[], Any] | None,
    backend_selector: Callable[[str, Any], Any] | None,
    child_process_runner: (
        Callable[
            [Sequence[str]],
            int | _RuntimeChildProcessResult,
        ]
        | None
    ),
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Compile and preflight without Isaac, then launch the isolated runtime child."""

    del stdout

    from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError

    if compiler is None:
        from isaac_autodata_interfaces.autonomous.task_compiler import compile_task_request

        compiler = compile_task_request
    try:
        resolved = compiler(args.request)
    except AutonomousValidationError as exc:
        failure = CliFailure(
            category="request_compilation",
            code="request_invalid",
            exit_code=ExitCode.REQUEST_COMPILATION_FAILED,
            message="The task request could not be compiled.",
            details={"issues": [issue.to_dict() for issue in exc.issues]},
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)
    except Exception as exc:
        failure = _unexpected_failure("request_compilation", exc)
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)

    requested_motion_backend = "unknown"
    capabilities: Any | None = None
    try:
        if capability_detector is None or backend_selector is None:
            from isaac_autodata_interfaces.motion_planners.curobo.backend_selection import (
                detect_curobo_runtime,
                select_schedulestream_backend,
            )

            capability_detector = capability_detector or detect_curobo_runtime
            backend_selector = backend_selector or select_schedulestream_backend
        requested_motion_backend = resolved.planner.motion_backend.value
        capabilities = capability_detector()
        compatibility = backend_selector(requested_motion_backend, capabilities)
        preflight = _preflight_to_dict(requested_motion_backend, compatibility)
    except Exception as exc:
        failure = CliFailure(
            category="runtime_preflight",
            code="backend_incompatible",
            exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
            message=_safe_exception_message(exc),
            details={
                "capabilities": _capabilities_to_dict(capabilities),
                "requested_motion_backend": requested_motion_backend,
                "remediation": (
                    "Use a reviewed runtime image containing the matching cuRobo and "
                    "ScheduleStream application. The host environment will not be modified."
                ),
            },
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)

    try:
        _attach_runtime_profile(resolved, preflight)
    except AutonomousValidationError as exc:
        failure = _runtime_capability_failure(exc)
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)
    except Exception as exc:
        failure = _unexpected_failure("runtime_capability", exc)
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)

    try:
        _require_fresh_output_targets(resolved)
    except Exception as exc:
        failure = CliFailure(
            category="output_preflight",
            code="output_conflict",
            exit_code=ExitCode.OUTPUT_CONFLICT,
            message=_safe_exception_message(exc),
            details={
                "dataset": str(resolved.output.dataset),
                "run_log": None if resolved.output.run_log is None else str(resolved.output.run_log),
            },
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)

    try:
        command = _runtime_child_command(args, resolved.digest, preflight)
    except Exception as exc:
        failure = _unexpected_failure("runtime_handoff", exc)
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)
    process_runner = child_process_runner or _run_runtime_child_process
    try:
        child_result = process_runner(command)
    except KeyboardInterrupt:
        failure = CliFailure(
            category="interrupted",
            code="operator_interrupt",
            exit_code=ExitCode.INTERRUPTED,
            message="Dataset generation was interrupted by the operator.",
            details={"phase": "runtime_child"},
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)
    except Exception as exc:
        failure = CliFailure(
            category="runtime_handoff",
            code="child_launch_failed",
            exit_code=ExitCode.INTERNAL_ERROR,
            message=f"The isolated runtime child could not be launched: {_safe_exception_message(exc)}",
            details={"exception_type": type(exc).__name__},
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)

    if isinstance(child_result, _RuntimeChildProcessResult):
        try:
            child_exit_code = _decode_runtime_status(child_result.status_payload)
        except _RuntimeChildStatusError as exc:
            failure = CliFailure(
                category="runtime_handoff",
                code=exc.code,
                exit_code=ExitCode.INTERNAL_ERROR,
                message=_safe_exception_message(exc),
                details={
                    "process_return_code": child_result.process_return_code,
                    "status_bytes": len(child_result.status_payload),
                },
            )
            _emit_failure(stderr, args.output_format, failure)
            return int(failure.exit_code)
        if child_exit_code == int(ExitCode.SUCCESS) and child_result.process_return_code != 0:
            failure = CliFailure(
                category="runtime_handoff",
                code="child_success_process_failed",
                exit_code=ExitCode.INTERNAL_ERROR,
                message="The runtime child reported success but then exited abnormally.",
                details={"process_return_code": child_result.process_return_code},
            )
            _emit_failure(stderr, args.output_format, failure)
            return int(failure.exit_code)
    else:
        # Integer results are retained only as a trusted dependency-injection seam for host tests.
        child_exit_code = child_result

    known_exit_codes = {int(code) for code in ExitCode}
    if isinstance(child_exit_code, bool) or not isinstance(child_exit_code, int):
        child_exit_code = -1
    if child_exit_code not in known_exit_codes:
        failure = CliFailure(
            category="runtime_handoff",
            code="child_exit_invalid",
            exit_code=ExitCode.INTERNAL_ERROR,
            message="The isolated runtime child exited without a recognized AutoData status.",
            details={"child_exit_code": child_exit_code},
        )
        _emit_failure(stderr, args.output_format, failure)
        return int(failure.exit_code)
    return child_exit_code


def _run_runtime_child(  # noqa: C901 - explicit phase/cleanup boundary is intentionally linear
    args: argparse.Namespace,
    *,
    compiler: Callable[[Path], Any] | None,
    capability_detector: Callable[[], Any] | None,
    backend_selector: Callable[[str, Any], Any] | None,
    app_launcher_factory_loader: Callable[[], Callable[[Mapping[str, Any]], Any]] | None,
    runtime_stack_loader: Callable[[], RuntimeStack] | None,
    terminal_status_writer: Callable[[int | None, int], None] | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Launch Isaac first, recompile and attest the request, then run bounded generation."""

    cleanup = _CleanupStack()
    summary: Any | None = None
    run_log_writer: Any | None = None
    output_transaction: Any | None = None
    request_anchor: Any | None = None
    published_artifacts: tuple[Any, ...] = ()
    resolved: Any | None = None
    preflight: dict[str, Any] | None = None
    primary_failure: CliFailure | None = None
    simulation_app: Any | None = None
    phase = "app_launch"

    try:
        if args.status_fd is not None:
            phase = "runtime_handoff"
            os.set_inheritable(args.status_fd, False)
        phase = "app_launch"
        launcher_loader = app_launcher_factory_loader or _load_app_launcher_factory
        app_launcher_factory = launcher_loader()
        launcher = app_launcher_factory(_app_launcher_options(args))
        simulation_app = launcher.app

        phase = "request_compilation"
        from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError

        if compiler is None:
            from isaac_autodata_core.autonomous.output_transaction import RequestDirectoryAnchor
            from isaac_autodata_interfaces.autonomous.task_compiler import compile_task_request

            request_anchor = RequestDirectoryAnchor.open(args.request)
            compiler = compile_task_request
        try:
            resolved = compiler(args.request)
        except AutonomousValidationError as exc:
            raise _CliAbort(
                CliFailure(
                    category="request_compilation",
                    code="request_invalid_in_runtime_child",
                    exit_code=ExitCode.REQUEST_COMPILATION_FAILED,
                    message="The task request could not be recompiled in the runtime child.",
                    details={"issues": [issue.to_dict() for issue in exc.issues]},
                )
            ) from None
        except Exception as exc:
            raise _CliAbort(_unexpected_failure("request_compilation", exc)) from None
        if request_anchor is not None:
            request_anchor.verify_current()
        if resolved.digest != args.expected_compiled_task_digest:
            raise _CliAbort(
                CliFailure(
                    category="runtime_handoff",
                    code="compiled_task_digest_mismatch",
                    exit_code=ExitCode.RUNTIME_SETUP_FAILED,
                    message="The runtime child compiled a different task and refused to execute it.",
                    details={
                        "actual_compiled_task_digest": resolved.digest,
                        "expected_compiled_task_digest": args.expected_compiled_task_digest,
                    },
                )
            )

        phase = "runtime_preflight"
        requested_motion_backend = resolved.planner.motion_backend.value
        capabilities: Any | None = None
        try:
            if capability_detector is None or backend_selector is None:
                from isaac_autodata_interfaces.motion_planners.curobo.backend_selection import (
                    detect_curobo_runtime,
                    select_schedulestream_backend,
                )

                capability_detector = capability_detector or detect_curobo_runtime
                backend_selector = backend_selector or select_schedulestream_backend
            capabilities = capability_detector()
            compatibility = backend_selector(requested_motion_backend, capabilities)
            preflight = _preflight_to_dict(requested_motion_backend, compatibility)
        except Exception as exc:
            raise _CliAbort(
                _runtime_preflight_failure(
                    exc,
                    requested_motion_backend=requested_motion_backend,
                    capabilities=capabilities,
                )
            ) from None
        if (
            preflight["selected_motion_backend"] != args.expected_motion_backend
            or preflight["schedulestream_application"] != args.expected_schedulestream_application
        ):
            raise _CliAbort(
                CliFailure(
                    category="runtime_handoff",
                    code="runtime_selection_mismatch",
                    exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
                    message="The runtime child selected a different planner runtime and refused to execute it.",
                    details={
                        "actual_motion_backend": preflight["selected_motion_backend"],
                        "actual_schedulestream_application": preflight["schedulestream_application"],
                        "expected_motion_backend": args.expected_motion_backend,
                        "expected_schedulestream_application": args.expected_schedulestream_application,
                    },
                )
            )

        try:
            _attach_runtime_profile(resolved, preflight)
        except AutonomousValidationError as exc:
            raise _CliAbort(_runtime_capability_failure(exc)) from None
        if preflight["runtime_profile_digest"] != args.expected_runtime_support_digest:
            raise _CliAbort(
                CliFailure(
                    category="runtime_handoff",
                    code="runtime_support_mismatch",
                    exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
                    message="The runtime child admitted a different runtime-support profile.",
                    details={
                        "actual_runtime_support_digest": preflight["runtime_profile_digest"],
                        "expected_runtime_support_digest": args.expected_runtime_support_digest,
                    },
                )
            )

        phase = "output_preflight"
        try:
            _require_fresh_output_targets(resolved)
        except Exception as exc:
            raise _CliAbort(_output_conflict_failure(resolved, exc)) from None

        phase = "runtime_setup"
        stack = (runtime_stack_loader or _load_runtime_stack)()
        transaction_arguments = {
            "dataset_path": resolved.output.dataset,
            "run_log_path": resolved.output.run_log,
            "keep_failed": resolved.output.keep_failed,
        }
        if request_anchor is None:
            transaction_arguments["request_directory"] = Path(args.request).expanduser().resolve(strict=False).parent
        else:
            transaction_arguments["request_anchor"] = request_anchor
        output_transaction = stack.output_transaction_factory(**transaction_arguments)
        if resolved.output.run_log is not None:
            run_log_writer = output_transaction.open_run_log_writer(stack.run_log_writer_factory)
            run_log_writer.append(
                _run_started_record(
                    resolved,
                    preflight,
                )
            )
        runtime_args = _arena_runtime_args(args, resolved)
        bundle = stack.arena_runtime_builder(
            resolved,
            runtime_args,
            recording_targets=output_transaction.recording_targets,
            allow_output_overwrite=False,
        )
        cleanup.push("arena_runtime", bundle.close)

        goal = stack.goal_projector(resolved)
        attachment_state = stack.attachment_state_factory()
        runtime = stack.runtime_factory(
            bundle.env,
            bundle.embodiment_adapter,
            graph_nodes=tuple(resolved.linked_graph.get("nodes", ())),
            attachment_state=attachment_state,
        )
        success_verifier = stack.success_verifier_factory(bundle.success_term)
        executor = stack.executor_factory(
            bundle.env,
            bundle.embodiment_adapter,
            success_verifier,
            attachment_state=attachment_state,
        )
        planner = stack.planner_factory(
            bundle,
            resolved,
            compatibility,
            attachment_state=attachment_state,
        )
        cleanup.push("episode_planner", planner.close)

        generator = stack.generator_factory(
            runtime,
            planner,
            executor,
            run_log_writer=run_log_writer,
        )
        generation_request = stack.generation_request_factory(
            request_digest=resolved.request_digest,
            goal=goal,
            successful_episodes=resolved.generation.successful_episodes,
            max_attempts=resolved.generation.max_attempts,
            base_seed=resolved.generation.seed,
            num_envs=resolved.generation.num_envs,
            keep_failed=resolved.output.keep_failed,
            expected_plan_backend=f"schedulestream_{preflight['schedulestream_application']}",
        )

        phase = "generation"
        generation = stack.run_loop(generator, generation_request, close=False)
        summary = asyncio.run(generation) if args.headless else _run_gui_generation_inline(generation)
        summary_dict = _summary_to_dict(summary)
        if run_log_writer is not None:
            run_log_writer.append({
                "record_type": "run_summary",
                "compiled_task_digest": resolved.digest,
                **summary_dict,
            })
    except _CliAbort as exc:
        primary_failure = exc.failure
    except KeyboardInterrupt:
        primary_failure = CliFailure(
            category="interrupted",
            code="operator_interrupt",
            exit_code=ExitCode.INTERRUPTED,
            message="Dataset generation was interrupted by the operator.",
            details={"phase": phase},
        )
    except Exception as exc:
        primary_failure = _phase_failure(phase, exc)
    finally:
        cleanup_issues = cleanup.close()

    if cleanup_issues:
        if primary_failure is None:
            primary_failure = CliFailure(
                category="cleanup",
                code="resource_cleanup_failed",
                exit_code=ExitCode.CLEANUP_FAILED,
                message="One or more owned runtime resources failed to close.",
                details={
                    "cleanup_failures": [issue.to_dict() for issue in cleanup_issues],
                    "generation_summary": None if summary is None else _summary_to_dict(summary),
                },
            )
        else:
            primary_failure = primary_failure.with_cleanup(cleanup_issues)

    if primary_failure is None:
        assert output_transaction is not None
        assert summary is not None
        assert resolved is not None
        phase = "output_publication"
        try:
            summary_dict = _summary_to_dict(summary)
            commit_record = (
                None
                if run_log_writer is None
                else {
                    "generation": summary_dict,
                    "record_type": "run_committed",
                    "request_digest": resolved.request_digest,
                    "compiled_task_digest": resolved.digest,
                }
            )
            published_artifacts = tuple(
                output_transaction.publish(
                    run_log_writer=run_log_writer,
                    commit_record=commit_record,
                    require_failed_dataset=bool(resolved.output.keep_failed and summary_dict["failures"]),
                    expected_successful_episodes=summary_dict["successes"],
                    expected_failed_episodes=(summary_dict["failures"] if resolved.output.keep_failed else None),
                )
            )
        except Exception as exc:
            primary_failure = _output_publication_failure(resolved, exc)

    # Any append/fsync ambiguity can mean the JSONL tail is partial or already durable. Never append
    # to that run log again in-process: a second write could concatenate onto a malformed line or
    # create contradictory terminal records. Recovery must inspect the existing tail.
    run_log_state_uncertain = primary_failure is not None and primary_failure.code in {
        "dataset_commit_uncertain",
        "run_log_write_uncertain",
    }
    if primary_failure is not None and run_log_writer is not None and not run_log_state_uncertain:
        try:
            run_log_writer.append(_terminal_failure_record(resolved, primary_failure))
        except Exception as exc:
            if type(exc).__name__ == "RunLogWriteUncertainError":
                primary_failure = _run_log_write_uncertain_failure(
                    "terminal_run_log",
                    exc,
                    preceding_failure=primary_failure,
                )
            else:
                primary_failure = primary_failure.with_cleanup((
                    CleanupIssue(
                        resource="terminal_run_log",
                        exception_type=type(exc).__name__,
                        message=_safe_exception_message(exc),
                    ),
                ))

    output_cleanup_issues = _close_output_resources(run_log_writer, output_transaction, request_anchor)
    if output_cleanup_issues:
        if primary_failure is None:
            primary_failure = CliFailure(
                category="cleanup",
                code="output_cleanup_failed",
                exit_code=ExitCode.CLEANUP_FAILED,
                message="One or more owned output resources failed to close.",
                details={
                    "cleanup_failures": [issue.to_dict() for issue in output_cleanup_issues],
                    "generation_summary": None if summary is None else _summary_to_dict(summary),
                },
            )
        else:
            primary_failure = primary_failure.with_cleanup(output_cleanup_issues)

    return _finish_runtime_child(
        args,
        simulation_app=simulation_app,
        summary=summary,
        resolved=resolved,
        preflight=preflight,
        published_artifacts=published_artifacts,
        primary_failure=primary_failure,
        terminal_status_writer=terminal_status_writer,
        stdout=stdout,
        stderr=stderr,
    )


def _finish_runtime_child(
    args: argparse.Namespace,
    *,
    simulation_app: Any | None,
    summary: Any | None,
    resolved: Any | None,
    preflight: Mapping[str, Any] | None,
    published_artifacts: tuple[Any, ...],
    primary_failure: CliFailure | None,
    terminal_status_writer: Callable[[int | None, int], None] | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Publish one terminal result before making SimulationApp shutdown the final operation."""

    if primary_failure is not None:
        terminal_exit_code = int(primary_failure.exit_code)
        try:
            _emit_failure(stderr, args.output_format, primary_failure)
        except Exception as exc:
            primary_failure = _unexpected_failure("terminal_output", exc)
            terminal_exit_code = int(primary_failure.exit_code)
            _emit_failure_best_effort(stderr, args.output_format, primary_failure)
    else:
        assert summary is not None
        assert resolved is not None
        assert preflight is not None
        try:
            result = _build_result(resolved, preflight, summary, published_artifacts=published_artifacts)
            if args.output_format == "json":
                _write_json(stdout, result)
            else:
                _write_human_summary(stdout, result)
            terminal_exit_code = int(ExitCode.SUCCESS if summary.target_reached else ExitCode.GENERATION_INCOMPLETE)
        except Exception as exc:
            primary_failure = _unexpected_failure("terminal_output", exc)
            terminal_exit_code = int(primary_failure.exit_code)
            _emit_failure(stderr, args.output_format, primary_failure)

    try:
        _flush_terminal_streams(stdout, stderr)
    except Exception as exc:
        primary_failure = CliFailure(
            category="terminal_output",
            code="terminal_flush_failed",
            exit_code=ExitCode.INTERNAL_ERROR,
            message=f"Terminal output could not be flushed: {_safe_exception_message(exc)}",
            details={"exception_type": type(exc).__name__},
        )
        terminal_exit_code = int(primary_failure.exit_code)
        _emit_failure_best_effort(stderr, args.output_format, primary_failure)
        _flush_best_effort(stdout, stderr)

    status_writer = terminal_status_writer or _write_runtime_status
    try:
        status_writer(args.status_fd, terminal_exit_code)
    except Exception as exc:
        status_failure = CliFailure(
            category="runtime_handoff",
            code="child_status_write_failed",
            exit_code=ExitCode.INTERNAL_ERROR,
            message=f"The runtime child could not publish terminal status: {_safe_exception_message(exc)}",
            details={"exception_type": type(exc).__name__},
        )
        terminal_exit_code = int(status_failure.exit_code)
        _emit_failure_best_effort(stderr, args.output_format, status_failure)
        _flush_best_effort(stdout, stderr)

    # SimulationApp.close() can terminate the process and never return. It must remain the final
    # operation, after ordinary resources, terminal output, and the out-of-band status record.
    if simulation_app is not None:
        simulation_app.close()
    return terminal_exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production dataset generation entrypoint."""

    return run_cli(argv)


def _runtime_child_command(
    args: argparse.Namespace,
    compiled_task_digest: str,
    preflight: Mapping[str, Any],
) -> tuple[str, ...]:
    """Build the complete allowlisted argv for the isolated, attested runtime child."""

    expected_digest = _sha256_argument(compiled_task_digest)
    motion_backend = preflight.get("selected_motion_backend")
    application = preflight.get("schedulestream_application")
    runtime_support_digest = preflight.get("runtime_profile_digest")
    if motion_backend not in ("curobo_v1", "curobo_v2"):
        raise ValueError("preflight returned no supported selected motion backend")
    if application not in ("custream", "custream2"):
        raise ValueError("preflight returned no supported ScheduleStream application")
    runtime_support_digest = _sha256_argument(runtime_support_digest)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        str(args.request.expanduser().resolve(strict=False)),
        "--runtime-child",
        "--expected-compiled-task-digest",
        expected_digest,
        "--expected-motion-backend",
        motion_backend,
        "--expected-schedulestream-application",
        application,
        "--expected-runtime-support-digest",
        runtime_support_digest,
        "--format",
        args.output_format,
        "--device",
        args.device,
        "--headless" if args.headless else "--no-headless",
    ]
    if args.enable_cameras:
        command.append("--enable-cameras")
    return tuple(command)


def _run_runtime_child_process(
    command: Sequence[str],
    *,
    wall_timeout_s: float = _RUNTIME_CHILD_WALL_TIMEOUT_S,
    terminate_grace_s: float = _RUNTIME_CHILD_TERMINATE_GRACE_S,
) -> _RuntimeChildProcessResult:
    """Supervise the child process group and its private bounded terminal-status pipe."""

    import os
    import signal
    import subprocess
    import time

    if "--status-fd" in command:
        raise ValueError("runtime child command must not provide its own status descriptor")
    if not command or any(not isinstance(argument, str) or "\x00" in argument for argument in command):
        raise ValueError("runtime child command contains an invalid argument")
    _validate_runtime_supervision_duration("wall_timeout_s", wall_timeout_s)
    _validate_runtime_supervision_duration("terminate_grace_s", terminate_grace_s)

    read_fd, write_fd = os.pipe()
    process: Any | None = None
    previous_signal_handlers: dict[int, Any] = {}
    forwarded_signals: list[int] = []
    cleanup_attempted = False
    try:
        child_command = (*command, "--status-fd", str(write_fd))
        try:
            process = subprocess.Popen(
                child_command,
                close_fds=True,
                pass_fds=(write_fd,),
                start_new_session=True,
            )
        finally:
            os.close(write_fd)
            write_fd = -1

        previous_signal_handlers = _install_runtime_signal_forwarders(process, forwarded_signals)
        deadline_s = time.monotonic() + wall_timeout_s
        try:
            timed_out = _wait_for_runtime_child(
                process,
                deadline_s=deadline_s,
                forwarded_signals=forwarded_signals,
            )
        except KeyboardInterrupt:
            cleanup_attempted = True
            _terminate_runtime_process_group(
                process,
                initial_signal=signal.SIGINT,
                terminate_grace_s=terminate_grace_s,
            )
            raise

        shutdown_interrupted = False
        if timed_out:
            cleanup_attempted = True
            shutdown_interrupted = _terminate_runtime_process_group(
                process,
                initial_signal=signal.SIGTERM,
                terminate_grace_s=terminate_grace_s,
            )
        elif forwarded_signals:
            # The temporary signal handler already forwarded the original signal verbatim.
            cleanup_attempted = True
            shutdown_interrupted = _terminate_runtime_process_group(
                process,
                initial_signal=None,
                terminate_grace_s=terminate_grace_s,
            )
        elif _runtime_process_group_exists(process.pid):
            # The direct child is the sole owner of this private session. A surviving group after
            # it exits is an unexpected descendant leak, even if the child published success.
            cleanup_attempted = True
            _terminate_runtime_process_group(
                process,
                initial_signal=signal.SIGTERM,
                terminate_grace_s=terminate_grace_s,
            )
            raise RuntimeError("runtime child exited while descendants remained in its process group")

        assert process.returncode is not None, "runtime child must be reaped before status collection"
        status_payload = _read_runtime_status_pipe(read_fd)
        result = _RuntimeChildProcessResult(
            process_return_code=process.returncode,
            status_payload=status_payload,
        )
        _restore_runtime_signal_handlers(previous_signal_handlers)
        previous_signal_handlers = {}
        if forwarded_signals or shutdown_interrupted:
            raise KeyboardInterrupt
        return result
    except BaseException:
        if (
            process is not None
            and not cleanup_attempted
            and (process.poll() is None or _runtime_process_group_exists(process.pid))
        ):
            cleanup_attempted = True
            _terminate_runtime_process_group(
                process,
                initial_signal=signal.SIGTERM,
                terminate_grace_s=terminate_grace_s,
            )
        raise
    finally:
        try:
            _restore_runtime_signal_handlers(previous_signal_handlers)
        finally:
            try:
                if write_fd >= 0:
                    os.close(write_fd)
            finally:
                os.close(read_fd)


def _validate_runtime_supervision_duration(name: str, value: float) -> None:
    """Reject supervision bounds that could disable or destabilize the parent guardrail."""

    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds")


def _install_runtime_signal_forwarders(process: Any, forwarded_signals: list[int]) -> dict[int, Any]:
    """Temporarily forward operator termination signals to the isolated child group."""

    import signal

    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        forwarded_signals.append(signum)
        _signal_runtime_process_group(process, signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handler = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
            previous_handlers[signum] = previous_handler
    except ValueError:
        # Python only permits signal-handler installation in the main thread. Direct callers in a
        # worker thread still get KeyboardInterrupt cleanup, but the production CLI runs here.
        _restore_runtime_signal_handlers(previous_handlers)
        return {}
    except BaseException:
        _restore_runtime_signal_handlers(previous_handlers)
        raise
    return previous_handlers


def _restore_runtime_signal_handlers(previous_handlers: Mapping[int, Any]) -> None:
    """Restore every signal disposition replaced by the runtime supervisor."""

    import signal

    for signum, previous_handler in previous_handlers.items():
        signal.signal(signum, previous_handler)


def _wait_for_runtime_child(
    process: Any,
    *,
    deadline_s: float,
    forwarded_signals: Sequence[int],
) -> bool:
    """Wait until exit, operator interruption, or the absolute runtime deadline."""

    import subprocess
    import time

    while process.poll() is None and not forwarded_signals:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0:
            return True
        try:
            process.wait(timeout=min(_RUNTIME_CHILD_WAIT_POLL_S, remaining_s))
        except subprocess.TimeoutExpired:
            continue
    return False


def _terminate_runtime_process_group(
    process: Any,
    *,
    initial_signal: int | None,
    terminate_grace_s: float,
) -> bool:
    """Gracefully stop, forcibly kill, and reap an isolated runtime process group."""

    import signal
    import subprocess
    import time

    def target_exists() -> bool:
        return process.poll() is None or _runtime_process_group_exists(process.pid)

    interrupted = False
    if target_exists() and initial_signal is not None:
        _signal_runtime_process_group(process, initial_signal)

    grace_deadline_s = time.monotonic() + terminate_grace_s
    while target_exists():
        remaining_s = grace_deadline_s - time.monotonic()
        if remaining_s <= 0:
            break
        poll_s = min(_RUNTIME_CHILD_WAIT_POLL_S, remaining_s)
        try:
            if process.poll() is None:
                process.wait(timeout=poll_s)
            else:
                time.sleep(poll_s)
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            interrupted = True
            _signal_runtime_process_group(process, signal.SIGINT)

    if target_exists():
        _signal_runtime_process_group(process, signal.SIGKILL)

    # SIGKILL cannot be handled. Verify the entire private group disappears within another bounded
    # grace window; reaping only the direct child is not evidence that descendants are gone.
    kill_deadline_s = time.monotonic() + terminate_grace_s
    while target_exists():
        remaining_s = kill_deadline_s - time.monotonic()
        poll_s = max(0.0, min(_RUNTIME_CHILD_WAIT_POLL_S, remaining_s))
        try:
            if process.poll() is None:
                process.wait(timeout=poll_s)
            elif remaining_s <= 0:
                raise RuntimeError("runtime process group remained alive after SIGKILL")
            time.sleep(poll_s)
        except subprocess.TimeoutExpired:
            if remaining_s <= 0:
                raise RuntimeError("runtime process group remained alive after SIGKILL") from None
            continue
        except KeyboardInterrupt:
            interrupted = True
            _signal_runtime_process_group(process, signal.SIGKILL)
    return interrupted


def _runtime_process_group_exists(process_group_id: int) -> bool:
    """Return whether the isolated runtime process group still has any member."""

    import os

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The group exists even if a changed credential prevents signaling it. Treat this as live
        # so supervision fails closed instead of claiming cleanup.
        return True
    return True


def _signal_runtime_process_group(process: Any, signum: int) -> None:
    """Signal the new-session process group, tolerating an exit race."""

    import os

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signum)


def _read_runtime_status_pipe(read_fd: int) -> bytes:
    """Read only currently available bounded bytes after the direct child has terminated."""

    import os

    os.set_blocking(read_fd, False)
    payload = bytearray()
    while len(payload) <= _MAX_RUNTIME_STATUS_BYTES:
        try:
            chunk = os.read(read_fd, _MAX_RUNTIME_STATUS_BYTES + 1 - len(payload))
        except BlockingIOError:
            break
        if not chunk:
            break
        payload.extend(chunk)
    return bytes(payload)


def _write_runtime_status(status_fd: int | None, exit_code: int) -> None:
    """Write and close one validated status record before SimulationApp termination."""

    if status_fd is None:
        return

    import os

    try:
        payload = _encode_runtime_status(exit_code)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(status_fd, remaining)
            if written <= 0:
                raise OSError("status pipe write made no progress")
            remaining = remaining[written:]
    finally:
        os.close(status_fd)


def _encode_runtime_status(exit_code: int) -> bytes:
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("runtime child exit code must be an integer")
    if exit_code not in {int(code) for code in ExitCode}:
        raise ValueError("runtime child exit code is not recognized")
    payload = (
        json.dumps(
            {
                "exit_code": exit_code,
                "protocol_version": _RUNTIME_STATUS_PROTOCOL_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert len(payload) <= _MAX_RUNTIME_STATUS_BYTES
    return payload


def _decode_runtime_status(payload: bytes) -> int:
    if not isinstance(payload, bytes):
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status record was not a byte sequence.",
        )
    if not payload:
        raise _RuntimeChildStatusError(
            "child_status_missing",
            "The runtime child exited without publishing terminal status; its process exit code is not trusted.",
        )
    if len(payload) > _MAX_RUNTIME_STATUS_BYTES:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status record exceeded its byte limit.",
        )
    if not payload.endswith(b"\n") or b"\n" in payload[:-1]:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status channel did not contain exactly one record.",
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_runtime_status_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status record was not valid UTF-8 JSON.",
        ) from exc
    if not isinstance(value, dict) or set(value) != {"exit_code", "protocol_version"}:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status record had an invalid schema.",
        )
    if type(value["protocol_version"]) is not int or value["protocol_version"] != _RUNTIME_STATUS_PROTOCOL_VERSION:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status protocol version was not supported.",
        )
    exit_code = value["exit_code"]
    if type(exit_code) is not int or exit_code not in {int(code) for code in ExitCode}:
        raise _RuntimeChildStatusError(
            "child_status_malformed",
            "The runtime child terminal-status exit code was not recognized.",
        )
    return exit_code


def _runtime_status_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate terminal-status field")
        value[key] = item
    return value


def _load_app_launcher_factory() -> Callable[[Mapping[str, Any]], Any]:
    """Import AppLauncher first in the fresh child, before Arena or request compilation."""

    from isaaclab.app import AppLauncher

    return AppLauncher


def _load_runtime_stack() -> RuntimeStack:
    """Import runtime and provider implementations only after SimulationApp starts."""

    from isaac_autodata_core.autonomous.attempt_generation import AttemptGenerator
    from isaac_autodata_core.autonomous.dataset_generation import DatasetGenerationRequest, generate_dataset
    from isaac_autodata_core.autonomous.output_transaction import OutputTransaction
    from isaac_autodata_core.autonomous.run_log import RunLogWriter
    from isaac_autodata_interfaces.autonomous.arena_environment import build_arena_runtime, goal_predicates_from_request
    from isaac_autodata_interfaces.autonomous.isaaclab_runtime import (
        AttachmentState,
        IsaacLabAttemptRuntime,
        IsaacLabPlanExecutor,
        success_term_verifier,
    )
    from isaac_autodata_interfaces.autonomous.schedulestream.episode_planner import (
        create_schedulestream_episode_planner,
    )

    return RuntimeStack(
        output_transaction_factory=OutputTransaction.reserve,
        arena_runtime_builder=build_arena_runtime,
        goal_projector=goal_predicates_from_request,
        attachment_state_factory=AttachmentState,
        runtime_factory=IsaacLabAttemptRuntime,
        success_verifier_factory=success_term_verifier,
        executor_factory=IsaacLabPlanExecutor,
        planner_factory=create_schedulestream_episode_planner,
        run_log_writer_factory=RunLogWriter,
        generator_factory=AttemptGenerator,
        generation_request_factory=DatasetGenerationRequest,
        run_loop=generate_dataset,
    )


def _app_launcher_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "device": args.device,
        "enable_cameras": args.enable_cameras,
        "headless": args.headless,
    }
    if not args.headless:
        # Current Isaac Lab resolves an omitted visualizer to headless execution. Selecting the Kit
        # visualizer is therefore the explicit GUI intent; ``headless=False`` alone is insufficient.
        options["visualizer"] = ["kit"]
    return options


def _arena_runtime_args(args: argparse.Namespace, resolved: Any) -> argparse.Namespace:
    """Return the allowlisted Arena builder arguments for deterministic source-free execution."""

    return argparse.Namespace(
        device=args.device,
        disable_fabric=False,
        enable_cameras=args.enable_cameras,
        env_spacing=30.0,
        headless=args.headless,
        language_instruction=None,
        mimic=False,
        num_envs=resolved.generation.num_envs,
        placement_seed=resolved.generation.seed,
        presets=None,
        random_yaw_init=False,
        resolve_on_reset=None,
        seed=resolved.generation.seed,
        solve_relations=True,
    )


def _preflight_to_dict(requested_motion_backend: str, compatibility: Any) -> dict[str, Any]:
    selected_motion_backend = getattr(compatibility, "motion_backend", None)
    application = getattr(compatibility, "schedulestream_application", None)
    selected_capabilities = getattr(compatibility, "capabilities", None)
    if selected_motion_backend not in ("curobo_v1", "curobo_v2"):
        raise ValueError("backend selector returned no supported motion backend")
    if application not in ("custream", "custream2"):
        raise ValueError("backend selector returned no supported ScheduleStream application")
    capabilities_dict = (
        selected_capabilities.to_dict()
        if callable(getattr(selected_capabilities, "to_dict", None))
        else {"summary": f"unavailable ({type(selected_capabilities).__name__})"}
    )
    if not isinstance(capabilities_dict, dict):
        raise ValueError("backend selector capabilities are not JSON compatible")
    return {
        "capabilities": capabilities_dict,
        "requested_motion_backend": requested_motion_backend,
        "schedulestream_application": application,
        "selected_motion_backend": selected_motion_backend,
        "status": "passed",
    }


def _capabilities_to_dict(capabilities: Any | None) -> dict[str, Any]:
    if capabilities is None:
        return {"status": "probe_failed_before_result"}
    serializer = getattr(capabilities, "to_dict", None)
    if not callable(serializer):
        return {"status": f"unavailable ({type(capabilities).__name__})"}
    try:
        value = serializer()
    except Exception as exc:
        return {
            "serialization_error": _safe_exception_message(exc),
            "status": "unavailable",
        }
    return value if isinstance(value, dict) else {"status": "invalid_capability_report"}


def _attach_runtime_profile(resolved: Any, preflight: dict[str, Any]) -> None:
    """Run the import-free product gate and attach its attestable identity to preflight."""

    from isaac_autodata_interfaces.autonomous.runtime_support import validate_runtime_support

    profile = validate_runtime_support(
        resolved,
        motion_backend=preflight["selected_motion_backend"],
        schedulestream_application=preflight["schedulestream_application"],
    )
    preflight["runtime_profile"] = profile.to_dict()
    preflight["runtime_profile_digest"] = profile.digest


def _runtime_capability_failure(exc: Any) -> CliFailure:
    issues = getattr(exc, "issues", ())
    return CliFailure(
        category="runtime_capability",
        code="request_capability_unsupported",
        exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
        message="The compiled task request is outside the currently executable product profile.",
        details={
            "issues": [issue.to_dict() for issue in issues],
            "remediation": (
                "Use the reviewed single-environment Franka PickAndPlace/on profile with "
                "curobo_v1/custream, or add and validate a new live runtime profile."
            ),
        },
    )


def _runtime_preflight_failure(
    exc: Exception,
    *,
    requested_motion_backend: str,
    capabilities: Any | None,
) -> CliFailure:
    return CliFailure(
        category="runtime_preflight",
        code="backend_incompatible",
        exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
        message=_safe_exception_message(exc),
        details={
            "capabilities": _capabilities_to_dict(capabilities),
            "requested_motion_backend": requested_motion_backend,
            "remediation": (
                "Use a reviewed runtime image containing the matching cuRobo and "
                "ScheduleStream application. The host environment will not be modified."
            ),
        },
    )


def _output_conflict_failure(resolved: Any, exc: Exception) -> CliFailure:
    return CliFailure(
        category="output_preflight",
        code="output_conflict",
        exit_code=ExitCode.OUTPUT_CONFLICT,
        message=_safe_exception_message(exc),
        details={
            "dataset": str(resolved.output.dataset),
            "run_log": None if resolved.output.run_log is None else str(resolved.output.run_log),
        },
    )


def _output_publication_failure(resolved: Any, exc: Exception) -> CliFailure:
    if isinstance(exc, FileExistsError):
        return _output_conflict_failure(resolved, exc)
    if type(exc).__name__ == "RunLogWriteUncertainError":
        return _run_log_write_uncertain_failure("output_publication", exc)
    commit_uncertain = type(exc).__name__ == "DatasetCommitUncertainError"
    return CliFailure(
        category="output_publication",
        code="dataset_commit_uncertain" if commit_uncertain else "dataset_publication_failed",
        exit_code=ExitCode.INTERNAL_ERROR,
        message=f"The staged dataset could not be durably published: {_safe_exception_message(exc)}",
        details={
            "dataset": str(resolved.output.dataset),
            "exception_type": type(exc).__name__,
            "recovery": (
                "Dataset links are durable but run-log commit durability is unknown; inspect the held run log "
                "and artifact digests before any retry or removal."
                if commit_uncertain
                else "No terminal commit was recorded; inspect the run log before retrying."
            ),
            "run_log": None if resolved.output.run_log is None else str(resolved.output.run_log),
        },
    )


def _terminal_failure_record(resolved: Any | None, failure: CliFailure) -> dict[str, Any]:
    assert failure.code not in {
        "dataset_commit_uncertain",
        "run_log_write_uncertain",
    }, "a run log with uncertain state must not be written again"
    if failure.code == "operator_interrupt":
        record_type = "run_interrupted"
    elif failure.category == "cleanup":
        record_type = "run_cleanup_failed"
    else:
        record_type = "run_aborted"
    return {
        "failure": {
            "category": failure.category,
            "code": failure.code,
            "exit_code": int(failure.exit_code),
            "message": failure.message,
        },
        "record_type": record_type,
        "request_digest": None if resolved is None else resolved.request_digest,
        "compiled_task_digest": None if resolved is None else resolved.digest,
    }


def _close_output_resources(
    run_log_writer: Any | None,
    output_transaction: Any | None,
    request_anchor: Any | None,
) -> tuple[CleanupIssue, ...]:
    issues: list[CleanupIssue] = []
    for resource, owned in (
        ("run_log_writer", run_log_writer),
        ("output_transaction", output_transaction),
        ("request_anchor", request_anchor),
    ):
        if owned is None:
            continue
        closer = getattr(owned, "close", None)
        if not callable(closer):
            issues.append(
                CleanupIssue(
                    resource=resource,
                    exception_type="TypeError",
                    message=f"{resource} does not expose close()",
                )
            )
            continue
        try:
            closer()
        except Exception as exc:
            issues.append(
                CleanupIssue(
                    resource=resource,
                    exception_type=type(exc).__name__,
                    message=_safe_exception_message(exc),
                )
            )
    return tuple(issues)


def _require_fresh_output_targets(resolved: Any) -> None:
    dataset_path = Path(resolved.output.dataset)
    paths = [("dataset", dataset_path)]
    if resolved.output.keep_failed:
        paths.append((
            "failed dataset",
            dataset_path.with_name(f"{dataset_path.stem}_failed{dataset_path.suffix}"),
        ))
    if resolved.output.run_log is not None:
        paths.append(("run_log", Path(resolved.output.run_log)))
    for label, path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing {label} output: {path}")


def _run_started_record(resolved: Any, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight": dict(preflight),
        "record_type": "run_started",
        "request_digest": resolved.request_digest,
        "compiled_task": resolved.to_dict(),
        "compiled_task_digest": resolved.digest,
    }


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    value = summary.to_dict()
    if not isinstance(value, dict):
        raise TypeError("generation summary must serialize to a mapping")
    if type(getattr(summary, "target_reached", None)) is not bool:
        raise TypeError("generation summary target_reached must be a boolean")
    return value


def _build_result(
    resolved: Any,
    preflight: Mapping[str, Any],
    summary: Any,
    *,
    published_artifacts: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "backend": dict(preflight),
        "cleanup": {"completed": True},
        "environment": {
            "graph_digest": resolved.graph_digest,
            "name": resolved.environment_name,
        },
        "generation": _summary_to_dict(summary),
        "output": {
            "artifacts": [_artifact_to_dict(artifact) for artifact in published_artifacts],
            "dataset": str(resolved.output.dataset),
            "run_log": None if resolved.output.run_log is None else str(resolved.output.run_log),
        },
        "request": {
            "digest": resolved.request_digest,
            "name": resolved.name,
            "compiled_task_digest": resolved.digest,
        },
        "status": "completed" if summary.target_reached else "incomplete",
    }


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    serializer = getattr(artifact, "to_dict", None)
    if not callable(serializer):
        raise TypeError("published artifact must expose to_dict()")
    value = serializer()
    if not isinstance(value, dict):
        raise TypeError("published artifact must serialize to a mapping")
    return value


def _phase_failure(phase: str, exc: Exception) -> CliFailure:
    if type(exc).__name__ == "RunLogWriteUncertainError":
        return _run_log_write_uncertain_failure(phase, exc)
    if phase == "app_launch":
        return CliFailure(
            category="app_launch",
            code="app_launch_failed",
            exit_code=ExitCode.APP_LAUNCH_FAILED,
            message=f"Isaac AppLauncher failed: {_safe_exception_message(exc)}",
            details={"exception_type": type(exc).__name__},
        )
    if phase == "runtime_setup":
        return CliFailure(
            category="runtime_setup",
            code="runtime_setup_failed",
            exit_code=ExitCode.RUNTIME_SETUP_FAILED,
            message=f"Runtime setup failed: {_safe_exception_message(exc)}",
            details={"exception_type": type(exc).__name__},
        )
    return CliFailure(
        category="generation",
        code="generation_failed",
        exit_code=ExitCode.GENERATION_INCOMPLETE,
        message=f"Dataset generation failed: {_safe_exception_message(exc)}",
        details={"exception_type": type(exc).__name__},
    )


def _run_log_write_uncertain_failure(
    phase: str,
    exc: Exception,
    *,
    preceding_failure: CliFailure | None = None,
) -> CliFailure:
    """Classify a run_log append whose durable tail state cannot be proven."""

    details: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "phase": phase,
        "recovery": (
            "Do not append to or automatically retry this run log. Inspect its existing JSONL tail "
            "and durable artifact identities to determine whether the attempted record is absent, partial, or "
            "already committed before any manual recovery."
        ),
    }
    if preceding_failure is not None:
        details["preceding_failure"] = {
            "category": preceding_failure.category,
            "code": preceding_failure.code,
            "exit_code": int(preceding_failure.exit_code),
        }
    return CliFailure(
        category="run_log",
        code="run_log_write_uncertain",
        exit_code=ExitCode.INTERNAL_ERROR,
        message=f"RunLog append durability is unknown: {_safe_exception_message(exc)}",
        details=details,
    )


def _unexpected_failure(category: str, exc: Exception) -> CliFailure:
    return CliFailure(
        category=category,
        code="internal_error",
        exit_code=ExitCode.INTERNAL_ERROR,
        message=f"Unexpected {type(exc).__name__}: {_safe_exception_message(exc)}",
        details={
            "remediation": "Report this bounded error to AutoData maintainers.",
        },
    )


def _emit_failure(stream: TextIO, output_format: str, failure: CliFailure) -> None:
    payload = {
        "category": failure.category,
        "code": failure.code,
        "details": dict(failure.details),
        "exit_code": int(failure.exit_code),
        "message": failure.message,
    }
    if output_format == "json":
        _write_json(stream, {"error": payload})
        return
    stream.write(f"ERROR [{failure.category}/{failure.code}] (exit {int(failure.exit_code)}): {failure.message}\n")
    issues = failure.details.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, Mapping):
                stream.write(
                    f"  - {issue.get('path', '$')} [{issue.get('code', 'invalid')}]: {issue.get('message', '')}\n"
                )
    remediation = failure.details.get("remediation")
    if isinstance(remediation, str):
        stream.write(f"  Remediation: {remediation}\n")
    stream.flush()


def _emit_failure_best_effort(stream: TextIO, output_format: str, failure: CliFailure) -> None:
    try:
        _emit_failure(stream, output_format, failure)
    except Exception:
        return


def _flush_terminal_streams(stdout: TextIO, stderr: TextIO) -> None:
    issues: list[str] = []
    for name, stream in (("stdout", stdout), ("stderr", stderr)):
        try:
            stream.flush()
        except Exception as exc:
            issues.append(f"{name}: {_safe_exception_message(exc)}")
    if issues:
        raise OSError("; ".join(issues))


def _flush_best_effort(stdout: TextIO, stderr: TextIO) -> None:
    for stream in (stdout, stderr):
        try:
            stream.flush()
        except Exception:
            continue


def _write_human_summary(stream: TextIO, result: Mapping[str, Any]) -> None:
    generation = result["generation"]
    backend = result["backend"]
    request = result["request"]
    output = result["output"]
    lines = [
        (
            "Dataset generation completed."
            if result["status"] == "completed"
            else "Dataset generation stopped before reaching its target."
        ),
        f"  Request: {json.dumps(request['name'], ensure_ascii=False)}",
        f"  Request digest: {request['digest']}",
        f"  Compiled task digest: {request['compiled_task_digest']}",
        f"  Motion backend: {backend['selected_motion_backend']}",
        f"  ScheduleStream application: {backend['schedulestream_application']}",
        (
            "  Generation: "
            f"attempts={generation['attempts']}, successes={generation['successes']}, "
            "failures="
            f"{generation['failures']}, requested_successes={generation['requested_successful_episodes']}"
        ),
        f"  Stop reason: {generation['stop_reason']}",
        f"  Target reached: {str(generation['target_reached']).lower()}",
        f"  Dataset: {json.dumps(output['dataset'], ensure_ascii=False)}",
        f"  RunLog: {json.dumps(output['run_log'], ensure_ascii=False)}",
        "  Cleanup completed: true",
    ]
    stream.write("\n".join(lines) + "\n")
    stream.flush()


def _write_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _safe_exception_message(exc: BaseException) -> str:
    message = " ".join(str(exc).splitlines()).replace("\x00", "").strip()
    return (message or "no additional details")[:2048]


def _device_argument(value: str) -> str:
    if value in ("cpu", "cuda"):
        return value
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise argparse.ArgumentTypeError("device must be cpu, cuda, or cuda:N")


def _sha256_argument(value: str) -> str:
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    raise argparse.ArgumentTypeError("resolved digest must be a lowercase SHA-256 hex string")


def _status_fd_argument(value: str) -> int:
    try:
        status_fd = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("status descriptor must be a decimal integer") from exc
    if status_fd < 3 or status_fd > 1_048_576:
        raise argparse.ArgumentTypeError("status descriptor is outside the allowed range")
    return status_fd


if __name__ == "__main__":
    raise SystemExit(main())
