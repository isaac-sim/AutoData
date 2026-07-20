#!/usr/bin/env python
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compile and preflight an AutoData task request without launching simulation.

This entrypoint is deliberately a product boundary, not a simulator launcher. It turns one
human-authored semantic YAML request into a deterministic compiled task request, optionally verifies
that the current Python runtime contains a compatible cuRobo/ScheduleStream pair, and then stops.
The later execution lane can consume the compiled task without silently changing user intent.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path, PureWindowsPath
from typing import Any, TextIO

MAX_ARTIFACT_PATH_LENGTH = 4096
_FORBIDDEN_ARTIFACT_DIRECTORIES = frozenset({".git", ".hg", ".svn"})
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)

# Direct ``python path/to/script.py`` execution puts only the examples directory on sys.path. Keep
# the documented repository-local invocation working without requiring an editable installation.
if __package__ in (None, ""):
    _REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, _REPOSITORY_ROOT)


class ExitCode(IntEnum):
    """Stable process exit codes exposed by the task compilation boundary."""

    SUCCESS = 0
    USAGE = 2
    REQUEST_COMPILATION_FAILED = 3
    RUNTIME_PREFLIGHT_FAILED = 4
    ARTIFACT_WRITE_FAILED = 5
    INTERNAL_ERROR = 6


@dataclass(frozen=True)
class RuntimePreflightReport:
    """Successful ScheduleStream runtime selection for one compiled task request."""

    requested_motion_backend: str
    selected_motion_backend: str
    schedulestream_application: str
    capabilities: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible preflight report."""

        return {
            "capabilities": dict(self.capabilities),
            "requested_motion_backend": self.requested_motion_backend,
            "schedulestream_application": self.schedulestream_application,
            "selected_motion_backend": self.selected_motion_backend,
            "status": "passed",
        }


class CompiledTaskWriteError(RuntimeError):
    """A safe, stable failure raised while validating or writing an artifact."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing Isaac, Arena, or planner packages."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "This command never launches Isaac Sim and never generates a dataset.\n\n"
            "Exit codes:\n"
            "  0  request compiled and requested checks passed\n"
            "  2  command-line usage error\n"
            "  3  request validation or Arena compilation failed\n"
            "  4  cuRobo/ScheduleStream runtime preflight failed\n"
            "  5  compiled task path validation or write failed\n"
            "  6  unexpected internal error"
        ),
    )
    parser.add_argument("request", type=Path, help="Path to an AutoData v1 semantic request YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and resolve the request, but skip the installed-runtime capability probe.",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        dest="output_format",
        help="Result and error output format (default: human).",
    )
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output_format",
        help="Shorthand for --format json.",
    )
    parser.add_argument(
        "--write-compiled",
        metavar="RELATIVE.json",
        help=(
            "Write canonical resolved JSON beneath the request file's directory. Existing files, "
            "absolute paths, traversal, symlinks, and repository-control directories are rejected."
        ),
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    compiler: Callable[[Path], Any] | None = None,
    capability_detector: Callable[[], Any] | None = None,
    backend_selector: Callable[[str, Any], Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Compile and preflight one request, returning a stable process exit code.

    Dependency injection keeps contract tests independent of Arena and installed motion stacks.
    Default dependencies are imported only after argument parsing, and the capability probe is
    imported only after semantic compilation succeeds.

    Args:
        argv: Arguments excluding the executable name.
        compiler: Optional semantic request compiler test seam.
        capability_detector: Optional import-free capability detector test seam.
        backend_selector: Optional backend selection test seam.
        stdout: Success output stream.
        stderr: Error output stream.

    Returns:
        A value from :class:`ExitCode`.
    """

    args = build_argument_parser().parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError

    if compiler is None:
        from isaac_autodata_interfaces.autonomous.task_compiler import compile_task_request

        compiler = compile_task_request

    try:
        resolved = compiler(args.request)
    except AutonomousValidationError as exc:
        _emit_error(
            stderr,
            args.output_format,
            category="request_compilation",
            code="request_invalid",
            exit_code=ExitCode.REQUEST_COMPILATION_FAILED,
            message="The semantic request could not be compiled.",
            details={"issues": [issue.to_dict() for issue in exc.issues]},
        )
        return int(ExitCode.REQUEST_COMPILATION_FAILED)
    except Exception as exc:
        _emit_unexpected_error(stderr, args.output_format, "request_compilation", exc)
        return int(ExitCode.INTERNAL_ERROR)

    preflight: RuntimePreflightReport | None = None
    if not args.dry_run:
        from isaac_autodata_interfaces.motion_planners.curobo.compat import (
            BackendCompatibilityError,
            detect_curobo_runtime,
            select_schedulestream_backend,
        )

        capability_detector = capability_detector or detect_curobo_runtime
        backend_selector = backend_selector or select_schedulestream_backend
        requested_motion_backend = resolved.planner.motion_backend.value
        capabilities: Any | None = None
        try:
            capabilities = capability_detector()
            selection = backend_selector(requested_motion_backend, capabilities)
            preflight = RuntimePreflightReport(
                requested_motion_backend=requested_motion_backend,
                selected_motion_backend=selection.motion_backend,
                schedulestream_application=selection.schedulestream_application,
                capabilities=selection.capabilities.to_dict(),
            )
        except BackendCompatibilityError as exc:
            _emit_error(
                stderr,
                args.output_format,
                category="runtime_preflight",
                code="backend_incompatible",
                exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
                message=_safe_exception_message(exc),
                details={
                    "capabilities": _capabilities_to_dict(capabilities),
                    "requested_motion_backend": requested_motion_backend,
                    "remediation": (
                        "Use motion_backend: auto with a reviewed pinned runtime, or select the "
                        "development image matching the requested cuRobo/ScheduleStream generation."
                    ),
                },
            )
            return int(ExitCode.RUNTIME_PREFLIGHT_FAILED)
        except Exception as exc:
            _emit_error(
                stderr,
                args.output_format,
                category="runtime_preflight",
                code="runtime_probe_failed",
                exit_code=ExitCode.RUNTIME_PREFLIGHT_FAILED,
                message=(
                    f"Runtime capability preflight failed with {type(exc).__name__}: {_safe_exception_message(exc)}"
                ),
                details={
                    "capabilities": _capabilities_to_dict(capabilities),
                    "requested_motion_backend": requested_motion_backend,
                    "remediation": "Inspect the pinned runtime image without modifying the AutoData host environment.",
                },
            )
            return int(ExitCode.RUNTIME_PREFLIGHT_FAILED)

    artifact_path: Path | None = None
    if args.write_compiled is not None:
        request_directory = args.request.expanduser().resolve(strict=False).parent
        try:
            artifact_path = _write_compiled_artifact(
                request_directory,
                args.write_compiled,
                resolved.canonical_json(),
            )
        except CompiledTaskWriteError as exc:
            _emit_error(
                stderr,
                args.output_format,
                category="compiled_task",
                code=exc.code,
                exit_code=ExitCode.ARTIFACT_WRITE_FAILED,
                message=str(exc),
                details={"request_directory": str(request_directory)},
            )
            return int(ExitCode.ARTIFACT_WRITE_FAILED)
        except Exception as exc:
            _emit_unexpected_error(stderr, args.output_format, "compiled_task", exc)
            return int(ExitCode.INTERNAL_ERROR)

    result = _build_success_result(resolved, preflight, artifact_path, dry_run=args.dry_run)
    if args.output_format == "json":
        _write_json(stdout, result)
    else:
        _write_human_result(stdout, resolved, preflight, artifact_path, dry_run=args.dry_run)
    return int(ExitCode.SUCCESS)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the task compilation CLI."""

    return run_cli(argv)


def _build_success_result(
    resolved: Any,
    preflight: RuntimePreflightReport | None,
    artifact_path: Path | None,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        runtime_preflight: dict[str, Any] = {
            "reason": "dry_run_requested",
            "status": "skipped",
        }
    else:
        assert preflight is not None
        runtime_preflight = preflight.to_dict()
    return {
        "execution": {"dataset_generated": False, "simulation_launched": False},
        "mode": "dry_run" if dry_run else "runtime_preflight",
        "compiled_task": resolved.to_dict(),
        "compiled_task_path": None if artifact_path is None else str(artifact_path),
        "compiled_task_digest": resolved.digest,
        "runtime_preflight": runtime_preflight,
    }


def _write_human_result(
    stream: TextIO,
    resolved: Any,
    preflight: RuntimePreflightReport | None,
    artifact_path: Path | None,
    *,
    dry_run: bool,
) -> None:
    generation = resolved.generation
    output = resolved.output
    lines = [
        "Task request compiled successfully.",
        f"  Name: {_quoted(resolved.name)}",
        f"  Environment: {_quoted(resolved.environment_name)}",
        f"  Request digest: {resolved.request_digest}",
        f"  Compiled task digest: {resolved.digest}",
        f"  Arena graph digest: {resolved.graph_digest}",
        f"  Planner: {resolved.planner.backend.value}",
        f"  Requested motion backend: {resolved.planner.motion_backend.value}",
        f"  Goal stages: {len(resolved.goal_stages)}",
        (
            "  Generation: "
            f"successful_episodes={generation.successful_episodes}, max_attempts={generation.max_attempts}, "
            f"num_envs={generation.num_envs}, seed={generation.seed}"
        ),
        f"  Dataset path: {_quoted(str(output.dataset))}",
        f"  RunLog path: {_quoted(None if output.run_log is None else str(output.run_log))}",
    ]
    if dry_run:
        lines.append("  Runtime preflight: skipped (--dry-run)")
    else:
        assert preflight is not None
        lines.extend([
            "  Runtime preflight: passed",
            f"  Selected motion backend: {preflight.selected_motion_backend}",
            f"  ScheduleStream application: {preflight.schedulestream_application}",
        ])
    if artifact_path is not None:
        lines.append(f"  Compiled task: {_quoted(str(artifact_path))}")
    lines.extend([
        "Simulation launched: no.",
        "Dataset generated: no.",
    ])
    stream.write("\n".join(lines) + "\n")
    stream.flush()


def _write_compiled_artifact(request_directory: Path, relative_name: str, canonical_json: str) -> Path:
    parts = _validate_artifact_relative_path(relative_name)
    root = request_directory.resolve(strict=False)
    try:
        parent_fd = _open_artifact_parent(root, parts[:-1])
    except CompiledTaskWriteError:
        raise
    except OSError as exc:
        raise CompiledTaskWriteError(
            "artifact_parent_unavailable",
            f"Could not open the request directory securely: {_safe_os_error(exc)}",
        ) from None

    target_name = parts[-1]
    temporary_name: str | None = None
    try:
        temporary_name, temporary_fd = _create_temporary_artifact(parent_fd)
        try:
            try:
                _write_all(temporary_fd, (canonical_json + "\n").encode("utf-8"))
                os.fsync(temporary_fd)
            except CompiledTaskWriteError:
                raise
            except OSError as exc:
                raise CompiledTaskWriteError(
                    "artifact_write_failed",
                    f"Could not write the compiled task: {_safe_os_error(exc)}",
                ) from None
        finally:
            with contextlib.suppress(OSError):
                os.close(temporary_fd)
        try:
            os.link(
                temporary_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise CompiledTaskWriteError(
                "artifact_exists",
                f"Refusing to overwrite existing compiled task {_quoted(relative_name)}.",
            ) from None
        except OSError as exc:
            raise CompiledTaskWriteError(
                "artifact_write_failed",
                f"Could not publish the compiled task: {_safe_os_error(exc)}",
            ) from None
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise CompiledTaskWriteError(
                "artifact_write_failed",
                f"The compiled task was published, but its directory could not be synchronized: {_safe_os_error(exc)}",
            ) from None
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error: OSError | None = None
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    cleanup_error = exc
        try:
            os.close(parent_fd)
        except OSError as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None and not active_error:
            raise CompiledTaskWriteError(
                "artifact_write_failed",
                f"Could not clean up the compiled task transaction: {_safe_os_error(cleanup_error)}",
            ) from None
    return root.joinpath(*parts)


def _validate_artifact_relative_path(raw_path: str) -> tuple[str, ...]:
    if not raw_path:
        raise CompiledTaskWriteError("artifact_path_invalid", "Compiled task path must not be empty.")
    if len(raw_path) > MAX_ARTIFACT_PATH_LENGTH:
        raise CompiledTaskWriteError(
            "artifact_path_invalid",
            f"Compiled task path exceeds {MAX_ARTIFACT_PATH_LENGTH} characters.",
        )
    if any(ord(character) < 32 for character in raw_path):
        raise CompiledTaskWriteError("artifact_path_invalid", "Compiled task path contains control characters.")
    if "\\" in raw_path:
        raise CompiledTaskWriteError(
            "artifact_path_invalid",
            "Compiled task path must use POSIX '/' separators.",
        )
    path = Path(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise CompiledTaskWriteError("artifact_path_absolute", "Compiled task path must be relative.")
    if raw_path.startswith("~"):
        raise CompiledTaskWriteError("artifact_path_invalid", "Home-directory expansion is not allowed.")
    raw_parts = tuple(raw_path.split("/"))
    if any(part in ("", ".", "..") for part in raw_parts):
        raise CompiledTaskWriteError(
            "artifact_path_traversal",
            "Compiled task path must not contain empty, '.' or '..' components.",
        )
    if any(part in _FORBIDDEN_ARTIFACT_DIRECTORIES for part in raw_parts):
        raise CompiledTaskWriteError(
            "artifact_path_forbidden",
            "Compiled tasks cannot be written inside repository-control directories.",
        )
    if Path(raw_parts[-1]).suffix != ".json":
        raise CompiledTaskWriteError("artifact_extension_invalid", "Compiled task path must end in '.json'.")
    return raw_parts


def _open_artifact_parent(root: Path, directory_parts: tuple[str, ...]) -> int:
    try:
        current_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise CompiledTaskWriteError(
            "artifact_parent_unavailable",
            f"Could not securely open request directory {_quoted(str(root))}: {_safe_os_error(exc)}",
        ) from None
    try:
        for part in directory_parts:
            try:
                next_fd = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                    next_fd = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
                except FileExistsError:
                    raise CompiledTaskWriteError(
                        "artifact_path_unsafe",
                        f"Artifact directory component {_quoted(part)} changed while it was being created.",
                    ) from None
                except OSError as exc:
                    raise CompiledTaskWriteError(
                        "artifact_parent_unavailable",
                        f"Could not create artifact directory component {_quoted(part)}: {_safe_os_error(exc)}",
                    ) from None
            except OSError as exc:
                code = (
                    "artifact_path_unsafe"
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR)
                    else "artifact_parent_unavailable"
                )
                raise CompiledTaskWriteError(
                    code,
                    f"Could not securely traverse artifact directory component {_quoted(part)}: {_safe_os_error(exc)}",
                ) from None
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _create_temporary_artifact(parent_fd: int) -> tuple[str, int]:
    for _ in range(128):
        name = f".autodata-compiled-{secrets.token_hex(12)}.tmp"
        try:
            file_fd = os.open(name, _FILE_CREATE_FLAGS, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise CompiledTaskWriteError(
                "artifact_write_failed",
                f"Could not create a temporary compiled task: {_safe_os_error(exc)}",
            ) from None
        return name, file_fd
    raise CompiledTaskWriteError(
        "artifact_write_failed",
        "Could not allocate a unique temporary compiled task name.",
    )


def _write_all(file_descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(file_descriptor, content[offset:])
        if written == 0:
            raise CompiledTaskWriteError("artifact_write_failed", "Compiled task write made no progress.")
        offset += written


def _capabilities_to_dict(capabilities: Any | None) -> dict[str, Any] | None:
    if capabilities is None:
        return None
    try:
        value = capabilities.to_dict()
    except Exception:
        return {"summary": f"unavailable ({type(capabilities).__name__})"}
    return value if isinstance(value, dict) else {"summary": f"invalid ({type(value).__name__})"}


def _emit_unexpected_error(stream: TextIO, output_format: str, category: str, exc: Exception) -> None:
    _emit_error(
        stream,
        output_format,
        category=category,
        code="internal_error",
        exit_code=ExitCode.INTERNAL_ERROR,
        message=f"Unexpected {type(exc).__name__}: {_safe_exception_message(exc)}",
        details={"remediation": "Re-run with validated inputs and report this bounded error to AutoData maintainers."},
    )


def _emit_error(
    stream: TextIO,
    output_format: str,
    *,
    category: str,
    code: str,
    exit_code: ExitCode,
    message: str,
    details: Mapping[str, Any],
) -> None:
    error = {
        "category": category,
        "code": code,
        "details": dict(details),
        "exit_code": int(exit_code),
        "message": message,
    }
    if output_format == "json":
        _write_json(stream, {"error": error})
        return
    stream.write(f"ERROR [{category}/{code}] (exit {int(exit_code)}): {message}\n")
    issues = details.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                stream.write(
                    f"  - {issue.get('path', '$')} [{issue.get('code', 'invalid')}]: {issue.get('message', '')}\n"
                )
    remediation = details.get("remediation")
    if isinstance(remediation, str):
        stream.write(f"  Remediation: {remediation}\n")
    capabilities = details.get("capabilities")
    if capabilities is not None:
        stream.write(f"  Detected capabilities: {json.dumps(capabilities, sort_keys=True, ensure_ascii=False)}\n")
    stream.flush()


def _write_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _safe_exception_message(exc: Exception) -> str:
    message = " ".join(str(exc).splitlines()).strip()
    return (message or "no additional details")[:2048]


def _safe_os_error(exc: OSError) -> str:
    message = exc.strerror or type(exc).__name__
    return " ".join(message.splitlines())[:512]


def _quoted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
