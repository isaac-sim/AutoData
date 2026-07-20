# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compile a validated task request through an injectable Arena intent bridge."""

from __future__ import annotations

from pathlib import Path

from isaac_autodata_interfaces.autonomous.arena_bridge import ArenaIntentBridge, LazyArenaIntentBridge
from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.task_request import TASK_REQUEST_SCHEMA_VERSION, load_task_request
from isaac_autodata_interfaces.autonomous.task_request_types import (
    CompiledTaskRequest,
    ResolvedOutputConfig,
    TaskRequest,
)

TASK_COMPILER_VERSION = "1.0.0"


def compile_loaded_task_request(
    request: TaskRequest,
    *,
    bridge: ArenaIntentBridge | None = None,
) -> CompiledTaskRequest:
    """Compile and link a task request's opaque Arena intent into a pure runtime input."""

    selected_bridge = bridge if bridge is not None else LazyArenaIntentBridge()
    try:
        arena_result = selected_bridge.compile_and_link(
            request.environment_intent,
            seed=request.generation.seed,
        )
    except AutonomousValidationError:
        raise
    except Exception as exc:
        safe_message = str(exc).replace("\n", " ")[:2048]
        raise AutonomousValidationError([
            ValidationIssue(
                ("environment", "intent"),
                "arena_bridge_failed",
                f"Arena intent bridge failed with {type(exc).__name__}: {safe_message}",
            )
        ]) from None

    request_dir = request.source_path.parent.resolve(strict=False)
    dataset = (request_dir / request.output.dataset).resolve(strict=False)
    run_log = None if request.output.run_log is None else (request_dir / request.output.run_log).resolve(strict=False)
    output = ResolvedOutputConfig(
        dataset=dataset,
        keep_failed=request.output.keep_failed,
        run_log=run_log,
    )
    return CompiledTaskRequest(
        schema_version=TASK_REQUEST_SCHEMA_VERSION,
        compiler_version=TASK_COMPILER_VERSION,
        name=request.name,
        canonical_request_json=request.canonical_json(),
        request_digest=request.digest,
        planner=request.planner,
        generation=request.generation,
        output=output,
        arena=arena_result,
        source_dataset_path=None,
    )


def compile_task_request(
    path: str | Path,
    *,
    bridge: ArenaIntentBridge | None = None,
) -> CompiledTaskRequest:
    """Load, validate, compile, and link one AutoData v1 task request file."""

    return compile_loaded_task_request(load_task_request(path), bridge=bridge)
