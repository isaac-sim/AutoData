# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure request envelope and lazy Arena bridge for autonomous AutoData generation."""

from isaac_autodata_interfaces.autonomous.arena_bridge import (
    REQUIRED_ARENA_CAPABILITY,
    REQUIRED_ARENA_COMMIT,
    ArenaIntentBridge,
    LazyArenaIntentBridge,
    build_arena_compilation_result,
    extract_goal_stages,
)
from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue
from isaac_autodata_interfaces.autonomous.runtime_support import (
    CURRENT_RUNTIME_SUPPORT,
    RUNTIME_SUPPORT_SCHEMA_VERSION,
    RuntimeSupportError,
    RuntimeSupportProfile,
    validate_runtime_support,
)
from isaac_autodata_interfaces.autonomous.task_compiler import (
    TASK_COMPILER_VERSION,
    compile_loaded_task_request,
    compile_task_request,
)
from isaac_autodata_interfaces.autonomous.task_request import (
    TASK_REQUEST_SCHEMA_VERSION,
    load_task_request,
    task_request_from_dict,
)
from isaac_autodata_interfaces.autonomous.task_request_types import (
    ArenaCompilationResult,
    CompiledTaskRequest,
    CompilerTraceEvent,
    GenerationConfig,
    GoalStage,
    MotionBackend,
    OutputConfig,
    PlannerBackend,
    PlannerConfig,
    ResolvedOutputConfig,
    SpatialGoalConstraint,
    TaskRequest,
    canonical_json,
    sha256_json,
)

__all__ = [
    "TASK_REQUEST_SCHEMA_VERSION",
    "TASK_COMPILER_VERSION",
    "RUNTIME_SUPPORT_SCHEMA_VERSION",
    "CURRENT_RUNTIME_SUPPORT",
    "REQUIRED_ARENA_CAPABILITY",
    "REQUIRED_ARENA_COMMIT",
    "TaskRequest",
    "RuntimeSupportError",
    "RuntimeSupportProfile",
    "ArenaCompilationResult",
    "ArenaIntentBridge",
    "AutonomousValidationError",
    "CompilerTraceEvent",
    "GenerationConfig",
    "GoalStage",
    "LazyArenaIntentBridge",
    "MotionBackend",
    "OutputConfig",
    "PlannerBackend",
    "PlannerConfig",
    "CompiledTaskRequest",
    "ResolvedOutputConfig",
    "SpatialGoalConstraint",
    "ValidationIssue",
    "task_request_from_dict",
    "build_arena_compilation_result",
    "canonical_json",
    "compile_task_request",
    "extract_goal_stages",
    "load_task_request",
    "compile_loaded_task_request",
    "sha256_json",
    "validate_runtime_support",
]
