# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure data contracts for the AutoData autonomous task envelope."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically, rejecting non-finite numbers."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    """Return a lowercase SHA-256 hex digest of canonical JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class PlannerBackend(StrEnum):
    """Task-and-motion planner selection understood by AutoData v1."""

    AUTO = "auto"
    SCHEDULESTREAM = "schedulestream"


class MotionBackend(StrEnum):
    """Motion backend selection understood by AutoData v1."""

    AUTO = "auto"
    CUROBO_V1 = "curobo_v1"
    CUROBO_V2 = "curobo_v2"


@dataclass(frozen=True)
class PlannerConfig:
    """Validated planner and motion-backend configuration."""

    backend: PlannerBackend
    motion_backend: MotionBackend
    collisions: bool
    max_time_s: float
    batch_size: int
    interpolation_dt_s: float
    profile: bool
    animate: bool

    def to_dict(self) -> dict[str, Any]:
        """Return canonical planner configuration."""

        return {
            "animate": self.animate,
            "backend": self.backend.value,
            "batch_size": self.batch_size,
            "collisions": self.collisions,
            "interpolation_dt_s": self.interpolation_dt_s,
            "max_time_s": self.max_time_s,
            "motion_backend": self.motion_backend.value,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class GenerationConfig:
    """Validated bounded-attempt generation configuration."""

    successful_episodes: int
    seed: int
    num_envs: int
    max_attempts: int

    def to_dict(self) -> dict[str, Any]:
        """Return canonical generation configuration."""

        return {
            "max_attempts": self.max_attempts,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "successful_episodes": self.successful_episodes,
        }


@dataclass(frozen=True)
class OutputConfig:
    """Validated request-relative output paths before absolute resolution."""

    dataset: str
    keep_failed: bool
    run_log: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return canonical request output configuration."""

        return {
            "dataset": self.dataset,
            "keep_failed": self.keep_failed,
            "run_log": self.run_log,
        }


@dataclass(frozen=True)
class ResolvedOutputConfig:
    """Output paths normalized beneath the semantic request directory."""

    dataset: Path
    keep_failed: bool
    run_log: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible absolute output paths."""

        return {
            "dataset": str(self.dataset),
            "keep_failed": self.keep_failed,
            "run_log": None if self.run_log is None else str(self.run_log),
        }


@dataclass(frozen=True)
class TaskRequest:
    """Validated AutoData v1 envelope around an opaque Arena environment intent.

    Arena's nested intent is retained as canonical JSON rather than re-modelled by AutoData.
    ``source_path`` is resolution context and is excluded from canonical request content.
    """

    schema_version: int
    name: str
    environment_intent_json: str
    planner: PlannerConfig
    generation: GenerationConfig
    output: OutputConfig
    source_path: Path

    @property
    def environment_intent(self) -> dict[str, Any]:
        """Return the opaque Arena intent as a fresh plain dictionary."""

        value = json.loads(self.environment_intent_json)
        assert isinstance(value, dict)
        return value

    def canonical_dict(self) -> dict[str, Any]:
        """Return normalized user-authored envelope content."""

        return {
            "environment": {"intent": self.environment_intent},
            "generation": self.generation.to_dict(),
            "name": self.name,
            "output": self.output.to_dict(),
            "planner": self.planner.to_dict(),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical request JSON."""

        return canonical_json(self.canonical_dict())

    @property
    def digest(self) -> str:
        """Return the canonical envelope SHA-256 digest."""

        return sha256_json(self.canonical_dict())


@dataclass(frozen=True)
class CompilerTraceEvent:
    """Plain, bounded projection of one Arena intent-resolution trace event."""

    stage: str
    query: str
    chosen: str | None
    note: str

    def to_dict(self) -> dict[str, str | None]:
        """Return the trace event as plain data."""

        return {"chosen": self.chosen, "note": self.note, "query": self.query, "stage": self.stage}


@dataclass(frozen=True)
class SpatialGoalConstraint:
    """One linked Arena success-state spatial constraint, without semantic reinterpretation."""

    id: str
    kind: str
    subject: str
    reference: str | None
    params_json: str

    @property
    def params(self) -> dict[str, Any]:
        """Return relation parameters as a fresh plain dictionary."""

        value = json.loads(self.params_json)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return the ordered spatial goal constraint."""

        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "reference": self.reference,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class GoalStage:
    """The ordered spatial success goal for one linked Arena task."""

    index: int
    task_id: str
    task_kind: str
    success_state_spec_id: str
    spatial_constraints: tuple[SpatialGoalConstraint, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the goal stage as plain data."""

        return {
            "index": self.index,
            "spatial_constraints": [constraint.to_dict() for constraint in self.spatial_constraints],
            "success_state_spec_id": self.success_state_spec_id,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
        }


@dataclass(frozen=True)
class ArenaCompilationResult:
    """Deterministic plain-data output of Arena intent validation, compilation, and linking."""

    initial_graph_json: str
    linked_graph_json: str
    compiler_trace: tuple[CompilerTraceEvent, ...]
    graph_digest: str
    goal_stages: tuple[GoalStage, ...]

    @property
    def initial_graph(self) -> dict[str, Any]:
        """Return the compiled initial graph as a fresh plain dictionary."""

        value = json.loads(self.initial_graph_json)
        assert isinstance(value, dict)
        return value

    @property
    def linked_graph(self) -> dict[str, Any]:
        """Return the linked graph as a fresh plain dictionary."""

        value = json.loads(self.linked_graph_json)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return all Arena compilation artifacts as plain data."""

        return {
            "compiler_trace": [event.to_dict() for event in self.compiler_trace],
            "goal_stages": [stage.to_dict() for stage in self.goal_stages],
            "graph_digest": self.graph_digest,
            "initial_graph": self.initial_graph,
            "linked_graph": self.linked_graph,
        }


@dataclass(frozen=True)
class CompiledTaskRequest:
    """Pure source-demo-free task ready for the autonomous runtime lane."""

    schema_version: int
    compiler_version: str
    name: str
    canonical_request_json: str
    request_digest: str
    planner: PlannerConfig
    generation: GenerationConfig
    output: ResolvedOutputConfig
    arena: ArenaCompilationResult
    source_dataset_path: None = None

    @property
    def canonical_request(self) -> dict[str, Any]:
        """Return canonical request content as a fresh mapping."""

        value = json.loads(self.canonical_request_json)
        assert isinstance(value, dict)
        return value

    @property
    def initial_graph(self) -> dict[str, Any]:
        """Return the resolved Arena initial graph."""

        return self.arena.initial_graph

    @property
    def linked_graph(self) -> dict[str, Any]:
        """Return the resolved Arena linked graph."""

        return self.arena.linked_graph

    @property
    def graph_digest(self) -> str:
        """Return the digest of the linked Arena graph."""

        return self.arena.graph_digest

    @property
    def goal_stages(self) -> tuple[GoalStage, ...]:
        """Return ordered task-success spatial goal stages."""

        return self.arena.goal_stages

    @property
    def environment_name(self) -> str:
        """Return the generated Arena graph environment name."""

        value = self.linked_graph.get("env_name")
        assert isinstance(value, str)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return the complete compiled task as canonical-JSON-compatible data."""

        return {
            "arena": self.arena.to_dict(),
            "canonical_request": self.canonical_request,
            "generation": self.generation.to_dict(),
            "name": self.name,
            "output": self.output.to_dict(),
            "planner": self.planner.to_dict(),
            "request_digest": self.request_digest,
            "compiler_version": self.compiler_version,
            "schema_version": self.schema_version,
            "source_dataset_path": self.source_dataset_path,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON for the compiled task."""

        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return a SHA-256 digest of the compiled task."""

        return sha256_json(self.to_dict())
