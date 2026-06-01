# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import yaml
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

from isaac_autodata_interfaces.tasks.subtask_constraint_spec import SubtaskConstraint
from isaac_autodata_interfaces.tasks.subtask_spec import ALGO_PARAMS_REGISTRY, Subtask, SubtaskAlgoParams
from isaac_autodata_interfaces.tasks.task_descriptor_utils import build_constraint, build_subtask, validate_task_dict


@dataclass
class TaskDescriptor:
    """Class used to describe a task for data generation.

    Args:
        name: Task identifier.
        description: Human/agent-readable task description.
        subtasks: Per-end-effector ordered subtask lists. Keys are eef names;
            each value is the ordered sequence of :class:`Subtask` for that eef.
        constraints: Cross-subtask coordination/sequential constraints (multi-eef tasks).
    """

    name: str = MISSING
    description: str = ""
    subtasks: dict[str, list[Subtask]] = field(default_factory=dict)
    constraints: list[SubtaskConstraint] = field(default_factory=list)
    env: Any = None

    def bind_env(self, env: Any) -> None:
        """Attach the Isaac Lab env after construction.

        Asserts the env was not previously bound — call exactly once.
        """

        assert self.env is None, "env already bound"
        self.env = env

    def get_eef_names(self) -> list[str]:
        """Return the end-effector names declared by this task."""

        return list(self.subtasks.keys())

    def get_subtasks(self, eef_name: str) -> list[Subtask]:
        """Return the list of subtasks for the given end-effector name."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return self.subtasks[eef_name]

    def get_object_refs(self, eef_name: str) -> list[str]:
        """Return all object reference names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.object_ref for st in self.subtasks[eef_name]]

    def get_start_signal_names(self, eef_name: str) -> list[str]:
        """Return subtask start-signal names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.subtask_start_signal for st in self.subtasks[eef_name]]

    def get_term_signal_names(self, eef_name: str) -> list[str]:
        """Return subtask termination-signal names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.subtask_term_signal for st in self.subtasks[eef_name]]

    def get_subtask_descriptions(self, eef_name: str) -> list[str]:
        """Return subtask descriptions for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.description for st in self.subtasks[eef_name]]

    def get_subtask_algo_params(self, eef_name: str) -> list[SubtaskAlgoParams]:
        """Return subtask algorithm parameters for the given eef, in subtask order."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.algo_params for st in self.subtasks[eef_name]]

    def get_task_constraints(self) -> list[SubtaskConstraint]:
        """Return the task's cross-subtask constraints, in declaration order.

        Each :class:`SubtaskConstraint` exposes ``generate_runtime_subtask_constraints()``, the
        verbatim expansion the data generator consumes — mirroring ``MimicEnvCfg.task_constraint_configs``.
        """

        return self.constraints

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskDescriptor:
        """Build a TaskDescriptor from a YAML config file.

        Expected schema:

            name: <str>
            description: <str>          # optional
            algo: <str>                 # key into ALGO_PARAMS_REGISTRY
            subtasks:
              <eef_name>:
                - object_ref: <str>            # optional
                  description: <str>           # optional
                  subtask_start_signal: <str>  # optional
                  subtask_term_signal: <str>   # optional

                  # algorithm-agnostic generation knobs:
                  selection_strategy: <str>
                  selection_strategy_kwargs: {<kwarg>: <value>}
                  first_subtask_start_offset_range: [<int>, <int>]
                  subtask_term_offset_range: [<int>, <int>]
                  action_noise: <float>
                  num_interpolation_steps: <int>
                  num_fixed_steps: <int>
                  apply_noise_during_interpolation: <bool>
                  algo_params:                 # optional, fields match the
                    <kwarg>: <value>           # chosen algo's dataclass
                - ...
              <other_eef>: [...]
            constraints:                        # optional, cross-subtask constraints
              - constraint_type: <str>          # "sequential" | "coordination"
                eef_subtask_constraint_tuple: [[<eef>, <int>], [<eef>, <int>]]
                sequential_min_time_diff: <int>            # sequential only
                coordination_scheme: <str>                 # coordination only
                coordination_scheme_pos_noise_scale: <float>
                coordination_scheme_rot_noise_scale: <float>
                coordination_synchronize_start: <bool>
              - ...
        """

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskDescriptor:
        """Build a TaskDescriptor from a parsed config dict.

        Schema is validated up-front; on a violation, raises
        :class:`AssertionError` with a message naming the offending field.
        """

        validate_task_dict(data)
        algo_cls = ALGO_PARAMS_REGISTRY[data["algo"]]

        subtasks: dict[str, list[Subtask]] = {
            eef_name: [build_subtask(st, algo_cls) for st in eef_subtasks]
            for eef_name, eef_subtasks in data["subtasks"].items()
        }
        constraints = [build_constraint(c) for c in data.get("constraints", [])]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            subtasks=subtasks,
            constraints=constraints,
        )
