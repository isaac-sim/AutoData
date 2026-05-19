# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import yaml
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

from isaac_autodata_interfaces.tasks.subtask_spec import ALGO_PARAMS_REGISTRY, Subtask, SubtaskAlgoParams
from isaac_autodata_interfaces.tasks.task_descriptor_utils import build_subtask, validate_task_dict


@dataclass
class TaskDescriptor:
    """Class used to describe a task for data generation.

    Args:
        name: Task identifier.
        description: Human/agent-readable task description.
        subtasks: Per-end-effector ordered subtask lists. Keys are eef names;
            each value is the ordered sequence of :class:`Subtask` for that eef.
    """

    name: str = MISSING
    description: str = ""
    subtasks: dict[str, list[Subtask]] = field(default_factory=dict)
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
                  algo_params:                 # optional; fields match the
                    <kwarg>: <value>           #   chosen algo's dataclass
                - ...
              <other_eef>: [...]
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

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            subtasks=subtasks,
        )
