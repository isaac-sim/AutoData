# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
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
        generation_policy: Data generation policy parameters.
    """

    name: str = MISSING
    description: str = ""
    subtasks: dict[str, list[Subtask]] = field(default_factory=dict)
    constraints: list[SubtaskConstraint] = field(default_factory=list)
    generation_policy: GenerationPolicy = field(default_factory=GenerationPolicy)
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

    def get_expected_attached_object(self, eef_name: str, subtask_index: int) -> str | None:
        """Return the object the EEF is expected to carry during a subtask, or ``None``.

        SkillGen plans collision-aware transit while the gripper holds a grasped object, so it
        needs the identity of the carried object for each subtask. A *stack* subtask is treated as
        carrying the object grasped in the immediately preceding *grasp* subtask; subtasks are
        classified by the ``"grasp"`` / ``"stack"`` substrings in their termination-signal names,
        so the result is derived purely from the descriptor's subtask metadata.

        Args:
            eef_name: End-effector to query.
            subtask_index: Index of the subtask whose carried object is requested.

        Returns:
            The held object's reference name, or ``None`` when the subtask carries nothing (a
            grasp/approach subtask, an out-of-range index, or an unknown eef).
        """

        if eef_name not in self.subtasks:
            return None
        subtasks = self.subtasks[eef_name]
        if not 0 <= subtask_index < len(subtasks):
            return None
        current = subtasks[subtask_index]
        if "stack" in current.subtask_term_signal.lower() and subtask_index > 0:
            prev = subtasks[subtask_index - 1]
            if "grasp" in prev.subtask_term_signal.lower():
                return prev.object_ref or None
        return None

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

    def get_generation_policy(self) -> GenerationPolicy:
        """Return the cross-cutting generation flags."""

        return self.generation_policy

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
            generation_policy:                  # optional; full parity with MimicEnvCfg.datagen_config
              name: <str>
              seed: <int>
              num_trials: <int>
              guarantee_success: <bool>
              keep_failed: <bool>
              max_num_failures: <int>
              source_dataset_path: <str>
              generation_path: <str>
              task_name: <str>
              use_skillgen: <bool>
              use_navigation_controller: <bool>
              select_src_per_subtask: <bool>
              select_src_per_arm: <bool>
              transform_first_robot_pose: <bool>
              interpolate_from_last_target_pose: <bool>
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
        generation_policy = GenerationPolicy(**data.get("generation_policy", {}))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            subtasks=subtasks,
            constraints=constraints,
            generation_policy=generation_policy,
        )
