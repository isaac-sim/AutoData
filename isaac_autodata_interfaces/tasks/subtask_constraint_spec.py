# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from dataclasses import dataclass


class SubTaskConstraintType(enum.IntEnum):
    """Type of constraint applied between two subtasks."""

    SEQUENTIAL = 0
    COORDINATION = 1

    _SEQUENTIAL_FORMER = 2
    _SEQUENTIAL_LATTER = 3


class SubTaskConstraintCoordinationScheme(enum.IntEnum):
    """Scheme used to coordinate two concurrent subtasks."""

    REPLAY = 0
    TRANSFORM = 1
    TRANSLATE = 2


@dataclass
class SubtaskConstraint:
    """A constraint between two ``(eef_name, subtask_index)`` pairs.

    Args:
        eef_subtask_constraint_tuple: The two associated ``(eef_name, subtask_index)`` pairs, in
            order. For ``SEQUENTIAL``, ``[0]`` is the precondition (former) and ``[1]`` is the
            constrained subtask (latter). For ``COORDINATION``, the two pairs run concurrently.
        constraint_type: Whether the constraint is sequential or coordination.
        sequential_min_time_diff: Minimum step gap before the latter subtask waits on the former;
            the latter executes until this many steps remain, then waits for the former to finish.
            ``-1`` means the latter starts only after the former finishes.
        coordination_scheme: Pose-coordination scheme, used by ``COORDINATION`` constraints.
        coordination_scheme_pos_noise_scale: Position noise scale applied during coordination.
        coordination_scheme_rot_noise_scale: Rotation noise scale applied during coordination.
        coordination_synchronize_start: Whether the two subtasks start on the same step.
    """

    eef_subtask_constraint_tuple: tuple[tuple[str, int], tuple[str, int]] = (("", 0), ("", 0))
    constraint_type: SubTaskConstraintType = SubTaskConstraintType.SEQUENTIAL
    sequential_min_time_diff: int = -1
    coordination_scheme: SubTaskConstraintCoordinationScheme = SubTaskConstraintCoordinationScheme.REPLAY
    coordination_scheme_pos_noise_scale: float = 0.0
    coordination_scheme_rot_noise_scale: float = 0.0
    coordination_synchronize_start: bool = False

    def generate_runtime_subtask_constraints(self) -> dict:
        """Expand this constraint into the runtime-constraint dict the data generator consumes.

        Returns a dict keyed by ``(eef_name, subtask_index)``. A sequential constraint produces a
        ``_SEQUENTIAL_LATTER`` entry (the waiting subtask) and a ``_SEQUENTIAL_FORMER`` entry (the
        precondition). A coordination constraint produces a ``COORDINATION`` entry for each of the
        two concurrent subtasks. The mutable flags (``fulfilled``, ``finished``,
        ``selected_src_demo_ind``, ``synchronous_steps``) are per-run state set during generation.
        """
        task_constraints_dict: dict = {}
        if self.constraint_type == SubTaskConstraintType.SEQUENTIAL:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(constrained_subtask_ind, int)
            pre_condition_task_spec_key, pre_condition_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(pre_condition_subtask_ind, int)
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_LATTER,
                pre_condition_task_spec_key=pre_condition_task_spec_key,
                pre_condition_subtask_ind=pre_condition_subtask_ind,
                min_time_diff=self.sequential_min_time_diff,
                fulfilled=False,
            )
            task_constraints_dict[(pre_condition_task_spec_key, pre_condition_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_FORMER,
                constrained_task_spec_key=constrained_task_spec_key,
                constrained_subtask_ind=constrained_subtask_ind,
            )
        elif self.constraint_type == SubTaskConstraintType.COORDINATION:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(constrained_subtask_ind, int)
            concurrent_task_spec_key, concurrent_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(concurrent_subtask_ind, int)
            assert self.coordination_scheme is not None, "Coordination scheme must be specified."
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                concurrent_task_spec_key=concurrent_task_spec_key,
                concurrent_subtask_ind=concurrent_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # calculated at runtime
            )
            task_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)] = dict(
                concurrent_task_spec_key=constrained_task_spec_key,
                concurrent_subtask_ind=constrained_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # calculated at runtime
            )
        else:
            raise ValueError(f"Constraint type not supported: {self.constraint_type!r}")

        return task_constraints_dict
