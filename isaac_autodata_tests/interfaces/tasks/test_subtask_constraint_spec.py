# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.tasks.subtask_constraint_spec`."""

import pytest

from isaac_autodata_interfaces.tasks.subtask_constraint_spec import (
    SubtaskConstraint,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)


def test_constraint_type_enum_values():
    assert int(SubTaskConstraintType.SEQUENTIAL) == 0
    assert int(SubTaskConstraintType.COORDINATION) == 1
    assert int(SubTaskConstraintType._SEQUENTIAL_FORMER) == 2
    assert int(SubTaskConstraintType._SEQUENTIAL_LATTER) == 3


def test_coordination_scheme_enum_values():
    assert int(SubTaskConstraintCoordinationScheme.REPLAY) == 0
    assert int(SubTaskConstraintCoordinationScheme.TRANSFORM) == 1
    assert int(SubTaskConstraintCoordinationScheme.TRANSLATE) == 2


def test_constraint_defaults():
    c = SubtaskConstraint()
    assert c.eef_subtask_constraint_tuple == (("", 0), ("", 0))
    assert c.constraint_type == SubTaskConstraintType.SEQUENTIAL
    assert c.sequential_min_time_diff == -1
    assert c.coordination_scheme == SubTaskConstraintCoordinationScheme.REPLAY
    assert c.coordination_scheme_pos_noise_scale == 0.0
    assert c.coordination_scheme_rot_noise_scale == 0.0
    assert c.coordination_synchronize_start is False


def test_sequential_runtime_constraints():
    # tuple[0] is the precondition (former); tuple[1] is the constrained (latter).
    c = SubtaskConstraint(
        eef_subtask_constraint_tuple=(("left", 0), ("right", 1)),
        constraint_type=SubTaskConstraintType.SEQUENTIAL,
        sequential_min_time_diff=5,
    )
    rt = c.generate_runtime_subtask_constraints()
    assert set(rt) == {("right", 1), ("left", 0)}

    latter = rt[("right", 1)]
    assert latter["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER
    assert latter["pre_condition_task_spec_key"] == "left"
    assert latter["pre_condition_subtask_ind"] == 0
    assert latter["min_time_diff"] == 5
    assert latter["fulfilled"] is False

    former = rt[("left", 0)]
    assert former["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER
    assert former["constrained_task_spec_key"] == "right"
    assert former["constrained_subtask_ind"] == 1


def test_sequential_default_min_time_diff_propagates():
    c = SubtaskConstraint(
        eef_subtask_constraint_tuple=(("a", 0), ("b", 1)),
        constraint_type=SubTaskConstraintType.SEQUENTIAL,
    )
    rt = c.generate_runtime_subtask_constraints()
    assert rt[("b", 1)]["min_time_diff"] == -1


def test_coordination_runtime_constraints():
    c = SubtaskConstraint(
        eef_subtask_constraint_tuple=(("left", 0), ("right", 0)),
        constraint_type=SubTaskConstraintType.COORDINATION,
        coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
        coordination_scheme_pos_noise_scale=0.1,
        coordination_scheme_rot_noise_scale=0.2,
        coordination_synchronize_start=True,
    )
    rt = c.generate_runtime_subtask_constraints()
    assert set(rt) == {("left", 0), ("right", 0)}

    left = rt[("left", 0)]
    assert left["type"] == SubTaskConstraintType.COORDINATION
    assert left["concurrent_task_spec_key"] == "right"
    assert left["concurrent_subtask_ind"] == 0
    assert left["coordination_scheme"] == SubTaskConstraintCoordinationScheme.TRANSFORM
    assert left["coordination_scheme_pos_noise_scale"] == 0.1
    assert left["coordination_scheme_rot_noise_scale"] == 0.2
    assert left["coordination_synchronize_start"] is True
    # Per-run mutable state is initialized but not yet resolved.
    assert left["fulfilled"] is False
    assert left["finished"] is False
    assert left["selected_src_demo_ind"] is None
    assert left["synchronous_steps"] is None

    # The concurrent entry points back at the constrained pair.
    right = rt[("right", 0)]
    assert right["type"] == SubTaskConstraintType.COORDINATION
    assert right["concurrent_task_spec_key"] == "left"
    assert right["concurrent_subtask_ind"] == 0


@pytest.mark.parametrize(
    "bad_type",
    [SubTaskConstraintType._SEQUENTIAL_FORMER, SubTaskConstraintType._SEQUENTIAL_LATTER],
)
def test_unsupported_constraint_type_raises(bad_type):
    c = SubtaskConstraint(constraint_type=bad_type)
    with pytest.raises(ValueError):
        c.generate_runtime_subtask_constraints()
