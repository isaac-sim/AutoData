# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.pool` (DataGenInfoPool).

Episodes are stubbed with a ``SimpleNamespace`` carrying a ``.data`` dict (the only attribute the
pool reads), so the boundary-parsing logic is exercised without touching HDF5.
"""

import torch
import types

import pytest

from isaac_autodata_core.pool import DataGenInfoPool
from isaac_autodata_interfaces.embodiments import DeltaPoseIKSingleArmAdapter, PoseObsKeys
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor


def _adapter() -> DeltaPoseIKSingleArmAdapter:
    return DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
    )


def _mimicgen_task(term_offset_first=(0, 0)) -> TaskDescriptor:
    return TaskDescriptor.from_dict({
        "name": "t",
        "algo": "mimicgen",
        "subtasks": {
            "franka": [
                {
                    "object_ref": "cube",
                    "subtask_term_signal": "grasp_1",
                    "subtask_term_offset_range": list(term_offset_first),
                },
                {"subtask_term_signal": ""},
            ]
        },
    })


def _skillgen_task() -> TaskDescriptor:
    return TaskDescriptor.from_dict({
        "name": "t",
        "algo": "skillgen",
        "subtasks": {
            "franka": [
                {"object_ref": "cube", "subtask_term_signal": "grasp_1"},
                {"object_ref": "cube", "subtask_term_signal": "stack_1"},
            ]
        },
    })


def _step_signal(length: int, edge: int) -> torch.Tensor:
    """A per-step step-function: False before ``edge``, True from ``edge`` on."""
    sig = torch.ones(length, dtype=torch.bool)
    sig[:edge] = False
    return sig


def _pool(task: TaskDescriptor, uses_start_signals: bool = False) -> DataGenInfoPool:
    return DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=_adapter(), device="cpu", uses_start_signals=uses_start_signals
    )


def _episode(*, actions_len: int, datagen_info: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(data={"actions": torch.zeros(actions_len, 7), "obs": {"datagen_info": datagen_info}})


def _mimicgen_episode(actions_len: int = 10, term_edge: int = 4) -> types.SimpleNamespace:
    return _episode(
        actions_len=actions_len,
        datagen_info={
            "eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "object_pose": {"cube": torch.zeros(actions_len, 4, 4)},
            "target_eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "subtask_term_signals": {"grasp_1": _step_signal(actions_len, term_edge)},
        },
    )


# ---------------------------------------------------------------------------------------------------
# __init__: subtask signal-name / offset-range tables built from the task descriptor
# ---------------------------------------------------------------------------------------------------
def test_init_builds_signal_and_offset_tables():
    pool = _pool(_mimicgen_task())
    assert pool.subtask_term_signal_names == {"franka": ["grasp_1", ""]}
    assert pool.subtask_term_offset_ranges == {"franka": [(0, 0), (0, 0)]}
    assert pool.subtask_start_offset_ranges == {"franka": [(0, 0), (0, 0)]}
    assert pool.num_datagen_infos == 0


# ---------------------------------------------------------------------------------------------------
# _add_episode: DatagenInfo extraction + MimicGen boundary parsing
# ---------------------------------------------------------------------------------------------------
def test_add_episode_parses_mimicgen_boundaries():
    pool = _pool(_mimicgen_task())
    pool._add_episode(_mimicgen_episode(actions_len=10, term_edge=4))
    assert pool.num_datagen_infos == 1
    # grasp_1 rising edge at step 4 -> subtask 0 ends at 4 + 1 = 5; last subtask runs to len (10).
    assert pool.subtask_boundaries["franka"] == [[(0, 5), (5, 10)]]


def test_add_episode_populates_datagen_info_and_passthrough():
    pool = _pool(_mimicgen_task())
    pool._add_episode(_mimicgen_episode(actions_len=8, term_edge=3))
    di = pool.datagen_infos[0]
    assert set(di.eef_pose) == {"franka"}
    assert set(di.object_poses) == {"cube"}
    assert set(di.target_eef_pose) == {"franka"}
    assert set(di.subtask_term_signals) == {"grasp_1"}
    # passthrough derived from actions via the adapter: single-arm gripper channel of width 1.
    assert set(di.passthrough_action) == {"franka"}
    assert di.passthrough_action["franka"].shape == (8, 1)


def test_add_episode_missing_datagen_info_raises():
    pool = _pool(_mimicgen_task())
    bad = types.SimpleNamespace(data={"actions": torch.zeros(5, 7), "obs": {}})
    with pytest.raises(ValueError, match="lacks 'datagen_info'"):
        pool._add_episode(bad)


def test_add_episode_rejects_offset_that_overlaps_in_worst_case():
    # Subtask 0's large term offset makes its worst-case boundary run past subtask 1's end.
    pool = _pool(_mimicgen_task(term_offset_first=(0, 5)))
    with pytest.raises(AssertionError, match="subtask boundary violation"):
        pool._add_episode(_mimicgen_episode(actions_len=10, term_edge=4))


# ---------------------------------------------------------------------------------------------------
# SkillGen path: start signals drive subtask starts
# ---------------------------------------------------------------------------------------------------
def test_add_episode_parses_skillgen_boundaries():
    pool = _pool(_skillgen_task(), uses_start_signals=True)
    actions_len = 10
    episode = _episode(
        actions_len=actions_len,
        datagen_info={
            "eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "object_pose": {"cube": torch.zeros(actions_len, 4, 4)},
            "target_eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "subtask_term_signals": {"grasp_1": _step_signal(actions_len, 4), "stack_1": _step_signal(actions_len, 8)},
            "subtask_start_signals": {"grasp_1": _step_signal(actions_len, 1), "stack_1": _step_signal(actions_len, 6)},
        },
    )
    pool._add_episode(episode)
    # start edges give the subtask starts (grasp_1 @1, stack_1 @6); grasp_1 term edge @4 -> end 5;
    # the final subtask runs to len (10).
    assert pool.subtask_boundaries["franka"] == [[(1, 5), (6, 10)]]
