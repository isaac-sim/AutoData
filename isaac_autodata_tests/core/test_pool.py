# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_core.pool` (DataGenInfoPool)."""

import torch
import types

import pytest

from isaac_autodata_core.pool import DataGenInfoPool
from isaac_autodata_interfaces.embodiments import (
    AbsolutePoseWholeBodyBimanualAdapter,
    BimanualEefConfig,
    DeltaPoseIKSingleArmAdapter,
    PoseObsKeys,
)
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor


def _single_arm_adapter() -> DeltaPoseIKSingleArmAdapter:
    return DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name="franka",
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
    )


def _bimanual_adapter() -> AbsolutePoseWholeBodyBimanualAdapter:
    def eef(name: str, gripper_indices: tuple[int, ...]) -> BimanualEefConfig:
        return BimanualEefConfig(
            name=name,
            pose_obs_keys=PoseObsKeys(pos=f"{name}_pos", quat=f"{name}_quat"),
            gripper_action_indices=gripper_indices,
        )

    return AbsolutePoseWholeBodyBimanualAdapter(
        name="g1",
        left=eef("left", (0, 1)),
        right=eef("right", (2, 3)),
        left_pose_slice=(0, 7),
        right_pose_slice=(7, 14),
        hand_joints_slice=(14, 18),
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


def _dexmimicgen_task() -> TaskDescriptor:
    return TaskDescriptor.from_dict({
        "name": "t",
        "algo": "dexmimicgen",
        "subtasks": {
            "left": [{"object_ref": "cube", "subtask_term_signal": "l_grasp"}, {"subtask_term_signal": ""}],
            "right": [{"object_ref": "cube", "subtask_term_signal": "r_grasp"}, {"subtask_term_signal": ""}],
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


def _softmimicgen_task() -> TaskDescriptor:
    return TaskDescriptor.from_dict({
        "name": "rope",
        "algo": "softmimicgen",
        "subtasks": {
            "franka": [
                {
                    "object_ref": "rope",
                    "subtask_term_signal": "grasp",
                    "algo_params": {"object_soft": True},
                },
                {
                    "object_ref": "rope",
                    "subtask_term_signal": "",
                    "algo_params": {"object_soft": True},
                },
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
        task_descriptor=task,
        embodiment_adapter=_single_arm_adapter(),
        device="cpu",
        uses_start_signals=uses_start_signals,
    )


def _episode(*, actions_len: int, datagen_info: dict, action_dim: int = 7) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data={"actions": torch.zeros(actions_len, action_dim), "obs": {"datagen_info": datagen_info}}
    )


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


def _dexmimicgen_episode(actions_len: int = 10, left_edge: int = 4, right_edge: int = 6) -> types.SimpleNamespace:
    return _episode(
        actions_len=actions_len,
        action_dim=18,  # bimanual adapter action layout width
        datagen_info={
            "eef_pose": {"left": torch.zeros(actions_len, 4, 4), "right": torch.zeros(actions_len, 4, 4)},
            "object_pose": {"cube": torch.zeros(actions_len, 4, 4)},
            "target_eef_pose": {"left": torch.zeros(actions_len, 4, 4), "right": torch.zeros(actions_len, 4, 4)},
            "subtask_term_signals": {
                "l_grasp": _step_signal(actions_len, left_edge),
                "r_grasp": _step_signal(actions_len, right_edge),
            },
        },
    )


def _skillgen_episode(
    actions_len: int = 10, start_edges: tuple[int, int] = (1, 6), term_edges: tuple[int, int] = (4, 8)
) -> types.SimpleNamespace:
    return _episode(
        actions_len=actions_len,
        datagen_info={
            "eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "object_pose": {"cube": torch.zeros(actions_len, 4, 4)},
            "target_eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "subtask_term_signals": {
                "grasp_1": _step_signal(actions_len, term_edges[0]),
                "stack_1": _step_signal(actions_len, term_edges[1]),
            },
            "subtask_start_signals": {
                "grasp_1": _step_signal(actions_len, start_edges[0]),
                "stack_1": _step_signal(actions_len, start_edges[1]),
            },
        },
    )


def _softmimicgen_episode(actions_len: int = 10, term_edge: int = 4) -> types.SimpleNamespace:
    return _episode(
        actions_len=actions_len,
        datagen_info={
            "eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "object_nodal_position": {"rope": torch.zeros(actions_len, 12, 3)},
            "target_eef_pose": {"franka": torch.zeros(actions_len, 4, 4)},
            "subtask_term_signals": {"grasp": _step_signal(actions_len, term_edge)},
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


def test_add_episode_accepts_deformable_state_without_rigid_object_pose():
    pool = _pool(_softmimicgen_task())
    pool._add_episode(_softmimicgen_episode(actions_len=8, term_edge=3))
    di = pool.datagen_infos[0]
    assert di.object_poses is None
    assert set(di.object_nodal_positions) == {"rope"}
    assert di.object_nodal_positions["rope"].shape == (8, 12, 3)
    assert pool.subtask_boundaries["franka"] == [[(0, 4), (4, 8)]]


def test_add_episode_missing_datagen_info_error():
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
# Multi-EEF (bimanual): each EEF's boundaries are parsed independently from its own term signal
# ---------------------------------------------------------------------------------------------------
def test_add_episode_parses_per_eef_boundaries_for_bimanual():
    pool = DataGenInfoPool(
        task_descriptor=_dexmimicgen_task(),
        embodiment_adapter=_bimanual_adapter(),
        device="cpu",
        uses_start_signals=False,
    )
    pool._add_episode(_dexmimicgen_episode(actions_len=10, left_edge=4, right_edge=6))
    # Each EEF's boundaries come from its own term signal, independently: left ends at 4+1=5,
    # right at 6+1=7; both final subtasks run to len (10).
    assert pool.subtask_boundaries["left"] == [[(0, 5), (5, 10)]]
    assert pool.subtask_boundaries["right"] == [[(0, 7), (7, 10)]]
    # The bimanual adapter de-interleaves the hand-joints slice into one gripper channel per arm.
    di = pool.datagen_infos[0]
    assert set(di.passthrough_action) == {"left", "right"}
    assert di.passthrough_action["left"].shape == (10, 2)


# ---------------------------------------------------------------------------------------------------
# SkillGen path: start signals drive subtask starts
# ---------------------------------------------------------------------------------------------------
def test_add_episode_parses_skillgen_boundaries():
    pool = _pool(_skillgen_task(), uses_start_signals=True)
    pool._add_episode(_skillgen_episode(actions_len=10, start_edges=(1, 6), term_edges=(4, 8)))
    # start edges give the subtask starts (grasp_1 @1, stack_1 @6); grasp_1 term edge @4 -> end 5;
    # the final subtask runs to len (10).
    assert pool.subtask_boundaries["franka"] == [[(1, 5), (6, 10)]]


def test_add_episode_skillgen_requires_start_signals():
    # A skillgen pool fed an episode without subtask_start_signals (e.g. a mimicgen recording) must
    # reject it rather than silently mis-parse boundaries.
    pool = _pool(_skillgen_task(), uses_start_signals=True)
    with pytest.raises(ValueError, match="subtask_start_signals missing"):
        pool._add_episode(_mimicgen_episode())


def test_add_episode_skillgen_rejects_start_offset_that_empties_subtask():
    # Subtask 0's worst-case start offset (+5) pushes its start (1) past its end (5), so the
    # subtask would be empty for some randomizations -> rejected at ingestion.
    task = TaskDescriptor.from_dict({
        "name": "t",
        "algo": "skillgen",
        "subtasks": {
            "franka": [
                {
                    "object_ref": "cube",
                    "subtask_term_signal": "grasp_1",
                    "algo_params": {"subtask_start_offset_range": [0, 5]},
                },
                {"object_ref": "cube", "subtask_term_signal": "stack_1"},
            ]
        },
    })
    pool = _pool(task, uses_start_signals=True)
    with pytest.raises(AssertionError, match="empty in worst case"):
        pool._add_episode(_skillgen_episode(actions_len=10, start_edges=(1, 6), term_edges=(4, 8)))
