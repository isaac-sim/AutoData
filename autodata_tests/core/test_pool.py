# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`autodata_core.pool` (DataGenInfoPool)."""

import torch
import types
from pathlib import Path

import pytest

from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.embodiments import (
    AbsolutePoseWholeBodyBimanualAdapter,
    BimanualEefConfig,
    DeltaPoseIKSingleArmAdapter,
    PoseObsKeys,
)
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor


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


def test_appending_after_a_subset_load_uses_an_unused_source_key():
    pool = _pool(_mimicgen_task())
    pool._add_episode(_mimicgen_episode(), demo_key="demo_1")
    pool._add_episode(_mimicgen_episode())
    assert pool.demo_keys == ("demo_1", "demo_2")


def test_single_skill_can_load_without_an_unused_termination_mapping():
    task = _skillgen_task()
    task.subtasks["franka"] = task.subtasks["franka"][:1]
    pool = _pool(task, uses_start_signals=True)
    episode = _skillgen_episode()
    del episode.data["obs"]["datagen_info"]["subtask_term_signals"]
    pool._add_episode(episode)
    assert pool.subtask_boundaries["franka"] == [[(1, 10)]]


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


@pytest.mark.parametrize("uses_start_signals", [False, True])
def test_add_episode_uses_first_trigger_when_signals_toggle(uses_start_signals):
    """Later predicate changes must not move the first source subtask boundary."""
    pool = _pool(_skillgen_task() if uses_start_signals else _mimicgen_task(), uses_start_signals)
    episode = _skillgen_episode() if uses_start_signals else _mimicgen_episode()
    annotations = episode.data["obs"]["datagen_info"]
    annotations["subtask_term_signals"]["grasp_1"] = torch.tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    if uses_start_signals:
        annotations["subtask_start_signals"]["grasp_1"] = torch.tensor([0, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    pool._add_episode(episode)

    first_start, second_start = (1, 6) if uses_start_signals else (0, 5)
    assert pool.subtask_boundaries["franka"] == [[(first_start, 5), (second_start, 10)]]


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


@pytest.mark.parametrize("sample", [0, 3, 9])
def test_object_event_uses_exact_source_sample(sample):
    """Object marks do not receive the completion signal's one-step end offset."""
    pool = _pool(_mimicgen_task())
    episode = _mimicgen_episode()
    episode.data["obs"]["datagen_info"]["object_event_signals"] = {"grasp": _step_signal(10, sample)}
    pool._add_episode(episode, demo_key="source")
    assert pool.get_object_event_index(0, "grasp") == sample
    assert "object_event_signals" in pool.datagen_infos[0].to_dict()
    with pytest.raises(AssertionError, match="missing object event"):
        pool.get_object_event_index(0, "release")


@pytest.mark.parametrize("signal", [torch.ones(9), torch.ones(1, 10), torch.full((10,), 0.5), torch.zeros(10)])
def test_object_event_rejects_invalid_signal(signal):
    """Malformed, nonbinary and unmarked events fail when the source is loaded."""
    pool = _pool(_mimicgen_task())
    episode = _mimicgen_episode()
    episode.data["obs"]["datagen_info"]["object_event_signals"] = {"grasp": signal}
    with pytest.raises((AssertionError, ValueError)):
        pool._add_episode(episode)


def test_required_object_events_are_checked_before_source_is_added():
    """A task requesting internal marks cannot silently fall back to skill boundaries."""
    task = TaskDescriptor.from_yaml(
        str(Path(__file__).resolve().parents[2] / "autodata_examples/tasks/gr1t2_nut_pour_schedulegen.yaml")
    )
    pool = DataGenInfoPool(task, _bimanual_adapter(), "cpu", uses_start_signals=True)
    with pytest.raises(AssertionError, match="missing object_event_signals"):
        pool._add_episode(_dexmimicgen_episode())
    assert pool.num_datagen_infos == 0


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


def test_add_episode_maps_legacy_single_arm_names_without_mutating_source():
    pool = _pool(_softmimicgen_task())
    episode = _softmimicgen_episode()
    info = episode.data["obs"]["datagen_info"]
    for key in ("eef_pose", "target_eef_pose"):
        info[key]["robot0"] = info[key].pop("franka")

    pool._add_episode(episode)
    recorded = pool.datagen_infos[0]
    for key in ("eef_pose", "target_eef_pose"):
        assert set(info[key]) == {"robot0"}
        assert set(getattr(recorded, key)) == {"franka"}
        torch.testing.assert_close(getattr(recorded, key)["franka"], info[key]["robot0"])


def test_add_episode_rejects_inconsistent_pose_names():
    episode = _softmimicgen_episode()
    info = episode.data["obs"]["datagen_info"]
    info["eef_pose"]["robot0"] = info["eef_pose"].pop("franka")
    with pytest.raises(AssertionError, match="do not match target_eef_pose"):
        _pool(_softmimicgen_task())._add_episode(episode)


def test_add_episode_rejects_ambiguous_arm_name_mapping():
    episode = _dexmimicgen_episode()
    with pytest.raises(AssertionError, match="multi-arm datasets require matching names"):
        _pool(_softmimicgen_task())._add_episode(episode)
