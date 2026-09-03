# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`autodata_interfaces.datastream.datastream.Datastream`."""

import torch

import pytest

from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.datastream import Datastream
from autodata_interfaces.embodiments import DeltaPoseIKSingleArmAdapter, PoseObsKeys
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from autodata_tests.interfaces.mocks import MockArticulationData, MockAsset, MockEnv, MockScene

_IDENTITY_QUAT_XYZW = [[0.0, 0.0, 0.0, 1.0]]
_SCENE_STATE_SENTINEL = {"articulation": {"robot": {"joint_position": "sentinel"}}}


def _task_dict() -> dict:
    return {
        "name": "stack",
        "algo": "mimicgen",
        "subtasks": {
            "franka": [
                {"object_ref": "cube_2", "description": "grasp", "subtask_term_signal": "grasp_1"},
                {"object_ref": "cube_1", "description": "stack", "subtask_term_signal": "stack_1"},
                {"object_ref": "cube_1", "description": "final", "subtask_term_signal": ""},
            ]
        },
        "generation_policy": {"num_trials": 5},
    }


def _embodiment_adapter(eef_name: str = "franka") -> DeltaPoseIKSingleArmAdapter:
    return DeltaPoseIKSingleArmAdapter(
        name="franka",
        eef_name=eef_name,
        pose_obs_keys=PoseObsKeys(pos="eef_pos", quat="eef_quat"),
        gripper_action_dim=1,
    )


def _env(env_origins: torch.Tensor | None = None) -> MockEnv:
    robot = MockAsset(
        MockArticulationData(
            joint_pos=torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            joint_names=[f"j{i}" for i in range(7)],
            root_pos_w=torch.tensor([[0.5, 0.0, 0.1]]),
            root_quat_w=torch.tensor(_IDENTITY_QUAT_XYZW),
        )
    )
    cube = MockAsset(
        MockArticulationData(
            root_pos_w=torch.tensor([[1.0, 1.0, 0.05]]),
            root_quat_w=torch.tensor(_IDENTITY_QUAT_XYZW),
        )
    )
    scene = MockScene(
        assets={"robot": robot},
        rigid_objects={"cube_1": cube},
        env_origins=torch.tensor([[0.0, 0.0, 0.0]]) if env_origins is None else env_origins,
        state=_SCENE_STATE_SENTINEL,
    )
    obs_buf = {"policy": {"eef_pos": torch.tensor([[0.2, 0.3, 0.4]]), "eef_quat": torch.tensor(_IDENTITY_QUAT_XYZW)}}
    return MockEnv(device="cpu", obs_buf=obs_buf, scene=scene)


def _datastream():
    task = TaskDescriptor.from_dict(_task_dict())
    embodiment_adapter = _embodiment_adapter()
    env = _env()
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    datastream = Datastream(
        env=env, task_descriptor=task, embodiment_adapter=embodiment_adapter, source_pool=pool, uses_start_signals=False
    )
    return datastream, task, embodiment_adapter, env


# ---------------------------------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------------------------------
def test_construction_binds_env_to_task_and_adapter():
    datastream, task, embodiment_adapter, env = _datastream()
    assert task.env is env
    assert embodiment_adapter.env is env
    assert datastream.get_env() is env
    assert datastream.device == "cpu"


def test_requires_exactly_one_demo_source():
    task = TaskDescriptor.from_dict(_task_dict())
    embodiment_adapter = _embodiment_adapter()
    env = _env()
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    # both provided
    with pytest.raises(AssertionError, match="exactly one of source_pool"):
        Datastream(
            env=env,
            task_descriptor=task,
            embodiment_adapter=embodiment_adapter,
            source_pool=pool,
            source_dataset_path="x.hdf5",
        )
    # neither provided
    with pytest.raises(AssertionError, match="exactly one of source_pool"):
        Datastream(
            env=env, task_descriptor=TaskDescriptor.from_dict(_task_dict()), embodiment_adapter=_embodiment_adapter()
        )


def test_eef_mismatch_error():
    task = TaskDescriptor.from_dict(_task_dict())  # eef "franka"
    embodiment_adapter = _embodiment_adapter(eef_name="other")
    env = _env()
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    with pytest.raises(AssertionError, match="EEF mismatch"):
        Datastream(env=env, task_descriptor=task, embodiment_adapter=embodiment_adapter, source_pool=pool)


def test_task_bound_to_different_env_error():
    task = TaskDescriptor.from_dict(_task_dict())
    task.bind_env(MockEnv())  # bound elsewhere
    embodiment_adapter = _embodiment_adapter()
    env = _env()
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    with pytest.raises(AssertionError, match="task descriptor is bound to a different env"):
        Datastream(env=env, task_descriptor=task, embodiment_adapter=embodiment_adapter, source_pool=pool)


def test_adapter_bound_to_different_env_error():
    task = TaskDescriptor.from_dict(_task_dict())
    embodiment_adapter = _embodiment_adapter()
    embodiment_adapter.bind_env(MockEnv())  # bound elsewhere
    env = _env()
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    with pytest.raises(AssertionError, match="embodiment adapter is bound to a different env"):
        Datastream(env=env, task_descriptor=task, embodiment_adapter=embodiment_adapter, source_pool=pool)


# ---------------------------------------------------------------------------------------------------
# Test task queries
# ---------------------------------------------------------------------------------------------------
def test_task_query_delegation():
    datastream, task, _, _ = _datastream()
    assert datastream.get_eef_names() == task.get_eef_names()
    assert datastream.get_subtasks("franka") == task.get_subtasks("franka")
    assert datastream.get_subtask("franka", 1) is task.get_subtasks("franka")[1]
    assert datastream.num_subtasks("franka") == 3
    assert datastream.get_object_refs("franka") == task.get_object_refs("franka")
    assert datastream.get_term_signal_names("franka") == ["grasp_1", "stack_1", ""]
    assert datastream.get_start_signal_names("franka") == task.get_start_signal_names("franka")
    assert datastream.get_subtask_descriptions("franka") == task.get_subtask_descriptions("franka")
    assert datastream.get_subtask_algo_params("franka") == task.get_subtask_algo_params("franka")
    assert datastream.get_task_constraints() == []
    assert datastream.get_generation_policy().num_trials == 5
    # grasp_1 (cube_2) -> stack_1 carries cube_2
    assert datastream.get_expected_attached_object("franka", 1) == "cube_2"


# ---------------------------------------------------------------------------------------------------
# Test embodiment queries
# ---------------------------------------------------------------------------------------------------
def test_embodiment_query_delegation():
    datastream, _, embodiment_adapter, _ = _datastream()
    assert torch.equal(
        datastream.get_robot_eef_pose(None, "franka"), embodiment_adapter.get_eef_poses(env_ids=None)["franka"]
    )
    assert torch.equal(datastream.get_robot_joint_positions(), embodiment_adapter.get_joint_positions())
    assert datastream.get_robot_joint_names() == embodiment_adapter.get_joint_names()


def test_action_pose_conversions_delegate():
    datastream, _, embodiment_adapter, _ = _datastream()
    action = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]])
    expected = embodiment_adapter.action_to_target_eef_pose(action)["franka"]
    assert torch.equal(datastream.action_to_target_eef_pose(action)["franka"], expected)

    actions = torch.randn(1, 4, 7)
    assert torch.equal(
        datastream.actions_to_passthrough_actions(actions)["franka"],
        embodiment_adapter.actions_to_passthrough_actions(actions)["franka"],
    )

    from autodata_utils import pose_math

    target = pose_math.make_pose(torch.tensor([0.2, 0.3, 0.4]), torch.eye(3))
    out = datastream.target_eef_pose_to_action({"franka": target}, {"franka": torch.tensor([0.5])}, env_id=0)
    assert out.shape == (7,)


# ---------------------------------------------------------------------------------------------------
# Runtime scene queries
# ---------------------------------------------------------------------------------------------------
def test_get_object_poses_is_env_relative():
    datastream, _, _, _ = _datastream()
    poses = datastream.get_object_poses()
    assert set(poses) == {"cube_1"}
    pose = poses["cube_1"]
    assert pose.shape == (1, 4, 4)
    # env origin is at 0, so env-relative position equals world position.
    assert torch.allclose(pose[0, :3, 3], torch.tensor([1.0, 1.0, 0.05]))


def test_get_object_poses_subtracts_env_origin():
    task = TaskDescriptor.from_dict(_task_dict())
    embodiment_adapter = _embodiment_adapter()
    env = _env(env_origins=torch.tensor([[1.0, 1.0, 0.0]]))  # shifted env origin
    pool = DataGenInfoPool(
        task_descriptor=task, embodiment_adapter=embodiment_adapter, device=env.device, uses_start_signals=False
    )
    datastream = Datastream(env=env, task_descriptor=task, embodiment_adapter=embodiment_adapter, source_pool=pool)
    pose = datastream.get_object_poses()["cube_1"]
    # world [1, 1, 0.05] - origin [1, 1, 0] = [0, 0, 0.05]
    assert torch.allclose(pose[0, :3, 3], torch.tensor([0.0, 0.0, 0.05]))


def test_get_robot_root_pose():
    datastream, _, _, _ = _datastream()
    pose = datastream.get_robot_root_pose(env_ids=[0])
    assert pose.shape == (1, 4, 4)
    assert torch.allclose(pose[0, :3, 3], torch.tensor([0.5, 0.0, 0.1]))


def test_get_scene_state_passthrough():
    datastream, _, _, _ = _datastream()
    assert datastream.get_scene_state(is_relative=True) is _SCENE_STATE_SENTINEL
