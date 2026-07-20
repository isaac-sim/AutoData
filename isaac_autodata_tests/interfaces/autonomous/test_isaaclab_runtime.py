# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
import torch
from dataclasses import replace
from types import SimpleNamespace

import pytest

from isaac_autodata_core.autonomous.attempt_generation import AttemptRequest, FailureStage
from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    DetachIntentSegment,
    GoalPredicate,
    GripperCommandMode,
    GripperCommandSegment,
    JointTrajectorySegment,
    TaskMotionPlan,
)
from isaac_autodata_interfaces.autonomous.isaaclab_runtime import (
    AttachmentState,
    IsaacLabAttemptRuntime,
    IsaacLabPlanExecutor,
)


def _identity(x: float = 0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _yaw(angle_rad: float):
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        (cosine, -sine, 0.0, 0.0),
        (sine, cosine, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _translation(x: float, y: float, z: float):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class _Recorder:
    def __init__(self):
        self.data = []
        self.reset_ids = None
        self.reset_calls = []
        self.post_reset_calls = []
        self.initial_state_supplier = lambda: "settled_state"
        self.success = None
        self.export_ids = None
        self.export_calls = 0

    def reset(self, env_ids):
        self.reset_ids = env_ids.clone()
        self.reset_calls.append(env_ids.clone())
        self.data = []

    def record_post_reset(self, env_ids):
        self.post_reset_calls.append(env_ids.clone())
        self.data.append(("initial_state", self.initial_state_supplier()))

    def set_success_to_episodes(self, env_ids, success):
        self.success = (env_ids.clone(), success.clone())

    def export_episodes(self, env_ids):
        self.export_calls += 1
        self.export_ids = env_ids.clone()


class _Env:
    def __init__(self):
        obj_data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        obj_cfg = SimpleNamespace(spawn=SimpleNamespace(usd_path="asset.usd"))
        robot_data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        self.scene = SimpleNamespace(
            env_origins=torch.tensor([[1.0, 2.0, 3.0]]),
            articulations={"robot": SimpleNamespace(data=robot_data)},
            rigid_objects={"cube": SimpleNamespace(data=obj_data, cfg=obj_cfg)},
        )
        self.action_space = SimpleNamespace(shape=(1, 7))
        self.num_envs = 1
        self.device = "cpu"
        self.step_dt = 0.05
        self.recorder_manager = _Recorder()
        self.reset_calls = []
        self.step_calls = []

    @property
    def unwrapped(self):
        return self

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.clone())

    def step(self, action):
        self.step_calls.append(action.clone())
        self.recorder_manager.data.append(("action", action.clone()))


class _Adapter:
    name = "fake_robot"
    gripper_action_dim = 1

    def __init__(self):
        self.env = None
        self.pose = torch.eye(4).unsqueeze(0)
        self.grippers = []

    def bind_env(self, env):
        self.env = env

    def get_eef_names(self):
        return ("tool",)

    def get_joint_positions(self, env_ids):
        return torch.tensor([[0.1, 0.2]])

    def get_joint_names(self):
        return ["j1", "j2"]

    def get_eef_poses(self, env_ids):
        return {"tool": self.pose.clone()}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict,
        gripper_action_dict,
        action_noise_dict,
        env_id,
    ):
        self.pose[0] = target_eef_pose_dict["tool"]
        self.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)


def _request():
    return AttemptRequest(
        request_digest="request",
        attempt_index=0,
        seed=3,
        env_id=0,
        goal=(GoalPredicate("on", "cube", "table"),),
        keep_failed=False,
    )


def _plan(*segments):
    return TaskMotionPlan(
        plan_id="plan",
        request_digest="request",
        snapshot_digest="snapshot",
        backend="fake",
        backend_version="1",
        seed=3,
        segments=segments,
        goal=(GoalPredicate("on", "cube", "table"),),
    )


def test_runtime_reset_snapshot_and_successful_export():
    env = _Env()
    adapter = _Adapter()
    env.recorder_manager.initial_state_supplier = lambda: {"simulator_steps": len(env.step_calls)}
    attachment_state = AttachmentState()
    runtime = IsaacLabAttemptRuntime(
        env,
        adapter,
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
        attachment_state=attachment_state,
    )

    reset_result = asyncio.run(runtime.reset_attempt(0))
    snapshot = runtime.capture_scene_snapshot(0, snapshot_id="snapshot")
    asyncio.run(runtime.finish_attempt(0, success=True, keep_failed=False))

    assert reset_result == {
        "env_id": 0,
        "recorder_excludes_reset_settling": True,
        "recorder_initial_state_recaptured_after_settling": True,
        "reset_completed": True,
        "reset_settle_duration_s": 0.5,
        "reset_settle_steps": 10,
    }
    assert snapshot.robot.robot_id == "robot"
    assert snapshot.objects[0].pose == _identity()
    assert snapshot.objects[0].geometry_ref == "asset.usd"
    assert snapshot.metadata["rigid_object_quaternion_convention"] == "xyzw"
    assert len(env.step_calls) == 10
    assert len(env.recorder_manager.reset_calls) == 2
    assert len(env.recorder_manager.post_reset_calls) == 1
    assert env.recorder_manager.data == [("initial_state", {"simulator_steps": 10})]
    assert adapter.grippers == [1.0] * 10
    assert env.recorder_manager.success[1].item()
    assert env.recorder_manager.export_ids.tolist() == [0]


def test_runtime_zero_settle_keeps_env_reset_initial_state_without_manual_recapture():
    class _ResetRecordingEnv(_Env):
        def reset(self, env_ids):
            super().reset(env_ids)
            self.recorder_manager.reset(env_ids)
            self.recorder_manager.record_post_reset(env_ids)

    env = _ResetRecordingEnv()
    env.recorder_manager.initial_state_supplier = lambda: {"simulator_steps": len(env.step_calls)}
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
        reset_settle_steps=0,
    )

    reset_result = asyncio.run(runtime.reset_attempt(0))

    assert reset_result["recorder_excludes_reset_settling"] is True
    assert reset_result["recorder_initial_state_recaptured_after_settling"] is False
    assert len(env.recorder_manager.post_reset_calls) == 1
    assert env.recorder_manager.data == [("initial_state", {"simulator_steps": 0})]
    assert env.step_calls == []


def test_runtime_settled_initial_state_precedes_next_task_sample():
    env = _Env()
    env.recorder_manager.initial_state_supplier = lambda: {"simulator_steps": len(env.step_calls)}
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )

    asyncio.run(runtime.reset_attempt(0))
    task_action = torch.zeros(env.action_space.shape)
    env.step(task_action)

    assert env.recorder_manager.data[0] == ("initial_state", {"simulator_steps": 10})
    assert env.recorder_manager.data[1][0] == "action"
    assert torch.equal(env.recorder_manager.data[1][1], task_action)


def test_runtime_initial_state_recapture_failure_propagates_after_clearing_setup_samples():
    env = _Env()
    recorder = env.recorder_manager

    def fail_recapture(env_ids):
        recorder.post_reset_calls.append(env_ids.clone())
        raise RuntimeError("settled initial-state capture failed")

    recorder.record_post_reset = fail_recapture
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )

    with pytest.raises(RuntimeError, match="settled initial-state capture failed"):
        asyncio.run(runtime.reset_attempt(0))

    assert len(env.step_calls) == 10
    assert len(recorder.reset_calls) == 2
    assert len(recorder.post_reset_calls) == 1
    assert recorder.data == []


def test_runtime_snapshot_decodes_isaaclab_xyzw_object_quaternion():
    env = _Env()
    half_angle = math.pi / 4
    env.scene.rigid_objects["cube"].data.root_quat_w = torch.tensor(
        [[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]]
    )
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
        reset_settle_steps=0,
    )

    snapshot = runtime.capture_scene_snapshot(0, snapshot_id="snapshot")

    assert torch.allclose(torch.tensor(snapshot.objects[0].pose), torch.tensor(_yaw(math.pi / 2)), atol=1e-6)


@pytest.mark.parametrize("reset_settle_steps", [True, -1, 101])
def test_runtime_rejects_unbounded_reset_settling(reset_settle_steps):
    with pytest.raises(ValueError, match="reset_settle_steps"):
        IsaacLabAttemptRuntime(
            _Env(),
            _Adapter(),
            graph_nodes=(
                {"id": "robot", "type": "embodiment"},
                {"id": "cube", "type": "object"},
            ),
            reset_settle_steps=reset_settle_steps,
        )


def test_runtime_reset_settling_rejects_saturated_action_before_simulator_step():
    class _SaturatedAdapter(_Adapter):
        def target_eef_pose_to_action(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    env = _Env()
    runtime = IsaacLabAttemptRuntime(
        env,
        _SaturatedAdapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )

    with pytest.raises(ValueError, match="clipping boundary"):
        asyncio.run(runtime.reset_attempt(0))

    assert env.step_calls == []
    assert len(env.recorder_manager.reset_calls) == 1


def test_failed_episode_is_not_exported_when_retention_is_disabled():
    env = _Env()
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )

    asyncio.run(runtime.finish_attempt(0, success=False, keep_failed=False))

    assert env.recorder_manager.export_ids is None


def test_attempt_finalization_is_idempotent_and_rejects_conflicting_retries():
    env = _Env()
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )

    asyncio.run(runtime.finish_attempt(0, success=True, keep_failed=False))
    asyncio.run(runtime.finish_attempt(0, success=True, keep_failed=False))

    assert env.recorder_manager.export_calls == 1
    with pytest.raises(RuntimeError, match="different retention state"):
        asyncio.run(runtime.finish_attempt(0, success=False, keep_failed=False))


def test_failed_second_attempt_reset_clears_prior_finalization_before_failure():
    env = _Env()
    runtime = IsaacLabAttemptRuntime(
        env,
        _Adapter(),
        graph_nodes=(
            {"id": "robot", "type": "embodiment"},
            {"id": "cube", "type": "object"},
        ),
    )
    asyncio.run(runtime.finish_attempt(0, success=True, keep_failed=False))

    def fail_reset(*, env_ids):
        del env_ids
        raise RuntimeError("second attempt reset failed")

    env.reset = fail_reset
    with pytest.raises(RuntimeError, match="second attempt reset failed"):
        asyncio.run(runtime.reset_attempt(0))

    asyncio.run(runtime.finish_attempt(0, success=False, keep_failed=False))
    asyncio.run(runtime.finish_attempt(0, success=False, keep_failed=False))

    assert env.recorder_manager.export_calls == 1
    assert env.recorder_manager.success[1].item() is False
    with pytest.raises(RuntimeError, match="different retention state"):
        asyncio.run(runtime.finish_attempt(0, success=True, keep_failed=False))


def test_runtime_rejects_unbound_semantic_objects_before_reset():
    env = _Env()

    with pytest.raises(ValueError, match="missing_bowl"):
        IsaacLabAttemptRuntime(
            env,
            _Adapter(),
            graph_nodes=(
                {"id": "robot", "type": "embodiment"},
                {"id": "cube", "type": "object"},
                {"id": "missing_bowl", "type": "object"},
            ),
        )


def test_executor_runs_cartesian_gripper_and_attachment_then_verifies_final_state():
    env = _Env()
    adapter = _Adapter()
    attachment_state = AttachmentState()
    attachment_state.reset(("tool",))
    segments = (
        CartesianTrajectorySegment(
            segment_id="move",
            eef_name="tool",
            frame="env_origin",
            poses=(_identity(0.1), _identity(0.2)),
        ),
        GripperCommandSegment(
            segment_id="close",
            depends_on=("move",),
            eef_name="tool",
            command=GripperCommandMode.CLOSE,
        ),
        AttachIntentSegment(
            segment_id="attach",
            depends_on=("close",),
            eef_name="tool",
            object_name="cube",
            verifier="test",
        ),
    )
    verifier_calls = []

    def verifier(_env, env_id):
        verifier_calls.append(env_id)
        return torch.tensor([True])

    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        verifier,
        attachment_state=attachment_state,
        attachment_verifier=lambda *_: True,
        final_settle_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(*segments)))

    assert result.success
    assert verifier_calls == [0, 0, 0]
    assert attachment_state.held_by_eef == {"tool": "cube"}
    assert result.final_observation["steps"] == 5
    assert adapter.grippers == [1.0, 1.0, -1.0, -1.0, -1.0]


def test_executor_rejects_false_final_state_without_or_accumulation():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: False, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "final_goal_not_satisfied"


def test_executor_rejects_joint_trajectory_for_ik_adapter_as_unrecoverable():
    env = _Env()
    adapter = _Adapter()
    segment = JointTrajectorySegment(
        segment_id="joint",
        joint_names=("j1",),
        positions=((0.0,),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_stage is FailureStage.EXECUTION
    assert result.failure_code == "joint_trajectory_not_supported_by_ik_executor"
    assert not result.recoverable


def test_executor_rejects_unsupported_late_segment_before_any_action():
    env = _Env()
    adapter = _Adapter()
    cartesian = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    joint = JointTrajectorySegment(
        segment_id="joint",
        depends_on=("move",),
        joint_names=("j1",),
        positions=((0.0,),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(cartesian, joint)))

    assert not result.success
    assert result.failure_code == "joint_trajectory_not_supported_by_ik_executor"
    assert env.step_calls == []


def test_executor_rejects_non_topological_segment_order_before_any_action():
    env = _Env()
    adapter = _Adapter()
    move = CartesianTrajectorySegment(
        segment_id="move",
        depends_on=("close",),
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert not result.success
    assert result.failure_code == "plan_not_topologically_ordered"
    assert env.step_calls == []


def test_executor_rejects_over_budget_plan_before_any_action():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1), _identity(0.2)),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0, max_steps=1)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "execution_step_limit"
    assert env.step_calls == []


def test_executor_rejects_eef_tracking_divergence_after_bounded_corrections():
    env = _Env()
    adapter = _Adapter()

    def action_without_observation_update(*_args, **_kwargs):
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_without_observation_update

    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_eef_position_error_m=0.01,
        max_tracking_correction_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "eef_position_tracking_error"
    assert len(env.step_calls) == 3
    assert result.final_observation["tracking_correction_steps"] == 2


def test_executor_holds_waypoint_until_bounded_tracking_converges():
    env = _Env()
    adapter = _Adapter()
    commanded_targets = []

    def action_with_one_step_lag(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        target = target_eef_pose_dict["tool"]
        commanded_targets.append(target.clone())
        adapter.pose[0, :3, 3] += 0.5 * (target[:3, 3] - adapter.pose[0, :3, 3])
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_with_one_step_lag
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_eef_position_error_m=0.03,
        max_tracking_correction_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success
    assert len(env.step_calls) == 5
    assert result.final_observation["tracking_correction_steps"] == 1
    assert result.final_observation["terminal_correction_steps"] == 3
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["target_segment_id"] == "move"
    assert boundary["target_sample_index"] == 0
    assert boundary["initial_position_error_m"] == pytest.approx(0.025)
    assert boundary["final_position_error_m"] == pytest.approx(0.003125, abs=1e-7)
    assert boundary["correction_steps"] == 3
    assert boundary["gripper_unchanged"] is True
    assert boundary["outcome"] == "converged"
    assert boundary["max_correction_steps"] == 32
    assert boundary["convergence_trace"] == [
        {
            "correction_steps": 0,
            "joint_path_verified": True,
            "position_error_m": pytest.approx(0.025, abs=1e-7),
            "rotation_error_rad": pytest.approx(0.0),
        },
        {
            "correction_steps": 1,
            "joint_path_verified": True,
            "position_error_m": pytest.approx(0.0125, abs=1e-7),
            "rotation_error_rad": pytest.approx(0.0),
        },
        {
            "correction_steps": 2,
            "joint_path_verified": True,
            "position_error_m": pytest.approx(0.00625, abs=1e-7),
            "rotation_error_rad": pytest.approx(0.0),
        },
        {
            "correction_steps": 3,
            "joint_path_verified": True,
            "position_error_m": pytest.approx(0.003125, abs=1e-7),
            "rotation_error_rad": pytest.approx(0.0),
        },
    ]
    assert boundary["post_settle_verification"] == {
        "gripper_unchanged": True,
        "joint_path_verified": True,
        "passed": True,
        "position_error_m": pytest.approx(0.003125, abs=1e-7),
        "rotation_error_rad": pytest.approx(0.0),
        "verification_samples": 1,
    }
    assert all(torch.equal(target, torch.tensor(_identity(0.1))) for target in commanded_targets)
    assert adapter.grippers == [1.0] * 5


def test_terminal_target_nonconvergence_fails_before_verification():
    env = _Env()
    adapter = _Adapter()
    verifier_calls = []

    def action_without_observation_update(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        del target_eef_pose_dict
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    def verifier(*_args):
        verifier_calls.append(True)
        return True

    adapter.target_eef_pose_to_action = action_without_observation_update
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.05),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        verifier,
        final_settle_steps=0,
        max_tracking_correction_steps=0,
        max_interaction_correction_steps=0,
        max_terminal_correction_steps=2,
    )
    assert executor.max_interaction_correction_steps == 0
    assert executor.max_terminal_correction_steps == 2

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_stage is FailureStage.EXECUTION
    assert result.failure_code == "terminal_target_not_reached"
    assert result.recoverable is True
    assert verifier_calls == []
    assert adapter.grippers == [1.0, 1.0, 1.0]
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["correction_steps"] == 2
    assert boundary["final_position_error_m"] == pytest.approx(0.05)
    assert boundary["outcome"] == "failed_tolerance_not_reached"
    assert boundary["max_correction_steps"] == 2
    assert boundary["convergence_trace"] == [
        {
            "correction_steps": step,
            "joint_path_verified": True,
            "position_error_m": pytest.approx(0.05),
            "rotation_error_rad": pytest.approx(0.0),
        }
        for step in range(3)
    ]


@pytest.mark.parametrize("max_terminal_correction_steps", [True, -1, 101])
def test_executor_rejects_unbounded_terminal_correction_budget(max_terminal_correction_steps):
    with pytest.raises(ValueError, match="max_terminal_correction_steps must be an integer in \\[0, 100\\]"):
        IsaacLabPlanExecutor(
            _Env(),
            _Adapter(),
            lambda *_: True,
            max_terminal_correction_steps=max_terminal_correction_steps,
        )


def test_terminal_trace_serializes_nonfinite_tracking_measurements_as_null():
    class _NonfiniteTerminalAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        def get_eef_poses(self, env_ids):
            result = super().get_eef_poses(env_ids)
            self.pose_reads += 1
            if 3 <= self.pose_reads <= 7:
                result["tool"][0, 0, 3] = float("nan")
            return result

    env = _Env()
    adapter = _NonfiniteTerminalAdapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_tracking_correction_steps=0,
        max_terminal_correction_steps=1,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_code == "terminal_target_not_reached"
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["initial_position_error_m"] is None
    assert boundary["final_position_error_m"] is None
    assert [sample["position_error_m"] for sample in boundary["convergence_trace"]] == [None, None]
    assert json.loads(json.dumps(boundary, allow_nan=False)) == boundary


def test_post_settle_cartesian_drift_fails_even_when_task_verifier_stays_true():
    class _PostSettleDriftEnv(_Env):
        def __init__(self, adapter):
            super().__init__()
            self.adapter = adapter

        def step(self, action):
            super().step(action)
            if len(self.step_calls) == 2:
                self.adapter.pose[0, 0, 3] += 0.01

    adapter = _Adapter()
    env = _PostSettleDriftEnv(adapter)
    verifier_calls = []

    def verifier(*_args):
        verifier_calls.append(True)
        return True

    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, verifier, final_settle_steps=1)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "terminal_target_not_stable"
    assert result.recoverable is True
    assert verifier_calls == [True, True]
    post_settle = result.final_observation["terminal_target_boundaries"][0]["post_settle_verification"]
    assert post_settle == {
        "gripper_unchanged": True,
        "joint_path_verified": True,
        "passed": False,
        "position_error_m": pytest.approx(0.01),
        "rotation_error_rad": pytest.approx(0.0),
        "verification_samples": 2,
    }


def test_post_settle_joint_drift_fails_terminal_stability_gate():
    env = _Env()
    adapter = _Adapter()
    joint_reads = 0

    def joint_positions(env_ids):
        nonlocal joint_reads
        del env_ids
        joint_reads += 1
        return torch.tensor([[0.1, 0.2]]) if joint_reads < 3 else torch.tensor([[0.6, 0.2]])

    adapter.get_joint_positions = joint_positions
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        joint_seed_names=("j1", "j2"),
        joint_seeds=((0.1, 0.2),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=1)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "terminal_target_not_stable"
    assert result.final_observation["maximum_joint_path_error_name"] == "j1"
    assert result.final_observation["maximum_joint_path_error_rad"] == pytest.approx(0.5)
    post_settle = result.final_observation["terminal_target_boundaries"][0]["post_settle_verification"]
    assert post_settle["position_error_m"] == pytest.approx(0.0)
    assert post_settle["joint_path_verified"] is False
    assert post_settle["passed"] is False


def test_post_settle_nonfinite_tracking_evidence_is_json_safe_and_fails_closed():
    class _NonfinitePostSettleAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        def get_eef_poses(self, env_ids):
            result = super().get_eef_poses(env_ids)
            self.pose_reads += 1
            if self.pose_reads in (7, 8):
                result["tool"][0, 0, 3] = float("nan")
            return result

    env = _Env()
    adapter = _NonfinitePostSettleAdapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=1)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "terminal_target_not_stable"
    post_settle = result.final_observation["terminal_target_boundaries"][0]["post_settle_verification"]
    assert post_settle["position_error_m"] is None
    assert post_settle["joint_path_verified"] is False
    assert post_settle["passed"] is False
    assert json.loads(json.dumps(post_settle, allow_nan=False)) == post_settle


def test_terminal_target_does_not_relax_joint_path_corridor():
    env = _Env()
    adapter = _Adapter()
    joint_reads = 0

    def joint_positions(env_ids):
        nonlocal joint_reads
        del env_ids
        joint_reads += 1
        return torch.tensor([[0.1, 0.2]]) if joint_reads == 1 else torch.tensor([[0.6, 0.2]])

    adapter.get_joint_positions = joint_positions
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        joint_seed_names=("j1", "j2"),
        joint_seeds=((0.1, 0.2),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_tracking_correction_steps=0,
        max_interaction_correction_steps=0,
        max_terminal_correction_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_code == "terminal_target_not_reached"
    assert result.final_observation["maximum_joint_path_error_name"] == "j1"
    assert result.final_observation["maximum_joint_path_error_rad"] == pytest.approx(0.5)
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["initial_joint_path_verified"] is False
    assert boundary["final_joint_path_verified"] is False
    assert boundary["gripper_unchanged"] is True
    assert [sample["joint_path_verified"] for sample in boundary["convergence_trace"]] == [False, False, False]


def test_terminal_target_boundary_is_a_clean_noop_without_cartesian_targets():
    env = _Env()
    adapter = _Adapter()
    segment = GripperCommandSegment(
        segment_id="initial_open",
        eef_name="tool",
        command=GripperCommandMode.OPEN,
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is True
    assert result.final_observation["terminal_target_boundaries"] == []
    assert result.final_observation["terminal_correction_steps"] == 0
    assert len(env.step_calls) == 1


def test_terminal_correction_budget_is_prevalidated_before_any_action():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_steps=2,
        max_tracking_correction_steps=0,
        max_interaction_correction_steps=0,
        max_terminal_correction_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success is False
    assert result.failure_code == "execution_step_limit"
    assert "approximately 3 simulator steps" in result.failure_message
    assert env.step_calls == []


def test_executor_enforces_finite_joint_path_bound_by_default_when_diagnostics_exist():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        joint_seed_names=("j1", "j2"),
        joint_seeds=((2.0, 0.2),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_tracking_correction_steps=0,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "joint_path_tracking_error"
    assert result.final_observation["maximum_joint_path_error_name"] == "j1"
    assert result.final_observation["maximum_joint_path_error_rad"] == pytest.approx(1.9)


def test_executor_cartesian_plan_without_joint_diagnostics_remains_supported():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.05),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert result.success
    assert len(env.step_calls) == 1
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["correction_steps"] == 0
    assert boundary["outcome"] == "already_converged"


def test_executor_rejects_disabling_finite_joint_path_bound():
    with pytest.raises(ValueError, match="must be finite and positive"):
        IsaacLabPlanExecutor(_Env(), _Adapter(), lambda *_: True, max_joint_path_error_rad=None)


def test_executor_rejects_saturated_dik_pose_action_before_env_step():
    env = _Env()
    adapter = _Adapter()

    def saturated_action(*_args, **_kwargs):
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    adapter.target_eef_pose_to_action = saturated_action
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "dik_pose_action_saturated"
    assert result.final_observation["steps"] == 0
    assert env.step_calls == []


def test_executor_rejects_nonfinite_dik_action_before_env_step():
    env = _Env()
    adapter = _Adapter()
    adapter.target_eef_pose_to_action = lambda *_args, **_kwargs: torch.tensor(
        [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "raw_dik_action_invalid"
    assert env.step_calls == []


def test_executor_rejects_cartesian_translation_discontinuity_before_any_action():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.11),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "cartesian_translation_discontinuity"
    assert env.step_calls == []


def test_executor_rejects_cartesian_linear_velocity_before_any_action():
    env = _Env()
    env.step_dt = 0.01
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.05),),
        metadata={"step_dt_s": 0.01},
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "cartesian_linear_velocity_limit"
    assert env.step_calls == []


def test_executor_rejects_cartesian_rotation_discontinuity_before_any_action():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_yaw(0.26),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "cartesian_rotation_discontinuity"
    assert env.step_calls == []


def test_executor_rejects_cartesian_angular_velocity_before_any_action():
    env = _Env()
    env.step_dt = 0.01
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_yaw(0.05),),
        metadata={"step_dt_s": 0.01},
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "cartesian_angular_velocity_limit"
    assert env.step_calls == []


def test_executor_rejects_planner_control_dt_mismatch_before_any_action():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        metadata={"step_dt_s": 0.02},
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "cartesian_step_dt_mismatch"
    assert env.step_calls == []


def test_executor_rejects_target_outside_base_relative_franka_workspace():
    env = _Env()
    adapter = _Adapter()
    adapter.pose[0] = torch.tensor(_identity(1.01))
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(1.01),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "eef_workspace_limit"
    assert env.step_calls == []


def test_executor_fails_closed_when_workspace_frame_is_unavailable():
    env = _Env()
    env.scene.articulations = {"robot": object()}
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "workspace_frame_unavailable"
    assert env.step_calls == []


def test_executor_enforces_explicit_joint_path_bound_when_configured():
    env = _Env()
    adapter = _Adapter()
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        joint_seed_names=("j1", "j2"),
        joint_seeds=((2.0, 0.2),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_joint_path_error_rad=0.1,
        max_tracking_correction_steps=1,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "joint_path_tracking_error"
    assert len(env.step_calls) == 2


def test_gripper_transition_corrects_cartesian_lag_before_close_with_old_gripper():
    env = _Env()
    adapter = _Adapter()

    def action_with_half_step_lag(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        target = target_eef_pose_dict["tool"]
        adapter.pose[0, :3, 3] += 0.5 * (target[:3, 3] - adapter.pose[0, :3, 3])
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_with_half_step_lag
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_interaction_correction_steps=5,
    )

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert result.success
    assert adapter.grippers == [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    assert result.final_observation["interaction_correction_steps"] == 4
    boundary = result.final_observation["interaction_boundaries"][0]
    assert boundary["outcome"] == "converged"
    assert boundary["correction_steps"] == 4
    assert boundary["initial_position_error_m"] == pytest.approx(0.05)
    assert boundary["final_position_error_m"] == pytest.approx(0.003125, abs=1e-7)
    assert boundary["target_segment_id"] == "move"
    assert boundary["target_sample_index"] == 0
    assert result.final_observation["gripper_milestones"][0]["interaction_boundary"] == boundary


def test_gripper_transition_corrects_rotation_to_strict_interaction_tolerance():
    env = _Env()
    adapter = _Adapter()

    def action_with_rotation_lag(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        target = target_eef_pose_dict["tool"]
        current_yaw = torch.atan2(adapter.pose[0, 1, 0], adapter.pose[0, 0, 0])
        target_yaw = torch.atan2(target[1, 0], target[0, 0])
        next_yaw = float(current_yaw + 0.5 * (target_yaw - current_yaw))
        adapter.pose[0] = torch.tensor(_yaw(next_yaw))
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_with_rotation_lag
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_yaw(0.12),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert result.success
    boundary = result.final_observation["interaction_boundaries"][0]
    assert boundary["correction_steps"] == 1
    assert boundary["initial_rotation_error_rad"] == pytest.approx(0.06, abs=1e-5)
    assert boundary["final_rotation_error_rad"] == pytest.approx(0.03, abs=1e-5)
    assert adapter.grippers == [1.0, 1.0, -1.0]


def test_gripper_transition_nonconvergence_fails_before_close_command():
    env = _Env()
    adapter = _Adapter()

    def action_without_observation_update(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        del target_eef_pose_dict
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_without_observation_update
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.05),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_interaction_correction_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert not result.success
    assert result.failure_code == "interaction_target_not_reached"
    assert adapter.grippers == [1.0, 1.0, 1.0]
    assert len(env.step_calls) == 3
    assert result.final_observation["gripper_milestones"] == []
    boundary = result.final_observation["interaction_boundaries"][0]
    assert boundary["converged"] is False
    assert boundary["correction_steps"] == 2
    assert boundary["outcome"] == "failed_tolerance_not_reached"
    assert result.final_observation["interaction_correction_steps"] == 2


def test_close_interaction_does_not_use_release_contact_tolerance() -> None:
    env = _Env()
    adapter = _Adapter()

    def action_stalled_nine_millimeters_from_target(target_eef_pose_dict, gripper_action_dict, *_args, **_kwargs):
        target = target_eef_pose_dict["tool"].clone()
        target[0, 3] -= 0.0095
        adapter.pose[0] = target
        adapter.grippers.append(float(gripper_action_dict["tool"][0]))
        return torch.zeros(7)

    adapter.target_eef_pose_to_action = action_stalled_nine_millimeters_from_target
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(0.1),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert result.success is False
    assert result.failure_code == "interaction_target_not_reached"
    assert all(command > 0 for command in adapter.grippers)
    boundary = result.final_observation["interaction_boundaries"][0]
    assert boundary["strict_target_converged"] is False
    assert boundary["release_contact_gate_required"] is False
    assert boundary["contact_constrained_task_success_accepted"] is False


def test_interaction_gate_preserves_joint_seed_corridor_before_close():
    env = _Env()
    adapter = _Adapter()
    joint_reads = 0

    def joint_positions(env_ids):
        nonlocal joint_reads
        del env_ids
        joint_reads += 1
        return torch.tensor([[0.1, 0.2]]) if joint_reads == 1 else torch.tensor([[1.0, 0.2]])

    adapter.get_joint_positions = joint_positions
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
        joint_seed_names=("j1", "j2"),
        joint_seeds=((0.1, 0.2),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_interaction_correction_steps=1,
    )

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert not result.success
    assert result.failure_code == "interaction_target_not_reached"
    assert adapter.grippers == [1.0, 1.0]
    assert result.final_observation["maximum_joint_path_error_name"] == "j1"
    assert result.final_observation["maximum_joint_path_error_rad"] == pytest.approx(0.9)
    assert result.final_observation["interaction_boundaries"][0]["converged"] is False


def test_initial_open_records_no_preceding_interaction_target():
    env = _Env()
    adapter = _Adapter()
    initial_open = GripperCommandSegment(
        segment_id="initial_open",
        eef_name="tool",
        command=GripperCommandMode.OPEN,
    )
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=0)

    result = asyncio.run(executor.execute(_request(), _plan(initial_open)))

    assert result.success
    boundary = result.final_observation["interaction_boundaries"][0]
    assert boundary["preceding_cartesian_target"] is False
    assert boundary["transition_required"] is False
    assert boundary["outcome"] == "not_required_no_preceding_target"
    assert result.final_observation["gripper_milestones"][0]["interaction_boundary"] == boundary


def test_interaction_correction_budget_is_prevalidated_before_any_action():
    env = _Env()
    adapter = _Adapter()
    move = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    close = GripperCommandSegment(
        segment_id="close",
        depends_on=("move",),
        eef_name="tool",
        command=GripperCommandMode.CLOSE,
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: True,
        final_settle_steps=0,
        max_steps=2,
        max_tracking_correction_steps=0,
        max_interaction_correction_steps=2,
        max_terminal_correction_steps=0,
    )

    result = asyncio.run(executor.execute(_request(), _plan(move, close)))

    assert not result.success
    assert result.failure_code == "execution_step_limit"
    assert env.step_calls == []


def test_final_goal_must_hold_for_consecutive_stability_window():
    env = _Env()
    adapter = _Adapter()
    outcomes = iter((True, False, True))
    segment = CartesianTrajectorySegment(
        segment_id="move",
        eef_name="tool",
        frame="env_origin",
        poses=(_identity(),),
    )
    executor = IsaacLabPlanExecutor(
        env,
        adapter,
        lambda *_: next(outcomes),
        final_settle_steps=2,
        final_stability_steps=2,
    )

    result = asyncio.run(executor.execute(_request(), _plan(segment)))

    assert not result.success
    assert result.failure_code == "final_goal_not_satisfied"
    assert result.final_observation["final_success_streak"] == 1
    assert result.final_observation["verification_samples"] == 3


class _SemanticAdapter(_Adapter):
    def get_joint_positions(self, env_ids):
        del env_ids
        aperture = 0.08 if not self.grippers or self.grippers[-1] > 0 else 0.056
        return torch.tensor([[aperture / 2, aperture / 2]])

    def get_joint_names(self):
        return ["panda_finger_joint1", "panda_finger_joint2"]


class _SemanticEnv(_Env):
    def __init__(self, adapter: _SemanticAdapter):
        super().__init__()
        self.adapter = adapter
        self.closed_once = False
        origin = self.scene.env_origins[0]
        cube = self.scene.rigid_objects["cube"]
        cube.data.root_pos_w = origin.unsqueeze(0).clone()
        cube.data.root_lin_vel_w = torch.zeros((1, 3))
        cube.data.root_ang_vel_w = torch.zeros((1, 3))
        bowl_data = SimpleNamespace(
            root_pos_w=(origin + torch.tensor([0.1, 0.0, 0.03])).unsqueeze(0),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            root_lin_vel_w=torch.zeros((1, 3)),
            root_ang_vel_w=torch.zeros((1, 3)),
        )
        bowl_cfg = SimpleNamespace(spawn=SimpleNamespace(usd_path="bowl.usd"))
        self.scene.rigid_objects["bowl"] = SimpleNamespace(data=bowl_data, cfg=bowl_cfg)

    def step(self, action):
        super().step(action)
        gripper = self.adapter.grippers[-1]
        cube = self.scene.rigid_objects["cube"]
        if gripper < 0:
            self.closed_once = True
            cube_position = self.adapter.pose[0, :3, 3]
        elif self.closed_once:
            cube_position = torch.tensor([0.1, 0.0, 0.03])
        else:
            cube_position = torch.zeros(3)
        cube.data.root_pos_w[0] = self.scene.env_origins[0] + cube_position


def _task_placement_geometry(object_id, asset_name, dimensions):
    return {
        "aabb_center_in_converted_object_origin_m": [0.0, 0.0, 0.0],
        "aabb_center_in_isaac_rigid_root_m": [0.0, 0.0, 0.0],
        "aabb_dimensions_m": list(dimensions),
        "aabb_kind": "converted_mesh_local_axis_aligned_bounding_box",
        "asset_name": asset_name,
        "converted_object_origin_from_aabb_center": [list(row) for row in _identity()],
        "isaac_rigid_root_from_aabb_center": [list(row) for row in _identity()],
        "object_id": object_id,
    }


def _destination_placement_geometry():
    return {
        "attested": True,
        "destination": _task_placement_geometry("bowl", "bowl_ycb_robolab", (0.16, 0.16, 0.05)),
        "frame_convention": "parent_from_child_homogeneous_4x4",
        "general_inside_semantics": False,
        "limitations": ["name_pinned_local_aabb_geometry_not_container_interior_geometry"],
        "placement_model": "destination_local_aabb_top_plane_shifted_downward",
        "predicted_aabb_center_offset_m": [0.0, 0.0, 0.03],
        "profile": "franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1",
        "relation": "on",
        "sampled_surface_extent_m": [0.0, 0.0, 0.0],
        "schema_version": 1,
        "subject": _task_placement_geometry(
            "cube",
            "rubiks_cube_hot3d_robolab",
            (0.06, 0.06, 0.06),
        ),
        "surface_config": {
            "implementation": "schedulestream.applications.custream.object.SurfaceConfig",
            "xy_extend_m": -0.16,
            "z_offset_m": -0.025,
        },
        "vertical_evidence_corridor_m": 0.04,
    }


def _grasp_geometry():
    primitives = [[list(row) for row in _yaw(angle)] for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    return {
        "asset_name": "rubiks_cube_hot3d_robolab",
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "converted_object_origin_from_aabb_center": [list(row) for row in _identity()],
        "link_from_object_transforms": primitives,
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": 4,
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": "cube",
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "primitive_link_from_aabb_center_transforms": primitives,
        "profile": "franka_rubiks_cube_offcenter_cuboid_top_v1",
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }


def _pick_place_plan() -> TaskMotionPlan:
    transport_poses = tuple(_translation(0.01 * index, 0.0, 0.004 * index) for index in range(1, 11))
    segments = (
        GripperCommandSegment(segment_id="initial_open", eef_name="tool", command=GripperCommandMode.OPEN),
        CartesianTrajectorySegment(
            segment_id="approach",
            depends_on=("initial_open",),
            eef_name="tool",
            frame="env_origin",
            poses=(_translation(0.0, 0.0, 0.0),),
        ),
        GripperCommandSegment(
            segment_id="close",
            depends_on=("approach",),
            eef_name="tool",
            command=GripperCommandMode.CLOSE,
        ),
        AttachIntentSegment(
            segment_id="attach",
            depends_on=("close",),
            eef_name="tool",
            object_name="cube",
            verifier="contact_and_relative_motion_v1",
        ),
        CartesianTrajectorySegment(
            segment_id="transport",
            depends_on=("attach",),
            eef_name="tool",
            frame="env_origin",
            poses=transport_poses,
        ),
        GripperCommandSegment(
            segment_id="release",
            depends_on=("transport",),
            eef_name="tool",
            command=GripperCommandMode.OPEN,
        ),
        DetachIntentSegment(
            segment_id="detach",
            depends_on=("release",),
            eef_name="tool",
            object_name="cube",
            verifier="contact_and_relative_motion_v1",
        ),
        CartesianTrajectorySegment(
            segment_id="retreat",
            depends_on=("detach",),
            eef_name="tool",
            frame="env_origin",
            poses=(_translation(0.14, 0.0, 0.07), _translation(0.18, 0.0, 0.10)),
        ),
    )
    return TaskMotionPlan(
        plan_id="pick_place_plan",
        request_digest="request",
        snapshot_digest="snapshot",
        backend="schedulestream_custream",
        backend_version="test",
        seed=3,
        segments=segments,
        goal=(GoalPredicate("on", "cube", "bowl"),),
        metadata={
            "arena_success_contract": {"attested": True},
            "schedulestream": {
                "attachment_events_preserved": True,
                "grasp_geometry": _grasp_geometry(),
                "destination_placement": _destination_placement_geometry(),
            },
        },
    )


class _ReleaseContactAdapter(_SemanticAdapter):
    def __init__(self, *, position_residual_m=0.0, rotation_residual_rad=0.0):
        super().__init__()
        self.position_residual_m = position_residual_m
        self.rotation_residual_rad = rotation_residual_rad

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict,
        gripper_action_dict,
        action_noise_dict,
        env_id,
    ):
        del action_noise_dict, env_id
        target = target_eef_pose_dict["tool"].clone()
        gripper = float(gripper_action_dict["tool"][0])
        if gripper < 0 and float(target[0, 3]) >= 0.099:
            target[0, 3] -= self.position_residual_m
            if self.rotation_residual_rad:
                target[:3, :3] = torch.tensor(_yaw(self.rotation_residual_rad))[:3, :3]
        self.pose[0] = target
        self.grippers.append(gripper)
        return torch.zeros(7)


class _OpenRetreatLagAdapter(_SemanticAdapter):
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict,
        gripper_action_dict,
        action_noise_dict,
        env_id,
    ):
        del action_noise_dict, env_id
        target = target_eef_pose_dict["tool"]
        gripper = float(gripper_action_dict["tool"][0])
        if gripper > 0 and any(value < 0 for value in self.grippers):
            self.pose[0, :3, 3] += 0.25 * (target[:3, 3] - self.pose[0, :3, 3])
            self.pose[0, :3, :3] = target[:3, :3]
        else:
            self.pose[0] = target
        self.grippers.append(gripper)
        return torch.zeros(7)


class _DisplacedDestinationSemanticEnv(_SemanticEnv):
    def step(self, action):
        super().step(action)
        if self.closed_once and self.adapter.grippers[-1] < 0 and float(self.adapter.pose[0, 0, 3]) >= 0.099:
            origin = self.scene.env_origins[0]
            self.scene.rigid_objects["bowl"].data.root_pos_w[0] = origin + torch.tensor([0.13, 0.0, 0.03])


class _NonfiniteReleaseSampleAdapter(_SemanticAdapter):
    def __init__(self):
        super().__init__()
        self.final_target_pose_reads = 0

    def get_eef_poses(self, env_ids):
        result = super().get_eef_poses(env_ids)
        if float(self.pose[0, 0, 3]) >= 0.099:
            self.final_target_pose_reads += 1
            if self.final_target_pose_reads == 5:
                result["tool"][0, 0, 3] = float("nan")
        return result


def _pick_place_plan_with_joint_seed() -> TaskMotionPlan:
    plan = _pick_place_plan()
    segments = list(plan.segments)
    transport_index = next(index for index, segment in enumerate(segments) if segment.segment_id == "transport")
    transport = segments[transport_index]
    assert isinstance(transport, CartesianTrajectorySegment)
    segments[transport_index] = replace(
        transport,
        joint_seed_names=("j1",),
        joint_seeds=tuple((0.1,) for _ in transport.poses),
    )
    return replace(plan, segments=tuple(segments))


def test_schedulestream_executor_requires_and_records_complete_pick_place_success_report():
    adapter = _SemanticAdapter()
    env = _SemanticEnv(adapter)

    def goal_verifier(_env, _env_id):
        cube = env.scene.rigid_objects["cube"].data.root_pos_w[0]
        bowl = env.scene.rigid_objects["bowl"].data.root_pos_w[0]
        return torch.linalg.norm(cube - bowl).item() < 1e-5

    executor = IsaacLabPlanExecutor(env, adapter, goal_verifier, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is True
    report = result.final_observation["task_success_report"]
    assert report["passed"] is True
    assert report["closed_transport"]["samples"] == 10
    assert report["plan_lifecycle"]["attach_segment_id"] == "attach"
    assert all(check["passed"] for check in report["checks"])
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["strict_target_converged"] is True
    assert release_boundary["contact_constrained_task_success_accepted"] is False
    assert release_boundary["convergence_mode"] == "strict_task_success_ready"
    assert release_boundary["task_success_release_gate"]["passed"] is True


def test_post_settle_terminal_drift_overrides_successful_task_verification():
    class _PostSettleSemanticDriftEnv(_SemanticEnv):
        def __init__(self, adapter):
            super().__init__(adapter)
            self.final_target_steps = 0

        def step(self, action):
            super().step(action)
            if self.adapter.grippers[-1] > 0 and float(self.adapter.pose[0, 0, 3]) >= 0.179:
                self.final_target_steps += 1
                if self.final_target_steps == 2:
                    self.adapter.pose[0, 0, 3] += 0.01

    adapter = _SemanticAdapter()
    env = _PostSettleSemanticDriftEnv(adapter)

    def goal_verifier(_env, _env_id):
        cube = env.scene.rigid_objects["cube"].data.root_pos_w[0]
        bowl = env.scene.rigid_objects["bowl"].data.root_pos_w[0]
        return torch.linalg.norm(cube - bowl).item() < 1e-5

    executor = IsaacLabPlanExecutor(env, adapter, goal_verifier, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is False
    assert result.failure_stage is FailureStage.VERIFICATION
    assert result.failure_code == "terminal_target_not_stable"
    assert result.final_observation["task_success_report"]["passed"] is True
    post_settle = result.final_observation["terminal_target_boundaries"][0]["post_settle_verification"]
    assert post_settle["position_error_m"] == pytest.approx(0.01)
    assert post_settle["passed"] is False


def test_checked_release_and_lagging_retreat_converge_before_physical_success():
    adapter = _OpenRetreatLagAdapter()
    env = _SemanticEnv(adapter)

    def goal_verifier(_env, _env_id):
        cube = env.scene.rigid_objects["cube"].data.root_pos_w[0]
        bowl = env.scene.rigid_objects["bowl"].data.root_pos_w[0]
        return torch.linalg.norm(cube - bowl).item() < 1e-5

    executor = IsaacLabPlanExecutor(env, adapter, goal_verifier, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is True
    assert result.final_observation["task_success_report"]["passed"] is True
    checks = {check["name"]: check for check in result.final_observation["task_success_report"]["checks"]}
    assert checks["final_eef_subject_separation_m"]["passed"] is True
    boundary = result.final_observation["terminal_target_boundaries"][0]
    assert boundary["target_segment_id"] == "retreat"
    assert boundary["target_sample_index"] == 1
    assert boundary["initial_position_error_m"] > 0.06
    assert boundary["final_position_error_m"] <= 0.005
    assert 1 <= boundary["correction_steps"] <= 12
    assert boundary["gripper_value"] == 1.0
    assert boundary["gripper_unchanged"] is True
    assert boundary["outcome"] == "converged"


@pytest.mark.parametrize("position_residual_m", [0.0095, 0.0200])
def test_release_contact_gate_accepts_bounded_contact_residual_with_explicit_evidence(position_residual_m):
    adapter = _ReleaseContactAdapter(position_residual_m=position_residual_m)
    env = _SemanticEnv(adapter)

    def goal_verifier(_env, _env_id):
        cube = env.scene.rigid_objects["cube"].data.root_pos_w[0]
        bowl = env.scene.rigid_objects["bowl"].data.root_pos_w[0]
        return torch.linalg.norm(cube - bowl).item() < 1e-5

    executor = IsaacLabPlanExecutor(env, adapter, goal_verifier, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is True
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["strict_target_converged"] is False
    assert release_boundary["contact_constrained_cartesian_ready"] is True
    assert release_boundary["contact_constrained_task_success_accepted"] is True
    assert release_boundary["convergence_mode"] == "contact_constrained_task_success_ready"
    assert release_boundary["release_contact_position_tolerance_m"] == 0.02
    assert release_boundary["release_contact_rotation_tolerance_rad"] == 0.05
    assert release_boundary["final_position_error_m"] == pytest.approx(position_residual_m, abs=1e-6)
    assert release_boundary["task_success_release_gate"]["passed"] is True


@pytest.mark.parametrize(
    ("position_residual_m", "rotation_residual_rad"),
    [(0.0201, 0.0), (0.0095, 0.051)],
)
def test_release_contact_gate_rejects_residual_outside_cartesian_contact_corridor(
    position_residual_m,
    rotation_residual_rad,
):
    adapter = _ReleaseContactAdapter(
        position_residual_m=position_residual_m,
        rotation_residual_rad=rotation_residual_rad,
    )
    env = _SemanticEnv(adapter)
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is False
    assert result.failure_code == "interaction_target_not_reached"
    assert adapter.grippers[-1] < 0
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["strict_target_converged"] is False
    assert release_boundary["contact_constrained_cartesian_ready"] is False
    assert release_boundary["contact_constrained_task_success_accepted"] is False
    assert release_boundary["task_success_release_gate"]["passed"] is True


def test_release_contact_gate_blocks_invalid_destination_even_at_strict_eef_target() -> None:
    adapter = _SemanticAdapter()
    env = _DisplacedDestinationSemanticEnv(adapter)
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is False
    assert result.failure_code == "release_task_success_not_ready"
    assert adapter.grippers[-1] < 0
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["strict_target_converged"] is True
    assert release_boundary["outcome"] == "failed_task_success_release_not_ready"
    checks = {check["name"]: check for check in release_boundary["task_success_release_gate"]["checks"]}
    assert checks["prerelease_subject_target_horizontal_radius_m"]["passed"] is False
    assert checks["prerelease_maximum_destination_drift_m"]["passed"] is False


def test_release_contact_gate_does_not_relax_joint_path_corridor() -> None:
    class _JointDriftContactAdapter(_ReleaseContactAdapter):
        def __init__(self):
            super().__init__(position_residual_m=0.0095)
            self.closed_final_target_calls = 0
            self.joint_drifted = False

        def get_joint_names(self):
            return ["j1", "panda_finger_joint1", "panda_finger_joint2"]

        def get_joint_positions(self, env_ids):
            del env_ids
            aperture = 0.08 if not self.grippers or self.grippers[-1] > 0 else 0.056
            j1 = 0.6 if self.joint_drifted else 0.1
            return torch.tensor([[j1, aperture / 2, aperture / 2]])

        def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id):
            target = target_eef_pose_dict["tool"]
            if float(gripper_action_dict["tool"][0]) < 0 and float(target[0, 3]) >= 0.099:
                self.closed_final_target_calls += 1
                self.joint_drifted = self.closed_final_target_calls >= 2
            return super().target_eef_pose_to_action(
                target_eef_pose_dict,
                gripper_action_dict,
                action_noise_dict,
                env_id,
            )

    adapter = _JointDriftContactAdapter()
    env = _SemanticEnv(adapter)
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan_with_joint_seed()))

    assert result.success is False
    assert result.failure_code == "interaction_target_not_reached"
    assert adapter.grippers[-1] < 0
    assert result.final_observation["maximum_joint_path_error_name"] == "j1"
    assert result.final_observation["maximum_joint_path_error_rad"] == pytest.approx(0.5)
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["contact_constrained_cartesian_ready"] is False
    assert release_boundary["task_success_release_gate"]["passed"] is True


def test_release_contact_gate_fails_closed_on_nonfinite_current_physical_state() -> None:
    adapter = _NonfiniteReleaseSampleAdapter()
    env = _SemanticEnv(adapter)
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: True, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is False
    assert result.failure_code == "task_success_state_unavailable"
    assert result.recoverable is False
    assert adapter.grippers[-1] < 0
    release_boundary = next(
        boundary
        for boundary in result.final_observation["interaction_boundaries"]
        if boundary["segment_id"] == "release"
    )
    assert release_boundary["outcome"] == "failed_task_success_unavailable"
    assert release_boundary["task_success_release_gate"] is None


def test_schedulestream_executor_rejects_proximity_only_empty_gripper_before_logical_attach():
    class _EmptyGripperAdapter(_SemanticAdapter):
        def get_joint_positions(self, env_ids):
            del env_ids
            aperture = 0.08 if not self.grippers or self.grippers[-1] > 0 else 0.006
            return torch.tensor([[aperture / 2, aperture / 2]])

    adapter = _EmptyGripperAdapter()
    env = _SemanticEnv(adapter)
    executor = IsaacLabPlanExecutor(env, adapter, lambda *_: False, final_settle_steps=4)

    result = asyncio.run(executor.execute(_request(), _pick_place_plan()))

    assert result.success is False
    assert result.failure_stage is FailureStage.EXECUTION
    assert result.failure_code == "grasp_not_verified"
    assert result.recoverable is True
    assert executor.attachment_state.held_by_eef == {"tool": None}
    report = result.final_observation["task_success_report"]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["attach_observed"]["passed"] is False
    assert checks["grasp_aperture_min_m"]["observed"] == pytest.approx(0.006)
