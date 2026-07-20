# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab runtime and typed-plan executor for autonomous generation.

This module intentionally uses duck-typed Isaac Lab objects. It can be imported and unit-tested
without launching SimulationApp; the live environment is required only when instances are used.
"""

from __future__ import annotations

import math
import time
import torch
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from isaac_autodata_core.autonomous.attempt_generation import (
    AttemptGenerationError,
    AttemptRequest,
    ExecutionResult,
    FailureStage,
)
from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    BarrierSegment,
    CartesianTrajectorySegment,
    ConcurrentGroupSegment,
    DetachIntentSegment,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionOutcome,
    GripperCommandMode,
    GripperCommandSegment,
    JointTrajectorySegment,
    RobotStateSnapshot,
    SceneObjectSnapshot,
    SceneSnapshot,
    TaskMotionPlan,
    WaitSegment,
    make_stable_id,
)
from isaac_autodata_interfaces.autonomous.pick_place_success import PickPlaceSuccessTracker

SuccessVerifier = Callable[[Any, int], bool | Sequence[bool] | torch.Tensor]
AttachmentVerifier = Callable[[Any, Any, int, str, str], bool]

# The current reviewed live executor is the normalized delta-pose Franka IK profile. Keep these
# limits local to that physical execution boundary until embodiment-specific limits become part of
# the public request contract. At 20 ms, the velocity limits admit at most 40 mm / 80 mrad between
# samples, well above the normal dense cuRobo interpolation while rejecting sparse teleports.
_RAW_DIK_POSE_ACTION_LIMIT = 1.0
_RAW_DIK_SATURATION_TOLERANCE = 1e-6
_MAX_CARTESIAN_TRANSLATION_STEP_M = 0.10
_MAX_CARTESIAN_ROTATION_STEP_RAD = 0.25
_MAX_CARTESIAN_LINEAR_VELOCITY_M_S = 2.0
_MAX_CARTESIAN_ANGULAR_VELOCITY_RAD_S = 4.0
_MAX_FRANKA_EEF_REACH_M = 1.0
_RELEASE_CONTACT_POSITION_TOLERANCE_M = 0.020
_RELEASE_CONTACT_ROTATION_TOLERANCE_RAD = 0.050
_PHYSICAL_BOUNDARY_ABS_TOLERANCE = 1e-7
_RETRYABLE_TRACKING_FAILURE_CODES = frozenset({
    "eef_position_tracking_error",
    "eef_rotation_tracking_error",
    "joint_path_tracking_error",
})


@dataclass(frozen=True)
class _CartesianTargetState:
    """Last collision-checked Cartesian target and optional joint-path diagnostics."""

    pose: torch.Tensor
    segment_id: str
    sample_index: int
    joint_seed_names: tuple[str, ...]
    joint_seed: tuple[float, ...] | None


@dataclass
class AttachmentState:
    """Executor-observed object held by each end effector."""

    held_by_eef: dict[str, str | None] = field(default_factory=dict)

    def reset(self, eef_names: Sequence[str]) -> None:
        """Clear all attachment state for a new episode."""

        self.held_by_eef = {name: None for name in eef_names}


class IsaacLabAttemptRuntime:
    """Reset, snapshot, and recorder lifecycle over one Isaac Lab environment."""

    def __init__(
        self,
        env: Any,
        embodiment_adapter: Any,
        *,
        graph_nodes: Sequence[Mapping[str, Any]],
        attachment_state: AttachmentState | None = None,
        reset_settle_steps: int = 10,
    ) -> None:
        if isinstance(reset_settle_steps, bool) or not isinstance(reset_settle_steps, int):
            raise ValueError("reset_settle_steps must be an integer")
        if not 0 <= reset_settle_steps <= 100:
            raise ValueError("reset_settle_steps must be in [0, 100]")
        self.env = _base_env(env)
        self.embodiment_adapter = embodiment_adapter
        self.graph_nodes = tuple(dict(node) for node in graph_nodes)
        self.attachment_state = attachment_state or AttachmentState()
        self.reset_settle_steps = reset_settle_steps
        if getattr(embodiment_adapter, "env", None) is None:
            embodiment_adapter.bind_env(self.env)
        self._validate_live_scene_bindings()
        self.attachment_state.reset(embodiment_adapter.get_eef_names())
        self._finalization_by_env: dict[int, tuple[str, bool, bool]] = {}

    async def reset_attempt(self, env_id: int) -> Mapping[str, Any]:
        """Clear recorder/attachment buffers and reset one environment instance."""

        # Beginning a new attempt invalidates the prior attempt's completion marker before any
        # fallible reset operation. If recorder or environment reset fails, the orchestrator still
        # owns a fresh attempt that must be finalized exactly once rather than being mistaken for
        # the preceding completed attempt.
        self._finalization_by_env.pop(env_id, None)
        env_ids = _env_id_tensor(self.env, env_id)
        recorder = getattr(self.env, "recorder_manager", None)
        if recorder is not None:
            recorder.reset(env_ids=env_ids)
        self.attachment_state.reset(self.embodiment_adapter.get_eef_names())
        self.env.reset(env_ids=env_ids)
        self._settle_reset_state(env_id)
        # The reset transient and its open/hold actions are setup, not generated task data.
        recorder_initial_state_recaptured = False
        if recorder is not None and self.reset_settle_steps:
            recorder.reset(env_ids=env_ids)
            # Recorder reset removes the setup samples and the initial state captured by
            # ``env.reset``. Re-run the public post-reset lifecycle at the settled state so the
            # surviving episode starts with exactly one matching ``initial_state``.
            recorder.record_post_reset(env_ids)
            recorder_initial_state_recaptured = True
        return {
            "env_id": env_id,
            "recorder_excludes_reset_settling": recorder is not None,
            "recorder_initial_state_recaptured_after_settling": recorder_initial_state_recaptured,
            "reset_completed": True,
            "reset_settle_duration_s": self.reset_settle_steps * _step_dt(self.env),
            "reset_settle_steps": self.reset_settle_steps,
        }

    def _settle_reset_state(self, env_id: int) -> None:
        """Hold the reset EEF pose with an open gripper while randomized objects settle."""

        if not self.reset_settle_steps:
            return
        eef_names = tuple(self.embodiment_adapter.get_eef_names())
        if not eef_names:
            raise ValueError("reset settling requires at least one end effector")
        observed = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])
        if set(observed) != set(eef_names):
            raise ValueError("reset settling requires one observed pose for every end effector")
        targets: dict[str, torch.Tensor] = {}
        for name in eef_names:
            pose = torch.as_tensor(observed[name][0], dtype=torch.float32, device=_device(self.env))
            if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all().item()):
                raise ValueError(f"reset settling observed an invalid pose for end effector {name!r}")
            targets[name] = pose.detach().clone()
        grippers = {
            name: torch.ones(
                int(self.embodiment_adapter.gripper_action_dim),
                dtype=torch.float32,
                device=_device(self.env),
            )
            for name in eef_names
        }
        for _ in range(self.reset_settle_steps):
            action = self.embodiment_adapter.target_eef_pose_to_action(
                target_eef_pose_dict=targets,
                gripper_action_dict=grippers,
                action_noise_dict=None,
                env_id=env_id,
            )
            _validate_raw_dik_action(self.env, self.embodiment_adapter, action, context="reset settling")
            action = torch.as_tensor(action)
            batch = torch.zeros(self.env.action_space.shape, dtype=action.dtype, device=_device(self.env))
            batch[env_id] = action
            self.env.step(batch)

    def capture_scene_snapshot(self, env_id: int, *, snapshot_id: str) -> SceneSnapshot:
        """Capture finite robot/object state in the environment-origin frame."""

        self._validate_live_scene_bindings()
        joint_positions = _row_to_tuple(self.embodiment_adapter.get_joint_positions(env_ids=[env_id])[0])
        joint_names = tuple(self.embodiment_adapter.get_joint_names())
        eef_poses = {
            name: _matrix_to_tuple(pose[0])
            for name, pose in self.embodiment_adapter.get_eef_poses(env_ids=[env_id]).items()
        }
        embodiment_node = next((node for node in self.graph_nodes if node.get("type") == "embodiment"), None)
        robot_id = (
            str(embodiment_node["id"])
            if embodiment_node is not None
            else str(getattr(self.embodiment_adapter, "name", "robot"))
        )
        robot = RobotStateSnapshot(
            robot_id=robot_id,
            joint_names=joint_names,
            joint_positions=joint_positions,
            eef_poses=eef_poses,
            held_objects=dict(self.attachment_state.held_by_eef),
        )

        graph_nodes = {str(node.get("id")): node for node in self.graph_nodes if node.get("id") is not None}
        origin = _env_origin(self.env, env_id)
        objects: list[SceneObjectSnapshot] = []
        for scene_id, scene_object in sorted(self.env.scene.rigid_objects.items()):
            position = _tensor_row(scene_object.data.root_pos_w, env_id) - origin
            quaternion_xyzw = _tensor_row(scene_object.data.root_quat_w, env_id)
            pose = _pose_from_xyzw(position, quaternion_xyzw)
            node = graph_nodes.get(scene_id)
            roles = () if node is None else (str(node.get("type", "object")),)
            geometry_ref = _geometry_ref(scene_object)
            objects.append(
                SceneObjectSnapshot(
                    semantic_id=scene_id,
                    scene_id=scene_id,
                    pose=_matrix_to_tuple(pose),
                    geometry_ref=geometry_ref,
                    roles=roles,
                )
            )

        return SceneSnapshot(
            snapshot_id=snapshot_id,
            captured_at_s=time.time(),
            env_id=env_id,
            robot=robot,
            objects=tuple(objects),
            metadata={
                "frame": "env_origin",
                "rigid_object_quaternion_convention": "xyzw",
                "step_dt_s": _step_dt(self.env),
                "scene_binding": "validated_exact_v1",
            },
        )

    def _validate_live_scene_bindings(self) -> None:
        """Require an exact semantic-to-live binding before planning can observe the scene."""

        node_ids = [node.get("id") for node in self.graph_nodes]
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise ValueError("every linked graph node must have a non-empty string id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("linked graph node ids must be unique")
        embodiment_nodes = [node for node in self.graph_nodes if node.get("type") == "embodiment"]
        if len(embodiment_nodes) != 1:
            raise ValueError(f"live scene binding requires exactly one embodiment node, got {len(embodiment_nodes)}")
        articulations = getattr(self.env.scene, "articulations", None)
        if not isinstance(articulations, Mapping) or len(articulations) != 1:
            names = () if not isinstance(articulations, Mapping) else tuple(articulations)
            raise ValueError(f"live scene binding requires exactly one articulation, got {names}")
        semantic_objects = {
            str(node["id"]) for node in self.graph_nodes if node.get("type") in ("object", "object_reference")
        }
        rigid_objects = getattr(self.env.scene, "rigid_objects", None)
        if not isinstance(rigid_objects, Mapping):
            raise ValueError("live scene does not expose a rigid-object mapping")
        live_objects = set(rigid_objects)
        if live_objects != semantic_objects:
            missing = sorted(semantic_objects - live_objects)
            unexpected = sorted(live_objects - semantic_objects)
            raise ValueError(
                "linked object ids must bind exactly to live rigid objects "
                f"(missing={missing}, unexpected={unexpected})"
            )
        for name, scene_object in rigid_objects.items():
            geometry_ref = _geometry_ref(scene_object)
            if not isinstance(geometry_ref, str) or not geometry_ref:
                raise ValueError(f"live rigid object {name!r} has no stable geometry reference")

    async def finish_attempt(self, env_id: int, *, success: bool, keep_failed: bool) -> None:
        """Stamp final success and export the recorder buffer according to retention policy."""

        requested = (success, keep_failed)
        prior = self._finalization_by_env.get(env_id)
        if prior is not None:
            state, prior_success, prior_keep_failed = prior
            if (prior_success, prior_keep_failed) != requested:
                raise RuntimeError(f"attempt for env {env_id} was already finalized with different retention state")
            if state == "complete":
                return
            raise RuntimeError(f"attempt finalization for env {env_id} is {state}; refusing a duplicate export")
        self._finalization_by_env[env_id] = ("in_progress", success, keep_failed)
        recorder = getattr(self.env, "recorder_manager", None)
        if recorder is None:
            self._finalization_by_env[env_id] = ("complete", success, keep_failed)
            return
        try:
            env_ids = _env_id_tensor(self.env, env_id)
            success_tensor = torch.tensor([[success]], dtype=torch.bool, device=_device(self.env))
            recorder.set_success_to_episodes(env_ids, success_tensor)
            if success or keep_failed:
                recorder.export_episodes(env_ids)
        except BaseException:
            self._finalization_by_env[env_id] = ("failed", success, keep_failed)
            raise
        self._finalization_by_env[env_id] = ("complete", success, keep_failed)


class IsaacLabPlanExecutor:
    """Execute a validated, currently single-arm task-motion plan in Isaac Lab."""

    def __init__(
        self,
        env: Any,
        embodiment_adapter: Any,
        success_verifier: SuccessVerifier,
        *,
        attachment_state: AttachmentState | None = None,
        attachment_verifier: AttachmentVerifier | None = None,
        final_settle_steps: int = 5,
        final_stability_steps: int | None = None,
        max_steps: int = 20_000,
        attachment_distance_m: float = 0.05,
        max_eef_position_error_m: float = 0.08,
        max_eef_rotation_error_rad: float = 0.50,
        max_joint_path_error_rad: float = 0.35,
        max_tracking_correction_steps: int = 12,
        interaction_position_tolerance_m: float = 0.005,
        interaction_rotation_tolerance_rad: float = 0.05,
        max_interaction_correction_steps: int = 12,
        max_terminal_correction_steps: int = 32,
    ) -> None:
        if final_settle_steps < 0:
            raise ValueError("final_settle_steps must be non-negative")
        if final_stability_steps is None:
            final_stability_steps = min(5, final_settle_steps + 1)
        if final_stability_steps < 1 or final_stability_steps > final_settle_steps + 1:
            raise ValueError("final_stability_steps must be in [1, final_settle_steps + 1]")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if (
            isinstance(max_tracking_correction_steps, bool)
            or not isinstance(max_tracking_correction_steps, int)
            or not 0 <= max_tracking_correction_steps <= 100
        ):
            raise ValueError("max_tracking_correction_steps must be an integer in [0, 100]")
        if (
            isinstance(max_interaction_correction_steps, bool)
            or not isinstance(max_interaction_correction_steps, int)
            or not 0 <= max_interaction_correction_steps <= 100
        ):
            raise ValueError("max_interaction_correction_steps must be an integer in [0, 100]")
        if (
            isinstance(max_terminal_correction_steps, bool)
            or not isinstance(max_terminal_correction_steps, int)
            or not 0 <= max_terminal_correction_steps <= 100
        ):
            raise ValueError("max_terminal_correction_steps must be an integer in [0, 100]")
        if not math.isfinite(attachment_distance_m) or attachment_distance_m <= 0:
            raise ValueError("attachment_distance_m must be finite and positive")
        for field_name, value in (
            ("max_eef_position_error_m", max_eef_position_error_m),
            ("max_eef_rotation_error_rad", max_eef_rotation_error_rad),
            ("interaction_position_tolerance_m", interaction_position_tolerance_m),
            ("interaction_rotation_tolerance_rad", interaction_rotation_tolerance_rad),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if (
            isinstance(max_joint_path_error_rad, bool)
            or not isinstance(max_joint_path_error_rad, (int, float))
            or not math.isfinite(max_joint_path_error_rad)
            or max_joint_path_error_rad <= 0
        ):
            raise ValueError("max_joint_path_error_rad must be finite and positive")
        if interaction_position_tolerance_m > max_eef_position_error_m:
            raise ValueError("interaction_position_tolerance_m must not exceed max_eef_position_error_m")
        if interaction_rotation_tolerance_rad > max_eef_rotation_error_rad:
            raise ValueError("interaction_rotation_tolerance_rad must not exceed max_eef_rotation_error_rad")
        self.env = _base_env(env)
        self.embodiment_adapter = embodiment_adapter
        self.success_verifier = success_verifier
        self.attachment_state = attachment_state or AttachmentState()
        self.attachment_verifier = attachment_verifier
        self.final_settle_steps = final_settle_steps
        self.final_stability_steps = final_stability_steps
        self.max_steps = max_steps
        self.attachment_distance_m = attachment_distance_m
        self.max_eef_position_error_m = max_eef_position_error_m
        self.max_eef_rotation_error_rad = max_eef_rotation_error_rad
        # ScheduleStream's native DIK PathController can carry cuRobo joint samples as diagnostics
        # while commanding only Cartesian poses. Enforce a finite null-space corridor whenever
        # those diagnostics are present; plans that legitimately omit them remain Cartesian-only.
        self.max_joint_path_error_rad = max_joint_path_error_rad
        self.max_tracking_correction_steps = max_tracking_correction_steps
        self.interaction_position_tolerance_m = interaction_position_tolerance_m
        self.interaction_rotation_tolerance_rad = interaction_rotation_tolerance_rad
        self.max_interaction_correction_steps = max_interaction_correction_steps
        self.max_terminal_correction_steps = max_terminal_correction_steps
        self._gripper_values: dict[str, float] = {}
        self._last_cartesian_targets: dict[str, _CartesianTargetState] = {}
        self._step_count = 0
        self._tracking_correction_steps = 0
        self._maximum_eef_position_error_m = 0.0
        self._maximum_eef_rotation_error_rad = 0.0
        self._maximum_joint_path_error_rad = 0.0
        self._maximum_joint_path_error_name: str | None = None
        self._gripper_milestones: list[dict[str, Any]] = []
        self._interaction_boundaries: list[dict[str, Any]] = []
        self._interaction_correction_steps = 0
        self._terminal_target_boundaries: list[dict[str, Any]] = []
        self._terminal_correction_steps = 0
        self._event_count = 0
        self._pick_place_success: PickPlaceSuccessTracker | None = None

    async def execute(self, request: AttemptRequest, plan: TaskMotionPlan) -> ExecutionResult:
        """Execute segments in declared order and verify only the settled final state."""

        events: list[ExecutionEvent] = []
        self._step_count = 0
        self._tracking_correction_steps = 0
        self._maximum_eef_position_error_m = 0.0
        self._maximum_eef_rotation_error_rad = 0.0
        self._maximum_joint_path_error_rad = 0.0
        self._maximum_joint_path_error_name = None
        self._gripper_milestones = []
        self._interaction_boundaries = []
        self._interaction_correction_steps = 0
        self._terminal_target_boundaries = []
        self._terminal_correction_steps = 0
        self._last_cartesian_targets = {}
        self._event_count = 0
        self._pick_place_success = None
        eef_names = tuple(self.embodiment_adapter.get_eef_names())
        if not eef_names:
            raise AttemptGenerationError(FailureStage.EXECUTION, "no_end_effector", "adapter has no end effector")
        self._gripper_values = {name: 1.0 for name in eef_names}
        if not self.attachment_state.held_by_eef:
            self.attachment_state.reset(eef_names)

        completed: set[str] = set()
        try:
            try:
                self._pick_place_success = PickPlaceSuccessTracker.from_plan(plan, eef_names)
            except ValueError as exc:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "pick_place_lifecycle_invalid",
                    str(exc),
                    recoverable=False,
                ) from exc
            # Validate the complete plan before taking the first action. In particular, a native
            # ScheduleStream stream may contain an otherwise valid Cartesian prefix followed by a
            # joint or concurrent segment that this IK executor cannot represent. Discovering that
            # only after executing the prefix would leave the robot in an unaccounted partial state.
            self._validate_plan(plan, eef_names, request.env_id)
            events.append(self._event(request, plan, None, ExecutionEventType.PLAN_STARTED, ExecutionOutcome.PENDING))
            for segment in plan.segments:
                missing = set(segment.depends_on) - completed
                if missing:
                    raise AttemptGenerationError(
                        FailureStage.EXECUTION,
                        "dependency_not_completed",
                        f"segment {segment.segment_id!r} dependencies were not completed: {sorted(missing)}",
                        recoverable=False,
                    )
                events.append(
                    self._event(
                        request,
                        plan,
                        segment.segment_id,
                        ExecutionEventType.SEGMENT_STARTED,
                        ExecutionOutcome.PENDING,
                    )
                )
                self._execute_segment(request.env_id, segment)
                completed.add(segment.segment_id)
                events.append(
                    self._event(
                        request,
                        plan,
                        segment.segment_id,
                        ExecutionEventType.SEGMENT_COMPLETED,
                        ExecutionOutcome.SUCCEEDED,
                    )
                )

            self._converge_terminal_targets(request.env_id, eef_names)
            success_streak = 0
            for verification_samples, settle_index in enumerate(range(self.final_settle_steps + 1), start=1):
                if settle_index:
                    self._hold(request.env_id, 1)
                goal_satisfied = _env_bool(self.success_verifier(self.env, request.env_id), request.env_id)
                if self._pick_place_success is not None:
                    self._observe_task_success_final(request.env_id, goal_satisfied=goal_satisfied)
                if goal_satisfied:
                    success_streak += 1
                else:
                    success_streak = 0
            self._verify_terminal_targets_after_settling(
                request.env_id,
                eef_names,
                verification_samples=verification_samples,
            )
            arena_success = success_streak >= self.final_stability_steps
            physical_observation = self._physical_observation(request.env_id)
            task_success_report = physical_observation.get("task_success_report")
            task_success = task_success_report is None or task_success_report["passed"] is True
            if not arena_success or not task_success:
                if arena_success:
                    failure_code = "task_success_checks_failed"
                    failure_message = "live task-success checks rejected the physical pick-and-place result"
                else:
                    failure_code = "final_goal_not_satisfied"
                    failure_message = "task success predicate was not stable after final settling"
                events.append(
                    self._event(
                        request,
                        plan,
                        None,
                        ExecutionEventType.TASK_REJECTED,
                        ExecutionOutcome.FAILED,
                        failure_code=failure_code,
                        message=failure_message,
                    )
                )
                return ExecutionResult(
                    success=False,
                    events=tuple(events),
                    final_observation={
                        "final_success": False,
                        "final_success_streak": success_streak,
                        "steps": self._step_count,
                        "tracking_correction_steps": self._tracking_correction_steps,
                        **self._tracking_observation(),
                        **physical_observation,
                        "verification_samples": verification_samples,
                    },
                    failure_stage=FailureStage.VERIFICATION,
                    failure_code=failure_code,
                    failure_message=failure_message,
                )
            events.append(
                self._event(request, plan, None, ExecutionEventType.TASK_VERIFIED, ExecutionOutcome.SUCCEEDED)
            )
            events.append(
                self._event(request, plan, None, ExecutionEventType.PLAN_COMPLETED, ExecutionOutcome.SUCCEEDED)
            )
            return ExecutionResult(
                success=True,
                events=tuple(events),
                final_observation={
                    "final_success": True,
                    "final_success_streak": success_streak,
                    "steps": self._step_count,
                    "tracking_correction_steps": self._tracking_correction_steps,
                    **self._tracking_observation(),
                    **physical_observation,
                    "verification_samples": verification_samples,
                },
            )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, AttemptGenerationError)
                else AttemptGenerationError(FailureStage.EXECUTION, "segment_execution_failed", str(exc))
            )
            events.append(
                self._event(
                    request,
                    plan,
                    None,
                    ExecutionEventType.PLAN_FAILED,
                    ExecutionOutcome.FAILED,
                    failure_code=error.code,
                    message=str(error)[:2048],
                )
            )
            return ExecutionResult(
                success=False,
                events=tuple(events),
                final_observation={
                    "final_success": False,
                    "steps": self._step_count,
                    "tracking_correction_steps": self._tracking_correction_steps,
                    **self._tracking_observation(),
                    **self._physical_observation(request.env_id),
                },
                failure_stage=error.stage,
                failure_code=error.code,
                failure_message=str(error),
                recoverable=error.recoverable,
            )

    def _validate_plan(self, plan: TaskMotionPlan, eef_names: tuple[str, ...], env_id: int) -> None:
        """Fail before execution when a segment or total step budget is unsupported."""

        estimated_steps = self.final_settle_steps
        known_eefs = set(eef_names)
        known_joints = set(self.embodiment_adapter.get_joint_names())
        predicted_poses: dict[str, torch.Tensor] | None = None
        robot_base_position: torch.Tensor | None = None
        predicted_gripper_values = {eef_name: 1.0 for eef_name in eef_names}
        eefs_with_cartesian_target: set[str] = set()
        prior_segment_ids: set[str] = set()
        for segment in plan.segments:
            missing_prior_dependencies = set(segment.depends_on) - prior_segment_ids
            if missing_prior_dependencies:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "plan_not_topologically_ordered",
                    f"segment {segment.segment_id!r} depends on later segments {sorted(missing_prior_dependencies)}",
                    recoverable=False,
                )
            if isinstance(segment, JointTrajectorySegment):
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "joint_trajectory_not_supported_by_ik_executor",
                    "joint trajectories require a joint-control embodiment executor",
                    recoverable=False,
                )
            if isinstance(segment, ConcurrentGroupSegment):
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "concurrent_group_not_supported_by_single_arm_executor",
                    "concurrent groups require a multi-arm executor",
                    recoverable=False,
                )
            if not isinstance(
                segment,
                (
                    CartesianTrajectorySegment,
                    GripperCommandSegment,
                    AttachIntentSegment,
                    DetachIntentSegment,
                    WaitSegment,
                    BarrierSegment,
                ),
            ):
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "unsupported_segment",
                    f"unsupported plan segment {type(segment).__name__}",
                    recoverable=False,
                )
            eef_name = getattr(segment, "eef_name", None)
            if eef_name is not None and eef_name not in known_eefs:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "unknown_end_effector",
                    f"plan references unknown end effector {eef_name!r}",
                    recoverable=False,
                )
            if isinstance(segment, CartesianTrajectorySegment):
                if segment.frame not in ("env", "env_origin", "world"):
                    raise AttemptGenerationError(
                        FailureStage.EXECUTION,
                        "unsupported_pose_frame",
                        f"executor supports only 'env_origin' and 'world' Cartesian poses, got {segment.frame!r}",
                        recoverable=False,
                    )
                missing_joints = set(segment.joint_seed_names) - known_joints
                if missing_joints:
                    raise AttemptGenerationError(
                        FailureStage.EXECUTION,
                        "joint_tracking_observation_missing",
                        f"joint tracking observations are missing {sorted(missing_joints)}",
                        recoverable=False,
                    )
                if predicted_poses is None:
                    predicted_poses = self._initial_eef_poses(env_id, eef_names)
                    robot_base_position = _robot_base_position_in_env(self.env, env_id)
                self._validate_cartesian_trajectory(
                    env_id,
                    segment,
                    predicted_poses,
                    robot_base_position,
                )
                eefs_with_cartesian_target.add(segment.eef_name)
                estimated_steps += len(segment.poses) * (1 + self.max_tracking_correction_steps)
            elif isinstance(segment, GripperCommandSegment):
                requested_gripper_value = _gripper_value(segment)
                previous_gripper_value = predicted_gripper_values[segment.eef_name]
                if requested_gripper_value != previous_gripper_value:
                    if segment.eef_name not in eefs_with_cartesian_target:
                        raise AttemptGenerationError(
                            FailureStage.EXECUTION,
                            "interaction_target_missing",
                            f"gripper transition {segment.segment_id!r} has no preceding collision-checked "
                            f"Cartesian target for {segment.eef_name!r}",
                            recoverable=False,
                        )
                    estimated_steps += self.max_interaction_correction_steps
                predicted_gripper_values[segment.eef_name] = requested_gripper_value
                estimated_steps += segment.settle_steps
            elif isinstance(segment, WaitSegment):
                estimated_steps += (
                    segment.steps
                    if segment.steps is not None
                    else max(1, math.ceil((segment.duration_s or 0.0) / _step_dt(self.env)))
                )
            prior_segment_ids.add(segment.segment_id)
        # The ordinary path corridor bounds transient DIK error but is intentionally wider than
        # the terminal physical-evidence boundary. Reserve one strict, bounded convergence window
        # for every end effector that received a collision-checked Cartesian target.
        estimated_steps += len(eefs_with_cartesian_target) * self.max_terminal_correction_steps
        if estimated_steps > self.max_steps:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "execution_step_limit",
                f"plan requires approximately {estimated_steps} simulator steps; limit is {self.max_steps}",
            )

    def _initial_eef_poses(self, env_id: int, eef_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        """Read finite live EEF poses used as the first trajectory continuity anchors."""

        try:
            observed = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])
        except Exception as exc:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "tracking_observation_missing",
                f"failed to read initial end-effector poses: {type(exc).__name__}: {str(exc)[:512]}",
                recoverable=False,
            ) from exc
        poses: dict[str, torch.Tensor] = {}
        for eef_name in eef_names:
            if eef_name not in observed:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "tracking_observation_missing",
                    f"no initial observed pose is available for end effector {eef_name!r}",
                    recoverable=False,
                )
            pose = torch.as_tensor(observed[eef_name][0], dtype=torch.float64, device=_device(self.env))
            if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all().item()):
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "tracking_observation_invalid",
                    f"initial observed pose for {eef_name!r} must be a finite 4x4 transform",
                    recoverable=False,
                )
            poses[eef_name] = pose
        return poses

    def _validate_cartesian_trajectory(
        self,
        env_id: int,
        segment: CartesianTrajectorySegment,
        predicted_poses: dict[str, torch.Tensor],
        robot_base_position: torch.Tensor,
    ) -> None:
        """Validate every Cartesian sample against live Franka execution limits."""

        step_dt_s = _cartesian_step_dt(segment, self.env)
        previous = predicted_poses[segment.eef_name]
        for sample_index, pose in enumerate(segment.poses):
            target = torch.tensor(pose, dtype=torch.float64, device=_device(self.env))
            target = _target_in_env_frame(self.env, env_id, target, segment.frame)
            distance_from_base = float(torch.linalg.norm(target[:3, 3] - robot_base_position).item())
            if not math.isfinite(distance_from_base) or distance_from_base > _MAX_FRANKA_EEF_REACH_M:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "eef_workspace_limit",
                    f"segment {segment.segment_id!r} sample {sample_index} is {distance_from_base:.6g}m "
                    f"from the robot base; Franka execution limit is {_MAX_FRANKA_EEF_REACH_M:.6g}m",
                )

            translation_step = float(torch.linalg.norm(target[:3, 3] - previous[:3, 3]).item())
            rotation_step = _rotation_distance_rad(previous[:3, :3], target[:3, :3])
            if translation_step > _MAX_CARTESIAN_TRANSLATION_STEP_M + 1e-9:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "cartesian_translation_discontinuity",
                    f"segment {segment.segment_id!r} sample {sample_index} translates {translation_step:.6g}m; "
                    f"limit is {_MAX_CARTESIAN_TRANSLATION_STEP_M:.6g}m per sample",
                )
            if rotation_step > _MAX_CARTESIAN_ROTATION_STEP_RAD + 1e-9:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "cartesian_rotation_discontinuity",
                    f"segment {segment.segment_id!r} sample {sample_index} rotates {rotation_step:.6g}rad; "
                    f"limit is {_MAX_CARTESIAN_ROTATION_STEP_RAD:.6g}rad per sample",
                )
            linear_velocity = translation_step / step_dt_s
            angular_velocity = rotation_step / step_dt_s
            if linear_velocity > _MAX_CARTESIAN_LINEAR_VELOCITY_M_S + 1e-6:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "cartesian_linear_velocity_limit",
                    f"segment {segment.segment_id!r} sample {sample_index} implies {linear_velocity:.6g}m/s; "
                    f"limit is {_MAX_CARTESIAN_LINEAR_VELOCITY_M_S:.6g}m/s",
                )
            if angular_velocity > _MAX_CARTESIAN_ANGULAR_VELOCITY_RAD_S + 1e-6:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "cartesian_angular_velocity_limit",
                    f"segment {segment.segment_id!r} sample {sample_index} implies {angular_velocity:.6g}rad/s; "
                    f"limit is {_MAX_CARTESIAN_ANGULAR_VELOCITY_RAD_S:.6g}rad/s",
                )
            previous = target
        predicted_poses[segment.eef_name] = previous

    def _execute_segment(self, env_id: int, segment: Any) -> None:
        if isinstance(segment, CartesianTrajectorySegment):
            self._require_eef(segment.eef_name)
            for sample_index, pose in enumerate(segment.poses):
                target = torch.tensor(pose, dtype=torch.float32, device=_device(self.env))
                target = _target_in_env_frame(self.env, env_id, target, segment.frame)
                joint_seed = None if not segment.joint_seeds else segment.joint_seeds[sample_index]
                self._step_target(
                    env_id,
                    segment.eef_name,
                    target,
                    joint_seed_names=segment.joint_seed_names,
                    joint_seed=joint_seed,
                )
                self._last_cartesian_targets[segment.eef_name] = _CartesianTargetState(
                    pose=target.detach().clone(),
                    segment_id=segment.segment_id,
                    sample_index=sample_index,
                    joint_seed_names=tuple(segment.joint_seed_names),
                    joint_seed=None if joint_seed is None else tuple(float(value) for value in joint_seed),
                )
            return
        if isinstance(segment, GripperCommandSegment):
            self._require_eef(segment.eef_name)
            requested_gripper_value = _gripper_value(segment)
            interaction_evidence = self._prepare_gripper_interaction(
                env_id,
                segment,
                requested_gripper_value,
            )
            if (
                self._pick_place_success is not None
                and segment.segment_id == self._pick_place_success.release_open_segment_id
            ):
                self._update_pick_place_success(
                    "begin_release",
                    segment.segment_id,
                    self._task_success_sample(env_id),
                )
            self._gripper_values[segment.eef_name] = requested_gripper_value
            self._hold(env_id, segment.settle_steps)
            milestone = self._capture_physical_state(env_id)
            milestone.update({
                "command": segment.command.value,
                "interaction_boundary": dict(interaction_evidence),
                "segment_id": segment.segment_id,
                "step": self._step_count,
            })
            self._gripper_milestones.append(milestone)
            if (
                self._pick_place_success is not None
                and segment.segment_id == self._pick_place_success.initial_open_segment_id
            ):
                initial_goal_satisfied = _env_bool(self.success_verifier(self.env, env_id), env_id)
                self._update_pick_place_success(
                    "settle_initial",
                    segment.segment_id,
                    self._task_success_sample(env_id),
                    goal_satisfied=initial_goal_satisfied,
                )
            return
        if isinstance(segment, AttachIntentSegment):
            self._require_eef(segment.eef_name)
            if self.attachment_state.held_by_eef.get(segment.eef_name) is not None:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "end_effector_already_holding",
                    f"{segment.eef_name!r} already holds an object",
                )
            verified = (
                self.attachment_verifier(
                    self.env,
                    self.embodiment_adapter,
                    env_id,
                    segment.eef_name,
                    segment.object_name,
                )
                if self.attachment_verifier is not None
                else self._verify_attachment_proximity(env_id, segment.eef_name, segment.object_name)
            )
            if not verified:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "grasp_not_verified",
                    f"attachment verifier {segment.verifier!r} rejected {segment.object_name!r}",
                )
            task_success_sample = None
            if self._pick_place_success is not None:
                task_success_sample = self._task_success_sample(env_id)
                physically_plausible = self._update_pick_place_success(
                    "attachment_is_plausible",
                    task_success_sample,
                )
                if physically_plausible is not True:
                    raise AttemptGenerationError(
                        FailureStage.EXECUTION,
                        "grasp_not_verified",
                        f"physical aperture/distance evidence rejected {segment.object_name!r}",
                    )
                self._update_pick_place_success(
                    "attach",
                    segment.segment_id,
                    segment.object_name,
                    segment.eef_name,
                    task_success_sample,
                )
            self.attachment_state.held_by_eef[segment.eef_name] = segment.object_name
            return
        if isinstance(segment, DetachIntentSegment):
            self._require_eef(segment.eef_name)
            held = self.attachment_state.held_by_eef.get(segment.eef_name)
            if held != segment.object_name:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "detach_object_mismatch",
                    f"cannot detach {segment.object_name!r}; {segment.eef_name!r} holds {held!r}",
                )
            if self._pick_place_success is not None:
                self._update_pick_place_success(
                    "detach",
                    segment.segment_id,
                    segment.object_name,
                    segment.eef_name,
                    self._task_success_sample(env_id),
                )
            self.attachment_state.held_by_eef[segment.eef_name] = None
            return
        if isinstance(segment, WaitSegment):
            steps = segment.steps
            if steps is None:
                steps = max(1, math.ceil((segment.duration_s or 0.0) / _step_dt(self.env)))
            self._hold(env_id, steps)
            return
        if isinstance(segment, BarrierSegment):
            return
        if isinstance(segment, JointTrajectorySegment):
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "joint_trajectory_not_supported_by_ik_executor",
                "joint trajectories require a joint-control embodiment executor",
                recoverable=False,
            )
        if isinstance(segment, ConcurrentGroupSegment):
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "concurrent_group_not_supported_by_single_arm_executor",
                "concurrent groups require a multi-arm executor",
                recoverable=False,
            )
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "unsupported_segment",
            f"unsupported plan segment {type(segment).__name__}",
            recoverable=False,
        )

    def _prepare_gripper_interaction(
        self,
        env_id: int,
        segment: GripperCommandSegment,
        requested_gripper_value: float,
    ) -> dict[str, Any]:
        """Converge to the preceding checked target before changing the gripper command."""

        previous_gripper_value = self._gripper_values[segment.eef_name]
        target_state = self._last_cartesian_targets.get(segment.eef_name)
        transition_required = requested_gripper_value != previous_gripper_value
        task_success_release = (
            self._pick_place_success is not None
            and segment.segment_id == self._pick_place_success.release_open_segment_id
        )
        evidence: dict[str, Any] = {
            "command": segment.command.value,
            "contact_constrained_cartesian_ready": None,
            "contact_constrained_task_success_accepted": False,
            "convergence_mode": None,
            "converged": None,
            "correction_steps": 0,
            "final_position_error_m": None,
            "final_rotation_error_rad": None,
            "initial_position_error_m": None,
            "initial_rotation_error_rad": None,
            "interaction_position_tolerance_m": self.interaction_position_tolerance_m,
            "interaction_rotation_tolerance_rad": self.interaction_rotation_tolerance_rad,
            "joint_diagnostics_present": target_state is not None and target_state.joint_seed is not None,
            "max_correction_steps": self.max_interaction_correction_steps,
            "outcome": "pending",
            "preceding_cartesian_target": target_state is not None,
            "previous_gripper_value": previous_gripper_value,
            "release_contact_gate_required": task_success_release,
            "release_contact_position_tolerance_m": (
                _RELEASE_CONTACT_POSITION_TOLERANCE_M if task_success_release else None
            ),
            "release_contact_rotation_tolerance_rad": (
                _RELEASE_CONTACT_ROTATION_TOLERANCE_RAD if task_success_release else None
            ),
            "requested_gripper_value": requested_gripper_value,
            "task_success_release_gate": None,
            "segment_id": segment.segment_id,
            "strict_target_converged": None,
            "target_sample_index": None if target_state is None else target_state.sample_index,
            "target_segment_id": None if target_state is None else target_state.segment_id,
            "transition_required": transition_required,
        }
        self._interaction_boundaries.append(evidence)
        if not transition_required:
            evidence["outcome"] = (
                "not_required_no_transition" if target_state is not None else "not_required_no_preceding_target"
            )
            return evidence
        if target_state is None:
            evidence["converged"] = False
            evidence["outcome"] = "failed_no_preceding_target"
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "interaction_target_missing",
                f"gripper transition {segment.segment_id!r} has no preceding collision-checked Cartesian target",
                recoverable=False,
            )

        position_error, rotation_error, path_verified, strict_target_converged = self._interaction_target_status(
            env_id,
            segment.eef_name,
            target_state,
        )
        evidence["initial_position_error_m"] = position_error
        evidence["initial_rotation_error_rad"] = rotation_error
        correction_steps = 0
        while not strict_target_converged and correction_steps < self.max_interaction_correction_steps:
            step_count_before = self._step_count
            try:
                # The requested value is intentionally not installed until this loop succeeds.
                self._step_targets(
                    env_id,
                    {segment.eef_name: target_state.pose},
                    joint_seed_names=target_state.joint_seed_names,
                    joint_seed=target_state.joint_seed,
                )
            except AttemptGenerationError as exc:
                executed_steps = self._step_count - step_count_before
                correction_steps += executed_steps
                self._interaction_correction_steps += executed_steps
                if exc.code not in _RETRYABLE_TRACKING_FAILURE_CODES:
                    evidence["converged"] = False
                    evidence["correction_steps"] = correction_steps
                    evidence["outcome"] = f"failed_{exc.code}"
                    raise
            else:
                executed_steps = self._step_count - step_count_before
                correction_steps += executed_steps
                self._interaction_correction_steps += executed_steps
            position_error, rotation_error, path_verified, strict_target_converged = self._interaction_target_status(
                env_id,
                segment.eef_name,
                target_state,
            )

        evidence["correction_steps"] = correction_steps
        evidence["final_position_error_m"] = position_error
        evidence["final_rotation_error_rad"] = rotation_error
        evidence["strict_target_converged"] = strict_target_converged
        contact_constrained_cartesian_ready = (
            task_success_release
            and path_verified
            and _finite_at_most(position_error, _RELEASE_CONTACT_POSITION_TOLERANCE_M)
            and _finite_at_most(rotation_error, _RELEASE_CONTACT_ROTATION_TOLERANCE_RAD)
        )
        evidence["contact_constrained_cartesian_ready"] = contact_constrained_cartesian_ready

        task_success_release_ready = False
        if task_success_release:
            try:
                task_success_gate = self._update_pick_place_success(
                    "release_contact_gate",
                    segment.segment_id,
                    self._task_success_sample(env_id),
                )
            except AttemptGenerationError:
                evidence["converged"] = False
                evidence["outcome"] = "failed_task_success_unavailable"
                raise
            evidence["task_success_release_gate"] = task_success_gate
            task_success_release_ready = task_success_gate.get("passed") is True
            if not task_success_release_ready:
                evidence["converged"] = False
                evidence["outcome"] = "failed_task_success_release_not_ready"
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "release_task_success_not_ready",
                    f"release transition {segment.segment_id!r} lacks validated pre-release placement/transport",
                )

        accepted = strict_target_converged and (not task_success_release or task_success_release_ready)
        if task_success_release and not accepted and contact_constrained_cartesian_ready and task_success_release_ready:
            accepted = True
            evidence["contact_constrained_task_success_accepted"] = True
            evidence["convergence_mode"] = "contact_constrained_task_success_ready"
        elif accepted and task_success_release:
            evidence["convergence_mode"] = "strict_task_success_ready"
        elif accepted:
            evidence["convergence_mode"] = "strict_target"

        evidence["converged"] = accepted
        if not accepted:
            evidence["outcome"] = "failed_tolerance_not_reached"
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "interaction_target_not_reached",
                f"gripper transition {segment.segment_id!r} remained {position_error:.6g}m / "
                f"{rotation_error:.6g}rad from its preceding Cartesian target after "
                f"{correction_steps} bounded correction steps",
            )
        evidence["outcome"] = evidence["convergence_mode"] if task_success_release else "converged"
        return evidence

    def _converge_terminal_targets(self, env_id: int, eef_names: Sequence[str]) -> None:
        """Converge final checked targets before settled task-success verification."""

        for eef_name in eef_names:
            target_state = self._last_cartesian_targets.get(eef_name)
            if target_state is None:
                continue
            gripper_value = self._gripper_values[eef_name]
            evidence: dict[str, Any] = {
                "correction_steps": 0,
                "eef_name": eef_name,
                "final_joint_path_verified": None,
                "final_position_error_m": None,
                "final_rotation_error_rad": None,
                "gripper_unchanged": None,
                "gripper_value": gripper_value,
                "initial_joint_path_verified": None,
                "initial_position_error_m": None,
                "initial_rotation_error_rad": None,
                "interaction_position_tolerance_m": self.interaction_position_tolerance_m,
                "interaction_rotation_tolerance_rad": self.interaction_rotation_tolerance_rad,
                "joint_diagnostics_present": target_state.joint_seed is not None,
                "max_correction_steps": self.max_terminal_correction_steps,
                "outcome": "pending",
                "target_sample_index": target_state.sample_index,
                "target_segment_id": target_state.segment_id,
                "convergence_trace": [],
            }
            self._terminal_target_boundaries.append(evidence)
            position_error, rotation_error, path_verified, converged = self._interaction_target_status(
                env_id,
                eef_name,
                target_state,
            )
            evidence["initial_position_error_m"] = _finite_or_none(position_error)
            evidence["initial_rotation_error_rad"] = _finite_or_none(rotation_error)
            evidence["initial_joint_path_verified"] = path_verified
            correction_steps = 0
            evidence["convergence_trace"].append({
                "correction_steps": correction_steps,
                "joint_path_verified": path_verified,
                "position_error_m": _finite_or_none(position_error),
                "rotation_error_rad": _finite_or_none(rotation_error),
            })
            while not converged and correction_steps < self.max_terminal_correction_steps:
                step_count_before = self._step_count
                try:
                    self._step_targets(
                        env_id,
                        {eef_name: target_state.pose},
                        joint_seed_names=target_state.joint_seed_names,
                        joint_seed=target_state.joint_seed,
                    )
                except AttemptGenerationError as exc:
                    executed_steps = self._step_count - step_count_before
                    correction_steps += executed_steps
                    self._terminal_correction_steps += executed_steps
                    if exc.code not in _RETRYABLE_TRACKING_FAILURE_CODES:
                        evidence["correction_steps"] = correction_steps
                        evidence["outcome"] = f"failed_{exc.code}"
                        raise
                else:
                    executed_steps = self._step_count - step_count_before
                    correction_steps += executed_steps
                    self._terminal_correction_steps += executed_steps
                position_error, rotation_error, path_verified, converged = self._interaction_target_status(
                    env_id,
                    eef_name,
                    target_state,
                )
                evidence["convergence_trace"].append({
                    "correction_steps": correction_steps,
                    "joint_path_verified": path_verified,
                    "position_error_m": _finite_or_none(position_error),
                    "rotation_error_rad": _finite_or_none(rotation_error),
                })

            evidence["correction_steps"] = correction_steps
            evidence["final_position_error_m"] = _finite_or_none(position_error)
            evidence["final_rotation_error_rad"] = _finite_or_none(rotation_error)
            evidence["final_joint_path_verified"] = path_verified
            evidence["gripper_unchanged"] = self._gripper_values[eef_name] == gripper_value
            assert len(evidence["convergence_trace"]) <= self.max_terminal_correction_steps + 1
            if not converged:
                evidence["outcome"] = "failed_tolerance_not_reached"
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "terminal_target_not_reached",
                    f"terminal target for {eef_name!r} remained {position_error:.6g}m / "
                    f"{rotation_error:.6g}rad from segment {target_state.segment_id!r} after "
                    f"{correction_steps} bounded correction steps",
                )
            evidence["outcome"] = "already_converged" if correction_steps == 0 else "converged"

    def _interaction_target_status(
        self,
        env_id: int,
        eef_name: str,
        target_state: _CartesianTargetState,
    ) -> tuple[float, float, bool, bool]:
        """Return strict interaction errors and combined Cartesian/joint convergence."""

        observed_poses = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])
        if eef_name not in observed_poses:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "tracking_observation_missing",
                f"no observed pose is available for end effector {eef_name!r}",
                recoverable=False,
            )
        observed = torch.as_tensor(
            observed_poses[eef_name][0],
            device=target_state.pose.device,
            dtype=target_state.pose.dtype,
        )
        position_error = float(torch.linalg.norm(observed[:3, 3] - target_state.pose[:3, 3]).item())
        rotation_error = _rotation_distance_rad(observed[:3, :3], target_state.pose[:3, :3])
        path_converged = True
        try:
            self._verify_path_tracking(
                env_id,
                {eef_name: target_state.pose},
                joint_seed_names=target_state.joint_seed_names,
                joint_seed=target_state.joint_seed,
            )
        except AttemptGenerationError as exc:
            if exc.code not in _RETRYABLE_TRACKING_FAILURE_CODES:
                raise
            path_converged = False
        strict_converged = (
            math.isfinite(position_error)
            and math.isfinite(rotation_error)
            and position_error <= self.interaction_position_tolerance_m
            and rotation_error <= self.interaction_rotation_tolerance_rad
        )
        return position_error, rotation_error, path_converged, path_converged and strict_converged

    def _verify_terminal_targets_after_settling(
        self,
        env_id: int,
        eef_names: Sequence[str],
        *,
        verification_samples: int,
    ) -> None:
        """Re-attest exact terminal targets after the final stability window."""

        boundaries_by_eef = {str(item["eef_name"]): item for item in self._terminal_target_boundaries}
        failures: list[str] = []
        for eef_name in eef_names:
            target_state = self._last_cartesian_targets.get(eef_name)
            if target_state is None:
                continue
            boundary = boundaries_by_eef[eef_name]
            position_error, rotation_error, path_verified, converged = self._interaction_target_status(
                env_id,
                eef_name,
                target_state,
            )
            gripper_unchanged = self._gripper_values[eef_name] == boundary["gripper_value"]
            passed = converged and gripper_unchanged
            boundary["post_settle_verification"] = {
                "gripper_unchanged": gripper_unchanged,
                "joint_path_verified": path_verified,
                "passed": passed,
                "position_error_m": _finite_or_none(position_error),
                "rotation_error_rad": _finite_or_none(rotation_error),
                "verification_samples": verification_samples,
            }
            if not passed:
                failures.append(
                    f"{eef_name!r} was {position_error:.6g}m / {rotation_error:.6g}rad from "
                    f"segment {target_state.segment_id!r} (joint_path_verified={path_verified}, "
                    f"gripper_unchanged={gripper_unchanged})"
                )
        if failures:
            raise AttemptGenerationError(
                FailureStage.VERIFICATION,
                "terminal_target_not_stable",
                "terminal target stability was lost during final settling: " + "; ".join(failures),
            )

    def _hold(self, env_id: int, steps: int) -> None:
        if steps <= 0:
            return
        poses = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])
        for _ in range(steps):
            targets = {name: pose[0] for name, pose in poses.items()}
            self._step_targets(env_id, targets)

    def _step_target(
        self,
        env_id: int,
        eef_name: str,
        target: torch.Tensor,
        *,
        joint_seed_names: Sequence[str] = (),
        joint_seed: Sequence[float] | None = None,
    ) -> None:
        # A collision-checked kinematic waypoint can take more than one simulator control step to
        # track because the articulated robot has closed-loop dynamics. Keep the waypoint fixed
        # while applying a small, bounded number of correction steps. This prevents tracking lag
        # from accumulating across a dense plan while retaining a hard corridor and step budget.
        for correction_index in range(self.max_tracking_correction_steps + 1):
            try:
                self._step_targets(
                    env_id,
                    {eef_name: target},
                    joint_seed_names=joint_seed_names,
                    joint_seed=joint_seed,
                )
            except AttemptGenerationError as exc:
                if (
                    exc.code not in _RETRYABLE_TRACKING_FAILURE_CODES
                    or correction_index >= self.max_tracking_correction_steps
                ):
                    raise
                self._tracking_correction_steps += 1
            else:
                return

    def _step_targets(
        self,
        env_id: int,
        targets: Mapping[str, torch.Tensor],
        *,
        joint_seed_names: Sequence[str] = (),
        joint_seed: Sequence[float] | None = None,
    ) -> None:
        if self._step_count >= self.max_steps:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "execution_step_limit",
                f"plan exceeded {self.max_steps} simulator steps",
            )
        eef_names = tuple(self.embodiment_adapter.get_eef_names())
        if set(targets) != set(eef_names):
            # The current executor is explicitly single-arm. This also prevents an accidental zero
            # command on an unmentioned arm from moving a multi-arm embodiment.
            if len(eef_names) != 1 or set(targets) != {eef_names[0]}:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "partial_multi_eef_action",
                    "executor requires targets for every end effector",
                    recoverable=False,
                )
        grippers = {
            name: torch.full(
                (int(self.embodiment_adapter.gripper_action_dim),),
                self._gripper_values[name],
                dtype=torch.float32,
                device=_device(self.env),
            )
            for name in eef_names
        }
        action = self.embodiment_adapter.target_eef_pose_to_action(
            target_eef_pose_dict=dict(targets),
            gripper_action_dict=grippers,
            action_noise_dict=None,
            env_id=env_id,
        )
        self._validate_raw_dik_action(action)
        batch = torch.zeros(self.env.action_space.shape, dtype=action.dtype, device=_device(self.env))
        batch[env_id] = action
        self._step_count += 1
        self.env.step(batch)
        if self._pick_place_success is not None:
            self._update_pick_place_success("observe_step", self._task_success_sample(env_id))
        self._verify_path_tracking(
            env_id,
            targets,
            joint_seed_names=joint_seed_names,
            joint_seed=joint_seed,
        )

    def _validate_raw_dik_action(self, action: torch.Tensor) -> None:
        """Reject malformed or saturated normalized IK actions before ``env.step``."""

        action = torch.as_tensor(action)
        action_shape = tuple(int(size) for size in self.env.action_space.shape)
        expected_action_dim = action_shape[-1] if action_shape else 0
        if action.ndim != 1 or action.shape[0] != expected_action_dim or not action.is_floating_point():
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "raw_dik_action_invalid",
                f"adapter returned action shape/dtype {tuple(action.shape)}/{action.dtype}; "
                f"expected one floating vector of width {expected_action_dim}",
                recoverable=False,
            )
        if not bool(torch.isfinite(action).all().item()):
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "raw_dik_action_invalid",
                "adapter returned a non-finite IK action",
                recoverable=False,
            )
        gripper_dim = int(self.embodiment_adapter.gripper_action_dim)
        pose_dim = expected_action_dim - gripper_dim
        if gripper_dim < 0 or pose_dim <= 0:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "raw_dik_action_invalid",
                f"action width {expected_action_dim} does not leave a positive IK pose slice after "
                f"{gripper_dim} gripper dimensions",
                recoverable=False,
            )
        maximum_pose_action = float(torch.max(torch.abs(action[:pose_dim])).item())
        if maximum_pose_action >= _RAW_DIK_POSE_ACTION_LIMIT - _RAW_DIK_SATURATION_TOLERANCE:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "dik_pose_action_saturated",
                f"normalized IK pose action magnitude {maximum_pose_action:.6g} reached the clipping boundary; "
                "the requested Cartesian step was not sent to the simulator",
            )
        gripper = action[pose_dim:]
        if gripper.numel() and bool(torch.any(torch.abs(gripper) > 1.0 + 1e-6).item()):
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "raw_gripper_action_out_of_range",
                "normalized gripper action must remain in [-1, 1]",
                recoverable=False,
            )

    def _verify_path_tracking(
        self,
        env_id: int,
        targets: Mapping[str, torch.Tensor],
        *,
        joint_seed_names: Sequence[str],
        joint_seed: Sequence[float] | None,
    ) -> None:
        """Reject DIK divergence from the collision-checked Cartesian/joint reference path."""

        observed_poses = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])
        for eef_name, target in targets.items():
            if eef_name not in observed_poses:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "tracking_observation_missing",
                    f"no observed pose is available for end effector {eef_name!r}",
                    recoverable=False,
                )
            observed = torch.as_tensor(observed_poses[eef_name][0], device=target.device, dtype=target.dtype)
            position_error = float(torch.linalg.norm(observed[:3, 3] - target[:3, 3]).item())
            relative_rotation = observed[:3, :3].transpose(0, 1) @ target[:3, :3]
            cosine = torch.clamp((torch.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            rotation_error = float(torch.acos(cosine).item())
            if math.isfinite(position_error):
                self._maximum_eef_position_error_m = max(self._maximum_eef_position_error_m, position_error)
            if math.isfinite(rotation_error):
                self._maximum_eef_rotation_error_rad = max(
                    self._maximum_eef_rotation_error_rad,
                    rotation_error,
                )
            if not math.isfinite(position_error) or position_error > self.max_eef_position_error_m:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "eef_position_tracking_error",
                    f"{eef_name!r} position error {position_error:.6g}m exceeds {self.max_eef_position_error_m:.6g}m",
                )
            if not math.isfinite(rotation_error) or rotation_error > self.max_eef_rotation_error_rad:
                raise AttemptGenerationError(
                    FailureStage.EXECUTION,
                    "eef_rotation_tracking_error",
                    f"{eef_name!r} rotation error {rotation_error:.6g}rad exceeds "
                    f"{self.max_eef_rotation_error_rad:.6g}rad",
                )

        if joint_seed is None:
            return
        observed_names = tuple(self.embodiment_adapter.get_joint_names())
        observed_positions = self.embodiment_adapter.get_joint_positions(env_ids=[env_id])[0]
        observed_by_name = {name: float(observed_positions[index]) for index, name in enumerate(observed_names)}
        missing = [name for name in joint_seed_names if name not in observed_by_name]
        if missing:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "joint_tracking_observation_missing",
                f"joint tracking observations are missing {missing}",
                recoverable=False,
            )
        joint_errors = [
            (
                abs(observed_by_name[name] - float(reference)),
                name,
                observed_by_name[name],
                float(reference),
            )
            for name, reference in zip(joint_seed_names, joint_seed)
        ]
        maximum_error, maximum_name, maximum_observed, maximum_reference = max(
            joint_errors,
            default=(0.0, "<none>", 0.0, 0.0),
        )
        if math.isfinite(maximum_error) and maximum_error > self._maximum_joint_path_error_rad:
            self._maximum_joint_path_error_rad = maximum_error
            self._maximum_joint_path_error_name = maximum_name
        if not math.isfinite(maximum_error) or maximum_error > self.max_joint_path_error_rad:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "joint_path_tracking_error",
                f"joint path error {maximum_error:.6g}rad on {maximum_name!r} exceeds "
                f"{self.max_joint_path_error_rad:.6g}rad "
                f"(observed={maximum_observed:.6g}, reference={maximum_reference:.6g})",
            )

    def _tracking_observation(self) -> dict[str, float | str | None]:
        """Return bounded path-tracking evidence for the attempt record."""

        return {
            "maximum_eef_position_error_m": self._maximum_eef_position_error_m,
            "maximum_eef_rotation_error_rad": self._maximum_eef_rotation_error_rad,
            "maximum_joint_path_error_name": self._maximum_joint_path_error_name,
            "maximum_joint_path_error_rad": self._maximum_joint_path_error_rad,
        }

    def _physical_observation(self, env_id: int) -> dict[str, Any]:
        """Return final physical state plus gripper-transition evidence for the attempt record."""

        observation = {
            "final_state": self._capture_physical_state(env_id),
            "gripper_milestones": list(self._gripper_milestones),
            "interaction_boundaries": [dict(item) for item in self._interaction_boundaries],
            "interaction_correction_steps": self._interaction_correction_steps,
            "terminal_correction_steps": self._terminal_correction_steps,
            "terminal_target_boundaries": [dict(item) for item in self._terminal_target_boundaries],
        }
        if self._pick_place_success is not None:
            eef_name = self._pick_place_success.eef_name
            observation["task_success_report"] = self._pick_place_success.report(
                logical_held_object=self.attachment_state.held_by_eef.get(eef_name)
            )
        return observation

    def _task_success_sample(self, env_id: int) -> dict[str, Any]:
        """Capture a required live sample for the fail-closed task-success checks."""

        sample = self._capture_physical_state(env_id)
        if "capture_error" in sample:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "task_success_state_unavailable",
                str(sample["capture_error"]),
                recoverable=False,
            )
        return sample

    def _observe_task_success_final(self, env_id: int, *, goal_satisfied: bool) -> None:
        """Record one post-settle task-success sample."""

        self._update_pick_place_success(
            "observe_final",
            self._task_success_sample(env_id),
            goal_satisfied=goal_satisfied,
            failure_stage=FailureStage.VERIFICATION,
        )

    def _update_pick_place_success(
        self,
        method_name: str,
        *args: Any,
        failure_stage: FailureStage = FailureStage.EXECUTION,
        **kwargs: Any,
    ) -> Any:
        """Convert malformed or unavailable task-success state into a classified failure."""

        assert self._pick_place_success is not None
        method = getattr(self._pick_place_success, method_name)
        try:
            return method(*args, **kwargs)
        except AttemptGenerationError:
            raise
        except Exception as exc:
            raise AttemptGenerationError(
                failure_stage,
                "task_success_state_unavailable",
                f"{method_name} failed: {type(exc).__name__}: {str(exc)[:1024]}",
                recoverable=False,
            ) from exc

    def _capture_physical_state(self, env_id: int) -> dict[str, Any]:
        """Capture a compact environment-origin observation without mutating simulator state."""

        try:
            origin = _env_origin(self.env, env_id)
            eef_poses = {
                name: [list(row) for row in _matrix_to_tuple(pose[0])]
                for name, pose in self.embodiment_adapter.get_eef_poses(env_ids=[env_id]).items()
            }
            objects = {}
            for name, scene_object in sorted(self.env.scene.rigid_objects.items()):
                position = _tensor_row(scene_object.data.root_pos_w, env_id) - origin
                quaternion = _tensor_row(scene_object.data.root_quat_w, env_id)
                object_observation = {
                    "position_env_m": [float(value) for value in position.tolist()],
                    "quaternion_xyzw": [float(value) for value in quaternion.tolist()],
                }
                linear_velocity = getattr(scene_object.data, "root_lin_vel_w", None)
                angular_velocity = getattr(scene_object.data, "root_ang_vel_w", None)
                if linear_velocity is not None:
                    object_observation["linear_speed_m_s"] = float(
                        torch.linalg.norm(_tensor_row(linear_velocity, env_id)).item()
                    )
                if angular_velocity is not None:
                    object_observation["angular_speed_rad_s"] = float(
                        torch.linalg.norm(_tensor_row(angular_velocity, env_id)).item()
                    )
                objects[name] = object_observation
            joint_names = tuple(self.embodiment_adapter.get_joint_names())
            joint_positions = self.embodiment_adapter.get_joint_positions(env_ids=[env_id])[0]
            return {
                "eef_poses_env": eef_poses,
                "gripper_commands": dict(self._gripper_values),
                "held_objects": dict(self.attachment_state.held_by_eef),
                "joint_positions": {name: float(joint_positions[index]) for index, name in enumerate(joint_names)},
                "objects": objects,
            }
        except Exception as exc:
            return {
                "capture_error": f"{type(exc).__name__}: {str(exc)[:512]}",
            }

    def _verify_attachment_proximity(self, env_id: int, eef_name: str, object_name: str) -> bool:
        if self._gripper_values[eef_name] >= 0:
            return False
        scene_object = self.env.scene.rigid_objects.get(object_name)
        if scene_object is None:
            return False
        object_position = _tensor_row(scene_object.data.root_pos_w, env_id) - _env_origin(self.env, env_id)
        eef_pose = self.embodiment_adapter.get_eef_poses(env_ids=[env_id])[eef_name][0]
        distance = torch.linalg.norm(object_position - eef_pose[:3, 3]).item()
        return math.isfinite(distance) and distance <= self.attachment_distance_m

    def _require_eef(self, name: str) -> None:
        if name not in self._gripper_values:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "unknown_end_effector",
                f"plan references unknown end effector {name!r}",
                recoverable=False,
            )

    def _event(
        self,
        request: AttemptRequest,
        plan: TaskMotionPlan,
        segment_id: str | None,
        event_type: ExecutionEventType,
        outcome: ExecutionOutcome,
        *,
        failure_code: str | None = None,
        message: str | None = None,
    ) -> ExecutionEvent:
        self._event_count += 1
        return ExecutionEvent(
            event_id=make_stable_id(
                "event",
                request.attempt_id,
                plan.plan_id,
                segment_id,
                event_type.value,
                self._event_count,
            ),
            attempt_id=request.attempt_id,
            plan_id=plan.plan_id,
            segment_id=segment_id,
            event_type=event_type,
            outcome=outcome,
            monotonic_time_s=time.monotonic(),
            failure_code=failure_code,
            message=message,
            metadata={"step": self._step_count},
        )


def success_term_verifier(success_term: Any) -> SuccessVerifier:
    """Wrap a saved Isaac Lab ``TerminationTermCfg`` as a final-state verifier."""

    if success_term is None or not callable(getattr(success_term, "func", None)):
        raise ValueError("success_term must expose a callable func")
    params = dict(getattr(success_term, "params", {}) or {})

    def verify(env: Any, _env_id: int) -> Any:
        return success_term.func(env, **params)

    return verify


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _device(env: Any) -> torch.device:
    return torch.device(getattr(env, "device", "cpu"))


def _env_id_tensor(env: Any, env_id: int) -> torch.Tensor:
    if env_id < 0 or env_id >= int(env.num_envs):
        raise ValueError(f"env_id {env_id} outside [0, {env.num_envs})")
    return torch.tensor([env_id], dtype=torch.int64, device=_device(env))


def _tensor_row(value: Any, env_id: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    return tensor[env_id].detach()


def _row_to_tuple(value: Any) -> tuple[float, ...]:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    result = tuple(float(item) for item in tensor.tolist())
    if not all(math.isfinite(item) for item in result):
        raise ValueError("snapshot tensor contains non-finite values")
    return result


def _matrix_to_tuple(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {tuple(tensor.shape)}")
    rows = tuple(tuple(float(item) for item in row) for row in tensor.tolist())
    return rows  # type: ignore[return-value]


def _pose_from_xyzw(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("root position/quaternion must have shapes (3,) and (4,)")
    if not bool(torch.isfinite(position).all().item()) or not bool(torch.isfinite(quaternion).all().item()):
        raise ValueError("root position/quaternion must be finite")
    norm = torch.linalg.norm(quaternion)
    if float(norm.item()) <= 1e-12:
        raise ValueError("root quaternion must have nonzero norm")
    quaternion = quaternion / norm
    x, y, z, w = quaternion.unbind()
    rotation = torch.stack((
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )).reshape(3, 3)
    pose = torch.eye(4, dtype=position.dtype, device=position.device)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def _validate_raw_dik_action(env: Any, embodiment_adapter: Any, action: Any, *, context: str) -> None:
    """Validate one normalized delta-IK action without mutating simulator state."""

    action = torch.as_tensor(action)
    action_shape = tuple(int(size) for size in env.action_space.shape)
    expected_action_dim = action_shape[-1] if action_shape else 0
    if action.ndim != 1 or action.shape[0] != expected_action_dim or not action.is_floating_point():
        raise ValueError(
            f"{context} adapter returned action shape/dtype {tuple(action.shape)}/{action.dtype}; "
            f"expected one floating vector of width {expected_action_dim}"
        )
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError(f"{context} adapter returned a non-finite IK action")
    gripper_dim = int(embodiment_adapter.gripper_action_dim)
    pose_dim = expected_action_dim - gripper_dim
    if gripper_dim < 0 or pose_dim <= 0:
        raise ValueError(
            f"{context} action width {expected_action_dim} does not leave a positive IK pose slice after "
            f"{gripper_dim} gripper dimensions"
        )
    maximum_pose_action = float(torch.max(torch.abs(action[:pose_dim])).item())
    if maximum_pose_action >= _RAW_DIK_POSE_ACTION_LIMIT - _RAW_DIK_SATURATION_TOLERANCE:
        raise ValueError(
            f"{context} normalized IK pose action magnitude {maximum_pose_action:.6g} reached the clipping boundary; "
            "the requested Cartesian step was not sent to the simulator"
        )
    gripper = action[pose_dim:]
    if gripper.numel() and bool(torch.any(torch.abs(gripper) > 1.0 + 1e-6).item()):
        raise ValueError(f"{context} normalized gripper action must remain in [-1, 1]")


def _env_origin(env: Any, env_id: int) -> torch.Tensor:
    return torch.as_tensor(env.scene.env_origins)[env_id].detach()


def _step_dt(env: Any) -> float:
    value = getattr(env, "step_dt", None)
    if value is None:
        value = getattr(getattr(env, "cfg", None), "decimation", 1) * getattr(env.sim, "physics_dt", 0.0)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("environment step_dt must be finite and positive")
    return value


def _cartesian_step_dt(segment: CartesianTrajectorySegment, env: Any) -> float:
    """Attest the planner sampling interval and return the live control interval [s]."""

    control_dt_s = _step_dt(env)
    value = segment.metadata.get("step_dt_s")
    if value is None:
        return control_dt_s
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "cartesian_step_dt_invalid",
            f"segment {segment.segment_id!r} step_dt_s must be a finite positive number",
            recoverable=False,
        )
    step_dt_s = float(value)
    if not math.isfinite(step_dt_s) or step_dt_s <= 0:
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "cartesian_step_dt_invalid",
            f"segment {segment.segment_id!r} step_dt_s must be a finite positive number",
            recoverable=False,
        )
    if not math.isclose(step_dt_s, control_dt_s, rel_tol=1e-6, abs_tol=1e-9):
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "cartesian_step_dt_mismatch",
            f"segment {segment.segment_id!r} declares step_dt_s={step_dt_s:.9g}, but the live control "
            f"interval is {control_dt_s:.9g}s",
            recoverable=False,
        )
    return control_dt_s


def _rotation_distance_rad(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the geodesic angle [rad] between two rotation matrices."""

    relative = first.transpose(0, 1) @ second
    cosine = torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(torch.acos(cosine).item())


def _finite_at_most(value: float, maximum: float) -> bool:
    """Apply one physical upper bound with only float32 roundoff tolerance."""

    return math.isfinite(value) and (
        value <= maximum or math.isclose(value, maximum, rel_tol=0.0, abs_tol=_PHYSICAL_BOUNDARY_ABS_TOLERANCE)
    )


def _finite_or_none(value: float) -> float | None:
    """Return a finite run-record measurement or ``None`` when unavailable."""

    return value if math.isfinite(value) else None


def _robot_base_position_in_env(env: Any, env_id: int) -> torch.Tensor:
    """Return the sole live articulation root in environment-origin coordinates [m]."""

    articulations = getattr(env.scene, "articulations", None)
    if not isinstance(articulations, Mapping) or len(articulations) != 1:
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "workspace_frame_unavailable",
            "Franka workspace validation requires exactly one live articulation root",
            recoverable=False,
        )
    articulation = next(iter(articulations.values()))
    root_pos_w = getattr(getattr(articulation, "data", None), "root_pos_w", None)
    if root_pos_w is None:
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "workspace_frame_unavailable",
            "Franka workspace validation requires live articulation root_pos_w",
            recoverable=False,
        )
    try:
        position = torch.as_tensor(root_pos_w, dtype=torch.float64, device=_device(env))[env_id]
        position = position - torch.as_tensor(_env_origin(env, env_id), dtype=torch.float64, device=_device(env))
    except Exception as exc:
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "workspace_frame_unavailable",
            f"failed to read the live articulation root: {type(exc).__name__}: {str(exc)[:512]}",
            recoverable=False,
        ) from exc
    if position.shape != (3,) or not bool(torch.isfinite(position).all().item()):
        raise AttemptGenerationError(
            FailureStage.EXECUTION,
            "workspace_frame_unavailable",
            "live articulation root position must be a finite 3-vector",
            recoverable=False,
        )
    return position


def _geometry_ref(scene_object: Any) -> str | None:
    value = getattr(getattr(getattr(scene_object, "cfg", None), "spawn", None), "usd_path", None)
    if value is None:
        return None
    value = str(value)
    return value if len(value) <= 2048 else None


def _target_in_env_frame(env: Any, env_id: int, target: torch.Tensor, frame: str) -> torch.Tensor:
    if frame in ("env", "env_origin"):
        return target
    if frame == "world":
        result = target.clone()
        result[:3, 3] -= _env_origin(env, env_id)
        return result
    raise AttemptGenerationError(
        FailureStage.EXECUTION,
        "unsupported_pose_frame",
        f"executor supports only 'env_origin' and 'world' poses, got {frame!r}",
        recoverable=False,
    )


def _gripper_value(segment: GripperCommandSegment) -> float:
    if segment.command is GripperCommandMode.OPEN:
        return 1.0
    if segment.command is GripperCommandMode.CLOSE:
        return -1.0
    if segment.command is GripperCommandMode.POSITION:
        assert segment.value is not None
        if not -1.0 <= segment.value <= 1.0:
            raise AttemptGenerationError(
                FailureStage.EXECUTION,
                "gripper_position_out_of_range",
                f"normalized gripper position must be in [-1, 1], got {segment.value}",
                recoverable=False,
            )
        return segment.value
    raise AttemptGenerationError(
        FailureStage.EXECUTION,
        "gripper_effort_not_supported",
        "the binary-gripper executor does not support effort commands",
        recoverable=False,
    )


def _env_bool(value: bool | Sequence[bool] | torch.Tensor, env_id: int) -> bool:
    if isinstance(value, bool):
        return value
    tensor = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    if env_id >= tensor.numel():
        raise ValueError(f"success verifier returned only {tensor.numel()} environment results")
    return bool(tensor[env_id].item())
