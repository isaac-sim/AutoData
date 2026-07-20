# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy semantic command-planner provider for ScheduleStream's cuRobo v1 IsaacLab bridge."""

from __future__ import annotations

import importlib
import math
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MethodType
from typing import Any, Protocol

from isaac_autodata_core.autonomous.dense_trace import (
    DenseAttachmentEvent,
    DensePlanTrace,
    task_motion_plan_from_dense_trace,
)
from isaac_autodata_core.autonomous.schedulestream_goal import (
    ScheduleStreamGoalSymbols,
    compile_schedulestream_goal,
    load_schedulestream_goal_symbols,
)
from isaac_autodata_core.autonomous.task_motion import GoalPredicate, Matrix4, TaskMotionPlan, matrix4
from isaac_autodata_interfaces.autonomous.schedulestream.command_types import (
    MalformedScheduleStreamCommandError,
    ScheduleStreamClosedError,
    ScheduleStreamImportError,
    ScheduleStreamLimitError,
    ScheduleStreamLoweringContext,
    ScheduleStreamLoweringLimits,
    ScheduleStreamProviderError,
    ScheduleStreamTimingError,
)

V1_FRANKA_USD_BASENAMES = frozenset({
    "franka_panda_hand_on_stand.usd",
    "panda_instanceable.usd",
})
V1_IK_JOINT_LIMIT_MARGIN_RAD = 1e-3
V1_FRAME_ATTESTATION_MAX_POSITION_ERROR_M = 0.005
V1_FRAME_ATTESTATION_MAX_ROTATION_ERROR_RAD = 0.01
V1_GRASP_GEOMETRY_MAX_POSITION_ROUNDOFF_M = 1e-7
V1_GRASP_GEOMETRY_MAX_ROTATION_ROUNDOFF_RAD = 1e-6
V1_REVIEWED_GRASPABLE_ASSET = "rubiks_cube_hot3d_robolab"
V1_REVIEWED_DESTINATION_ASSET = "bowl_ycb_robolab"
V1_DESTINATION_PLACEMENT_PROFILE = "franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1"
V1_GRASP_GEOMETRY_PROFILE = "franka_rubiks_cube_offcenter_cuboid_top_v1"
V1_DESIRED_AABB_CENTER_VERTICAL_OFFSET_M = 0.03
V1_VERTICAL_EVIDENCE_CORRIDOR_M = 0.04
V1_REVIEWED_GRASPABLE_AABB_DIMENSION_BOUNDS_M = ((0.05, 0.065),) * 3
V1_REVIEWED_DESTINATION_AABB_DIMENSION_BOUNDS_M = (
    (0.14, 0.17),
    (0.14, 0.17),
    (0.04, 0.07),
)

_ModuleLoader = Callable[[], Any]
_GoalSymbolsLoader = Callable[[str], ScheduleStreamGoalSymbols]
_WorldFactory = Callable[..., Any]
_WorldCloser = Callable[[Any], None]
_EefPoseReader = Callable[[int, str], Any]
_PrimitiveGraspGenerator = Callable[..., Iterable[Any]]


class _SeedSetter(Protocol):
    """Keyword-only seed hook matching ``custream.utils.set_seed(**kwargs)``."""

    def __call__(self, *, seed: int) -> None:
        """Set the deterministic seed immediately before a planning attempt."""

        ...


class ScheduleStreamCommandPlanner(Protocol):
    """Minimal planner-provider interface consumed before command lowering."""

    application: str

    def plan_commands(self, env_id: int = 0) -> tuple[Any, ...] | None:
        """Return native commands, or ``None`` when the solver proves no plan."""

        ...

    def hold_commands(self, env_id: int = 0, *, steps: int) -> tuple[Any, ...]:
        """Return explicit Configuration commands for a bounded hold."""

        ...

    def plan_dense_trace(
        self,
        context: ScheduleStreamLoweringContext,
        env_id: int = 0,
        *,
        link_name: str | None = None,
    ) -> DensePlanTrace | None:
        """Return the upstream controller as an aligned, calibrated dense trace."""

        ...

    def plan_task_motion_plan(
        self,
        context: ScheduleStreamLoweringContext,
        env_id: int = 0,
        *,
        link_name: str | None = None,
    ) -> TaskMotionPlan | None:
        """Return an IK-executor-compatible Cartesian task-motion plan."""

        ...

    def close(self) -> None:
        """Release or forget owned planner resources."""

        ...


@dataclass(frozen=True)
class V1IsaacLabPlannerConfig:
    """Bounded settings for the concrete cuRobo v1 IsaacLab command provider."""

    batch_size: int = 10
    scale_dt: float = 5.0
    collisions: bool = True
    max_time_s: float = 60.0
    profile: bool = False
    animate: bool = False
    video: bool = False
    verbose: bool = False
    visualize_spheres: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.batch_size > 65_536:
            raise ValueError("batch_size must not exceed 65536")
        for field_name in ("scale_dt", "max_time_s"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"{field_name} must be positive and finite")
            object.__setattr__(self, field_name, normalized)
        if self.max_time_s > 86_400:
            raise ValueError("max_time_s must not exceed 86400")
        for field_name in (
            "collisions",
            "profile",
            "animate",
            "video",
            "verbose",
            "visualize_spheres",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")


class V1IsaacLabCommandPlanner:
    """Owned wrapper around a semantic subclass of upstream IsaacLab ``Planner``."""

    application = "custream"

    def __init__(
        self,
        native_planner: Any,
        *,
        eef_pose_reader: _EefPoseReader | None = None,
        seed_setter: _SeedSetter | None = None,
        world_closer: _WorldCloser | None = None,
    ) -> None:
        self._native_planner = native_planner
        self._eef_pose_reader = eef_pose_reader
        self._seed_setter = seed_setter
        self._world_closer = world_closer
        self._closed = False
        self._last_frame_evidence: dict[str, Any] = {}

    @property
    def native_planner(self) -> Any:
        """Return the native planner while it remains open."""

        self._require_open()
        return self._native_planner

    @property
    def arm(self) -> str:
        """ScheduleStream arm identifier used to compile the semantic goal."""

        return self.native_planner.semantic_arm

    @property
    def world(self) -> Any:
        """Native custream world, exposed for explicit integration wiring."""

        return self.native_planner.world

    @property
    def planning_diagnostics(self) -> dict[str, Any]:
        """Return bounded native planning diagnostics accumulated for the current world."""

        diagnostics = getattr(self.world, "autodata_ik_joint_limit_evidence", None)
        return dict(diagnostics) if isinstance(diagnostics, Mapping) else {}

    def plan_commands(self, env_id: int = 0) -> tuple[Any, ...] | None:
        """Plan from one live environment and return unexecuted native commands."""

        self._require_open()
        _require_env_id(env_id)
        return self._native_planner.plan_commands(env_id)

    def hold_commands(self, env_id: int = 0, *, steps: int) -> tuple[Any, ...]:
        """Capture the current configuration and repeat it for ``steps`` executor ticks."""

        self._require_open()
        _require_env_id(env_id)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or steps > 1_000_000:
            raise ValueError("steps must be an integer in [1, 1000000]")
        return self._native_planner.hold_commands(env_id, steps=steps)

    def plan_dense_trace(
        self,
        context: ScheduleStreamLoweringContext,
        env_id: int = 0,
        *,
        link_name: str | None = None,
        limits: ScheduleStreamLoweringLimits | None = None,
    ) -> DensePlanTrace | None:
        """Lower upstream ``PathController`` arrays after exact live body-offset attestation."""

        self._require_open()
        _require_env_id(env_id)
        if context.frame != "world":
            raise ScheduleStreamProviderError(
                "upstream v1 PathController produces world-frame poses; lowering context.frame must be 'world'"
            )
        if self._eef_pose_reader is None:
            raise ScheduleStreamProviderError(
                "dense v1 lowering requires an eef_pose_reader for cuRobo-to-Isaac frame attestation"
            )
        if self._seed_setter is None:
            raise ScheduleStreamProviderError("dense v1 lowering requires a custream seed_setter")
        limits = limits or ScheduleStreamLoweringLimits()
        observed_eef = _matrix_from_native(self._eef_pose_reader(env_id, context.eef_name), "observed_eef_pose")
        try:
            self._seed_setter(seed=context.seed)
        except Exception as exc:
            raise ScheduleStreamProviderError(f"failed to seed custream for attempt seed {context.seed}") from exc
        controller = self._native_planner.plan_controller(env_id)
        if controller is None:
            return None
        links = tuple(controller.link_poses)
        if link_name is None:
            matching = [link for link, eef in context.eef_by_link.items() if eef == context.eef_name and link in links]
            if len(matching) == 1:
                link_name = matching[0]
            elif len(links) == 1:
                link_name = links[0]
            else:
                raise ScheduleStreamProviderError(
                    f"cannot select one action link for EEF {context.eef_name!r}; controller links are {links}"
                )
        if link_name not in controller.link_poses:
            raise ScheduleStreamProviderError(
                f"requested action link {link_name!r} is absent; controller links are {links}"
            )
        eef_offset, frame_attestation = _attest_action_to_eef_transform(
            self._native_planner,
            action_link_name=link_name,
            observed_eef=observed_eef,
        )
        raw_link_rows = _rows_from_native(
            controller.link_poses[link_name],
            "controller.link_poses",
            maximum_rows=limits.max_samples_per_segment,
            expected_columns=7,
        )
        if not raw_link_rows:
            raise MalformedScheduleStreamCommandError("PathController contains no link poses")
        raw_initial_body = _pose_vector_to_matrix(raw_link_rows[0], "controller.link_poses[0]")
        initial_position_error, initial_rotation_error = _transform_error(
            raw_initial_body,
            frame_attestation["current_action_link"],
        )
        if initial_position_error > 0.005 or initial_rotation_error > 0.01:
            raise ScheduleStreamProviderError(
                "v1 controller initial action-link pose does not attest against the live planner "
                f"state (position={initial_position_error:.6g}m, rotation={initial_rotation_error:.6g}rad)"
            )
        poses = []
        raw_action_poses = []
        for index, row in enumerate(raw_link_rows):
            raw_body = _pose_vector_to_matrix(row, f"controller.link_poses[{index}]")
            raw_action_poses.append(raw_body)
            lowered_eef = _matrix_multiply(raw_body, eef_offset)
            poses.append(matrix4(lowered_eef, f"controller.link_poses[{index}]"))

        joint_names = _bounded_names(controller.joints, "controller.joints", limits.max_joints)
        joint_positions = _rows_from_native(
            controller.joint_positions,
            "controller.joint_positions",
            maximum_rows=limits.max_samples_per_segment,
            expected_columns=len(joint_names),
        )
        gripper_rows = _vector_from_native(
            controller.gripper_actions,
            "controller.gripper_actions",
            maximum_values=limits.max_samples_per_segment,
        )
        sample_count = len(poses)
        if len(joint_positions) != sample_count or len(gripper_rows) != sample_count:
            raise MalformedScheduleStreamCommandError(
                "PathController poses, joint positions, and gripper actions must align one-to-one"
            )
        evidence_indices = {0, sample_count - 1}
        evidence_indices.update(
            index for index in range(1, sample_count) if abs(gripper_rows[index] - gripper_rows[index - 1]) > 1e-6
        )
        self._last_frame_evidence = {
            "samples": [
                {
                    "gripper": gripper_rows[index],
                    "index": index,
                    "lowered_eef": [list(row) for row in poses[index]],
                    "raw_action_link": [list(row) for row in raw_action_poses[index]],
                }
                for index in sorted(evidence_indices)
            ],
            "object_root_to_mesh": getattr(self._native_planner, "object_pose_offset_evidence", {}),
            "ik_joint_limit_filter": self.planning_diagnostics,
            "observed_initial_eef": [list(row) for row in observed_eef],
            "planned_link_name": link_name,
            "planner_reference_frame": getattr(
                self._native_planner.world,
                "autodata_reference_frame_evidence",
                {},
            ),
            "destination_placement": getattr(
                self._native_planner,
                "destination_placement_geometry",
                {},
            ),
            "grasp_geometry": getattr(self._native_planner, "grasp_geometry", {}),
            "world_graspability": getattr(self._native_planner, "grasp_configuration", {}),
            "static_frame_attestation": {
                "action_to_observed_eef": [list(row) for row in frame_attestation["observed_offset"]],
                "configured_action_offset": [list(row) for row in frame_attestation["configured_offset"]],
                "configured_offset_position_error_m": frame_attestation["configured_position_error_m"],
                "configured_offset_rotation_error_rad": frame_attestation["configured_rotation_error_rad"],
                "initial_action_position_error_m": initial_position_error,
                "initial_action_rotation_error_rad": initial_rotation_error,
            },
        }
        if sample_count > limits.max_total_samples:
            raise ScheduleStreamLimitError(
                f"PathController has {sample_count} samples; total limit is {limits.max_total_samples}"
            )
        native_dt_s = _positive_float(self._native_planner.world.time_step, "world.time_step")
        tolerance = max(1e-9, context.step_dt_s * 1e-6)
        if abs(native_dt_s - context.step_dt_s) > tolerance:
            raise ScheduleStreamTimingError(
                f"v1 controller dt {native_dt_s:g}s does not match executor dt {context.step_dt_s:g}s"
            )
        if sample_count * native_dt_s > limits.max_duration_s:
            raise ScheduleStreamLimitError("PathController duration exceeds configured lowering limit")
        attachment_events = _infer_attachment_events(
            gripper_rows,
            context.goal,
            graspable_object=getattr(self._native_planner, "semantic_graspable_object", None),
        )
        return DensePlanTrace(
            eef_name=context.eef_name,
            frame=context.frame,
            poses=tuple(poses),
            gripper_values=tuple(gripper_rows),
            step_dt_s=native_dt_s,
            gripper_settle_steps=max(1, math.ceil(0.2 / native_dt_s)),
            joint_names=joint_names,
            joint_positions=joint_positions,
            attachment_events=attachment_events,
        )

    def plan_task_motion_plan(
        self,
        context: ScheduleStreamLoweringContext,
        env_id: int = 0,
        *,
        link_name: str | None = None,
        limits: ScheduleStreamLoweringLimits | None = None,
    ) -> TaskMotionPlan | None:
        """Plan and lower through the executable dense Cartesian controller representation."""

        trace = self.plan_dense_trace(context, env_id, link_name=link_name, limits=limits)
        if trace is None:
            return None
        metadata = dict(context.metadata)
        metadata["schedulestream"] = {
            "application": "custream",
            "attachment_event_source": "aligned_binary_gripper_transitions",
            "attachment_events_preserved": True,
            "destination_placement": getattr(
                self._native_planner,
                "destination_placement_geometry",
                {},
            ),
            "grasp_geometry": getattr(self._native_planner, "grasp_geometry", {}),
            "frame_evidence": self._last_frame_evidence,
            "frame_calibration": "configured_action_offset_attested_live",
            "source": "isaaclab.PathController",
        }
        return task_motion_plan_from_dense_trace(
            trace,
            request_digest=context.request_digest,
            snapshot_digest=context.snapshot_digest,
            backend="schedulestream_custream",
            backend_version=context.backend_version or "unknown",
            seed=context.seed,
            goal=context.goal,
            metadata=metadata,
        )

    def close(self) -> None:
        """Invoke an injected native closer, then drop all strong native references."""

        if self._closed:
            return
        self._closed = True
        planner = self._native_planner
        self._native_planner = None
        world = getattr(planner, "world", None)
        try:
            if world is not None and self._world_closer is not None:
                self._world_closer(world)
        except Exception as exc:
            message = str(exc).replace("\n", " ")[:500]
            raise ScheduleStreamProviderError(f"v1 world closer failed ({type(exc).__name__}: {message})") from exc
        finally:
            if planner is not None:
                frames = getattr(planner, "frames", None)
                if hasattr(frames, "clear"):
                    frames.clear()
                planner.goal = None
                planner.world = None
                planner.env = None

    def __enter__(self) -> V1IsaacLabCommandPlanner:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise ScheduleStreamClosedError("v1 IsaacLab command planner is closed")


def _initialize_semantic_v1_planner(
    planner: Any,
    *,
    env: Any,
    module: Any,
    predicates: tuple[GoalPredicate, ...],
    graspable_object: str,
    destination_object: str,
    graspable_asset_name: str,
    destination_asset_name: str,
    arm: str | None,
    config: V1IsaacLabPlannerConfig,
    world_factory: _WorldFactory,
    goal_symbols_loader: _GoalSymbolsLoader,
    world_closer: _WorldCloser | None,
) -> None:
    planner.env = env
    planner.world = None
    planner.collisions = config.collisions
    planner.max_time = config.max_time_s
    planner.animate = config.animate
    planner.video = config.video
    planner.verbose = config.verbose
    planner.kwargs = {}
    native_world = None
    try:
        _validate_goal_subjects_are_dynamic(planner.scene, predicates)
        native_world = world_factory(
            module,
            planner.scene,
            config,
            graspable_object=graspable_object,
            destination_object=destination_object,
            graspable_asset_name=graspable_asset_name,
            destination_asset_name=destination_asset_name,
        )
        native_world.set_retract_conf()
        native_world.initialize(batch_size=config.batch_size)
        _install_v1_ik_joint_limit_filter(native_world)
        planner.world = native_world
        grasp_configuration = _validate_world_graspability(native_world, graspable_object)
        grasp_geometry = grasp_configuration["grasp_geometry"]
        object_pose_offsets, object_pose_offset_evidence = _capture_v1_object_pose_offsets(planner, module)
        destination_placement_geometry = _validate_world_destination_placement(
            native_world,
            graspable_object=graspable_object,
            destination_object=destination_object,
            object_pose_offset_evidence=object_pose_offset_evidence,
        )
        semantic_arm = _select_semantic_arm(tuple(native_world.arms), arm)
        symbols = goal_symbols_loader("custream")
        semantic_goal = compile_schedulestream_goal(predicates, symbols, arm=semantic_arm)
    except Exception as exc:
        planner.world = None
        planner.env = None
        _close_failed_staged_world(native_world, world_closer, exc)
        raise
    planner.world = native_world
    planner.object_pose_offsets = object_pose_offsets
    planner.object_pose_offset_evidence = object_pose_offset_evidence
    planner.grasp_configuration = grasp_configuration
    planner.grasp_geometry = grasp_geometry
    planner.destination_placement_geometry = destination_placement_geometry
    planner.semantic_graspable_object = graspable_object
    planner.semantic_destination_object = destination_object
    planner.semantic_arm = semantic_arm
    planner.goal = semantic_goal
    planner.frames = []
    planner.errors = Counter()


def _install_v1_ik_joint_limit_filter(world: Any) -> None:
    """Reject v1 IK candidates that cuRobo motion generation would reject at its boundary.

    The pinned v1 stack can report an IK seed as successful even when a waypoint lies just outside
    the robot model's position limits. ScheduleStream otherwise selects that seed and later fails
    the whole symbolic skeleton during motion refinement. Filtering only those native candidates
    preserves collision checking and lets the stream consider another already-computed IK seed.
    """

    original = getattr(world, "link_iterative_inverse_kinematics", None)
    if not callable(original):
        raise ScheduleStreamProviderError("v1 world has no callable link iterative IK method")
    world.autodata_ik_joint_limit_evidence = {
        "accepted_candidates": 0,
        "calls": 0,
        "candidate_solutions": 0,
        "joint_limit_margin_rad": V1_IK_JOINT_LIMIT_MARGIN_RAD,
        "policy": "all_waypoints_inside_native_position_limits",
        "rejected_candidates": 0,
    }

    def filtered_link_ik(_world: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        try:
            joint_state, distances = original(*args, **kwargs)
            return _filter_v1_ik_joint_limits(_world, joint_state, distances)
        except ScheduleStreamProviderError:
            raise
        except Exception as exc:
            raise ScheduleStreamProviderError("failed to enforce v1 IK joint-limit compatibility") from exc

    world.link_iterative_inverse_kinematics = MethodType(filtered_link_ik, world)


def _filter_v1_ik_joint_limits(world: Any, joint_state: Any, distances: Any) -> tuple[Any, Any]:
    """Retain only finite, in-limit IK candidates and record aggregate evidence.

    Native non-solutions may use non-finite distance sentinels. Normalize every sentinel to
    positive infinity so NaN or negative infinity can never win downstream candidate selection.
    """

    import torch

    positions = getattr(joint_state, "position", None)
    if not isinstance(positions, torch.Tensor) or positions.ndim != 4:
        raise ScheduleStreamProviderError("v1 iterative IK positions must have shape [batch, seeds, steps, joints]")
    if not isinstance(distances, torch.Tensor) or distances.shape != positions.shape[:2]:
        raise ScheduleStreamProviderError("v1 iterative IK distances must align with batch and seed dimensions")
    if distances.dtype not in (torch.float32, torch.float64):
        raise ScheduleStreamProviderError("v1 iterative IK distances must use float32 or float64")
    if not torch.isfinite(positions).all():
        raise ScheduleStreamProviderError("v1 iterative IK returned non-finite joint positions")
    limit_distances = world.get_limit_distances(joint_state)
    if not isinstance(limit_distances, torch.Tensor) or limit_distances.shape != positions.shape:
        raise ScheduleStreamProviderError("v1 world returned malformed joint-limit distances")
    if not torch.isfinite(limit_distances).all():
        raise ScheduleStreamProviderError("v1 world returned non-finite joint-limit distances")

    maximum_violation = torch.amax(limit_distances, dim=(-1, -2))
    native_candidates = torch.isfinite(distances)
    inside_limits = maximum_violation <= -V1_IK_JOINT_LIMIT_MARGIN_RAD
    rejected = native_candidates & ~inside_limits
    accepted = native_candidates & inside_limits
    filtered_distances = torch.where(accepted, distances, torch.full_like(distances, torch.inf))

    diagnostics = getattr(world, "autodata_ik_joint_limit_evidence", None)
    if not isinstance(diagnostics, dict):
        raise ScheduleStreamProviderError("v1 IK joint-limit diagnostics were not initialized")
    diagnostics["calls"] += 1
    diagnostics["candidate_solutions"] += int(native_candidates.sum().item())
    diagnostics["accepted_candidates"] += int(accepted.sum().item())
    diagnostics["rejected_candidates"] += int(rejected.sum().item())
    if bool(native_candidates.any().item()):
        minimum_margin = float((-maximum_violation[native_candidates]).min().item())
        prior_margin = diagnostics.get("minimum_observed_margin_rad")
        diagnostics["minimum_observed_margin_rad"] = (
            minimum_margin if prior_margin is None else min(float(prior_margin), minimum_margin)
        )
    return joint_state, filtered_distances


def _infer_attachment_events(
    gripper_values: tuple[float, ...],
    goal: tuple[GoalPredicate, ...],
    *,
    graspable_object: Any,
) -> tuple[DenseAttachmentEvent, ...]:
    """Recover reviewed symbolic attachment intent from an aligned binary pick/place trace.

    The upstream v1 ``PathController`` does not expose its symbolic Attach/Detach actions, but it
    does preserve their binary gripper transitions at exact controller sample indices. The live
    profile admits one task-selected movable object, so those transitions have an unambiguous
    object binding. Anything other than one open-close-open place lifecycle (or open-close for a
    terminal holding goal) is rejected instead of silently producing an unverifiable plan.
    """

    if (
        not isinstance(graspable_object, str)
        or not graspable_object
        or len(graspable_object) > 512
        or "\x00" in graspable_object
    ):
        raise MalformedScheduleStreamCommandError(
            "v1 planner did not retain the task-selected graspable object for attachment lowering"
        )
    if not goal:
        raise MalformedScheduleStreamCommandError("v1 attachment lowering requires a non-empty semantic goal")
    subjects = {predicate.subject for predicate in goal}
    if subjects != {graspable_object}:
        raise MalformedScheduleStreamCommandError(
            "v1 attachment object does not exactly match the semantic goal subject"
        )

    states: list[str] = []
    for index, value in enumerate(gripper_values):
        if value >= 1.0 - 1e-6:
            states.append("open")
        elif value <= -1.0 + 1e-6:
            states.append("closed")
        else:
            raise MalformedScheduleStreamCommandError(
                f"v1 binary-gripper trace has unsupported value {value:.6g} at sample {index}"
            )
    transitions = [(0, states[0])]
    transitions.extend((index, state) for index, state in enumerate(states[1:], start=1) if state != states[index - 1])
    relations = {predicate.relation.lower() for predicate in goal}
    expected_states = ("open", "closed") if relations == {"holding"} else ("open", "closed", "open")
    observed_states = tuple(state for _, state in transitions)
    if observed_states != expected_states:
        raise MalformedScheduleStreamCommandError(
            "v1 binary-gripper lifecycle does not match the semantic goal; "
            f"expected {expected_states}, got {observed_states}"
        )

    events = [
        DenseAttachmentEvent(
            sample_index=transitions[1][0],
            operation="attach",
            object_name=graspable_object,
        )
    ]
    if len(transitions) == 3:
        events.append(
            DenseAttachmentEvent(
                sample_index=transitions[2][0],
                operation="detach",
                object_name=graspable_object,
            )
        )
    return tuple(events)


def _validate_world_graspability(world: Any, graspable_object: str) -> dict[str, Any]:
    """Prove the constructed TAMP world exposes exactly the task-selected movable object."""

    raw_names = getattr(world, "movable_names", None)
    if isinstance(raw_names, str):
        raise ScheduleStreamProviderError("v1 world movable_names must be a materialized name collection")
    try:
        movable_names = tuple(raw_names)
    except TypeError as exc:
        raise ScheduleStreamProviderError("v1 world has no materialized movable_names collection") from exc
    if any(not isinstance(name, str) or not name or len(name) > 512 for name in movable_names):
        raise ScheduleStreamProviderError("v1 world movable_names contains an invalid scene ID")
    if movable_names != (graspable_object,):
        raise ScheduleStreamProviderError(
            "v1 world graspability does not match the task-selected pickup object; "
            f"expected {(graspable_object,)}, got {movable_names}"
        )
    return {
        "grasp_geometry": _validate_world_grasp_geometry(world, graspable_object),
        "task_selected_graspable_object": graspable_object,
        "world_movable_names": list(movable_names),
    }


def _validate_world_grasp_geometry(world: Any, graspable_object: str) -> dict[str, Any]:
    """Validate the staged world's finite grasp transforms for the off-center mesh."""

    raw = getattr(world, "autodata_grasp_geometry", None)
    expected_static = {
        "asset_name": V1_REVIEWED_GRASPABLE_ASSET,
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": 4,
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": graspable_object,
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "profile": V1_GRASP_GEOMETRY_PROFILE,
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }
    dynamic_keys = {
        "converted_object_origin_from_aabb_center",
        "link_from_object_transforms",
        "primitive_link_from_aabb_center_transforms",
    }
    if not isinstance(raw, Mapping) or set(raw) != set(expected_static) | dynamic_keys:
        raise ScheduleStreamProviderError("v1 world grasp geometry was not exactly validated")
    if any(raw.get(key) != value for key, value in expected_static.items()):
        raise ScheduleStreamProviderError("v1 world grasp geometry was not exactly validated")
    primitive_values = raw.get("primitive_link_from_aabb_center_transforms")
    link_from_object_values = raw.get("link_from_object_transforms")
    if not isinstance(primitive_values, list) or not isinstance(link_from_object_values, list):
        raise ScheduleStreamProviderError("v1 grasp transforms must be materialized lists")
    if len(primitive_values) != 4 or len(link_from_object_values) != 4:
        raise ScheduleStreamProviderError("v1 grasp geometry must contain exactly four transforms")
    try:
        object_from_aabb = _matrix_from_native(
            raw["converted_object_origin_from_aabb_center"],
            "grasp_geometry.converted_object_origin_from_aabb_center",
        )
        aabb_from_object = _rigid_inverse(object_from_aabb)
        primitive_transforms = tuple(
            _matrix_from_native(value, f"grasp_geometry.primitive[{index}]")
            for index, value in enumerate(primitive_values)
        )
        link_from_object_transforms = tuple(
            _matrix_from_native(value, f"grasp_geometry.link_from_object[{index}]")
            for index, value in enumerate(link_from_object_values)
        )
    except Exception as exc:
        raise ScheduleStreamProviderError("v1 grasp geometry contains a malformed transform") from exc
    if len(set(primitive_transforms)) != 4:
        raise ScheduleStreamProviderError("v1 grasp geometry primitive transforms are not unique")
    if len(set(link_from_object_transforms)) != 4:
        raise ScheduleStreamProviderError("v1 grasp geometry transforms are not unique")
    for index, (primitive, link_from_object) in enumerate(zip(primitive_transforms, link_from_object_transforms)):
        expected = _matrix_multiply(primitive, aabb_from_object)
        position_error, rotation_error = _transform_error(expected, link_from_object)
        if (
            position_error > V1_GRASP_GEOMETRY_MAX_POSITION_ROUNDOFF_M
            or rotation_error > V1_GRASP_GEOMETRY_MAX_ROTATION_ROUNDOFF_RAD
        ):
            raise ScheduleStreamProviderError(
                f"v1 grasp transform {index} does not match the validated composition formula "
                f"(position={position_error:.6g}m, rotation={rotation_error:.6g}rad)"
            )
    try:
        world_object = world.get_object(graspable_object)
        grasp_config = world_object.grasp_config
        configured_poses = grasp_config.generator
    except Exception as exc:
        raise ScheduleStreamProviderError("v1 world has no inspectable grasp generator") from exc
    if (
        getattr(grasp_config, "primitive", None) != "cuboid"
        or getattr(grasp_config, "pitch_interval", None) != "top"
        or not isinstance(configured_poses, tuple)
        or len(configured_poses) != 4
    ):
        raise ScheduleStreamProviderError("v1 world grasp generator config is incompatible")
    configured_transforms = tuple(
        _native_pose_matrix(pose, f"world_grasp_config.generator[{index}]")
        for index, pose in enumerate(configured_poses)
    )
    for index, (configured, link_from_object) in enumerate(zip(configured_transforms, link_from_object_transforms)):
        position_error, rotation_error = _transform_error(configured, link_from_object)
        if (
            position_error > V1_GRASP_GEOMETRY_MAX_POSITION_ROUNDOFF_M
            or rotation_error > V1_GRASP_GEOMETRY_MAX_ROTATION_ROUNDOFF_RAD
        ):
            raise ScheduleStreamProviderError(
                f"v1 world grasp generator pose {index} does not match its geometry record "
                f"(position={position_error:.6g}m, rotation={rotation_error:.6g}rad)"
            )
    return dict(raw)


def _validate_world_destination_placement(
    world: Any,
    *,
    graspable_object: str,
    destination_object: str,
    object_pose_offset_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate placement geometry and derive live rigid-root-to-AABB-center transforms."""

    raw = getattr(world, "autodata_destination_placement_geometry", None)
    if not isinstance(raw, Mapping) or raw.get("attested") is not True:
        raise ScheduleStreamProviderError("v1 world destination placement profile was not attested")
    if (
        raw.get("schema_version") != 1
        or raw.get("profile") != V1_DESTINATION_PLACEMENT_PROFILE
        or raw.get("relation") != "on"
        or raw.get("general_inside_semantics") is not False
    ):
        raise ScheduleStreamProviderError("v1 world destination placement geometry is incompatible")
    result = dict(raw)
    for role, expected_name in (("subject", graspable_object), ("destination", destination_object)):
        raw_geometry = raw.get(role)
        if not isinstance(raw_geometry, Mapping) or raw_geometry.get("object_id") != expected_name:
            raise ScheduleStreamProviderError(f"v1 world destination placement {role} identity does not match")
        if expected_name not in object_pose_offset_evidence:
            raise ScheduleStreamProviderError(f"v1 world has no rigid-root offset evidence for {expected_name!r}")
        try:
            root_from_converted = _matrix_from_native(
                object_pose_offset_evidence[expected_name],
                f"object_root_to_converted[{expected_name}]",
            )
            converted_from_aabb = _matrix_from_native(
                raw_geometry["converted_object_origin_from_aabb_center"],
                f"converted_from_aabb[{expected_name}]",
            )
            root_from_aabb = _matrix_multiply(root_from_converted, converted_from_aabb)
        except Exception as exc:
            raise ScheduleStreamProviderError(
                f"failed to derive rigid-root-to-AABB-center transform for {expected_name!r}"
            ) from exc
        geometry = dict(raw_geometry)
        geometry["aabb_center_in_isaac_rigid_root_m"] = [root_from_aabb[index][3] for index in range(3)]
        geometry["isaac_rigid_root_from_aabb_center"] = [list(row) for row in root_from_aabb]
        result[role] = geometry
    return result


def _select_semantic_arm(available_arms: tuple[str, ...], requested_arm: str | None) -> str:
    if requested_arm is None:
        if len(available_arms) != 1:
            raise ScheduleStreamProviderError(
                f"v1 semantic goal requires an explicit arm; world arms are {available_arms}"
            )
        return available_arms[0]
    if requested_arm not in available_arms:
        raise ScheduleStreamProviderError(
            f"requested arm {requested_arm!r} is not present; world arms are {available_arms}"
        )
    return requested_arm


def _capture_v1_object_pose_offsets(planner: Any, module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture immutable rigid-root-to-converted-mesh transforms before live synchronization."""

    multiply_poses = getattr(module, "multiply_poses", None)
    if not callable(multiply_poses):
        raise ScheduleStreamProviderError("v1 planner module has no callable multiply_poses")
    names = tuple(planner.scene.rigid_objects)
    if len(names) > 10_000:
        raise ScheduleStreamProviderError("IsaacLab scene has more than 10000 rigid objects")
    offsets: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for name in names:
        try:
            root_pose = planner.pose(name)
            mesh_pose = planner.world.get_object_pose(name)
            offset = multiply_poses(root_pose.inverse(), mesh_pose)
            [offset_matrix] = offset.get_numpy_matrix()
            matrix = _matrix_from_native(offset_matrix, f"object_root_to_mesh[{name}]")
        except Exception as exc:
            if isinstance(exc, ScheduleStreamProviderError):
                raise
            raise ScheduleStreamProviderError(
                f"failed to capture fixed rigid-root-to-mesh transform for {name!r}"
            ) from exc
        offsets[name] = offset
        evidence[name] = [list(row) for row in matrix]
    return offsets, evidence


def _restore_v1_object_mesh_poses(planner: Any, module: Any, *, state: Any | None) -> None:
    """Synchronize converted collision meshes without discarding their USD root offsets."""

    offsets = getattr(planner, "object_pose_offsets", None)
    if not isinstance(offsets, Mapping):
        raise ScheduleStreamProviderError("v1 planner object pose offsets were not initialized")
    scene_names = tuple(planner.scene.rigid_objects)
    if set(offsets) != set(scene_names):
        raise ScheduleStreamProviderError("v1 planner object pose offsets do not match the live rigid-object set")
    multiply_poses = getattr(module, "multiply_poses", None)
    if not callable(multiply_poses):
        raise ScheduleStreamProviderError("v1 planner module has no callable multiply_poses")
    for name in scene_names:
        try:
            root_pose = planner.pose(name, state)
            planner.world.set_object_pose(name, multiply_poses(root_pose, offsets[name]))
        except Exception as exc:
            if isinstance(exc, ScheduleStreamProviderError):
                raise
            raise ScheduleStreamProviderError(
                f"failed to synchronize converted mesh pose for rigid object {name!r}"
            ) from exc


def _scene_state_with_curobo_pose_order(state: Any, name: str) -> dict[str, Any]:
    """Shallow-copy one IsaacLab scene pose while converting XYZW to cuRobo v1 WXYZ.

    Args:
        state: Live ``InteractiveScene.state`` mapping.
        name: Articulation or rigid-object scene name whose root pose will be consumed by upstream
            ScheduleStream.

    Returns:
        A state mapping identical to ``state`` except for the selected root-pose quaternion order.

    Raises:
        ScheduleStreamProviderError: If the one-environment live state is malformed.
    """

    import torch

    if not isinstance(state, Mapping):
        raise ScheduleStreamProviderError("IsaacLab scene state must be a mapping")
    if not isinstance(name, str) or not name:
        raise ScheduleStreamProviderError("IsaacLab scene pose name must be a non-empty string")
    for body_type, bodies in state.items():
        if not isinstance(bodies, Mapping) or name not in bodies:
            continue
        body_state = bodies[name]
        if not isinstance(body_state, Mapping):
            raise ScheduleStreamProviderError(f"IsaacLab state for {name!r} must be a mapping")
        root_pose = body_state.get("root_pose")
        if (
            not isinstance(root_pose, torch.Tensor)
            or root_pose.ndim != 2
            or tuple(root_pose.shape) != (1, 7)
            or root_pose.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(root_pose).all().item())
        ):
            raise ScheduleStreamProviderError(
                f"IsaacLab root pose for {name!r} must be one finite float32/float64 XYZ+XYZW row"
            )
        root_pose_wxyz = torch.cat((root_pose[:, :3], root_pose[:, 6:7], root_pose[:, 3:6]), dim=1)
        converted_body_state = dict(body_state)
        converted_body_state["root_pose"] = root_pose_wxyz
        converted_bodies = dict(bodies)
        converted_bodies[name] = converted_body_state
        converted_state = dict(state)
        converted_state[body_type] = converted_bodies
        return converted_state
    raise ScheduleStreamProviderError(f"IsaacLab scene state has no root pose for {name!r}")


def _close_failed_staged_world(
    world: Any | None,
    world_closer: _WorldCloser | None,
    initialization_error: Exception,
) -> None:
    if world is None or world_closer is None:
        return
    try:
        world_closer(world)
    except Exception as cleanup_error:
        initialization = (
            f"{type(initialization_error).__name__}: {str(initialization_error).replace(chr(10), ' ')[:300]}"
        )
        cleanup = f"{type(cleanup_error).__name__}: {str(cleanup_error).replace(chr(10), ' ')[:300]}"
        raise ScheduleStreamProviderError(
            f"v1 planner initialization failed ({initialization}) and world cleanup failed ({cleanup})"
        ) from cleanup_error


def _validate_v1_destination_request(
    predicates: tuple[GoalPredicate, ...],
    *,
    graspable_object: str,
    destination_object: str,
    graspable_asset_name: str,
    destination_asset_name: str,
    arm: str | None,
) -> None:
    """Fail closed unless the direct API describes the one reviewed placement capability."""

    if not predicates or any(not isinstance(item, GoalPredicate) for item in predicates):
        raise ValueError("goal must contain GoalPredicate instances")
    for field_name, value in (("graspable_object", graspable_object), ("destination_object", destination_object)):
        if not isinstance(value, str) or not value.strip() or len(value) > 512 or "\x00" in value:
            raise ValueError(f"{field_name} must be a bounded non-empty scene ID")
    if graspable_object == destination_object:
        raise ValueError("graspable_object and destination_object must be distinct")
    if graspable_asset_name != V1_REVIEWED_GRASPABLE_ASSET:
        raise ValueError(f"graspable_asset_name must be the reviewed asset {V1_REVIEWED_GRASPABLE_ASSET!r}")
    if destination_asset_name != V1_REVIEWED_DESTINATION_ASSET:
        raise ValueError(f"destination_asset_name must be the reviewed asset {V1_REVIEWED_DESTINATION_ASSET!r}")
    expected_goal = ("on", graspable_object, destination_object)
    observed_goals = tuple((predicate.relation, predicate.subject, predicate.target) for predicate in predicates)
    if observed_goals != (expected_goal,):
        raise ValueError(
            "the v1 destination placement profile requires exactly "
            f"on({graspable_object}, {destination_object}); got {observed_goals}"
        )
    if arm is not None and (not isinstance(arm, str) or not arm.strip() or "\x00" in arm):
        raise ValueError("arm must be null or a non-empty string")


def create_v1_isaaclab_command_planner(
    env: Any,
    goal: Iterable[GoalPredicate],
    *,
    graspable_object: str,
    destination_object: str,
    graspable_asset_name: str,
    destination_asset_name: str,
    arm: str | None = None,
    config: V1IsaacLabPlannerConfig | None = None,
    module_loader: _ModuleLoader | None = None,
    goal_symbols_loader: _GoalSymbolsLoader | None = None,
    world_factory: _WorldFactory | None = None,
    eef_pose_reader: _EefPoseReader | None = None,
    seed_setter: _SeedSetter | None = None,
    world_closer: _WorldCloser | None = None,
) -> V1IsaacLabCommandPlanner:
    """Create the concrete semantic v1 provider without importing heavy modules beforehand.

    Args:
        env: Live IsaacLab manager-based environment.
        goal: Resolved backend-neutral goal predicates.
        graspable_object: Exact live scene ID selected by the task as ``pick_up_object``.
        destination_object: Exact live scene ID selected by the task as ``destination_location``.
        graspable_asset_name: Linked Arena asset name for the pickup object.
        destination_asset_name: Linked Arena asset name for the destination object.
        arm: Optional custream arm ID; inferred only when the world has exactly one arm.
        config: Bounded v1 runtime settings.
        module_loader: Test hook returning the upstream IsaacLab planner module.
        goal_symbols_loader: Test hook returning injected goal-language symbols.
        world_factory: Test hook or alternate reviewed custream world construction function.
        eef_pose_reader: Callback returning the live observed EEF 4x4 pose for frame attestation.
        seed_setter: Test hook or custream RNG seeding callback invoked for every attempt.
        world_closer: Optional explicit GPU/resource cleanup callback. Upstream v1 exposes no close.
    """

    if env is None:
        raise ValueError("env must not be None")
    predicates = tuple(goal)
    _validate_v1_destination_request(
        predicates,
        graspable_object=graspable_object,
        destination_object=destination_object,
        graspable_asset_name=graspable_asset_name,
        destination_asset_name=destination_asset_name,
        arm=arm,
    )
    config = config or V1IsaacLabPlannerConfig()
    module_loader = module_loader or _load_v1_isaaclab_planner_module
    goal_symbols_loader = goal_symbols_loader or load_schedulestream_goal_symbols
    world_factory = world_factory or build_v1_isaaclab_world
    seed_setter = seed_setter or _load_v1_seed_setter()
    try:
        module = module_loader()
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamImportError(
            f"failed to import v1 IsaacLab planner ({type(exc).__name__}: {message})"
        ) from exc
    _validate_v1_module(module)

    class SemanticV1Planner(module.Planner):
        """Upstream-compatible planner whose goal and world are supplied by AutoData."""

        def __init__(self) -> None:
            _initialize_semantic_v1_planner(
                self,
                env=env,
                module=module,
                predicates=predicates,
                graspable_object=graspable_object,
                destination_object=destination_object,
                graspable_asset_name=graspable_asset_name,
                destination_asset_name=destination_asset_name,
                arm=arm,
                config=config,
                world_factory=world_factory,
                goal_symbols_loader=goal_symbols_loader,
                world_closer=world_closer,
            )

        def _pose(self, name: str, state: Any | None = None) -> Any:
            """Bridge IsaacLab 6 XYZW scene state into pinned cuRobo v1 WXYZ poses."""

            source_state = self.scene.state if state is None else state
            converted_state = _scene_state_with_curobo_pose_order(source_state, name)
            return super()._pose(name, state=converted_state)

        def set_env_state(self, env_id: int, state: Any | None = None, **kwargs: Any) -> None:
            super().set_env_state(env_id, state=state, **kwargs)
            _restore_v1_object_mesh_poses(self, module, state=state)

        def solve_commands(self, env_id: int) -> tuple[Any, Any | None]:
            self.set_env_state(env_id)
            state = self.world.state()
            try:
                with module.timeout_context(timeout=2 * self.max_time):
                    commands = module.solve_tamp(
                        state,
                        self.goal,
                        collisions=self.collisions,
                        max_time=self.max_time,
                        profile=config.profile,
                    )
            except Exception as exc:
                failure_name = type(exc).__name__
                self.errors[failure_name] += 1
                message = str(exc).replace("\n", " ")[:500]
                raise ScheduleStreamProviderError(
                    f"v1 ScheduleStream planning failed ({failure_name}: {message})"
                ) from exc
            if self.animate or self.video:
                self.frames.extend(module.animate_commands(state, commands, frequency=1, record=self.video))
            return state, commands

        def plan_commands(self, env_id: int) -> tuple[Any, ...] | None:
            state, commands = self.solve_commands(env_id)
            state.set()
            if commands is None:
                return None
            try:
                return tuple(module.Commands.flatten([commands]))
            except Exception as exc:
                raise ScheduleStreamProviderError("v1 planner returned a malformed command container") from exc

        def plan_controller(self, env_id: int) -> Any | None:
            state, commands = self.solve_commands(env_id)
            if commands is None:
                state.set()
                return None
            try:
                controller = module.create_controller(self, state, commands)
            except Exception as exc:
                raise ScheduleStreamProviderError("failed to construct aligned v1 PathController") from exc
            finally:
                state.set()
            if controller is None:
                raise ScheduleStreamProviderError("v1 create_controller returned None for a solved command stream")
            return controller

        def current_link_matrix(self, link_name: str) -> Matrix4:
            try:
                pose = self.from_reference(self.world.get_node_pose(link_name))
                [native_matrix] = pose.get_numpy_matrix()
            except Exception as exc:
                raise ScheduleStreamProviderError(
                    f"failed to read current world pose for action link {link_name!r}"
                ) from exc
            return _matrix_from_native(native_matrix, f"current_link[{link_name}]")

        def hold_commands(self, env_id: int, *, steps: int) -> tuple[Any, ...]:
            self.set_env_state(env_id)
            command = self.world.configuration()
            return (command,) * steps

    try:
        native_planner = SemanticV1Planner()
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamProviderError(
            f"failed to construct semantic v1 planner ({type(exc).__name__}: {message})"
        ) from exc
    return V1IsaacLabCommandPlanner(
        native_planner,
        eef_pose_reader=eef_pose_reader,
        seed_setter=seed_setter,
        world_closer=world_closer,
    )


def build_v1_isaaclab_world(
    planner_module: Any,
    scene: Any,
    config: V1IsaacLabPlannerConfig,
    *,
    graspable_object: str,
    destination_object: str,
    graspable_asset_name: str,
    destination_asset_name: str,
    surface_config_factory: Callable[..., Any] | None = None,
    primitive_grasp_generator: _PrimitiveGraspGenerator | None = None,
) -> Any:
    """Construct the reviewed cube-to-bowl custream world and placement surrogate."""

    articulations = tuple(scene.articulations)
    if len(articulations) != 1:
        raise ScheduleStreamProviderError(
            f"v1 IsaacLab provider requires exactly one articulation, got {articulations}"
        )
    robot = articulations[0]
    articulation = scene.articulations[robot]
    spawn = getattr(articulation.cfg, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    if not isinstance(usd_path, str) or not usd_path:
        raise ScheduleStreamProviderError("robot articulation has no USD path")
    usd_name = os.path.basename(usd_path)
    if usd_name not in V1_FRANKA_USD_BASENAMES:
        raise ScheduleStreamProviderError(
            f"unsupported v1 robot USD {usd_name!r}; supported: {sorted(V1_FRANKA_USD_BASENAMES)}"
        )
    objects = planner_module.create_objects(scene, env_id=0)
    grasp_geometry = _repair_converted_rigid_object_mobility(
        planner_module,
        scene,
        objects,
        graspable_object=graspable_object,
        graspable_asset_name=graspable_asset_name,
        primitive_grasp_generator=primitive_grasp_generator,
    )
    destination_placement_geometry = _configure_v1_destination_placement(
        objects,
        graspable_object=graspable_object,
        destination_object=destination_object,
        graspable_asset_name=graspable_asset_name,
        destination_asset_name=destination_asset_name,
        surface_config_factory=surface_config_factory or getattr(planner_module, "SurfaceConfig", None),
    )
    robot_reference_pose = _live_robot_reference_pose(planner_module, scene, robot)
    # Upstream object conversion uses the enclosing /Robot prim as its reference while Arena's
    # stand USD places panda_link0 at a nontrivial transform beneath that prim. Rebase every
    # converted obstacle into panda_link0 coordinates and keep cuRobo's robot base at identity.
    # The inherited Planner.to_reference/from_reference methods then add the live articulation
    # root exactly once when crossing between planner and world coordinates.
    robot_config = planner_module.load_franka_config(base_poses=None)
    sim_dt = float(scene.sim.get_physics_dt())
    if not math.isfinite(sim_dt) or sim_dt <= 0:
        raise ScheduleStreamProviderError("IsaacLab physics dt must be positive and finite")
    interpolation_dt = config.scale_dt * sim_dt
    world = planner_module.World(
        robot_config,
        objects,
        visualize_spheres=config.visualize_spheres,
        interpolation_dt=interpolation_dt,
    )
    world.autodata_destination_placement_geometry = destination_placement_geometry
    world.autodata_grasp_geometry = grasp_geometry
    try:
        _rebase_v1_world_objects(world, planner_module, robot_reference_pose)
        positions = scene.state["articulation"][robot]["joint_position"][0]
        world.set_joint_positions(articulation.joint_names, positions)
        world.set_camera_pose(planner_module.CAMERA_POSE)
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to synchronize initial IsaacLab robot state") from exc
    return world


def _configure_v1_destination_placement(
    converted_objects: Any,
    *,
    graspable_object: str,
    destination_object: str,
    graspable_asset_name: str,
    destination_asset_name: str,
    surface_config_factory: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Assign and attest the one reviewed v1 shifted-top-plane placement profile."""

    if graspable_object == destination_object:
        raise ScheduleStreamProviderError("graspable and destination objects must be distinct")
    if graspable_asset_name != V1_REVIEWED_GRASPABLE_ASSET:
        raise ScheduleStreamProviderError("v1 placement source is not the reviewed Rubik's cube asset")
    if destination_asset_name != V1_REVIEWED_DESTINATION_ASSET:
        raise ScheduleStreamProviderError("v1 placement destination is not the reviewed YCB bowl asset")
    source = _unique_converted_object(converted_objects, graspable_object)
    destination = _unique_converted_object(converted_objects, destination_object)
    source_geometry = _converted_aabb_geometry(source, graspable_object)
    destination_geometry = _converted_aabb_geometry(destination, destination_object)
    _require_reviewed_aabb_dimension_bounds(
        source_geometry["aabb_dimensions_m"],
        V1_REVIEWED_GRASPABLE_AABB_DIMENSION_BOUNDS_M,
        graspable_object,
    )
    _require_reviewed_aabb_dimension_bounds(
        destination_geometry["aabb_dimensions_m"],
        V1_REVIEWED_DESTINATION_AABB_DIMENSION_BOUNDS_M,
        destination_object,
    )
    source_geometry["aabb_dimension_bounds_m"] = [
        list(bounds) for bounds in V1_REVIEWED_GRASPABLE_AABB_DIMENSION_BOUNDS_M
    ]
    destination_geometry["aabb_dimension_bounds_m"] = [
        list(bounds) for bounds in V1_REVIEWED_DESTINATION_AABB_DIMENSION_BOUNDS_M
    ]
    source_dimensions = source_geometry["aabb_dimensions_m"]
    destination_dimensions = destination_geometry["aabb_dimensions_m"]
    requested_xy_extend = -max(destination_dimensions[:2])
    requested_z_offset = V1_DESIRED_AABB_CENTER_VERTICAL_OFFSET_M - 0.5 * (
        destination_dimensions[2] + source_dimensions[2]
    )

    factory = surface_config_factory or _load_v1_surface_config_factory()
    if not callable(factory):
        raise ScheduleStreamProviderError("v1 SurfaceConfig factory is not callable")
    try:
        surface_config = factory(
            xy_extend=requested_xy_extend,
            z_offset=requested_z_offset,
        )
        actual_xy_extend = _placement_float(surface_config.xy_extend, "SurfaceConfig.xy_extend")
        actual_z_offset = _placement_float(surface_config.z_offset, "SurfaceConfig.z_offset")
        surface_extend = _placement_vector3(surface_config.surface_extend, "SurfaceConfig.surface_extend")
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to construct the reviewed v1 SurfaceConfig") from exc
    if not math.isclose(actual_xy_extend, requested_xy_extend, rel_tol=0.0, abs_tol=1e-12):
        raise ScheduleStreamProviderError("v1 SurfaceConfig changed the reviewed xy_extend")
    if not math.isclose(actual_z_offset, requested_z_offset, rel_tol=0.0, abs_tol=1e-12):
        raise ScheduleStreamProviderError("v1 SurfaceConfig changed the reviewed z_offset")
    expected_surface_extend = (actual_xy_extend, actual_xy_extend, 0.0)
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in zip(surface_extend, expected_surface_extend)
    ):
        raise ScheduleStreamProviderError("v1 SurfaceConfig.surface_extend has unexpected semantics")

    top_surface_dimensions = tuple(destination_dimensions[:2]) + (0.0,)
    sampled_surface_extent = tuple(
        max(0.0, dimension + extension) for dimension, extension in zip(top_surface_dimensions, surface_extend)
    )
    if any(abs(value) > 1e-9 for value in sampled_surface_extent):
        raise ScheduleStreamProviderError(
            "reviewed v1 destination profile did not collapse the bowl AABB surface to its exact center"
        )
    try:
        for item in converted_objects:
            item.surface_config = None
        destination.surface_config = surface_config
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to assign the reviewed destination SurfaceConfig") from exc

    source_height = source_dimensions[2]
    destination_height = destination_dimensions[2]
    predicted_vertical_offset = destination_height / 2.0 + actual_z_offset + source_height / 2.0
    if not math.isclose(
        predicted_vertical_offset,
        V1_DESIRED_AABB_CENTER_VERTICAL_OFFSET_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ScheduleStreamProviderError("reviewed placement derivation did not preserve its AABB-center target")
    if abs(predicted_vertical_offset) > V1_VERTICAL_EVIDENCE_CORRIDOR_M:
        raise ScheduleStreamProviderError("reviewed placement profile exceeds the physical-evidence vertical corridor")
    return {
        "attested": True,
        "destination": {
            "asset_name": destination_asset_name,
            "object_id": destination_object,
            **destination_geometry,
        },
        "frame_convention": "parent_from_child_homogeneous_4x4",
        "general_inside_semantics": False,
        "limitations": [
            "name_pinned_local_aabb_geometry_not_container_interior_geometry",
            "negative_top_plane_offset_is_a_gpu_validation_surrogate",
            "pinned_v1_placement_collision_check_excludes_the_destination_parent",
        ],
        "placement_model": "destination_local_aabb_top_plane_shifted_downward",
        "predicted_aabb_center_offset_m": [0.0, 0.0, predicted_vertical_offset],
        "profile": V1_DESTINATION_PLACEMENT_PROFILE,
        "relation": "on",
        "sampled_surface_extent_m": list(sampled_surface_extent),
        "schema_version": 1,
        "subject": {
            "asset_name": graspable_asset_name,
            "object_id": graspable_object,
            **source_geometry,
        },
        "surface_config": {
            "derivation": {
                "desired_aabb_center_vertical_offset_m": V1_DESIRED_AABB_CENTER_VERTICAL_OFFSET_M,
                "inputs": "attested_converted_local_aabb_dimensions_m",
                "xy_extend_formula": "-max(destination_aabb_width_m,destination_aabb_depth_m)",
                "z_offset_formula": "desired_center_z_m-0.5*(destination_aabb_height_m+subject_aabb_height_m)",
            },
            "implementation": "schedulestream.applications.custream.object.SurfaceConfig",
            "xy_extend_m": actual_xy_extend,
            "z_offset_m": actual_z_offset,
        },
        "vertical_evidence_corridor_m": V1_VERTICAL_EVIDENCE_CORRIDOR_M,
    }


def _unique_converted_object(converted_objects: Any, name: str) -> Any:
    if not isinstance(converted_objects, (list, tuple)):
        raise ScheduleStreamProviderError("v1 converted objects must be a materialized list or tuple")
    matches = [item for item in converted_objects if getattr(item, "name", None) == name]
    if not matches:
        raise ScheduleStreamProviderError(f"reviewed placement object {name!r} did not bind to a converted object")
    if len(matches) != 1:
        raise ScheduleStreamProviderError(
            f"reviewed placement object {name!r} bound ambiguously to {len(matches)} converted objects"
        )
    return matches[0]


def _converted_aabb_geometry(converted_object: Any, name: str) -> dict[str, Any]:
    try:
        bounding_box = converted_object.bounding_box
        dimensions = _placement_vector3(bounding_box.dimensions, f"converted_aabb[{name}].dimensions")
        [native_transform] = bounding_box.pose.get_numpy_matrix()
        transform = _matrix_from_native(native_transform, f"converted_aabb[{name}].pose")
    except Exception as exc:
        raise ScheduleStreamProviderError(f"failed to attest converted local AABB for {name!r}") from exc
    if any(value <= 0.0 for value in dimensions):
        raise ScheduleStreamProviderError(f"converted local AABB for {name!r} must have positive dimensions")
    return {
        "aabb_center_in_converted_object_origin_m": [transform[index][3] for index in range(3)],
        "aabb_dimensions_m": list(dimensions),
        "aabb_kind": "converted_mesh_local_axis_aligned_bounding_box",
        "converted_object_origin_from_aabb_center": [list(row) for row in transform],
    }


def _require_reviewed_aabb_dimension_bounds(
    actual: Any,
    expected_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    name: str,
) -> None:
    if any(not lower <= value <= upper for value, (lower, upper) in zip(actual, expected_bounds)):
        raise ScheduleStreamProviderError(
            f"converted local AABB for {name!r} is outside the reviewed per-axis bounds; "
            f"expected {expected_bounds}, got {tuple(actual)}"
        )


def _placement_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ScheduleStreamProviderError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleStreamProviderError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ScheduleStreamProviderError(f"{field_name} must be finite")
    return result


def _placement_vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    materialized = _to_builtin(value, field_name)
    if not isinstance(materialized, (list, tuple)) or len(materialized) != 3:
        raise ScheduleStreamProviderError(f"{field_name} must contain three values")
    return tuple(_placement_float(item, field_name) for item in materialized)  # type: ignore[return-value]


def _live_robot_reference_pose(planner_module: Any, scene: Any, robot: str) -> Any:
    """Return the enclosing robot-prim to live articulation-root transform.

    The reviewed profile has exactly one environment. Its finite world origin is mandatory because
    silently assuming zero would move every collision object when the scene uses a translated origin.
    IsaacLab's Warp-backed pose is ``XYZ + XYZW`` while cuRobo v1 consumes ``XYZ + WXYZ``;
    the quaternion reorder is an explicit compatibility boundary.
    """

    converter = getattr(planner_module, "to_pose", None)
    if not callable(converter):
        raise ScheduleStreamProviderError("v1 planner module has no callable to_pose converter")
    try:
        root_pose_value = scene.state["articulation"][robot]["root_pose"]
        [root_pose] = _rows_from_native(
            root_pose_value,
            f"scene.state.articulation[{robot}].root_pose",
            maximum_rows=1,
            expected_columns=7,
        )
        env_origins = getattr(scene, "env_origins", None)
        if env_origins is None:
            raise ScheduleStreamProviderError("scene.env_origins is required for v1 reference-frame attestation")
        [env_origin] = _rows_from_native(
            env_origins,
            "scene.env_origins",
            maximum_rows=1,
            expected_columns=3,
        )
        relative_position = tuple(root_pose[index] - env_origin[index] for index in range(3))
        quaternion_wxyz = (root_pose[6], root_pose[3], root_pose[4], root_pose[5])
        pose = converter(relative_position + quaternion_wxyz)
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        raise ScheduleStreamProviderError(f"failed to derive live robot reference pose for {robot!r}") from exc
    if pose is None:
        raise ScheduleStreamProviderError("v1 to_pose converter returned null for the robot base")
    return pose


def _rebase_v1_world_objects(world: Any, planner_module: Any, robot_reference_pose: Any) -> None:
    """Move converted obstacles from the enclosing robot prim into articulation-root coordinates."""

    multiply_poses = getattr(planner_module, "multiply_poses", None)
    if not callable(multiply_poses):
        raise ScheduleStreamProviderError("v1 planner module has no callable multiply_poses")
    raw_names = getattr(world, "object_names", None)
    if isinstance(raw_names, str):
        raise ScheduleStreamProviderError("v1 world object_names must be a materialized name collection")
    try:
        object_names = tuple(raw_names)
    except TypeError as exc:
        raise ScheduleStreamProviderError("v1 world has no materialized object_names collection") from exc
    if (
        not object_names
        or len(object_names) > 100_000
        or len(set(object_names)) != len(object_names)
        or any(not isinstance(name, str) or not name or len(name) > 1024 for name in object_names)
    ):
        raise ScheduleStreamProviderError("v1 world object_names is empty, duplicated, or malformed")
    inverse = getattr(robot_reference_pose, "inverse", None)
    if not callable(inverse):
        raise ScheduleStreamProviderError("v1 robot reference pose has no callable inverse")
    try:
        reference_inverse = inverse()
        [reference_matrix] = robot_reference_pose.get_numpy_matrix()
        matrix = _matrix_from_native(reference_matrix, "robot_prim_to_articulation_root")
        for name in object_names:
            object_pose = world.get_object_pose(name)
            world.set_object_pose(name, multiply_poses(reference_inverse, object_pose))
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to rebase v1 obstacles into articulation-root coordinates") from exc
    world.autodata_reference_frame_evidence = {
        "converted_scene_frame": "enclosing_robot_prim",
        "curobo_pose_quaternion_order": "wxyz",
        "isaaclab_pose_quaternion_order": "xyzw",
        "object_count": len(object_names),
        "planner_frame": "articulation_root",
        "robot_prim_to_articulation_root": [list(row) for row in matrix],
    }


def _repair_converted_rigid_object_mobility(
    planner_module: Any,
    scene: Any,
    converted_objects: Any,
    *,
    graspable_object: str,
    graspable_asset_name: str,
    primitive_grasp_generator: _PrimitiveGraspGenerator | None,
) -> dict[str, Any]:
    """Make only the task-selected object graspable with the reviewed cube strategy."""

    entries = _scene_rigid_object_entries(scene)
    if not isinstance(converted_objects, (list, tuple)):
        raise ScheduleStreamProviderError("v1 create_objects must return a materialized list or tuple")
    if len(converted_objects) > 100_000:
        raise ScheduleStreamProviderError("v1 create_objects returned more than 100000 objects")
    grasp_config_type = getattr(planner_module, "GraspConfig", None)
    if entries and not callable(grasp_config_type):
        raise ScheduleStreamProviderError("v1 planner module has no callable GraspConfig")
    entries_by_name = {name: is_kinematic for name, _, is_kinematic in entries}
    if graspable_object not in entries_by_name:
        raise ScheduleStreamProviderError(
            f"task-selected graspable object {graspable_object!r} is absent from IsaacLab scene.rigid_objects"
        )
    if entries_by_name[graspable_object]:
        raise ScheduleStreamProviderError(
            f"task-selected graspable object {graspable_object!r} is explicitly kinematic"
        )

    converted_by_name = {}
    for scene_name, _, _ in entries:
        matches = [item for item in converted_objects if getattr(item, "name", None) == scene_name]
        if not matches:
            raise ScheduleStreamProviderError(
                f"Arena rigid object {scene_name!r} did not bind to any converted custream object"
            )
        if len(matches) != 1:
            raise ScheduleStreamProviderError(
                f"Arena rigid object {scene_name!r} bound ambiguously to {len(matches)} converted custream objects"
            )
        converted = matches[0]
        converted_by_name[scene_name] = converted
        try:
            converted.grasp_config = None
        except Exception as exc:
            raise ScheduleStreamProviderError(
                f"failed to classify converted custream object {scene_name!r} as non-graspable"
            ) from exc
    link_from_object_poses, grasp_geometry = _build_v1_analytical_grasps(
        planner_module,
        converted_by_name[graspable_object],
        graspable_object=graspable_object,
        graspable_asset_name=graspable_asset_name,
        primitive_grasp_generator=primitive_grasp_generator,
    )
    try:
        converted_by_name[graspable_object].grasp_config = grasp_config_type(
            primitive="cuboid",
            pitch_interval="top",
            generator=link_from_object_poses,
        )
    except Exception as exc:
        raise ScheduleStreamProviderError(
            f"failed to install analytical grasp generator for {graspable_object!r}"
        ) from exc
    return grasp_geometry


def _build_v1_analytical_grasps(
    planner_module: Any,
    converted_object: Any,
    *,
    graspable_object: str,
    graspable_asset_name: str,
    primitive_grasp_generator: _PrimitiveGraspGenerator | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Materialize the reviewed analytical top grasps in the object's true mesh frame."""

    if graspable_asset_name != V1_REVIEWED_GRASPABLE_ASSET:
        raise ScheduleStreamProviderError("v1 analytical grasp source is not the reviewed Rubik's cube asset")
    geometry = _converted_aabb_geometry(converted_object, graspable_object)
    _require_reviewed_aabb_dimension_bounds(
        geometry["aabb_dimensions_m"],
        V1_REVIEWED_GRASPABLE_AABB_DIMENSION_BOUNDS_M,
        graspable_object,
    )
    generator = primitive_grasp_generator or getattr(planner_module, "primitive_grasp_generator", None)
    if generator is None:
        generator = _load_v1_primitive_grasp_generator()
    if not callable(generator):
        raise ScheduleStreamProviderError("v1 primitive_grasp_generator is not callable")
    multiply_poses = getattr(planner_module, "multiply_poses", None)
    if not callable(multiply_poses):
        raise ScheduleStreamProviderError("v1 planner module has no callable multiply_poses")
    try:
        raw_generator = iter(generator("cuboid", geometry["aabb_dimensions_m"], "top"))
        primitive_poses = []
        for _ in range(5):
            try:
                primitive_poses.append(next(raw_generator))
            except StopIteration:
                break
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to materialize reviewed cuboid/top primitive grasps") from exc
    if len(primitive_poses) != 4:
        raise ScheduleStreamProviderError(
            f"reviewed cuboid/top primitive grasp generator must yield exactly four poses, got {len(primitive_poses)}"
        )
    try:
        primitive_transforms = tuple(
            _native_pose_matrix(pose, f"primitive_grasp[{index}]") for index, pose in enumerate(primitive_poses)
        )
        if len(set(primitive_transforms)) != 4:
            raise ScheduleStreamProviderError("reviewed cuboid/top primitive grasps must contain four unique poses")
        bounding_box_from_object = converted_object.bounding_box.pose.inverse()
        link_from_object_poses = tuple(multiply_poses(pose, bounding_box_from_object) for pose in primitive_poses)
        link_from_object_transforms = tuple(
            _native_pose_matrix(pose, f"grasp_geometry.link_from_object[{index}]")
            for index, pose in enumerate(link_from_object_poses)
        )
    except ScheduleStreamProviderError:
        raise
    except Exception as exc:
        raise ScheduleStreamProviderError("failed to compose off-center cube grasp transforms") from exc
    return link_from_object_poses, {
        "asset_name": graspable_asset_name,
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "converted_object_origin_from_aabb_center": geometry["converted_object_origin_from_aabb_center"],
        "link_from_object_transforms": [[list(row) for row in transform] for transform in link_from_object_transforms],
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": len(link_from_object_poses),
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": graspable_object,
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "primitive_link_from_aabb_center_transforms": [
            [list(row) for row in transform] for transform in primitive_transforms
        ],
        "profile": V1_GRASP_GEOMETRY_PROFILE,
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }


def _native_pose_matrix(pose: Any, field_name: str) -> Matrix4:
    try:
        [native_matrix] = pose.get_numpy_matrix()
        return _matrix_from_native(native_matrix, field_name)
    except Exception as exc:
        raise ScheduleStreamProviderError(f"{field_name} must be one finite rigid pose") from exc


def _validate_goal_subjects_are_dynamic(scene: Any, predicates: tuple[GoalPredicate, ...]) -> None:
    entries = {name: is_kinematic for name, _, is_kinematic in _scene_rigid_object_entries(scene)}
    for predicate in predicates:
        subject = predicate.subject
        if subject not in entries:
            raise ScheduleStreamProviderError(
                f"goal subject {subject!r} does not bind to an IsaacLab scene.rigid_objects entry"
            )
        if entries[subject]:
            raise ScheduleStreamProviderError(
                f"goal subject {subject!r} is explicitly kinematic and cannot be planned as a movable object"
            )


def _scene_rigid_object_entries(scene: Any) -> tuple[tuple[str, Any, bool], ...]:
    rigid_objects = getattr(scene, "rigid_objects", None)
    if not isinstance(rigid_objects, Mapping):
        raise ScheduleStreamProviderError("IsaacLab scene.rigid_objects must be a mapping")
    if len(rigid_objects) > 10_000:
        raise ScheduleStreamProviderError("IsaacLab scene has more than 10000 rigid objects")
    entries = []
    for name, rigid_object in rigid_objects.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 512 or "\x00" in name:
            raise ScheduleStreamProviderError("IsaacLab rigid-object names must be bounded non-empty strings")
        cfg = getattr(rigid_object, "cfg", None)
        spawn = None if cfg is None else getattr(cfg, "spawn", None)
        if spawn is None:
            raise ScheduleStreamProviderError(f"IsaacLab rigid object {name!r} has no reviewed spawn config")
        rigid_props = getattr(spawn, "rigid_props", None)
        kinematic = None if rigid_props is None else getattr(rigid_props, "kinematic_enabled", None)
        if kinematic is not None and type(kinematic) is not bool:
            raise ScheduleStreamProviderError(
                f"IsaacLab rigid object {name!r} has a non-boolean kinematic_enabled value"
            )
        entries.append((name, rigid_object, kinematic is True))
    return tuple(entries)


def _load_v1_isaaclab_planner_module() -> Any:
    return importlib.import_module("schedulestream.applications.isaaclab.planner")


def _load_v1_surface_config_factory() -> Callable[..., Any]:
    try:
        module = importlib.import_module("schedulestream.applications.custream.object")
        factory = module.SurfaceConfig
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamImportError(
            f"failed to load custream SurfaceConfig ({type(exc).__name__}: {message})"
        ) from exc
    if not callable(factory):
        raise ScheduleStreamImportError("custream SurfaceConfig symbol is not callable")
    return factory


def _load_v1_primitive_grasp_generator() -> _PrimitiveGraspGenerator:
    try:
        module = importlib.import_module("schedulestream.applications.custream.grasp")
        generator = module.primitive_grasp_generator
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamImportError(
            f"failed to load custream primitive grasp generator ({type(exc).__name__}: {message})"
        ) from exc
    if not callable(generator):
        raise ScheduleStreamImportError("custream primitive_grasp_generator symbol is not callable")
    return generator


def _load_v1_seed_setter() -> _SeedSetter:
    try:
        module = importlib.import_module("schedulestream.applications.custream.utils")
        setter = module.set_seed
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamImportError(
            f"failed to load custream seed setter ({type(exc).__name__}: {message})"
        ) from exc
    if not callable(setter):
        raise ScheduleStreamImportError("custream set_seed symbol is not callable")
    return setter


def _validate_v1_module(module: Any) -> None:
    required = (
        "CAMERA_POSE",
        "Commands",
        "GraspConfig",
        "Planner",
        "World",
        "animate_commands",
        "create_objects",
        "create_controller",
        "load_franka_config",
        "multiply_poses",
        "solve_tamp",
        "timeout_context",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ScheduleStreamImportError(f"v1 IsaacLab planner module is missing symbols {missing}")


def _require_env_id(env_id: int) -> None:
    if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
        raise ValueError("env_id must be a non-negative integer")


def _action_body_offset(env: Any, link_name: str) -> Matrix4:
    action_manager = getattr(env, "action_manager", None)
    active_terms = () if action_manager is None else getattr(action_manager, "active_terms", ())
    matches = []
    for term_name in active_terms:
        try:
            term = action_manager.get_term(term_name)
            cfg = term.cfg
        except Exception as exc:
            raise ScheduleStreamProviderError(f"failed to inspect IsaacLab action term {term_name!r}") from exc
        if getattr(cfg, "body_name", None) == link_name:
            matches.append(cfg)
    if len(matches) != 1:
        raise ScheduleStreamProviderError(f"expected one IK action term for body {link_name!r}, found {len(matches)}")
    offset = getattr(matches[0], "body_offset", None)
    if offset is None:
        return _identity_matrix()
    position = getattr(offset, "pos", (0.0, 0.0, 0.0))
    quaternion = getattr(offset, "rot", (0.0, 0.0, 0.0, 1.0))
    try:
        position_values = tuple(position)
        q_x, q_y, q_z, q_w = tuple(quaternion)
    except TypeError as exc:
        raise ScheduleStreamProviderError("IK action body_offset pos/rot must be sequences") from exc
    except ValueError as exc:
        raise ScheduleStreamProviderError("IK action body_offset rot must contain four XYZW values") from exc
    vector = position_values + (q_w, q_x, q_y, q_z)
    return _pose_vector_to_matrix(vector, f"action_term[{link_name}].body_offset")


def _attest_action_to_eef_transform(
    native_planner: Any,
    *,
    action_link_name: str,
    observed_eef: Matrix4,
) -> tuple[Matrix4, dict[str, Any]]:
    """Require the live action-link-to-EEF transform to match the configured action offset."""

    configured_offset = _action_body_offset(native_planner.base_env, action_link_name)
    try:
        current_action_link = native_planner.current_link_matrix(action_link_name)
    except Exception as exc:
        if isinstance(exc, ScheduleStreamProviderError):
            raise
        raise ScheduleStreamProviderError(f"failed to read live action-link pose {action_link_name!r}") from exc
    _require_rigid_transform(current_action_link, f"current_action_link[{action_link_name}]")
    _require_rigid_transform(observed_eef, "observed_eef_pose")
    _require_rigid_transform(configured_offset, f"configured_action_offset[{action_link_name}]")
    observed_offset = _matrix_multiply(_rigid_inverse(current_action_link), observed_eef)
    _require_rigid_transform(observed_offset, f"action_to_observed_eef[{action_link_name}]")
    configured_position_error, configured_rotation_error = _transform_error(
        configured_offset,
        observed_offset,
    )
    if (
        configured_position_error > V1_FRAME_ATTESTATION_MAX_POSITION_ERROR_M
        or configured_rotation_error > V1_FRAME_ATTESTATION_MAX_ROTATION_ERROR_RAD
    ):
        observed_summary = tuple(tuple(round(value, 6) for value in row) for row in observed_offset)
        raise ScheduleStreamProviderError(
            "live action-to-EEF transform does not match the configured v1 action offset "
            f"(position={configured_position_error:.6g}m, rotation={configured_rotation_error:.6g}rad, "
            f"observed_action_to_eef={observed_summary})"
        )
    return configured_offset, {
        "configured_offset": configured_offset,
        "configured_position_error_m": configured_position_error,
        "configured_rotation_error_rad": configured_rotation_error,
        "current_action_link": current_action_link,
        "observed_offset": observed_offset,
    }


def _matrix_from_native(value: Any, field_name: str) -> Matrix4:
    materialized = _to_builtin(value, field_name)
    if not isinstance(materialized, (list, tuple)) or len(materialized) != 4:
        raise MalformedScheduleStreamCommandError(f"{field_name} must have shape [4, 4]")
    rows = []
    for row in materialized:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise MalformedScheduleStreamCommandError(f"{field_name} must have shape [4, 4]")
        rows.append(tuple(_finite_number(item, field_name) for item in row))
    try:
        return matrix4(tuple(rows), field_name)
    except ValueError as exc:
        raise MalformedScheduleStreamCommandError(str(exc)) from exc


def _rows_from_native(
    value: Any,
    field_name: str,
    *,
    maximum_rows: int,
    expected_columns: int,
) -> tuple[tuple[float, ...], ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        try:
            shape = tuple(int(item) for item in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MalformedScheduleStreamCommandError(f"{field_name} has an invalid shape") from exc
        if len(shape) != 2 or not 1 <= shape[0] <= maximum_rows or shape[1] != expected_columns:
            raise MalformedScheduleStreamCommandError(
                f"{field_name} shape {shape} violates [1..{maximum_rows}, {expected_columns}]"
            )
    materialized = _to_builtin(value, field_name)
    if not isinstance(materialized, (list, tuple)) or not 1 <= len(materialized) <= maximum_rows:
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain between 1 and {maximum_rows} rows")
    rows = []
    for row in materialized:
        if not isinstance(row, (list, tuple)) or len(row) != expected_columns:
            raise MalformedScheduleStreamCommandError(f"every {field_name} row must contain {expected_columns} values")
        rows.append(tuple(_finite_number(item, field_name) for item in row))
    return tuple(rows)


def _vector_from_native(
    value: Any,
    field_name: str,
    *,
    maximum_values: int,
) -> tuple[float, ...]:
    materialized = _to_builtin(value, field_name)
    if not isinstance(materialized, (list, tuple)) or not 1 <= len(materialized) <= maximum_values:
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain between 1 and {maximum_values} values")
    result = []
    for item in materialized:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise MalformedScheduleStreamCommandError(f"{field_name} must have shape [N] or [N, 1]")
            item = item[0]
        result.append(_finite_number(item, field_name))
    return tuple(result)


def _bounded_names(value: Any, field_name: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain between 1 and {maximum} names")
    names = tuple(value)
    if any(not isinstance(name, str) or not name.strip() or len(name) > 512 or "\x00" in name for name in names):
        raise MalformedScheduleStreamCommandError(f"{field_name} entries must be bounded strings")
    if len(names) != len(set(names)):
        raise MalformedScheduleStreamCommandError(f"{field_name} entries must be unique")
    return names


def _pose_vector_to_matrix(value: Any, field_name: str) -> Matrix4:
    materialized = _to_builtin(value, field_name)
    if not isinstance(materialized, (list, tuple)) or len(materialized) != 7:
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain [x, y, z, qw, qx, qy, qz]")
    x_pos, y_pos, z_pos, q_w, q_x, q_y, q_z = (_finite_number(item, field_name) for item in materialized)
    norm = math.sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
    if norm < 1e-12 or abs(norm - 1.0) > 1e-2:
        raise MalformedScheduleStreamCommandError(f"{field_name} quaternion must have unit norm")
    q_w, q_x, q_y, q_z = (item / norm for item in (q_w, q_x, q_y, q_z))
    result = (
        (
            1.0 - 2.0 * (q_y * q_y + q_z * q_z),
            2.0 * (q_x * q_y - q_z * q_w),
            2.0 * (q_x * q_z + q_y * q_w),
            x_pos,
        ),
        (
            2.0 * (q_x * q_y + q_z * q_w),
            1.0 - 2.0 * (q_x * q_x + q_z * q_z),
            2.0 * (q_y * q_z - q_x * q_w),
            y_pos,
        ),
        (
            2.0 * (q_x * q_z - q_y * q_w),
            2.0 * (q_y * q_z + q_x * q_w),
            1.0 - 2.0 * (q_x * q_x + q_y * q_y),
            z_pos,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )
    return matrix4(result, field_name)


def _matrix_multiply(first: Matrix4, second: Matrix4) -> Matrix4:
    result = tuple(
        tuple(sum(first[row][inner] * second[inner][column] for inner in range(4)) for column in range(4))
        for row in range(4)
    )
    return matrix4(result, "pose_product")


def _transform_error(first: Matrix4, second: Matrix4) -> tuple[float, float]:
    position_error = math.sqrt(sum((first[row][3] - second[row][3]) ** 2 for row in range(3)))
    relative_trace = sum(first[row][column] * second[row][column] for row in range(3) for column in range(3))
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    return position_error, math.acos(cosine)


def _require_rigid_transform(value: Matrix4, field_name: str, *, tolerance: float = 1e-3) -> None:
    for first_column in range(3):
        for second_column in range(3):
            dot = sum(value[row][first_column] * value[row][second_column] for row in range(3))
            expected = 1.0 if first_column == second_column else 0.0
            if abs(dot - expected) > tolerance:
                raise ScheduleStreamProviderError(f"{field_name} rotation is not orthonormal")
    determinant = (
        value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ScheduleStreamProviderError(f"{field_name} rotation determinant is not +1")


def _rigid_inverse(value: Matrix4) -> Matrix4:
    rotation_t = tuple(tuple(value[column][row] for column in range(3)) for row in range(3))
    translation = tuple(value[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_t[row][column] * translation[column] for column in range(3)) for row in range(3)
    )
    result = tuple(
        tuple(rotation_t[row][column] for column in range(3)) + (inverse_translation[row],) for row in range(3)
    ) + (
        (0.0, 0.0, 0.0, 1.0),
    )
    return matrix4(result, "pose_inverse")


def _identity_matrix() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _to_builtin(value: Any, field_name: str) -> Any:
    result = value
    for method_name in ("detach", "cpu", "numpy", "tolist"):
        method = getattr(result, method_name, None)
        if callable(method):
            try:
                result = method()
            except Exception as exc:
                raise MalformedScheduleStreamCommandError(
                    f"failed to materialize {field_name} via {method_name}"
                ) from exc
    return result


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise MalformedScheduleStreamCommandError(f"{field_name} must not contain booleans")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain numbers") from exc
    if not math.isfinite(result):
        raise MalformedScheduleStreamCommandError(f"{field_name} must contain finite values")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if result <= 0:
        raise ScheduleStreamTimingError(f"{field_name} must be positive")
    return result
