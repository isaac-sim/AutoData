# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision discovery and composed-trajectory checks for ScheduleGen."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

from autodata_core.schedulegen.scheduling import get_scheduled_path_indices
from autodata_core.schedulegen.types import (
    ArmPath,
    ArmWaypointConflict,
    EpisodeGenerationError,
    ObjectStateSample,
    ScheduledRobotContact,
    ScheduledWorldContact,
)
from autodata_core.scheduling import Schedule
from autodata_core.waypoint import Waypoint
from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory
from autodata_interfaces.motion_planners.curobo.collision_spheres import ObjectCollisionSpheres
from autodata_interfaces.tasks.schedule_constraint_spec import SegmentId
from autodata_utils import pose_math

if TYPE_CHECKING:
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner


def find_waypoint_conflicts(
    first_path: ArmPath,
    second_path: ArmPath,
    first_planner: CuroboPlanner,
    second_planner: CuroboPlanner,
    *,
    collision_margin_m: float = 0.0,
    densify_factor: int = 1,
    pair_batch_size: int = 256,
    sphere_batch_size: int = 64,
) -> tuple[ArmWaypointConflict, ...]:
    """Return sampled original-waypoint conflicts between two arm paths.

    Joint paths are optionally linearly densified. Each colliding evaluated interior sample is
    mapped to both original waypoints that bound its interpolation interval, so the scheduler
    conservatively blocks that sampled interval. This is a finite-sampling check, not a continuous
    swept-volume guarantee.

    Both planners must express their arm-local spheres in the same planning-root frame. Shared
    ancestry stays in each planner so cuRobo can avoid arm-body collisions, but is removed from the
    cross-arm distance query. Shared-joint proposals are averaged per candidate pair before FK so
    both sides describe one physically consistent robot state; the embodiment's shared-joint solver
    remains authoritative.

    Args:
        first_path: Complete first-arm path.
        second_path: Complete second-arm path.
        first_planner: Initialized ordinary cuRobo planner for ``first_path``.
        second_planner: Initialized ordinary cuRobo planner for ``second_path``.
        collision_margin_m: Additional required sphere-surface separation [m].
        densify_factor: Linear samples per original path interval.
        pair_batch_size: Maximum dense waypoint pairs evaluated together.
        sphere_batch_size: Maximum spheres per arm-side distance chunk.

    Returns:
        Deduplicated conflicts ordered by the two original waypoint indices.
    """

    assert isinstance(first_path, ArmPath) and isinstance(second_path, ArmPath)
    assert first_path.eef_name != second_path.eef_name, "collision paths must belong to different lanes"
    assert math.isfinite(collision_margin_m) and collision_margin_m >= 0.0
    assert type(densify_factor) is int and densify_factor >= 1
    assert type(pair_batch_size) is int and pair_batch_size >= 1
    assert type(sphere_batch_size) is int and sphere_batch_size >= 1
    _planning_roots_match(first_planner, second_planner)

    first_dense, first_support = _path_densify(first_path.joint_positions, densify_factor)
    second_dense, second_support = _path_densify(second_path.joint_positions, densify_factor)
    first_joint_names = tuple(getattr(first_planner, "active_joint_names", ()))
    second_joint_names = tuple(getattr(second_planner, "active_joint_names", ()))
    if first_joint_names or second_joint_names:
        assert len(first_joint_names) == first_dense.shape[1]
        assert len(second_joint_names) == second_dense.shape[1]
    second_joint_index = {name: index for index, name in enumerate(second_joint_names)}
    shared_joint_indices = tuple(
        (index, second_joint_index[name]) for index, name in enumerate(first_joint_names) if name in second_joint_index
    )
    second_count = second_dense.shape[0]
    pair_count = first_dense.shape[0] * second_count
    conflicts: dict[tuple[int, int], float] = {}

    for pair_start in range(0, pair_count, pair_batch_size):
        pair_end = min(pair_start + pair_batch_size, pair_count)
        flat_indices = torch.arange(pair_start, pair_end, dtype=torch.long)
        first_dense_indices = torch.div(flat_indices, second_count, rounding_mode="floor")
        second_dense_indices = torch.remainder(flat_indices, second_count)
        first_positions = first_dense.index_select(0, first_dense_indices.to(first_dense.device))
        second_positions = second_dense.index_select(0, second_dense_indices.to(second_dense.device))
        for first_index, second_index in shared_joint_indices:
            shared = 0.5 * (
                first_positions[:, first_index]
                + second_positions[:, second_index].to(first_positions.device, first_positions.dtype)
            )
            first_positions[:, first_index] = shared
            second_positions[:, second_index] = shared.to(second_positions.device, second_positions.dtype)
        first_spheres = first_planner.joint_positions_to_collision_spheres(first_positions)
        second_spheres = second_planner.joint_positions_to_collision_spheres(second_positions).to(
            device=first_spheres.device,
            dtype=first_spheres.dtype,
        )
        first_links = first_planner.collision_sphere_link_names(first_spheres.shape[1])
        second_links = second_planner.collision_sphere_link_names(second_spheres.shape[1])
        shared_links = set(first_links) & set(second_links)
        first_spheres = first_spheres[:, [name not in shared_links for name in first_links]]
        second_spheres = second_spheres[:, [name not in shared_links for name in second_links]]
        assert first_spheres.shape[1] and second_spheres.shape[1], "each lane must own collision spheres"
        minimum_clearance = _minimum_clearance(first_spheres, second_spheres, sphere_batch_size)
        colliding = torch.nonzero(minimum_clearance <= collision_margin_m, as_tuple=False).flatten().cpu()

        for local_index in colliding.tolist():
            first_dense_index = int(first_dense_indices[local_index])
            second_dense_index = int(second_dense_indices[local_index])
            clearance = float(minimum_clearance[local_index].item())
            for first_index in first_support[first_dense_index]:
                for second_index in second_support[second_dense_index]:
                    key = (first_index, second_index)
                    conflicts[key] = min(conflicts.get(key, math.inf), clearance)

    return tuple(
        ArmWaypointConflict(
            first_eef_name=first_path.eef_name,
            first_waypoint_index=first_index,
            second_eef_name=second_path.eef_name,
            second_waypoint_index=second_index,
            clearance_m=clearance,
        )
        for (first_index, second_index), clearance in sorted(conflicts.items())
    )


def _path_densify(path: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Interpolate joint samples and map each sample to its supporting original waypoints."""

    dense, ticks = _joint_timeline_densify(path, factor)
    supports = tuple((int(tick),) if tick.is_integer() else (math.floor(tick), math.ceil(tick)) for tick in ticks)
    return dense, supports


def _minimum_clearance(first: torch.Tensor, second: torch.Tensor, sphere_batch_size: int) -> torch.Tensor:
    """Compute the minimum active sphere-surface clearance for each paired sample.

    Args:
        first: First sphere batch with shape ``(T, S1, 4)`` [m], storing XYZ and radius.
        second: Second sphere batch with shape ``(T, S2, 4)`` [m], in the same frame.
        sphere_batch_size: Maximum spheres per chunk in each pairwise distance calculation.

    Returns:
        Signed minimum clearances with shape ``(T,)`` [m]; negative values indicate penetration.
        Spheres with nonpositive radii are ignored.

    Raises:
        AssertionError: Any paired sample has no finite active-sphere clearance.
    """

    batch_size = first.shape[0]
    minimum = torch.full((batch_size,), math.inf, device=first.device, dtype=first.dtype)
    for first_start in range(0, first.shape[1], sphere_batch_size):
        first_chunk = first[:, first_start : first_start + sphere_batch_size]
        for second_start in range(0, second.shape[1], sphere_batch_size):
            second_chunk = second[:, second_start : second_start + sphere_batch_size]
            distances = torch.linalg.vector_norm(
                first_chunk[:, :, None, :3] - second_chunk[:, None, :, :3],
                dim=-1,
            )
            clearance = distances - first_chunk[:, :, None, 3] - second_chunk[:, None, :, 3]
            valid = (first_chunk[:, :, None, 3] > 0.0) & (second_chunk[:, None, :, 3] > 0.0)
            clearance = torch.where(valid, clearance, math.inf)
            minimum = torch.minimum(minimum, clearance.flatten(1).amin(dim=1))
    assert bool(torch.isfinite(minimum).all()), "each arm path sample must contain an active collision sphere"
    return minimum


def _planning_roots_match(first_planner: Any, second_planner: Any) -> None:
    """Require matching planning-root names when both planners expose them."""

    first_root = getattr(first_planner, "planning_root_frame_name", None)
    second_root = getattr(second_planner, "planning_root_frame_name", None)
    if first_root is not None and second_root is not None:
        assert first_root == second_root, f"arm planners use different planning roots: {first_root!r}, {second_root!r}"


@dataclass(frozen=True)
class _ScheduledRobotState:
    """Combined joint states and tool poses used by scheduled collision checks."""

    joint_positions: torch.Tensor
    initial_joint_positions: torch.Tensor
    collision_joint_positions: torch.Tensor
    collision_sample_ticks: tuple[float, ...]
    tool_poses_by_eef: Mapping[str, torch.Tensor]
    self_collision_sample_count: int
    visualization_index: int | None


def arm_path_fk_validate(
    eef_name: str,
    waypoints: Sequence[Waypoint],
    joint_positions: torch.Tensor,
    planner: CuroboPlanner,
    segment_bounds: Mapping[SegmentId, tuple[int, int]] | None = None,
) -> None:
    """Check that path joint states reproduce their waypoint tool poses.

    Args:
        eef_name: End-effector name included in failure diagnostics.
        waypoints: Tool-pose targets in path order; translation is in [m].
        joint_positions: Planner-ordered joint states with shape (T, J) [rad, m].
        planner: Arm planner providing FK and pose tolerances.
        segment_bounds: Optional half-open path bounds used to label failures.

    Raises:
        EpisodeGenerationError: A joint state is invalid or FK exceeds a pose tolerance.
    """

    assert waypoints and len(waypoints) == len(joint_positions), "path poses and joint states must align"
    if not bool(torch.isfinite(joint_positions).all()):
        raise EpisodeGenerationError("forward_kinematics", f"{eef_name} path contains non-finite joint positions")
    try:
        actual_poses = planner.joint_positions_to_ee_poses(joint_positions)
    except Exception as error:
        raise EpisodeGenerationError("forward_kinematics", f"{eef_name} path FK failed: {error}") from error

    target_poses = torch.stack([waypoint.pose for waypoint in waypoints]).to(
        device=actual_poses.device,
        dtype=actual_poses.dtype,
    )
    assert actual_poses.shape == target_poses.shape, "path FK must return one 4x4 pose per joint state"
    finite = torch.isfinite(actual_poses).all(dim=(-2, -1)) & torch.isfinite(target_poses).all(dim=(-2, -1))
    position_error, rotation_error = pose_math.pose_errors(actual_poses, target_poses)
    rejected = (
        ~finite
        | (position_error > planner.config.position_threshold)
        | (rotation_error > planner.config.rotation_threshold)
    )
    if not bool(rejected.any()):
        return

    first = int(torch.nonzero(rejected, as_tuple=False)[0, 0].item())
    position_error = torch.nan_to_num(position_error, nan=math.inf, posinf=math.inf, neginf=math.inf)
    rotation_error = torch.nan_to_num(rotation_error, nan=math.inf, posinf=math.inf, neginf=math.inf)
    location = f"path waypoint {first}/{len(waypoints)}"
    for segment_id, (start, end) in (segment_bounds or {}).items():
        if start <= first < end:
            location = f"{segment_id} waypoint {first - start}/{end - start} (path index {first})"
            break

    raise EpisodeGenerationError(
        "forward_kinematics",
        f"{eef_name} path FK rejected {location}: "
        f"position_error={float(position_error[first]):.6f} m, "
        f"rotation_error={float(rotation_error[first]):.6f} rad",
    )


def scheduled_object_spheres(
    object_states: tuple[ObjectStateSample, ...],
    object_geometry: Mapping[str, ObjectCollisionSpheres],
    *,
    makespan_ticks: int,
) -> dict[str, torch.Tensor]:
    """Transform fitted object spheres along the predicted object trajectory.

    Args:
        object_states: Object poses at each execution boundary.
        object_geometry: Fitted object-frame collision spheres keyed by object name.
        makespan_ticks: Exclusive terminal execution tick.

    Returns:
        Object name to sphere tensor with shape (makespan_ticks + 1, S, 4).
        Each sphere stores its environment-frame center and radius [m].
    """

    sphere_timelines = {}
    for object_ref, geometry in sorted(object_geometry.items()):
        samples = tuple(state for state in object_states if state.object_ref == object_ref)
        assert tuple(sample.boundary_tick for sample in samples) == tuple(range(makespan_ticks + 1))
        poses = torch.tensor(tuple(sample.env_from_object for sample in samples), dtype=torch.float32)
        centers = poses.new_tensor(tuple((*sphere.center, 1.0) for sphere in geometry.spheres))
        positions = torch.matmul(poses, centers.transpose(0, 1)).transpose(1, 2)[..., :3]
        radii = poses.new_tensor(tuple(sphere.radius for sphere in geometry.spheres)).expand(len(samples), -1)
        sphere_timelines[object_ref] = torch.cat((positions, radii.unsqueeze(-1)), dim=-1)
    assert sphere_timelines, "object sphere projection requires at least one sphere geometry"
    return sphere_timelines


def scheduled_robot_object_validate(
    full_state_planner: CuroboPlanner,
    scheduled_robot: _ScheduledRobotState,
    object_spheres: Mapping[str, torch.Tensor],
    *,
    collision_margin_m: float,
    object_manipulation_starts: Mapping[str, int],
    robot_contacts: tuple[ScheduledRobotContact, ...],
    batch_size: int = 256,
) -> tuple[int, int]:
    """Check projected objects against the robot using grasp, carry and release permissions.

    Args:
        full_state_planner: Collision model for the combined robot.
        scheduled_robot: Scheduled joint states and tool poses.
        object_spheres: Predicted environment-frame sphere centers and radii [m].
        collision_margin_m: Additional required surface separation [m].
        object_manipulation_starts: Boundary where each projected object replaces its original world mesh.
            Robot/world checks cover the original mesh before this boundary.
        robot_contacts: Object/link pairs and intervals where contact is permitted.
        batch_size: Maximum number of states per collision query.

    Returns:
        Number of checked object-state samples and samples with permitted contacts.

    Raises:
        EpisodeGenerationError: Geometry is missing or an unpermitted collision is found.
    """

    assert type(batch_size) is int and batch_size > 0
    assert set(object_manipulation_starts) == set(object_spheres)
    robot_states = torch.cat(
        (scheduled_robot.initial_joint_positions.unsqueeze(0), scheduled_robot.joint_positions),
        dim=0,
    )
    sample_count = 0
    waived_contact_sample_count = 0
    contacts_by_object: dict[str, list[ScheduledRobotContact]] = {}
    for contact in robot_contacts:
        contacts_by_object.setdefault(contact.spec.object_ref, []).append(contact)

    missing = set(contacts_by_object) - set(object_spheres)
    if missing:
        raise EpisodeGenerationError(
            "full_state_robot_contact_resolution", f"missing object geometry: {sorted(missing)}"
        )
    for object_ref, spheres in sorted(object_spheres.items()):
        contacts = contacts_by_object.get(object_ref, [])
        assert len(spheres) == len(robot_states), "robot and object boundary timelines must align"
        if any(contact.stop_tick > len(spheres) for contact in contacts):
            raise EpisodeGenerationError(
                "full_state_robot_contact_resolution",
                f"robot-contact interval for {object_ref!r} exceeds its projected timeline",
            )
        start_tick = object_manipulation_starts[object_ref]
        assert type(start_tick) is int and 0 <= start_tick <= len(spheres)
        allowed_links_by_tick = tuple(
            tuple(
                sorted({
                    link_name
                    for contact in contacts
                    if contact.start_tick <= tick < contact.stop_tick
                    for link_name in contact.spec.robot_link_names
                })
            )
            for tick in range(start_tick, len(spheres))
        )
        start = start_tick
        while start < len(spheres):
            allowed_links = allowed_links_by_tick[start - start_tick]
            stop = start + 1
            while (
                stop < len(spheres)
                and stop - start < batch_size
                and allowed_links_by_tick[stop - start_tick] == allowed_links
            ):
                stop += 1
            try:
                strict_collision_free = full_state_planner.joint_positions_environment_spheres_collision_free(
                    robot_states[start:stop],
                    spheres[start:stop],
                    collision_margin_m,
                )
                if allowed_links:
                    collision_free = full_state_planner.joint_positions_environment_spheres_collision_free(
                        robot_states[start:stop],
                        spheres[start:stop],
                        collision_margin_m,
                        allowed_robot_link_names=allowed_links,
                    )
                    waived_contact_sample_count += int(
                        torch.count_nonzero(~strict_collision_free & collision_free).item()
                    )
                else:
                    collision_free = strict_collision_free
            except Exception as error:
                raise EpisodeGenerationError(
                    "full_state_robot_object_collision",
                    f"cuRobo robot/object query failed for {object_ref!r} at boundary tick {start}: "
                    f"{type(error).__name__}: {error}",
                ) from error
            sample_count += stop - start
            rejected = torch.nonzero(~collision_free, as_tuple=False)
            if len(rejected):
                tick = start + int(rejected[0, 0].item())
                witness_message = ""
                try:
                    witness = full_state_planner.joint_positions_environment_spheres_collision_witness(
                        robot_states[tick : tick + 1],
                        spheres[tick : tick + 1],
                        collision_margin_m,
                        allowed_robot_link_names=allowed_links,
                    )
                    if witness is not None:
                        link_name, robot_sphere_index, object_sphere_index = witness
                        witness_message = (
                            f"; {link_name}[{robot_sphere_index}] versus {object_ref}[{object_sphere_index}]"
                        )
                except Exception as error:
                    witness_message = f"; witness unavailable: {type(error).__name__}: {error}"
                raise EpisodeGenerationError(
                    "full_state_robot_object_collision",
                    f"full robot intersects projected object {object_ref!r} outside reviewed contact at "
                    f"boundary tick {tick}/{len(spheres) - 1}{witness_message}",
                )
            start = stop
    return sample_count, waived_contact_sample_count


def scheduled_world_validate(
    full_state_planner: CuroboPlanner,
    scheduled_robot: _ScheduledRobotState,
    object_spheres: Mapping[str, torch.Tensor],
    object_geometry: Mapping[str, ObjectCollisionSpheres],
    *,
    collision_margin_m: float,
    object_manipulation_starts: Mapping[str, int],
    world_contacts: tuple[ScheduledWorldContact, ...] = (),
    batch_size: int = 256,
) -> tuple[int, int, int]:
    """Check the robot and predicted object poses against the planning world.

    Args:
        full_state_planner: Collision model for the combined robot.
        scheduled_robot: Scheduled joint states and tool poses.
        object_spheres: Predicted environment-frame sphere centers and radii [m].
        object_geometry: Fitted object geometry and corresponding world obstacle names.
        collision_margin_m: Additional required surface separation [m].
        object_manipulation_starts: First manipulation-segment tick per projected object.
            Its original mesh remains an obstacle before that tick.
        world_contacts: Object/fixture pairs and intervals where contact is permitted.
        batch_size: Maximum number of states per collision query.

    Returns:
        Robot/world, object/world and permitted-contact sample counts.

    Raises:
        EpisodeGenerationError: World synchronization fails or an unpermitted collision is found.
    """

    assert type(batch_size) is int and batch_size > 0
    assert set(object_spheres) == set(object_geometry), "projected object geometry must be complete"
    assert set(object_manipulation_starts) == set(object_geometry)
    assert all(
        type(tick) is int and 0 <= tick <= len(scheduled_robot.joint_positions) + 1
        for tick in object_manipulation_starts.values()
    )
    excluded_paths = tuple(sorted(geometry.obstacle_path for geometry in object_geometry.values()))
    assert len(excluded_paths) == len(set(excluded_paths)), "projected objects must resolve to unique obstacle paths"
    try:
        full_state_planner.update_world()
    except Exception as error:
        raise EpisodeGenerationError(
            "full_state_world_sync",
            f"cuRobo world synchronization failed: {type(error).__name__}: {error}",
        ) from error

    robot_states = scheduled_robot.collision_joint_positions
    robot_sample_ticks = scheduled_robot.collision_sample_ticks

    def exclusions_at(tick: float) -> tuple[str, ...]:
        """Exclude each original mesh only after manipulation begins at this boundary."""
        return tuple(
            sorted(
                geometry.obstacle_path
                for name, geometry in object_geometry.items()
                if object_manipulation_starts[name] <= tick
            )
        )

    exclusions_by_tick = tuple(exclusions_at(tick) for tick in robot_sample_ticks)
    exclusions_by_boundary = tuple(exclusions_at(tick) for tick in range(len(scheduled_robot.joint_positions) + 1))
    start = 0
    while start < len(robot_states):
        stop = start + 1
        while (
            stop < len(robot_states)
            and stop - start < batch_size
            and exclusions_by_tick[stop] == exclusions_by_tick[start]
        ):
            stop += 1
        try:
            collision_free = full_state_planner.joint_positions_world_collision_free(
                robot_states[start:stop],
                collision_margin_m,
                excluded_obstacle_paths=exclusions_by_tick[start],
            )
        except Exception as error:
            raise EpisodeGenerationError(
                "full_state_robot_world_collision",
                f"cuRobo robot/world query failed at schedule tick {start}: {type(error).__name__}: {error}",
            ) from error
        rejected = torch.nonzero(~collision_free, as_tuple=False)
        if len(rejected):
            sample = start + int(rejected[0, 0].item())
            tick = robot_sample_ticks[sample]
            witness_message = ""
            try:
                witness = full_state_planner.joint_positions_world_collision_witness(
                    robot_states[sample : sample + 1],
                    collision_margin_m,
                    excluded_obstacle_paths=exclusions_by_tick[start],
                )
                if witness is not None:
                    link_name, sphere_index, obstacle_path = witness
                    witness_message = f"; {link_name}[{sphere_index}] versus {obstacle_path}"
            except Exception as error:
                witness_message = f"; witness unavailable: {type(error).__name__}: {error}"
            raise EpisodeGenerationError(
                "full_state_robot_world_collision",
                "full robot intersects the configured world at schedule tick "
                f"{tick:g}/{len(scheduled_robot.joint_positions)}{witness_message}",
            )
        start = stop

    resolved_contacts: list[tuple[ScheduledWorldContact, tuple[str, ...]]] = []
    for contact in world_contacts:
        object_ref = contact.spec.object_ref
        if object_ref not in object_spheres:
            raise EpisodeGenerationError(
                "full_state_world_contact_resolution",
                f"world-contact object {object_ref!r} has no projected sphere geometry",
            )
        if contact.stop_tick > len(object_spheres[object_ref]):
            raise EpisodeGenerationError(
                "full_state_world_contact_resolution",
                f"world-contact interval for {object_ref!r} exceeds its projected timeline",
            )
        try:
            fixture_paths = full_state_planner.world_object_obstacle_paths(contact.spec.fixture_ref)
        except Exception as error:
            raise EpisodeGenerationError(
                "full_state_world_contact_resolution",
                f"fixture {contact.spec.fixture_ref!r} could not be resolved: {type(error).__name__}: {error}",
            ) from error
        overlap = set(fixture_paths) & set(excluded_paths)
        if overlap:
            raise EpisodeGenerationError(
                "full_state_world_contact_resolution",
                f"fixture {contact.spec.fixture_ref!r} overlaps projected objects: {sorted(overlap)}",
            )
        resolved_contacts.append((contact, fixture_paths))

    object_sample_count = 0
    waived_contact_sample_count = 0
    for object_ref, spheres in sorted(object_spheres.items()):
        object_sample_count += len(spheres)
        allowed_paths_by_tick = tuple(
            tuple(
                sorted({
                    path
                    for contact, fixture_paths in resolved_contacts
                    if contact.spec.object_ref == object_ref and contact.start_tick <= tick < contact.stop_tick
                    for path in fixture_paths
                })
            )
            for tick in range(len(spheres))
        )
        start = 0
        while start < len(spheres):
            allowed_paths = allowed_paths_by_tick[start]
            stop = start + 1
            while (
                stop < len(spheres)
                and stop - start < batch_size
                and allowed_paths_by_tick[stop] == allowed_paths
                and exclusions_by_boundary[stop] == exclusions_by_boundary[start]
            ):
                stop += 1
            strict_exclusions = tuple(
                sorted({object_geometry[object_ref].obstacle_path, *exclusions_by_boundary[start]})
            )
            query_exclusions = tuple(sorted(set(strict_exclusions) | set(allowed_paths)))
            try:
                strict_collision_free = full_state_planner.environment_spheres_world_collision_free(
                    spheres[start:stop],
                    collision_margin_m,
                    excluded_obstacle_paths=strict_exclusions,
                )
                if allowed_paths:
                    collision_free = full_state_planner.environment_spheres_world_collision_free(
                        spheres[start:stop],
                        collision_margin_m,
                        excluded_obstacle_paths=query_exclusions,
                    )
                    waived_contact_sample_count += int(
                        torch.count_nonzero(~strict_collision_free & collision_free).item()
                    )
                else:
                    collision_free = strict_collision_free
            except Exception as error:
                raise EpisodeGenerationError(
                    "full_state_object_world_collision",
                    f"cuRobo object/world query failed for {object_ref!r} at boundary tick {start}: "
                    f"{type(error).__name__}: {error}",
                ) from error
            rejected = torch.nonzero(~collision_free, as_tuple=False)
            if len(rejected):
                tick = start + int(rejected[0, 0].item())
                witness_message = ""
                try:
                    witness = full_state_planner.environment_spheres_world_collision_witness(
                        spheres[tick : tick + 1],
                        collision_margin_m,
                        excluded_obstacle_paths=query_exclusions,
                    )
                    if witness is not None:
                        sphere_index, obstacle_path = witness
                        witness_message = f"; {object_ref}[{sphere_index}] versus {obstacle_path}"
                except Exception as error:
                    witness_message = f"; witness unavailable: {type(error).__name__}: {error}"
                raise EpisodeGenerationError(
                    "full_state_object_world_collision",
                    f"projected object {object_ref!r} intersects the configured world at "
                    f"boundary tick {tick}/{len(spheres) - 1}{witness_message}",
                )
            start = stop
    _object_pairs_validate(object_spheres, object_manipulation_starts, collision_margin_m, batch_size)
    return len(robot_states), object_sample_count, waived_contact_sample_count


def _object_pairs_validate(
    object_spheres: Mapping[str, torch.Tensor],
    object_manipulation_starts: Mapping[str, int],
    collision_margin_m: float,
    batch_size: int,
) -> None:
    """Reject collisions between projected object timelines.

    Args:
        object_spheres: Environment-frame spheres with shape ``(T, N, 4)`` per object [m].
        object_manipulation_starts: Boundary where each projected object replaces its original world mesh.
        collision_margin_m: Additional required surface separation [m].
        batch_size: Maximum number of states per collision query.
    """

    for first_name, second_name in combinations(sorted(object_spheres), 2):
        first, second = object_spheres[first_name], object_spheres[second_name]
        assert len(first) == len(second), "projected object timelines must align"
        # Until both original meshes are excluded, the native object/world queries above
        # cover this pair without substituting another surface-sphere approximation.
        pair_start = max(object_manipulation_starts[first_name], object_manipulation_starts[second_name])
        for start in range(pair_start, len(first), batch_size):
            clearances = _minimum_clearance(first[start : start + batch_size], second[start : start + batch_size], 64)
            rejected = torch.nonzero(clearances <= collision_margin_m, as_tuple=False)
            if len(rejected):
                tick = start + int(rejected[0, 0])
                raise EpisodeGenerationError(
                    "full_state_object_object_collision",
                    f"projected objects {first_name!r} and {second_name!r} collide at boundary tick {tick}",
                )


def scheduled_robot_validate(
    datastream: Any,
    planners: Mapping[str, CuroboPlanner],
    full_state_planner: CuroboPlanner,
    arm_paths: Mapping[str, ArmPath],
    schedule: Schedule,
    initial_joint_positions: torch.Tensor,
    *,
    joint_trajectory: NamedJointTrajectory | None = None,
    collision_margin_m: float,
    densify_factor: int,
    batch_size: int = 256,
) -> _ScheduledRobotState:
    """Combine arm paths and check tool FK and robot self-collision.

    Args:
        datastream: Interface providing the robot joint names and controller frames.
        planners: Arm planners keyed by end-effector name.
        full_state_planner: Collision model containing the combined robot joints.
        arm_paths: Complete unscheduled arm paths.
        schedule: Execution ticks for every segment waypoint.
        initial_joint_positions: Initial articulation state in live joint order [rad, m].
        joint_trajectory: Optional coupled-controller trajectory in named joint order.
        collision_margin_m: Additional required surface separation [m].
        densify_factor: Linear samples per waypoint interval for collision checks.
        batch_size: Maximum number of states per collision query.

    Returns:
        Combined joint states, tool poses and collision-check details.

    Raises:
        EpisodeGenerationError: Joint states, tool poses or sampled self-collisions are invalid.
    """

    live_names = tuple(datastream.get_robot_joint_names())
    assert initial_joint_positions.shape == (len(live_names),)
    full_names = full_state_planner.active_joint_names
    live_index = {name: index for index, name in enumerate(live_names)}
    missing = set(full_names) - set(live_names)
    assert not missing, f"full-state joints are absent from the live articulation: {sorted(missing)}"
    for joint_name, compiled_position in full_state_planner.locked_joint_positions.items():
        if joint_name not in live_index:
            raise EpisodeGenerationError(
                "full_state_locked_joint_state",
                f"locked joint {joint_name!r} is absent from the live articulation",
            )
        live_position = float(initial_joint_positions[live_index[joint_name]].item())
        if not math.isfinite(live_position) or not math.isclose(
            live_position,
            compiled_position,
            abs_tol=1.0e-6,
            rel_tol=0.0,
        ):
            raise EpisodeGenerationError(
                "full_state_locked_joint_state",
                f"locked joint {joint_name!r} changed from compiled {compiled_position:.8f} "
                f"to live {live_position:.8f}",
            )
    initial_full_state = initial_joint_positions[[live_index[name] for name in full_names]]
    states = initial_full_state.repeat(schedule.makespan_ticks, 1)
    full_index = {name: index for index, name in enumerate(full_names)}
    written: dict[str, str] = {}
    timeline_indices: dict[str, tuple[int, ...]] = {}

    if joint_trajectory is not None:
        assert joint_trajectory.positions.shape[0] == schedule.makespan_ticks
        missing = set(joint_trajectory.joint_names) - set(full_names)
        assert not missing, f"coupled trajectory contains joints outside the full-state context: {sorted(missing)}"
        trajectory = joint_trajectory.positions.to(device=states.device, dtype=states.dtype)
        for trajectory_index, joint_name in enumerate(joint_trajectory.joint_names):
            states[:, full_index[joint_name]] = trajectory[:, trajectory_index]
            written[joint_name] = "whole_body"

    for eef_name, path in arm_paths.items():
        planner = planners[eef_name]
        indices = get_scheduled_path_indices(path, schedule)
        timeline_indices[eef_name] = indices
        source_indices = torch.tensor(indices, device=path.joint_positions.device)
        lane_states = path.joint_positions.index_select(0, source_indices).to(device=states.device, dtype=states.dtype)
        assert lane_states.shape == (schedule.makespan_ticks, len(planner.active_joint_names))
        for lane_index, joint_name in enumerate(planner.active_joint_names):
            assert joint_name in full_index, f"group joint {joint_name!r} is absent from the full-state context"
            if written.get(joint_name) == "whole_body":
                continue
            column = full_index[joint_name]
            if joint_name in written and not torch.allclose(
                states[:, column], lane_states[:, lane_index], atol=1.0e-6, rtol=0.0
            ):
                raise EpisodeGenerationError(
                    "full_state_composition",
                    f"groups {written[joint_name]!r} and {eef_name!r} disagree on shared joint {joint_name!r}",
                )
            states[:, column] = lane_states[:, lane_index]
            written[joint_name] = eef_name
    unwritten = set(full_names) - set(written)
    assert not unwritten, f"full-state active joints have no trajectory owner: {sorted(unwritten)}"

    tool_links = tuple(planners[name].config.ee_link_name for name in arm_paths)
    assert all(tool_links), "every planning group must declare its physical tool link"
    boundary_states = torch.cat((initial_full_state.unsqueeze(0), states), dim=0)
    try:
        full_state_planner.planning_root_pose_refresh()
        full_tool_poses = full_state_planner.joint_positions_to_link_poses(
            boundary_states,
            tool_links,
        )
    except Exception as error:
        raise EpisodeGenerationError(
            "full_state_forward_kinematics",
            f"{type(error).__name__}: {error}",
        ) from error
    for eef_name, tool_link in zip(arm_paths, tool_links, strict=True):
        path = arm_paths[eef_name]
        target_poses = torch.stack([path.waypoints[index].pose for index in timeline_indices[eef_name]])
        actual_poses = full_tool_poses[tool_link][1:]
        target_poses = target_poses.to(device=actual_poses.device, dtype=actual_poses.dtype)
        position_error, rotation_error = pose_math.pose_errors(actual_poses, target_poses)
        rejected = (
            ~torch.isfinite(actual_poses).all(dim=(-2, -1))
            | ~torch.isfinite(target_poses).all(dim=(-2, -1))
            | (position_error > planners[eef_name].config.position_threshold)
            | (rotation_error > planners[eef_name].config.rotation_threshold)
        )
        if bool(rejected.any()):
            tick = int(torch.nonzero(rejected, as_tuple=False)[0, 0].item())
            raise EpisodeGenerationError(
                "full_state_forward_kinematics",
                f"{eef_name} full-state FK rejected schedule tick {tick}: "
                f"position_error={float(position_error[tick]):.6f} m, "
                f"rotation_error={float(rotation_error[tick]):.6f} rad",
            )

    visualization_index = None
    if full_state_planner.plan_visualizer is not None:
        full_state_planner.world_visualization_refresh()
        spheres = full_state_planner.joint_positions_to_collision_spheres(boundary_states)
        visualization_index = full_state_planner.plan_visualizer.animate_collision_spheres(spheres)

    dense_states, sample_ticks = _joint_timeline_densify(boundary_states, densify_factor)
    for start in range(0, len(dense_states), batch_size):
        try:
            collision_free = full_state_planner.joint_positions_self_collision_free(
                dense_states[start : start + batch_size],
                collision_margin_m,
            )
        except Exception as error:
            raise EpisodeGenerationError(
                "full_state_self_collision",
                f"cuRobo query failed at validation sample {start}: {type(error).__name__}: {error}",
            ) from error
        rejected = torch.nonzero(~collision_free, as_tuple=False)
        if len(rejected):
            sample = start + int(rejected[0, 0].item())
            witness_message = ""
            try:
                witness = full_state_planner.joint_positions_self_collision_witness(
                    dense_states[sample : sample + 1],
                    collision_margin_m,
                )
                if witness is not None:
                    first_link, first_sphere, second_link, second_sphere, clearance = witness
                    witness_message = (
                        f"; {first_link}[{first_sphere}] versus {second_link}[{second_sphere}], "
                        f"clearance={clearance:.6f} m"
                    )
            except Exception as error:
                witness_message = f"; witness unavailable: {type(error).__name__}: {error}"
            raise EpisodeGenerationError(
                "full_state_self_collision",
                f"full robot self collision at validation sample {sample}/{len(dense_states)} "
                f"near schedule tick {sample_ticks[sample]:.3f}{witness_message}",
            )
    return _ScheduledRobotState(
        joint_positions=states,
        initial_joint_positions=initial_full_state,
        collision_joint_positions=dense_states,
        collision_sample_ticks=sample_ticks,
        tool_poses_by_eef={
            eef_name: full_tool_poses[tool_link].detach().clone()
            for eef_name, tool_link in zip(arm_paths, tool_links, strict=True)
        },
        self_collision_sample_count=len(dense_states),
        visualization_index=visualization_index,
    )


def _joint_timeline_densify(states: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Subdivide joint-space intervals while retaining all original samples.

    Args:
        states: Joint positions with shape ``(T, J)`` in consistent joint order [rad, m].
        factor: Equal subdivisions per adjacent pair; one leaves the timeline unchanged.

    Returns:
        Interpolated joint positions and their fractional ticks on the original timeline.
        The final original sample is included exactly once.
    """

    assert states.ndim == 2 and len(states) > 0
    assert type(factor) is int and factor >= 1
    if factor == 1 or len(states) == 1:
        return states, tuple(float(index) for index in range(len(states)))
    samples: list[torch.Tensor] = []
    ticks: list[float] = []
    for index in range(len(states) - 1):
        for subdivision in range(factor):
            alpha = subdivision / factor
            samples.append(torch.lerp(states[index], states[index + 1], alpha))
            ticks.append(index + alpha)
    samples.append(states[-1])
    ticks.append(float(len(states) - 1))
    return torch.stack(samples), tuple(ticks)


__all__ = ["find_waypoint_conflicts"]
