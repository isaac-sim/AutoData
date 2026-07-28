# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reviewed constants for the live Franka cube-into-bowl task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlaceSuccessThresholds:
    """Physical thresholds for the reviewed cube-into-bowl success check.

    These limits are intentionally asset-specific. They are not general container semantics:
    Arena's validated success term remains the semantic authority while the executor requires
    independent physical evidence for grasp, transport, release, and stable placement.
    """

    attachment_distance_m: float = 0.05
    grasp_aperture_min_m: float = 0.040
    grasp_aperture_max_m: float = 0.075
    minimum_closed_samples: int = 10
    minimum_lift_m: float = 0.030
    minimum_transport_m: float = 0.050
    maximum_relative_translation_drift_m: float = 0.020
    maximum_relative_rotation_drift_rad: float = 0.35
    minimum_success_streak: int = 5
    maximum_final_linear_speed_m_s: float = 0.05
    maximum_final_angular_speed_rad_s: float = 0.5
    maximum_final_horizontal_radius_m: float = 0.028
    maximum_final_vertical_offset_m: float = 0.040
    maximum_destination_drift_m: float = 0.020
    minimum_final_eef_separation_m: float = 0.060
    minimum_final_aperture_m: float = 0.075


@dataclass(frozen=True)
class AutonomousTaskProfile:
    """Immutable facts shared by admission, planning, execution, and verification."""

    name: str
    motion_backend: str
    schedulestream_application: str
    schedulestream_plan_backend: str
    embodiment_name: str
    task_kind: str
    success_relation: str
    background_asset: str
    graspable_asset: str
    destination_asset: str
    required_task_params: frozenset[str]
    maximum_batch_size: int
    maximum_successful_episodes: int
    maximum_attempts_per_success: int
    maximum_planner_time_s: float
    interpolation_dt_s: float
    command_to_observation_offset_m: tuple[float, float, float]
    runtime_asset_profile: str
    registry_usd_basename: str
    runtime_usd_basename: str
    runtime_usd_path: str
    runtime_usd_bytes: int
    runtime_usd_sha256: str
    runtime_usd_max_bytes: int
    franka_usd_basenames: frozenset[str]
    ik_joint_limit_margin_rad: float
    frame_max_position_error_m: float
    frame_max_rotation_error_rad: float
    grasp_max_position_roundoff_m: float
    grasp_max_rotation_roundoff_rad: float
    grasp_geometry_profile: str
    destination_placement_profile: str
    desired_aabb_center_vertical_offset_m: float
    vertical_evidence_corridor_m: float
    graspable_aabb_dimension_bounds_m: tuple[tuple[float, float], ...]
    destination_aabb_dimension_bounds_m: tuple[tuple[float, float], ...]
    finger_joint_names: tuple[str, str]
    success_profile: str
    success_thresholds: PickPlaceSuccessThresholds
    arena_success_force_threshold_n: float
    arena_success_velocity_threshold_m_s: float


FRANKA_PICK_CUBE_INTO_BOWL = AutonomousTaskProfile(
    name="franka_pick_and_place_custream_v1",
    motion_backend="curobo_v1",
    schedulestream_application="custream",
    schedulestream_plan_backend="schedulestream_custream",
    embodiment_name="franka_ik",
    task_kind="PickAndPlaceTask",
    success_relation="on",
    background_asset="maple_table_robolab",
    graspable_asset="rubiks_cube_hot3d_robolab",
    destination_asset="bowl_ycb_robolab",
    required_task_params=frozenset({"background_scene", "destination_location", "pick_up_object"}),
    maximum_batch_size=1_024,
    maximum_successful_episodes=10,
    maximum_attempts_per_success=5,
    maximum_planner_time_s=60.0,
    interpolation_dt_s=0.02,
    command_to_observation_offset_m=(0.0, 0.0, -0.0036),
    runtime_asset_profile="franka_ik_custream_v1_official_root_usd",
    registry_usd_basename="franka_panda_hand_on_stand.usd",
    runtime_usd_basename="panda_instanceable.usd",
    runtime_usd_path=(
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
        "Robots/FrankaEmika/panda_instanceable.usd"
    ),
    runtime_usd_bytes=8_038,
    runtime_usd_sha256="7f5a0c0aa6760cfbd348e08bc464d4b94341f027f51c2d9e42406ceefcc7787f",
    runtime_usd_max_bytes=1 << 20,
    franka_usd_basenames=frozenset({"franka_panda_hand_on_stand.usd", "panda_instanceable.usd"}),
    ik_joint_limit_margin_rad=1e-3,
    frame_max_position_error_m=0.005,
    frame_max_rotation_error_rad=0.01,
    grasp_max_position_roundoff_m=1e-7,
    grasp_max_rotation_roundoff_rad=1e-6,
    grasp_geometry_profile="franka_rubiks_cube_offcenter_cuboid_top_v1",
    destination_placement_profile="franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1",
    desired_aabb_center_vertical_offset_m=0.03,
    vertical_evidence_corridor_m=0.04,
    graspable_aabb_dimension_bounds_m=((0.05, 0.065),) * 3,
    destination_aabb_dimension_bounds_m=((0.14, 0.17), (0.14, 0.17), (0.04, 0.07)),
    finger_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
    success_profile="franka_rubiks_cube_into_ycb_bowl_v1",
    success_thresholds=PickPlaceSuccessThresholds(),
    arena_success_force_threshold_n=0.1,
    arena_success_velocity_threshold_m_s=0.1,
)

__all__ = [
    "FRANKA_PICK_CUBE_INTO_BOWL",
    "AutonomousTaskProfile",
    "PickPlaceSuccessThresholds",
]
