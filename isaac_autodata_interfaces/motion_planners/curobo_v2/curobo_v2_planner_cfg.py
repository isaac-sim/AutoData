# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the cuRobo v2 motion planner.

Named presets are built by the factory methods at the end of the class and selected either by
profile name or by task id, so an environment profile resolves the same name against any planner
backend.

This module imports neither cuRobo nor Isaac Lab, so it can be loaded without a simulator.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CuroboV2PlannerCfg:
    """Configuration for the cuRobo v2 motion planner.

    Args:
        robot_config_file: cuRobo robot config (e.g. ``"franka.yml"``), resolved through cuRobo's
            robot config search path. Absolute paths are also accepted.
        robot_name: Robot identifier used in logs and visualization.
        gripper_open_positions: Joint positions [m] the gripper holds while moving empty-handed.
        gripper_closed_positions: Joint positions [m] the gripper holds while carrying an object.
            Leave this or the open positions empty to keep the robot config's own width.
        hand_link_names: Gripper links, used to populate ``contact_disable_collision_links``.
        attached_object_link_name: Link a grasped object is attached to.
        extra_collision_spheres: Collision spheres to allocate per link, applied on top of the
            robot config's own values.
        scene_model: cuRobo scene config holding fixed collision geometry, or None for none.
            Obstacles are read from the USD stage regardless.
        static_objects: Substrings of scene object names to treat as fixed. Matching obstacles
            are placed once and never moved again; everything else follows its object.
        world_ignore_substrings: Substrings of USD prim paths to skip when reading obstacles.
        obstacle_representation: ``"mesh"`` keeps each obstacle's triangles, so concave shapes
            such as a bin are represented faithfully and the gripper can reach inside.
            ``"obb"`` reduces every obstacle to a bounding box, which is exact for boxes and
            cheaper, but unsuitable for concave geometry.
        num_ik_seeds: Number of inverse-kinematics seeds.
        num_trajopt_seeds: Number of trajectory-optimization seeds.
        position_tolerance: Goal position tolerance [m].
        orientation_tolerance: Goal orientation tolerance [rad].
        optimizer_collision_activation_distance: Distance at which collision costs engage [m].
        use_cuda_graph: Whether to use CUDA graphs for speed.
        random_seed: Seed for reproducible planning.
        collision_cache: Obstacle slots to pre-allocate per type. Required when no
            ``scene_model`` is given so obstacles can be loaded later.
        motion_noise_scale: Action noise applied per waypoint during execution. Read by the
            SkillGen algorithm, not by cuRobo.
        surface_sphere_radius: Amount each fitted sphere is inflated by [m].
        sphere_fit_type: How spheres are fitted to a grasped object. One of ``"SURFACE"``,
            ``"VOXEL"``, or ``"MORPHIT"``.
        attached_object_num_spheres: Spheres to fit to a grasped object. Must not exceed the
            attached link's allocation in ``extra_collision_spheres``, or attaching fails.
        approach_distance: Distance short of the goal the approach phase stops at [m].
        retreat_distance: Distance the gripper backs off before crossing free space [m].
        contact_disable_collision_links: Links whose collisions are ignored during the phases
            that end in contact, so the gripper may touch what it is reaching for.
        visualize_spheres: Not implemented; use ``visualize_plan``.
        visualize_plan: Draw the plan in Rerun: end-effector path, goal, obstacles, and
            collision spheres. Applies to env 0 only and requires the ``rerun-sdk`` package.
    """

    # Robot
    robot_config_file: str | None = None
    robot_name: str = ""

    # Gripper. The open and closed positions are loaded into the robot's locked joints as the
    # grasp state changes, so the gripper is collision-checked at the width it actually has.
    gripper_open_positions: dict[str, float] = field(default_factory=dict)
    gripper_closed_positions: dict[str, float] = field(default_factory=dict)
    hand_link_names: list[str] = field(default_factory=list)
    attached_object_link_name: str = "attached_object"
    # The stock Franka config allocates 4 spheres to the attached link, too coarse a model of a
    # grasped object to plan through the narrow clearances of a bin.
    extra_collision_spheres: dict[str, int] = field(default_factory=lambda: {"attached_object": 100})

    # Scene
    scene_model: str | None = None
    static_objects: list[str] = field(default_factory=list)
    world_ignore_substrings: list[str] = field(
        default_factory=lambda: ["/World/defaultGroundPlane", "/curobo", "/Robot"]
    )
    obstacle_representation: str = "mesh"

    # Planning
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.03
    use_cuda_graph: bool = True
    random_seed: int = 123
    collision_cache: dict[str, int] = field(default_factory=lambda: {"obb": 64, "mesh": 64})
    motion_noise_scale: float = 0.0

    # Attachment
    surface_sphere_radius: float = 0.005
    sphere_fit_type: str = "SURFACE"
    attached_object_num_spheres: int = 100

    # Approach and contact
    approach_distance: float = 0.05
    retreat_distance: float = 0.05
    contact_disable_collision_links: list[str] = field(default_factory=list)

    # Visualization
    visualize_spheres: bool = False
    visualize_plan: bool = False

    def __post_init__(self) -> None:
        """Validate cross-field constraints that would otherwise fail deep inside cuRobo."""
        assert self.obstacle_representation in (
            "mesh",
            "obb",
        ), f"obstacle_representation must be 'mesh' or 'obb', got {self.obstacle_representation!r}"
        # When the allocation comes from the robot config instead, its size is unknown here and
        # is checked by cuRobo at attach time.
        if self.attached_object_link_name in self.extra_collision_spheres:
            allocated = self.extra_collision_spheres[self.attached_object_link_name]
            assert self.attached_object_num_spheres <= allocated, (
                f"attached_object_num_spheres ({self.attached_object_num_spheres}) exceeds the "
                f"{self.attached_object_link_name!r} allocation in extra_collision_spheres ({allocated})"
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def franka_config(cls) -> CuroboV2PlannerCfg:
        """Create a configuration for the Franka Panda arm and its parallel gripper."""
        return cls(
            robot_config_file="franka.yml",
            robot_name="franka",
            gripper_open_positions={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            gripper_closed_positions={"panda_finger_joint1": 0.023, "panda_finger_joint2": 0.023},
            hand_link_names=["panda_leftfinger", "panda_rightfinger", "panda_hand"],
            motion_noise_scale=0.02,
        )

    @classmethod
    def franka_stack_cube_config(cls) -> CuroboV2PlannerCfg:
        """Create a configuration for the Franka cube-stacking task.

        The table is fixed while the cubes follow their scene objects, and the gripper links are
        allowed to touch during the contact phases so the fingers can close on a cube.
        """
        cfg = cls.franka_config()
        cfg.static_objects = ["table"]
        cfg.optimizer_collision_activation_distance = 0.01
        cfg.approach_distance = 0.05
        cfg.retreat_distance = 0.05
        cfg.surface_sphere_radius = 0.01
        cfg.collision_cache = {"obb": 150, "mesh": 150}
        cfg.contact_disable_collision_links = list(cfg.hand_link_names)
        return cfg

    @classmethod
    def franka_stack_cube_bin_config(cls) -> CuroboV2PlannerCfg:
        """Create a configuration for stacking cubes inside a sorting bin.

        The bin joins the table as fixed geometry. Clearances are tighter than on an open table,
        so the collision margin is wider, the retreat longer, and the modeled closed-finger width
        slightly larger for margin against the walls. Obstacles keep their own triangles, letting
        the planner avoid the bin walls while the gripper reaches inside.
        """
        cfg = cls.franka_stack_cube_config()
        cfg.static_objects = ["blue_sorting_bin", "bin", "table"]
        cfg.optimizer_collision_activation_distance = 0.02
        cfg.approach_distance = 0.05
        cfg.retreat_distance = 0.07
        cfg.surface_sphere_radius = 0.01
        cfg.gripper_closed_positions = {"panda_finger_joint1": 0.024, "panda_finger_joint2": 0.024}
        return cfg

    @classmethod
    def from_profile(cls, profile_name: str) -> CuroboV2PlannerCfg:
        """Create a configuration from a named planner profile.

        Profiles let an environment profile name the tuning that suits its scene, rather than
        inferring it from the task id.

        Args:
            profile_name: Key into :data:`PLANNER_PROFILES`.

        Returns:
            Configuration for the named profile.
        """
        assert (
            profile_name in PLANNER_PROFILES
        ), f"Unknown planner profile {profile_name!r}. Registered: {sorted(PLANNER_PROFILES)}"
        return PLANNER_PROFILES[profile_name]()

    @classmethod
    def from_task_name(cls, task_name: str) -> CuroboV2PlannerCfg:
        """Create a configuration by matching substrings of the task id.

        Args:
            task_name: Task id, e.g. ``"Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"``.

        Returns:
            Configuration for the task, falling back to the plain Franka preset.
        """
        lower = task_name.lower()
        if "stack-cube-bin" in lower:
            return cls.franka_stack_cube_bin_config()
        if "stack-cube" in lower:
            return cls.franka_stack_cube_config()
        logging.getLogger(__name__).warning(
            "No planner preset matches task %r; falling back to the Franka configuration.", task_name
        )
        return cls.franka_config()

    def to_v2_kwargs(self) -> dict[str, Any]:
        """Return the fields :meth:`MotionPlannerCfg.create` accepts, ready to splat into it.

        Fields read elsewhere, such as ``motion_noise_scale`` and the visualization flags, are
        left out.
        """
        return {
            "robot": self.robot_config_file,
            "scene_model": self.scene_model,
            "collision_cache": self.collision_cache,
            "num_ik_seeds": self.num_ik_seeds,
            "num_trajopt_seeds": self.num_trajopt_seeds,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "optimizer_collision_activation_distance": self.optimizer_collision_activation_distance,
            "use_cuda_graph": self.use_cuda_graph,
            "random_seed": self.random_seed,
        }


PLANNER_PROFILES: dict[str, Callable[[], CuroboV2PlannerCfg]] = {
    "franka": CuroboV2PlannerCfg.franka_config,
    "franka_stack_cube": CuroboV2PlannerCfg.franka_stack_cube_config,
    "franka_stack_cube_bin": CuroboV2PlannerCfg.franka_stack_cube_bin_config,
}
"""Maps a planner-profile name to the factory that builds its configuration.

Keys match those of the other planner backends, so an environment profile's ``planner`` name
resolves whichever backend is selected. Register new profiles here."""
