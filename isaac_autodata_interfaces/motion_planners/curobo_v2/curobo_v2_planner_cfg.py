# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the cuRobo v2 motion planner backend.

Mirrors the public shape of the v1 :class:`CuroboPlannerCfg` where the underlying API permits
(robot identification, gripper/hand metadata, scene composition, motion-noise scale). v2-specific
knobs are exposed by their v2 names (e.g. ``num_ik_seeds``, ``num_trajopt_seeds``,
``position_tolerance``); v1-only knobs that have no v2 counterpart are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CuroboV2PlannerCfg:
    """Configuration for the cuRobo v2 motion planner.

    Args:
        robot_config_file: Path to a cuRobo robot YAML (e.g. ``"franka.yml"``). Resolved through
            cuRobo's robot config search path; absolute paths are also accepted.
        robot_name: Human-readable robot identifier used in logs and visualization.
        ee_link_name: Tool-link name FK targets and goal poses are expressed against. ``None``
            picks the robot YAML's default tool frame.
        gripper_joint_names: Joint names that control gripper opening/closing.
        gripper_open_positions: Map ``joint_name -> position`` for the open pose.
        gripper_closed_positions: Map ``joint_name -> position`` for the closed pose.
        hand_link_names: Robot links that are part of the gripper. Used by grasp detection.
        attached_object_link_name: Virtual link used to attach grasped objects.
        scene_model: Optional path to a cuRobo scene YAML defining static collision geometry,
            or ``None`` for an empty scene.
        static_objects: Names of scene objects that are not updated each tick.
        world_ignore_substrings: Substrings of USD prim paths to ignore when extracting world
            obstacles from the live simulator.
        num_ik_seeds: Number of IK optimization seeds (v2: ``MotionPlannerCfg.num_ik_seeds``).
        num_trajopt_seeds: Number of trajectory-optimization seeds.
        position_tolerance: Goal-pose position tolerance in meters [m].
        orientation_tolerance: Goal-pose orientation tolerance in radians [rad].
        optimizer_collision_activation_distance: Distance at which optimizer collision costs
            engage [m].
        use_cuda_graph: Whether the planner uses CUDA graphs for speed.
        random_seed: RNG seed for reproducibility.
        motion_noise_scale: Per-waypoint action-noise scalar applied at execution time.
            Consumed by the SkillGen algorithm; not used by cuRobo internally.
        motion_step_size: Joint-space retiming step size in radians [rad]. ``None`` disables
            retiming. Surfaced as :attr:`CuroboV2Planner.step_size` for SkillGen.
        visualize_spheres: Reserved for future plan-visualization parity with v1; currently unused.
        visualize_plan: Reserved for future plan-visualization parity with v1; currently unused.
    """

    # Robot
    robot_config_file: str | None = None
    robot_name: str = ""
    ee_link_name: str | None = None

    # Gripper / hand
    gripper_joint_names: list[str] = field(default_factory=list)
    gripper_open_positions: dict[str, float] = field(default_factory=dict)
    gripper_closed_positions: dict[str, float] = field(default_factory=dict)
    hand_link_names: list[str] = field(default_factory=list)
    attached_object_link_name: str = "attached_object"

    # Scene. Collision geometry is extracted from the live USD stage at construction (see
    # :meth:`CuroboV2Planner._initialize_static_world`); there is no hardcoded geometry.
    scene_model: str | None = None
    # Substrings of live scene object names whose obstacles are treated as fixed furniture and
    # never pose-synced (e.g. a welded table). Objects not listed here but present in the live
    # scene are synced every plan.
    static_objects: list[str] = field(default_factory=list)
    # Substrings of USD prim paths to drop during obstacle extraction (robot, ground plane, and
    # cuRobo's own debug prims). The env's robot prim is excluded automatically via the
    # extraction's reference frame, but listing it here is harmless and explicit.
    world_ignore_substrings: list[str] = field(
        default_factory=lambda: ["/World/defaultGroundPlane", "/curobo", "/Robot"]
    )

    # Planner params
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.03
    use_cuda_graph: bool = True
    random_seed: int = 123

    # Collision-cache pre-allocation. Must be non-``None`` when no static ``scene_model`` is
    # provided so the underlying SceneCollision checker is constructed up-front and accepts
    # subsequent :meth:`MotionPlanner.update_world` calls.
    collision_cache: dict[str, int] = field(default_factory=lambda: {"obb": 64, "mesh": 64})

    # Execution-side knobs consumed by the SkillGen algorithm
    motion_noise_scale: float = 0.0
    motion_step_size: float | None = None

    # Attachment sphere fitting. ``surface_sphere_radius`` is the inflation applied to each
    # fitted sphere (meters); ``sphere_fit_type`` is the algorithm name. Valid values are the
    # member names of :class:`curobo._src.geom.sphere_fit.types.SphereFitType`: ``"SURFACE"``,
    # ``"VOXEL"``, ``"MORPHIT"``. ``attached_object_num_spheres`` caps the number of spheres fit
    # to the grasped object; it must not exceed the ``attached_object`` link's sphere allocation
    # in the robot YAML (``extra_collision_spheres``, 4 for the stock Franka), or the attach fails.
    surface_sphere_radius: float = 0.005
    sphere_fit_type: str = "SURFACE"
    attached_object_num_spheres: int = 4

    # Approach / retreat / contact configuration. The planner uses :meth:`MotionPlanner.plan_grasp`
    # to produce three phases: approach (from current EEF to a pose offset from the goal),
    # grasp/place (linear motion from approach pose to goal), and lift (linear motion from goal
    # to a lift pose). Distances are in meters [m] along ``approach_axis`` in tool frame.
    # ``contact_disable_collision_links`` lists robot links whose collision is disabled during
    # the grasp and lift phases so the gripper can close on the object without aborting.
    approach_distance: float = 0.05
    retreat_distance: float = 0.05
    approach_axis: str = "z"
    approach_in_tool_frame: bool = True
    contact_disable_collision_links: list[str] = field(default_factory=list)

    # Reserved (v1-style visualization)
    visualize_spheres: bool = False
    visualize_plan: bool = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def franka_config(cls) -> CuroboV2PlannerCfg:
        """Default cuRobo v2 config for the Franka Panda arm + parallel gripper."""
        return cls(
            robot_config_file="franka.yml",
            robot_name="franka",
            gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            gripper_open_positions={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            gripper_closed_positions={"panda_finger_joint1": 0.023, "panda_finger_joint2": 0.023},
            hand_link_names=["panda_leftfinger", "panda_rightfinger", "panda_hand"],
            motion_noise_scale=0.02,
        )

    @classmethod
    def franka_stack_cube_config(cls) -> CuroboV2PlannerCfg:
        """cuRobo v2 config tuned for the Franka cube-stack task.

        Collision geometry (table + cubes) is extracted from the live USD stage at construction;
        the cubes are then pose-synced from the live scene every plan, while the table — declared
        static — is left at its extracted pose. The gripper links are collision-disabled during
        the contact phases so the fingers can close on a cube.
        """
        cfg = cls.franka_config()
        cfg.static_objects = ["table"]
        # Mirror the v1 franka_stack_cube_config tuning.
        cfg.optimizer_collision_activation_distance = 0.01
        cfg.approach_distance = 0.05
        cfg.retreat_distance = 0.05
        cfg.surface_sphere_radius = 0.01
        cfg.collision_cache = {"obb": 150, "mesh": 150}
        cfg.contact_disable_collision_links = list(cfg.hand_link_names)
        return cfg

    @classmethod
    def franka_bin_stack_cube_config(cls) -> CuroboV2PlannerCfg:
        """cuRobo v2 config for the Franka *bin* cube-stack task (obstacle avoidance).

        Mirrors the v1 ``franka_stack_cube_bin_config`` tuning: the bin is declared static (loaded
        once, not pose-synced), the collision-activation margin is widened and the retreat lengthened
        to keep clear of the bin walls, and the gripper closes slightly more on contact. All
        obstacles (bin + cubes) are modeled as their real meshes, so the planner avoids the concave
        bin walls while the gripper reaches *inside* to stack.
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
    def from_task_name(cls, task_name: str) -> CuroboV2PlannerCfg:
        """Create configuration by matching substrings of the gym task name.

        Mirrors the v1 :meth:`CuroboPlannerCfg.from_task_name` dispatch so callers can swap
        backends without changing their entry-point logic.

        Args:
            task_name: Gym task ID (e.g. ``"Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"``).

        Returns:
            A configuration appropriate for the requested task.
        """
        lower = task_name.lower()
        if "bin" in lower:
            return cls.franka_bin_stack_cube_config()
        if "stack-cube" in lower:
            return cls.franka_stack_cube_config()
        return cls.franka_config()

    def to_v2_kwargs(self) -> dict[str, Any]:
        """Project the fields cuRobo v2's :meth:`MotionPlannerCfg.create` understands.

        Returns a kwargs dict ready to splat into ``MotionPlannerCfg.create(**kwargs)``. Fields
        that are consumed at execution time (e.g. ``motion_noise_scale``) or that exist only for
        v1 parity (e.g. ``visualize_*``) are deliberately excluded.
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
