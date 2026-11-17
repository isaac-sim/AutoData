# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import numpy as np
import torch
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import warp as wp
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.cuda_robot_model.types import JointLimits
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenStatus
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import MeshSphereCfg, spawn_mesh_sphere

from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory
from autodata_interfaces.motion_planners.curobo.collision_spheres import CollisionSphere, ObjectCollisionSpheres
from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase
from autodata_utils import pose_math as PoseUtils

if TYPE_CHECKING:
    from autodata_interfaces.datastream.datastream import Datastream
    from autodata_interfaces.motion_planners.curobo.plan_visualizer import PlanVisualizer


def _usd_mesh_missing_scales_fill(stage: Any) -> None:
    """Supply identity scale where cuRobo's USD reader requires an authored mesh value."""

    from pxr import Gf, Sdf, UsdGeom

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        scale = prim.GetAttribute("xformOp:scale")
        if scale and scale.Get() is not None:
            continue
        if not scale:
            scale = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
        scale.Set(Gf.Vec3f(1.0))


def _self_collision_fallback_buffers_reset_install(motion_gen: MotionGen) -> None:
    """Clear pinned cuRobo fallback outputs before every self-collision evaluation."""

    visited: set[int] = set()
    for rollout in motion_gen.get_all_rollout_instances():
        for attribute in ("robot_self_collision_cost", "robot_self_collision_constraint"):
            cost = getattr(rollout, attribute, None)
            if cost is None or id(cost) in visited:
                continue
            visited.add(id(cost))
            config = getattr(cost, "self_collision_kin_config", None)
            if config is None or config.experimental_kernel:
                continue
            if getattr(cost, "_autodata_buffer_reset_installed", False):
                continue
            original_forward = cost.forward

            def forward(robot_spheres: torch.Tensor, *, _cost=cost, _original=original_forward):
                """Clear reused self-collision buffers before evaluating the sphere batch."""
                _cost.update_batch_size(robot_spheres)
                batch, horizon, sphere_count, width = robot_spheres.shape
                expected_shapes = {
                    "_out_distance": (batch, horizon),
                    "_out_vec": (batch, horizon, sphere_count, width),
                    "_sparse_sphere_idx": (batch, horizon, sphere_count),
                }
                for name, shape in expected_shapes.items():
                    buffer = getattr(_cost, name, None)
                    assert (
                        isinstance(buffer, torch.Tensor) and buffer.shape == shape
                    ), f"cuRobo self-collision buffer {name!r} changed contract"
                    buffer.zero_()
                return _original(robot_spheres)

            cost.forward = forward
            cost._autodata_buffer_reset_installed = True


# cuRobo ebb71702 still uses this namespace in WorldMeshCollision. Warp 1.16
# exposes the same bridge at the top level and no longer provides warp.torch.
if not hasattr(wp, "torch"):
    wp.torch = SimpleNamespace(device_from_torch=wp.device_from_torch)


class PlannerLogger:
    """Logger class for motion planner debugging and monitoring.

    This class provides standard logging functionality while maintaining isolation from
    the main application's logging configuration. The logger supports configurable verbosity
    levels and formats messages consistently for debugging motion planning operations,
    collision checking, and object manipulation.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize the logger with specified name and level.

        Args:
            name: Logger name for identification in log messages
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self._name = name
        self._level = level
        self._logger = None

    @property
    def logger(self):
        """Get the underlying logger instance, initializing it if needed.

        Returns:
            Configured Python logger instance with stream handler and formatter
        """
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self._level)
        return self._logger

    def debug(self, msg, *args, **kwargs):
        """Log debug-level message for detailed internal state information.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log info-level message for important operational events.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log warning-level message for potentially problematic conditions.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log error-level message for serious problems and failures.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.error(msg, *args, **kwargs)


@dataclass
class Attachment:
    """Stores object attachment information for robot manipulation.

    This dataclass tracks the relative pose between an attached object and its parent link,
    enabling the robot to maintain consistent object positioning during motion planning.
    """

    pose: Pose  # Relative pose from parent link to object
    parent: str  # Parent link name


@dataclass(frozen=True)
class _CollisionOnlyBackend:
    """cuRobo kinematics paired with an already initialized collision world."""

    kinematics: CudaRobotModel
    world_coll_checker: Any


class CuroboPlanner(MotionPlannerBase):
    """Motion planner for robot manipulation using cuRobo.

    This planner provides collision-aware motion planning capabilities for robotic manipulation tasks.
    It integrates with Isaac Lab environments to:

    - Update collision world from current stage state
    - Plan collision-free paths to target poses
    - Handle object attachment and detachment during manipulation
    - Execute planned motions with proper collision checking

    The planner uses cuRobo for fast motion generation and supports
    multi-phase planning for contact scenarios like grasping and placing objects.
    """

    plan_visualizer: PlanVisualizer | None

    def __init__(
        self,
        datastream: Datastream,
        config: CuroboPlannerCfg,
        task_name: str | None = None,
        env_id: int = 0,
        collision_checker: CollisionCheckerType = CollisionCheckerType.MESH,
        num_trajopt_seeds: int = 12,
        num_graph_seeds: int = 12,
        interpolation_dt: float = 0.05,
        collision_world_source: CuroboPlanner | None = None,
    ) -> None:
        """Initialize the motion planner for a specific environment.

        Sets up the cuRobo motion generator with collision checking, configures the robot model,
        and prepares visualization components if enabled. The planner is isolated to CUDA device
        regardless of Isaac Lab's device configuration.

        Args:
            datastream: Read interface over the env, task, and embodiment. World state (collision
                geometry source, object poses, joint configuration) is read through it.
            config: Configuration object containing planner parameters and settings
            task_name: Legacy argument retained for compatibility; configure the planner through ``config``.
            env_id: Environment ID for multi-environment setups (0 to num_envs-1)
            collision_checker: Legacy argument; use ``config.collision_checker_type``.
            num_trajopt_seeds: Legacy argument; use ``config.num_trajopt_seeds``.
            num_graph_seeds: Legacy argument; use ``config.num_graph_seeds``.
            interpolation_dt: Legacy argument; use ``config.interpolation_dt`` [s].
            collision_world_source: Initialized planner whose synchronized world is reused by a
                collision-only context.

        Raises:
            ValueError: If ``robot_config_file`` is not provided
        """
        # Initialize base class. The base derives the raw env/robot escape-hatch handles from
        # the datastream; the v1 backend still uses those for USD extraction and joint mutation.
        super().__init__(datastream=datastream, env_id=env_id, debug=config.debug_planner)

        # Initialize planner logger with debug level based on config
        log_level = logging.DEBUG if config.debug_planner else logging.INFO
        self.logger = PlannerLogger(f"CuroboPlanner_{env_id}", log_level)

        # Store instance variables
        self.config: CuroboPlannerCfg = config
        self.n_repeat: int | None = self.config.n_repeat
        self.step_size: float | None = self.config.motion_step_size
        self.visualize_plan: bool = self.config.visualize_plan
        self.visualize_spheres: bool = self.config.visualize_spheres
        self.plan_visualizer: PlanVisualizer | None = None

        # Log the config parameter values
        self.logger.info(f"Config parameter values: {self.config}")

        # Store attached objects as Attachment objects
        self.attached_objects: dict[str, Attachment] = {}  # object_name -> Attachment
        self._object_collision_spheres: dict[str, ObjectCollisionSpheres] = {}

        # Initialize cuRobo components - FORCE CUDA DEVICE FOR ISOLATION
        setup_curobo_logger("warn")

        # Force cuRobo to always use CUDA device regardless of Isaac Lab device
        # This isolates the motion planner from Isaac Lab's device configuration
        self.tensor_args: TensorDeviceType
        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")
        else:
            # Fallback to CPU if CUDA not available, but this may cause issues
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA not available, cuRobo using CPU - this may cause device compatibility issues")

        # Load robot configuration
        if self.config.robot_config_file is None:
            raise ValueError("robot_config_file is required")
        robot_cfg_file = self.config.robot_config_file
        robot_cfg: dict[str, Any] = load_yaml(robot_cfg_file)["robot_cfg"]
        self.logger.info(f"Loaded robot configuration from {robot_cfg_file}")

        # Configure collision spheres
        if self.config.collision_spheres_file:
            robot_cfg["kinematics"]["collision_spheres"] = self.config.collision_spheres_file

        # Configure extra collision spheres
        if self.config.extra_collision_spheres:
            robot_cfg["kinematics"]["extra_collision_spheres"] = self.config.extra_collision_spheres

        configured_self_collision_ignore = {
            str(name): tuple(str(ignored_name) for ignored_name in ignored_names)
            for name, ignored_names in robot_cfg["kinematics"]["self_collision_ignore"].items()
        }
        self._self_collision_ignore = MappingProxyType(configured_self_collision_ignore)
        if not self.config.use_native_self_collision:
            collision_links = tuple(str(name) for name in robot_cfg["kinematics"]["collision_link_names"])
            robot_cfg["kinematics"]["self_collision_ignore"] = {name: list(collision_links) for name in collision_links}

        self.robot_cfg: dict[str, Any] = robot_cfg
        self._locked_joint_positions = MappingProxyType(
            {str(name): float(value) for name, value in robot_cfg["kinematics"].get("lock_joints", {}).items()}
        )
        self.planning_root_frame_name = str(robot_cfg["kinematics"]["base_link"])
        self.root_pose_source_frame_name = config.root_pose_source_frame_name

        # Load world configuration using the config's method
        world_cfg: WorldConfig = self.config.get_world_config()
        self._configured_world_config = world_cfg.get_collision_check_world()

        # Synchronize before rollout construction: exported zero velocities are not valid bounds.
        robot_configuration = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self._joint_limits_sync(robot_configuration.kinematics.get_joint_limits())

        if collision_world_source is None:
            assert self.config.planning_enabled, "collision-only contexts require an initialized world source"
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_configuration,
                world_cfg,
                tensor_args=self.tensor_args,
                collision_checker_type=self.config.collision_checker_type,
                num_trajopt_seeds=self.config.num_trajopt_seeds,
                num_graph_seeds=self.config.num_graph_seeds,
                interpolation_dt=self.config.interpolation_dt,
                collision_cache=self.config.collision_cache_size,
                trajopt_tsteps=self.config.trajopt_tsteps,
                collision_activation_distance=self.config.collision_activation_distance,
                position_threshold=self.config.position_threshold,
                rotation_threshold=self.config.rotation_threshold,
                store_debug_in_result=self.config.debug_planner,
                self_collision_check=self.config.use_native_self_collision,
                self_collision_opt=self.config.use_native_self_collision,
            )
            self.motion_gen: MotionGen | _CollisionOnlyBackend = MotionGen(motion_gen_config)
            _self_collision_fallback_buffers_reset_install(self.motion_gen)
        else:
            assert not self.config.planning_enabled, "planning backends cannot reuse another planner's world"
            assert isinstance(collision_world_source.motion_gen, MotionGen)
            assert collision_world_source.env_id == self.env_id
            assert collision_world_source.datastream is self.datastream
            robot_model = CudaRobotModel(robot_configuration.kinematics)
            self.motion_gen = _CollisionOnlyBackend(
                kinematics=robot_model,
                world_coll_checker=collision_world_source.motion_gen.world_coll_checker,
            )
        self._collision_world_owner: CuroboPlanner = collision_world_source or self
        self._collision_world_active_context: CuroboPlanner | None = None
        self._kinematics = self.motion_gen.kinematics
        self._kinematic_robot_cfg = robot_cfg
        if config.kinematic_robot_config_file is not None:
            self._kinematic_robot_cfg = load_yaml(config.kinematic_robot_config_file)["robot_cfg"]
            for field in ("base_link", "ee_link", "collision_link_names"):
                assert self._kinematic_robot_cfg["kinematics"][field] == robot_cfg["kinematics"][field], field
            query_config = RobotConfig.from_dict(self._kinematic_robot_cfg, self.tensor_args)
            self._joint_limits_sync(query_config.kinematics.get_joint_limits())
            self._kinematics = CudaRobotModel(query_config.kinematics)
            self._self_collision_ignore = MappingProxyType({
                str(name): tuple(str(ignored) for ignored in ignored_names)
                for name, ignored_names in self._kinematic_robot_cfg["kinematics"]["self_collision_ignore"].items()
            })
        self._active_joint_names = tuple(str(name) for name in self._kinematics.joint_names)
        planning_names = set(self.motion_gen.kinematics.joint_names)
        assert planning_names <= set(self._active_joint_names), "planning joints must belong to the state-query model"
        self._auxiliary_joint_indices = {
            name: index for index, name in enumerate(self._active_joint_names) if name not in planning_names
        }
        assert set(self._auxiliary_joint_indices) <= set(self._locked_joint_positions)
        self._locked_joint_positions = MappingProxyType({
            name: value for name, value in self._locked_joint_positions.items() if name not in self._active_joint_names
        })
        self._self_collision_pair_cache: tuple[torch.Tensor, torch.Tensor, tuple[str, ...]] | None = None

        # Create plan config with parameters from configuration
        self.plan_config: MotionGenPlanConfig = MotionGenPlanConfig(
            enable_graph=self.config.enable_graph,
            enable_opt=self.config.enable_opt,
            enable_graph_attempt=self.config.enable_graph_attempt,
            max_attempts=self.config.max_planning_attempts,
            enable_finetune_trajopt=self.config.enable_finetune_trajopt,
            time_dilation_factor=self.config.time_dilation_factor,
        )

        # Create USD helper. The collision-geometry source (USD stage) is read through the
        # Datastream so the planner never reaches into ``env.scene`` directly.
        self.usd_helper: UsdHelper = UsdHelper()
        self.usd_helper.load_stage(self.datastream.get_usd_stage())

        # Initialize planning state
        self._current_plan: JointState | None = None
        self._failed_joint_state: JointState | None = None
        self._plan_index: int = 0

        # Initialize visualization state
        self.frame_counter: int = 0
        self.spheres: list[tuple[str, float]] | None = None
        self.sphere_update_freq: int = self.config.sphere_update_freq

        # cuRobo plans in its kinematic root frame, while the rest of AutoData exchanges
        # env-relative poses. Establish that transform before loading either static or dynamic
        # collision geometry so every pose follows the same frame path.
        identity_pose = torch.eye(4, dtype=torch.float32, device=self.datastream.device)
        self._base_pose_in_env: torch.Tensor = identity_pose
        self._env_to_base_transform: torch.Tensor = identity_pose.clone()
        self._base_frame_is_identity: bool = True
        self._refresh_base_frame_transform()

        if self.visualize_plan:
            from autodata_interfaces.motion_planners.curobo.plan_visualizer import PlanVisualizer as _PlanVisualizer

            self.plan_visualizer = _PlanVisualizer(
                robot_name=self.config.robot_name,
                recording_id=f"curobo_plan_{env_id}",
                debug=config.debug_planner,
                base_pose_in_env=self._base_pose_in_env,
            )
            self.plan_visualizer.set_motion_generator_reference(self.motion_gen)

        if self.config.planning_enabled:
            assert isinstance(self.motion_gen, MotionGen)
            self.logger.info("Warming up motion planner...")
            self.motion_gen.warmup(enable_graph=self.config.enable_graph, warmup_js_trajopt=False)

        self._initialize_static_world(activate=collision_world_source is None)

        # Defer object validation baseline until first update_world() call when scene is fully loaded
        self._expected_objects: set[str] | None = None

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

        # Cache object mappings
        # Only recompute when objects are added/removed, not when poses change
        self._cached_object_mappings: dict[str, str] | None = None

    # =====================================================================================
    # DEVICE CONVERSION UTILITIES
    # =====================================================================================

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management.

        Ensures all tensors used by cuRobo are on CUDA device, providing device isolation
        from Isaac Lab's potentially different device configuration. This prevents device
        mismatch errors and optimizes cuRobo performance.

        Args:
            tensor: Input tensor (may be on any device)

        Returns:
            Contiguous tensor on cuRobo's CUDA device with the appropriate dtype.
        """
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype).contiguous()

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor back to environment device for Isaac Lab compatibility.

        Converts cuRobo tensors back to the environment's device to ensure compatibility
        with Isaac Lab operations that expect tensors on the environment's configured device.

        Args:
            tensor: Input tensor from cuRobo operations (typically on CUDA)

        Returns:
            Tensor converted to environment's device while preserving dtype
        """
        return tensor.to(device=self.datastream.device, dtype=tensor.dtype)

    # =====================================================================================
    # FRAME RECONCILIATION
    # =====================================================================================

    def planning_root_pose_refresh(self) -> None:
        """Refresh the live planning-root transform used by FK, planning, and visualization."""

        self._refresh_base_frame_transform()

    def _refresh_base_frame_transform(self) -> None:
        """Refresh the cached transform between the env-relative and robot base frames.

        cuRobo plans in the robot base frame (the kinematic root at its origin), while the
        planner exchanges poses with callers in the env-relative frame. This reads the robot
        base pose (env-relative) and caches the forward and inverse transforms used to map
        object, goal, and planned poses between the two frames.
        """
        if self.root_pose_source_frame_name is None:
            self._base_pose_in_env = self.datastream.get_robot_root_pose(env_ids=[self.env_id])[0]
        else:
            self._base_pose_in_env = self.datastream.get_robot_planning_root_pose(
                self.root_pose_source_frame_name,
                env_ids=[self.env_id],
            )[0]
        self._env_to_base_transform = torch.linalg.inv(self._base_pose_in_env)
        identity = torch.eye(4, dtype=self._base_pose_in_env.dtype, device=self._base_pose_in_env.device)
        self._base_frame_is_identity = bool(torch.allclose(self._base_pose_in_env, identity, atol=1e-6))
        if self.plan_visualizer is not None:
            self.plan_visualizer.set_base_pose_in_env(self._base_pose_in_env)

    def _pose_env_to_base(self, pose_env: torch.Tensor) -> torch.Tensor:
        """Express an env-relative pose in the robot base frame.

        The planner exchanges poses with callers in the env-relative frame, while cuRobo plans
        in the robot base frame. This applies the cached env-to-base transform and is a no-op
        when the robot base coincides with the env origin.

        Args:
            pose_env: Pose as a 4x4 transformation matrix in the env-relative frame.

        Returns:
            The same pose as a 4x4 transformation matrix in the robot base frame.
        """
        if self._base_frame_is_identity:
            return pose_env
        transform = self._env_to_base_transform.to(device=pose_env.device, dtype=pose_env.dtype)
        return transform @ pose_env

    def _pose_base_to_env(self, pose_base: torch.Tensor) -> torch.Tensor:
        """Express a robot-base-frame pose in the env-relative frame.

        Inverse of :meth:`_pose_env_to_base`, used to map planned poses back into the
        env-relative frame callers expect. This is a no-op when the robot base coincides with
        the env origin.

        Args:
            pose_base: Pose as a 4x4 transformation matrix in the robot base frame.

        Returns:
            The same pose as a 4x4 transformation matrix in the env-relative frame.
        """
        if self._base_frame_is_identity:
            return pose_base
        transform = self._base_pose_in_env.to(device=pose_base.device, dtype=pose_base.dtype)
        return transform @ pose_base

    # =====================================================================================
    # INITIALIZATION AND CONFIGURATION
    # =====================================================================================

    def _joint_limits_sync(self, curobo_limits: JointLimits) -> None:
        """Synchronize positions and fill missing speed bounds from the live articulation."""
        curobo_joint_names = tuple(curobo_limits.joint_names)
        live_joint_names = tuple(self.datastream.get_robot_joint_names())
        missing = set(curobo_joint_names) - set(live_joint_names)
        assert not missing, f"cuRobo joints are absent from the live articulation: {sorted(missing)}"
        live_index = {name: index for index, name in enumerate(live_joint_names)}
        selected = [live_index[name] for name in curobo_joint_names]
        live_limits = self.datastream.get_robot_joint_position_limits(env_ids=[self.env_id])[0, selected]
        synchronized = self._to_curobo_device(live_limits).T.contiguous()
        assert synchronized.shape == curobo_limits.position.shape
        mismatch_count = int(torch.sum(torch.abs(curobo_limits.position - synchronized) > 0.01).item())
        curobo_limits.position.copy_(synchronized)
        live_speed = self._to_curobo_device(
            self.datastream.get_robot_joint_velocity_limits(env_ids=[self.env_id])[0, selected]
        )
        assert bool((live_speed > 0.0).all()), "Live joint speed limits must be positive."
        exported_speed = curobo_limits.velocity.abs().amax(dim=0)
        speed = torch.where(exported_speed > 0.0, torch.minimum(exported_speed, live_speed), live_speed)
        if self.config.max_joint_velocity is not None:
            cap = self.config.max_joint_velocity
            assert math.isfinite(cap) and cap > 0.0, "max_joint_velocity must be finite and positive."
            speed = speed.clamp(max=cap)
        assert bool(torch.isfinite(speed).all()), "Unbounded joint speeds require an explicit max_joint_velocity."
        curobo_limits.velocity.copy_(torch.stack((-speed, speed)))
        self.logger.debug(
            "Synchronized %d cuRobo joint limits from the live articulation (%d bound differences)",
            len(curobo_joint_names),
            mismatch_count,
        )

    def _initialize_static_world(self, *, activate: bool = True) -> None:
        """Initialize static world geometry from USD stage.

        Reads static environment geometry once during planner initialization to establish
        the base collision world. This includes walls, tables, bins, and other fixed obstacles
        that don't change during the simulation. Dynamic objects are synchronized separately
        in update_world() to maintain performance.
        """
        _usd_mesh_missing_scales_fill(self.usd_helper.stage)
        # Prim scoping for obstacle extraction comes from the Datastream (collision-world source),
        # not from direct env access.
        env_prim_path = self.datastream.get_env_prim_path(self.env_id)
        robot_prim_path = self.config.robot_prim_path or self.datastream.get_robot_prim_path(self.env_id)

        ignore_list = list(
            self.config.world_ignore_substrings
            or [
                f"{env_prim_path}/target",
                "/World/defaultGroundPlane",
                "/curobo",
            ]
        )
        if robot_prim_path not in ignore_list:
            ignore_list.append(robot_prim_path)

        # USD articulation-link transforms need not equal the live controller root. Extract in
        # the env frame, then use the same live root transform as goals and dynamic objects.
        self._static_world_config = self.usd_helper.get_obstacles_from_stage(
            only_paths=[env_prim_path],
            reference_prim_path=env_prim_path,
            ignore_substring=ignore_list,
        )
        self._transform_world_config_to_base_frame(self._static_world_config)
        self._static_world_config = self._static_world_config.get_collision_check_world()
        if self.config.include_configured_world:
            combined_world = self._configured_world_config.clone()
            for obstacle in self._static_world_config:
                combined_world.add_obstacle(obstacle)
            self._static_world_config = combined_world

        if activate:
            self._static_world_activate()

    def _static_world_activate(self) -> None:
        """Install this context's world in its shared fixed-capacity collision checker."""

        owner = self._collision_world_owner
        if owner._collision_world_active_context is self:
            return
        # Rebuild the cache so pinned cuRobo cannot retain an unlisted bootstrap cuboid.
        owner.motion_gen.clear_world_cache()
        owner.motion_gen.update_world(self._static_world_config)
        owner._collision_world_active_context = self

    def _transform_world_config_to_base_frame(self, world_config: WorldConfig) -> None:
        """Express env-relative cuRobo obstacles in the live planning-root frame."""

        for obstacle in world_config:
            if obstacle.pose is None:
                continue
            pose = torch.as_tensor(obstacle.pose, dtype=torch.float32, device=self.datastream.device)
            quat_xyzw = torch.cat((pose[4:], pose[3:4]))
            pose_env = PoseUtils.make_pose(pose[:3], PoseUtils.matrix_from_quat(quat_xyzw))
            obstacle.pose = self._object_pose_to_curobo_list(self._pose_env_to_base(pose_env))

    # =====================================================================================
    # PROPERTIES AND BASIC GETTERS
    # =====================================================================================

    @property
    def attached_link(self) -> str:
        """Default link name for object attachment operations."""
        return self.config.attached_object_link_name

    @property
    def attachment_links(self) -> set[str]:
        """Set of parent link names that currently have attached objects."""
        return {attachment.parent for attachment in self.attached_objects.values()}

    @property
    def current_plan(self) -> JointState | None:
        """Current plan from cuRobo motion generator."""
        return self._current_plan

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        """Joint names in state-query order, including separately commanded auxiliary joints."""
        return self._active_joint_names

    @property
    def locked_joint_positions(self) -> Mapping[str, float]:
        """Exact named joint positions compiled as fixed in this cuRobo model [rad, m]."""

        return self._locked_joint_positions

    @property
    def current_joint_position_path(self) -> torch.Tensor | None:
        """Current arm-local joint-position path in :attr:`active_joint_names` order."""
        if self._current_plan is None:
            return None
        plan = self._current_plan.get_ordered_joint_state(self.active_joint_names)
        return self._to_env_device(plan.position).detach().clone()

    def get_planned_joint_trajectory(self) -> NamedJointTrajectory | None:
        """Return the complete planned joint trajectory.

        Returns:
            Joint names and sampled positions with shape (T, J) [rad, m],
            or None when no plan is available.
        """
        positions = self.current_joint_position_path
        return None if positions is None else NamedJointTrajectory(self.active_joint_names, positions)

    def joint_positions_to_ee_poses(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Compute env-relative end-effector poses for arm-local joint positions.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].

        Returns:
            End-effector poses with shape ``(T, 4, 4)`` in the env-relative frame.
        """

        ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"]
        return self.joint_positions_to_link_poses(joint_positions, (ee_link,))[ee_link]

    def joint_positions_to_link_poses(
        self,
        joint_positions: torch.Tensor,
        link_names: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Compute env-relative poses for configured robot links.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].
            link_names: Configured links whose poses are requested.

        Returns:
            Link-name mapping to poses with shape ``(T, 4, 4)`` in the env-relative frame.
        """

        query = self._joint_positions_query(joint_positions)
        assert link_names and len(link_names) == len(set(link_names)), "link_names must be non-empty and unique"
        state = self._kinematics.get_state(query, calculate_jacobian=False)
        available = {str(name): index for index, name in enumerate(state.link_names)}
        missing = set(link_names) - set(available)
        assert not missing, f"cuRobo state does not expose links: {sorted(missing)}"
        result: dict[str, torch.Tensor] = {}
        for link_name in link_names:
            link_index = available[link_name]
            position = self._to_env_device(state.links_position[..., link_index, :])
            quaternion_wxyz = self._to_env_device(state.links_quaternion[..., link_index, :])
            quaternion_xyzw = torch.cat((quaternion_wxyz[..., 1:], quaternion_wxyz[..., :1]), dim=-1)
            pose_base = PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quaternion_xyzw))
            result[link_name] = self._pose_base_to_env(pose_base).detach().clone()
        return result

    def joint_positions_to_collision_spheres(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Return configured collision spheres in the planning-root frame.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].

        Returns:
            Sphere tensor with shape ``(T, S, 4)``; the final coordinate is radius [m].
        """

        query = self._joint_positions_query(joint_positions)
        state = self._kinematics.get_state(query, calculate_jacobian=False)
        spheres = state.link_spheres_tensor
        assert isinstance(spheres, torch.Tensor), "cuRobo kinematics returned no collision spheres"
        return spheres.reshape(query.shape[0], -1, 4).detach().clone()

    def joint_positions_self_collision_free(
        self,
        joint_positions: torch.Tensor,
        collision_margin_m: float = 0.0,
    ) -> torch.Tensor:
        """Check configured full-robot self collision for batched states.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].
            collision_margin_m: Required sphere-surface separation [m].

        Returns:
            Boolean collision freedom with shape ``(T,)`` on the environment device.
        """

        assert math.isfinite(collision_margin_m) and collision_margin_m >= 0.0
        spheres = self.joint_positions_to_collision_spheres(joint_positions)
        if not self.config.use_native_self_collision or self._auxiliary_joint_indices:
            collisions = self._collision_spheres_self_collisions(spheres, collision_margin_m)
            return self._to_env_device(~collisions).detach().clone()
        if collision_margin_m:
            spheres[..., 3] += torch.where(
                spheres[..., 3] > 0.0,
                0.5 * collision_margin_m,
                0.0,
            )
        spheres = spheres.unsqueeze(1)
        constraint = self.motion_gen.rollout_fn.robot_self_collision_constraint
        assert constraint.enabled, "cuRobo self-collision constraint is disabled"
        cost = constraint.forward(spheres)
        assert cost.shape == (len(joint_positions), 1), "cuRobo returned unexpected self-collision shape"
        return self._to_env_device(cost[:, 0] <= 0.0).detach().clone()

    def _collision_spheres_self_collisions(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float,
    ) -> torch.Tensor:
        """Check configured robot-sphere pairs in bounded sample batches."""

        pair_indices, sphere_offsets, _ = self._self_collision_pair_data(spheres.shape[1])
        collisions = torch.zeros(len(spheres), dtype=torch.bool, device=spheres.device)
        if not len(pair_indices):
            return collisions
        pair_count = len(pair_indices)
        sample_batch_size = max(1, 2_000_000 // pair_count)
        first_indices, second_indices = pair_indices.T
        for start in range(0, len(spheres), sample_batch_size):
            stop = min(start + sample_batch_size, len(spheres))
            batch = spheres[start:stop]
            first = batch[:, first_indices]
            second = batch[:, second_indices]
            clearance = (
                torch.linalg.vector_norm(first[..., :3] - second[..., :3], dim=-1)
                - first[..., 3]
                - second[..., 3]
                - sphere_offsets[first_indices]
                - sphere_offsets[second_indices]
            )
            active = (first[..., 3] > 0.0) & (second[..., 3] > 0.0)
            collisions[start:stop] = torch.any(active & (clearance <= collision_margin_m), dim=1)
        return collisions

    def _self_collision_pair_data(
        self,
        sphere_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        """Build direct collision pairs without cuRobo's fallback scan kernel."""

        if self._self_collision_pair_cache is not None:
            pairs, offsets, link_names = self._self_collision_pair_cache
            assert len(link_names) == sphere_count
            return pairs, offsets, link_names

        link_names = self.collision_sphere_link_names(sphere_count)
        unknown = set(link_names) - set(self._self_collision_ignore)
        assert not unknown, f"collision links have no self-collision policy: {sorted(unknown)}"
        first_indices, second_indices = np.triu_indices(sphere_count, k=1)
        enabled = np.fromiter(
            (
                link_names[first] != link_names[second]
                and link_names[second] not in self._self_collision_ignore[link_names[first]]
                for first, second in zip(first_indices, second_indices, strict=True)
            ),
            dtype=np.bool_,
            count=len(first_indices),
        )
        pair_indices = torch.as_tensor(
            np.stack((first_indices[enabled], second_indices[enabled]), axis=1),
            dtype=torch.long,
            device=self.tensor_args.device,
        )
        buffers = self._kinematic_robot_cfg["kinematics"]["self_collision_buffer"]
        sphere_offsets = torch.tensor(
            [float(buffers[name]) for name in link_names],
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        self._self_collision_pair_cache = pair_indices, sphere_offsets, link_names
        return self._self_collision_pair_cache

    def joint_positions_world_collision_free(
        self,
        joint_positions: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        excluded_obstacle_paths: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Check robot states against the synchronized cuRobo world.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].
            collision_margin_m: Required sphere-to-world separation [m].
            excluded_obstacle_paths: Exact cuRobo obstacle paths omitted only for this query.

        Returns:
            Boolean collision freedom with shape ``(T,)`` on the environment device.
        """

        spheres = self.joint_positions_to_collision_spheres(joint_positions)
        return self._planning_spheres_world_collision_free(
            spheres,
            collision_margin_m,
            excluded_obstacle_paths=excluded_obstacle_paths,
        )

    def joint_positions_environment_spheres_collision_free(
        self,
        joint_positions: torch.Tensor,
        environment_spheres: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        allowed_robot_link_names: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Check robot states against matched environment-frame object spheres.

        Args:
            joint_positions: Joint positions with shape ``(T, J)`` in
                :attr:`active_joint_names` order [rad, m].
            environment_spheres: Object spheres with shape ``(T, O, 4)`` in the
                environment frame, with radius last [m].
            collision_margin_m: Required robot-to-object sphere separation [m].
            allowed_robot_link_names: Exact robot links whose overlap with these
                object spheres is authorized.

        Returns:
            Boolean collision freedom with shape ``(T,)`` on the environment device.
        """

        collisions, _ = self._joint_positions_environment_spheres_collisions(
            joint_positions,
            environment_spheres,
            collision_margin_m,
            allowed_robot_link_names=allowed_robot_link_names,
        )
        return ~collisions

    def joint_positions_environment_spheres_collision_witness(
        self,
        joint_positions: torch.Tensor,
        environment_spheres: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        allowed_robot_link_names: tuple[str, ...] = (),
    ) -> tuple[str, int, int] | None:
        """Return the first forbidden robot-to-object sphere overlap.

        Args:
            joint_positions: One joint state shaped ``(1, J)`` in
                :attr:`active_joint_names` order [rad, m].
            environment_spheres: One object sphere state shaped ``(1, O, 4)`` in
                the environment frame, with radius last [m].
            collision_margin_m: Required robot-to-object sphere separation [m].
            allowed_robot_link_names: Exact robot links whose overlap with these
                object spheres is authorized.

        Returns:
            Robot link name, robot sphere index, and object sphere index, or ``None``.
        """

        assert joint_positions.shape == (1, len(self.active_joint_names))
        assert environment_spheres.shape[0] == 1
        _, witness = self._joint_positions_environment_spheres_collisions(
            joint_positions,
            environment_spheres,
            collision_margin_m,
            allowed_robot_link_names=allowed_robot_link_names,
        )
        return witness

    def _joint_positions_environment_spheres_collisions(
        self,
        joint_positions: torch.Tensor,
        environment_spheres: torch.Tensor,
        collision_margin_m: float,
        *,
        allowed_robot_link_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, tuple[str, int, int] | None]:
        """Check exact robot/object sphere pairs in bounded sample batches."""

        assert math.isfinite(collision_margin_m) and collision_margin_m >= 0.0
        robot_spheres = self.joint_positions_to_collision_spheres(joint_positions)
        object_spheres = self._collision_spheres_validate(environment_spheres).to(
            device=robot_spheres.device,
            dtype=robot_spheres.dtype,
        )
        assert robot_spheres.shape[0] == object_spheres.shape[0], "robot and object batches must match"

        transform = self._env_to_base_transform.to(device=object_spheres.device, dtype=object_spheres.dtype)
        object_spheres[..., :3] = object_spheres[..., :3] @ transform[:3, :3].T + transform[:3, 3]

        robot_link_names = self.collision_sphere_link_names(robot_spheres.shape[1])
        assert isinstance(allowed_robot_link_names, tuple) and len(allowed_robot_link_names) == len(
            set(allowed_robot_link_names)
        )
        assert all(
            isinstance(name, str) and name and name == name.strip() for name in allowed_robot_link_names
        ), "allowed_robot_link_names must contain unique, non-empty, trimmed strings"
        missing = set(allowed_robot_link_names) - set(robot_link_names)
        assert not missing, f"allowed robot links have no cuRobo collision spheres: {sorted(missing)}"
        allowed = torch.tensor(
            [name in allowed_robot_link_names for name in robot_link_names],
            dtype=torch.bool,
            device=robot_spheres.device,
        )

        collisions = torch.zeros(len(robot_spheres), dtype=torch.bool, device=robot_spheres.device)
        witness: tuple[str, int, int] | None = None
        for start in range(0, len(robot_spheres), 128):
            stop = min(start + 128, len(robot_spheres))
            robot_batch = robot_spheres[start:stop]
            object_batch = object_spheres[start:stop]
            clearance = (
                torch.cdist(robot_batch[..., :3], object_batch[..., :3])
                - robot_batch[..., 3].unsqueeze(-1)
                - object_batch[..., 3].unsqueeze(-2)
            )
            active = (robot_batch[..., 3].unsqueeze(-1) > 0.0) & (object_batch[..., 3].unsqueeze(-2) > 0.0)
            forbidden = active & ~allowed.view(1, -1, 1) & (clearance <= collision_margin_m)
            collisions[start:stop] = torch.any(forbidden.flatten(start_dim=1), dim=1)
            if len(robot_spheres) == 1 and witness is None:
                indices = torch.nonzero(forbidden[0], as_tuple=False)
                if len(indices):
                    robot_index, object_index = (int(index.item()) for index in indices[0])
                    witness = robot_link_names[robot_index], robot_index, object_index
        return self._to_env_device(collisions).detach().clone(), witness

    def collision_sphere_link_names(self, sphere_count: int) -> tuple[str, ...]:
        """Map every configured cuRobo collision sphere to its exact robot link.

        Args:
            sphere_count: Number of spheres returned by cuRobo kinematics.

        Returns:
            Link name for each sphere in cuRobo tensor order.
        """

        kinematics = self._kinematics.kinematics_config
        sphere_links = kinematics.link_sphere_idx_map
        assert sphere_links.shape == (sphere_count,)
        names_by_index = {int(index): str(name) for name, index in kinematics.link_name_to_idx_map.items()}
        missing = {int(index.item()) for index in sphere_links} - set(names_by_index)
        assert not missing, f"cuRobo sphere links have no names: {sorted(missing)}"
        return tuple(names_by_index[int(index.item())] for index in sphere_links)

    def world_object_obstacle_paths(self, object_ref: str) -> tuple[str, ...]:
        """Resolve one logical scene object to its exact cuRobo obstacle inventory.

        Args:
            object_ref: Exact key exposed by :meth:`Datastream.get_object_poses`.

        Returns:
            Canonically ordered cuRobo obstacle paths comprising that scene object.
        """

        assert object_ref and object_ref == object_ref.strip(), "object_ref must be non-empty and trimmed"
        root_path = self.datastream.get_object_prim_path(object_ref, self.env_id)
        matches = tuple(
            path
            for path in sorted(set(self._get_world_object_names()))
            if path == root_path or path.startswith(f"{root_path}/")
        )
        if not matches:
            raise KeyError(f"scene object {object_ref!r} has no cuRobo obstacle inventory")
        return matches

    def environment_spheres_world_collision_free(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        excluded_obstacle_paths: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Check arbitrary environment-frame spheres against the synchronized cuRobo world.

        Args:
            spheres: Sphere states shaped ``(T, S, 4)`` with environment-relative centers and
                radius last [m].
            collision_margin_m: Required sphere-to-world separation [m].
            excluded_obstacle_paths: Exact cuRobo obstacle paths omitted only for this query.

        Returns:
            Boolean collision freedom with shape ``(T,)`` on the environment device.
        """

        query = self._collision_spheres_validate(spheres)
        transform = self._env_to_base_transform.to(device=query.device, dtype=query.dtype)
        query[..., :3] = query[..., :3] @ transform[:3, :3].T + transform[:3, 3]
        return self._planning_spheres_world_collision_free(
            query,
            collision_margin_m,
            excluded_obstacle_paths=excluded_obstacle_paths,
        )

    def environment_spheres_world_collision_witness(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        excluded_obstacle_paths: tuple[str, ...] = (),
    ) -> tuple[int, str] | None:
        """Return the first environment-sphere and configured-obstacle collision.

        Args:
            spheres: One environment-frame sphere state shaped ``(1, S, 4)`` with radius last [m].
            collision_margin_m: Required sphere-to-world separation [m].
            excluded_obstacle_paths: Exact cuRobo obstacle paths omitted from attribution.

        Returns:
            Sphere index and exact obstacle path, or ``None``.
        """

        assert spheres.shape[0] == 1
        query = self._collision_spheres_validate(spheres)
        transform = self._env_to_base_transform.to(device=query.device, dtype=query.dtype)
        query[..., :3] = query[..., :3] @ transform[:3, :3].T + transform[:3, 3]
        return self._planning_spheres_world_collision_witness(
            query,
            collision_margin_m,
            excluded_obstacle_paths=excluded_obstacle_paths,
        )

    def _planning_spheres_world_collision_free(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float,
        *,
        excluded_obstacle_paths: tuple[str, ...],
    ) -> torch.Tensor:
        """Return collision freedom for each sphere batch in the planner's root frame.

        Args:
            spheres: Sphere states with shape (T, S, 4), storing XYZ and radius [m].
            collision_margin_m: Required sphere-to-world separation [m].
            excluded_obstacle_paths: Exact obstacle paths omitted from this query.

        Returns:
            One collision-free flag per batch sample.
        """
        collisions = self._planning_spheres_world_collisions(
            spheres,
            collision_margin_m,
            excluded_obstacle_paths=excluded_obstacle_paths,
        )
        return ~torch.any(collisions, dim=1)

    def _planning_spheres_world_collisions(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float,
        *,
        excluded_obstacle_paths: tuple[str, ...],
    ) -> torch.Tensor:
        """Return the native binary collision result for every query sphere."""

        assert math.isfinite(collision_margin_m) and collision_margin_m >= 0.0
        assert isinstance(excluded_obstacle_paths, tuple) and len(excluded_obstacle_paths) == len(
            set(excluded_obstacle_paths)
        )
        assert all(path and path == path.strip() for path in excluded_obstacle_paths)
        query = self._collision_spheres_validate(spheres)
        if collision_margin_m:
            query[..., 3] += torch.where(query[..., 3] > 0.0, collision_margin_m, 0.0)

        query = query.unsqueeze(1)
        checker = self.motion_gen.world_coll_checker
        buffer = CollisionQueryBuffer()
        buffer.update_buffer_shape(query.shape, self.tensor_args, checker.collision_types)
        with self._world_obstacles_disabled(excluded_obstacle_paths):
            collisions = checker.get_sphere_collision(
                query,
                buffer,
                weight=self.tensor_args.to_device([1.0]),
                activation_distance=self.tensor_args.to_device([0.0]),
            )
        assert collisions.shape == query.shape[:-1], "cuRobo returned unexpected world-collision shape"
        return self._to_env_device(torch.any(collisions > 0.0, dim=1)).detach().clone()

    @contextmanager
    def _world_obstacles_disabled(self, obstacle_paths: tuple[str, ...]) -> Iterator[None]:
        """Temporarily disable an exact, validated cuRobo obstacle set."""

        assert isinstance(obstacle_paths, tuple) and len(obstacle_paths) == len(set(obstacle_paths))
        assert all(
            isinstance(path, str) and path and path == path.strip() for path in obstacle_paths
        ), "obstacle_paths must contain unique, non-empty, trimmed strings"
        disabled_paths: list[str] = []
        try:
            for path in obstacle_paths:
                self.motion_gen.world_coll_checker.enable_obstacle(path, enable=False)
                disabled_paths.append(path)
            yield
        finally:
            for path in reversed(disabled_paths):
                self.motion_gen.world_coll_checker.enable_obstacle(path, enable=True)

    def joint_positions_world_collision_witness(
        self,
        joint_positions: torch.Tensor,
        collision_margin_m: float = 0.0,
        *,
        excluded_obstacle_paths: tuple[str, ...] = (),
    ) -> tuple[str, int, str] | None:
        """Return the first exact robot-link sphere and configured obstacle collision.

        Args:
            joint_positions: One joint state shaped ``(1, J)`` in
                :attr:`active_joint_names` order [rad, m].
            collision_margin_m: Required sphere-to-world separation [m].
            excluded_obstacle_paths: Exact cuRobo obstacle paths omitted from attribution.

        Returns:
            Robot link name, global sphere index, and exact obstacle path, or ``None``.
        """

        assert joint_positions.shape == (1, len(self.active_joint_names))
        spheres = self.joint_positions_to_collision_spheres(joint_positions)
        witness = self._planning_spheres_world_collision_witness(
            spheres,
            collision_margin_m,
            excluded_obstacle_paths=excluded_obstacle_paths,
        )
        if witness is None:
            return None
        sphere_index, obstacle_path = witness
        return self.collision_sphere_link_names(spheres.shape[1])[sphere_index], sphere_index, obstacle_path

    def _planning_spheres_world_collision_witness(
        self,
        spheres: torch.Tensor,
        collision_margin_m: float,
        *,
        excluded_obstacle_paths: tuple[str, ...],
    ) -> tuple[int, str] | None:
        """Isolate the first collision against the synchronized obstacle inventory."""

        assert spheres.shape[0] == 1
        object_mappings = self._get_object_mappings()
        attached_paths = {
            object_mappings[object_name] for object_name in self.attached_objects if object_name in object_mappings
        }
        candidate_paths = tuple(
            sorted(set(self._get_world_object_names()) - set(excluded_obstacle_paths) - attached_paths)
        )
        inventory_exclusions = tuple(sorted(set(excluded_obstacle_paths) | set(candidate_paths)))
        residual = self._planning_spheres_world_collisions(
            spheres,
            collision_margin_m,
            excluded_obstacle_paths=inventory_exclusions,
        )
        if bool(residual.any()):
            raise RuntimeError("world collision remains after disabling the synchronized obstacle inventory")
        for obstacle_path in candidate_paths:
            isolated_exclusions = tuple(sorted(set(excluded_obstacle_paths) | (set(candidate_paths) - {obstacle_path})))
            collisions = self._planning_spheres_world_collisions(
                spheres,
                collision_margin_m,
                excluded_obstacle_paths=isolated_exclusions,
            )
            colliding = torch.nonzero(collisions[0], as_tuple=False)
            if not len(colliding):
                continue
            sphere_index = int(colliding[0, 0].item())
            return sphere_index, obstacle_path
        return None

    def _collision_spheres_validate(self, spheres: torch.Tensor) -> torch.Tensor:
        """Check sphere batches and copy them to the planner's device.

        Args:
            spheres: Sphere states with shape (T, S, 4), storing XYZ and radius [m].

        Returns:
            A detached copy on the cuRobo device, safe for query-specific changes.
        """
        assert isinstance(spheres, torch.Tensor), "spheres must be a torch.Tensor"
        assert spheres.ndim == 3 and spheres.shape[0] > 0 and spheres.shape[1] > 0 and spheres.shape[2] == 4
        assert bool(torch.isfinite(spheres).all()), "spheres must be finite"
        assert bool((spheres[..., 3] > 0.0).any(dim=1).all()), "every sample must contain an active sphere"
        return self._to_curobo_device(spheres).detach().clone()

    def joint_positions_self_collision_witness(
        self,
        joint_positions: torch.Tensor,
        collision_margin_m: float = 0.0,
    ) -> tuple[str, int, str, int, float] | None:
        """Return the deepest enabled sphere overlap for one robot state.

        Args:
            joint_positions: One joint state shaped ``(1, J)`` in :attr:`active_joint_names` order [rad, m].
            collision_margin_m: Required sphere-surface separation [m].

        Returns:
            First link, first sphere index, second link, second sphere index, and clearance [m],
            or ``None`` when the state satisfies the requested margin.
        """

        assert joint_positions.shape == (1, len(self.active_joint_names))
        assert math.isfinite(collision_margin_m) and collision_margin_m >= 0.0
        spheres = self.joint_positions_to_collision_spheres(joint_positions)[0]
        sphere_count = len(spheres)
        pair_indices, sphere_offsets, link_names = self._self_collision_pair_data(sphere_count)
        if not len(pair_indices):
            return None
        first_indices, second_indices = pair_indices.T
        radii = spheres[:, 3]
        clearance = (
            torch.linalg.vector_norm(spheres[first_indices, :3] - spheres[second_indices, :3], dim=-1)
            - radii[first_indices]
            - radii[second_indices]
            - sphere_offsets[first_indices]
            - sphere_offsets[second_indices]
        )
        active = (radii[first_indices] > 0.0) & (radii[second_indices] > 0.0)
        minimum, pair_index = torch.where(active, clearance, torch.inf).min(dim=0)
        if not bool(torch.isfinite(minimum)) or float(minimum.item()) > collision_margin_m:
            return None

        first_sphere = int(first_indices[pair_index].item())
        second_sphere = int(second_indices[pair_index].item())
        return (
            link_names[first_sphere],
            first_sphere,
            link_names[second_sphere],
            second_sphere,
            float(minimum.item()),
        )

    def _joint_positions_query(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Check joint samples and place them on the planner's device.

        Args:
            joint_positions: Samples with shape (T, J) in active-joint order [rad, m].

        Returns:
            Finite joint samples on the cuRobo device.
        """
        assert isinstance(joint_positions, torch.Tensor), "joint_positions must be a torch.Tensor"
        assert (
            joint_positions.ndim == 2 and joint_positions.shape[0] > 0
        ), "joint_positions must have non-empty shape (T, J)"
        assert joint_positions.shape[1] == len(
            self.active_joint_names
        ), "joint_positions width must match active_joint_names"
        assert bool(torch.isfinite(joint_positions).all()), "joint_positions must be finite"
        return self._to_curobo_device(joint_positions)

    # =====================================================================================
    # WORLD AND OBJECT MANAGEMENT, ATTACHMENT, AND DETACHMENT
    # =====================================================================================

    def get_object_pose(self, object_name: str) -> Pose | None:
        """Retrieve object pose from cuRobo's collision world model.

        Searches the collision world model for the specified object and returns its current
        pose. This is useful for attachment calculations and debugging collision world state.
        The method handles both mesh and cuboid object types automatically.

        Args:
            object_name: Short object name used in Isaac Lab scene (e.g., "cube_1")

        Returns:
            Object pose in cuRobo coordinate frame, or None if object not found
        """
        # Get cached object mappings
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model

        object_path = object_mappings.get(object_name)
        if not object_path:
            self.logger.debug(f"Object {object_name} not found in world model")
            return None

        # Search for object in world model
        for obj_list, _ in [
            (world_model.mesh, "mesh"),
            (world_model.cuboid, "cuboid"),
        ]:
            if not obj_list:
                continue

            for obj in obj_list:
                if obj.name and object_path == str(obj.name):
                    if obj.pose is not None:
                        return Pose.from_list(obj.pose, tensor_args=self.tensor_args)

        self.logger.debug(f"Object {object_name} found in mappings but pose not available")
        return None

    def get_attached_pose(self, link_name: str, joint_state: JointState | None = None) -> Pose:
        """Calculate pose of specified link using forward kinematics.

        Computes the world pose of any robot link at the given joint configuration.
        This is essential for attachment calculations where we need to know the exact
        pose of the parent link to compute relative object positions.

        Args:
            link_name: Name of the robot link to get pose for
            joint_state: Joint configuration to use for calculation, uses current state if None

        Returns:
            World pose of the specified link in cuRobo coordinate frame

        Raises:
            KeyError: If link_name is not found in the computed link poses
        """
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get all link states using the robot model
        link_state = self.motion_gen.kinematics.get_state(
            q=joint_state.position.detach().clone().to(device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            calculate_jacobian=False,
        )

        # Extract all link poses
        link_poses = {}
        if link_state.links_position is not None and link_state.links_quaternion is not None:
            for i, link in enumerate(link_state.link_names):
                # cuRobo kinematics returns quaternions in (w, x, y, z) format
                link_poses[link] = self._make_pose(
                    position=link_state.links_position[..., i, :],
                    quaternion=link_state.links_quaternion[..., i, :],
                    name=link,
                    quat_is_xyzw=False,
                )

        # For attached object link, use ee_link from robot config as parent
        if link_name == self.config.attached_object_link_name:
            ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"]
            if ee_link in link_poses:
                self.logger.debug(f"Using {ee_link} for {link_name}")
                return link_poses[ee_link]

        # Return directly for other links
        if link_name in link_poses:
            return link_poses[link_name]
        raise KeyError(f"Link {link_name} not found in computed link poses")

    def create_attachment(
        self, object_name: str, link_name: str | None = None, joint_state: JointState | None = None
    ) -> Attachment:
        """Create attachment relationship between object and robot link.

        Computes the relative pose between an object and a robot link to enable the robot
        to carry the object consistently during motion planning. The attachment stores the transform
        from the parent link frame to the object frame, which remains constant while grasped.

        Args:
            object_name: Name of the object to attach
            link_name: Parent link for attachment, uses default attached_object_link if None
            joint_state: Robot configuration for calculation, uses current state if None

        Returns:
            Attachment object containing relative pose and parent link information
        """
        if link_name is None:
            link_name = self.attached_link
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get current link pose
        link_pose = self.get_attached_pose(link_name, joint_state)
        self.logger.info(f"Getting object pose for {object_name}")
        obj_pose = self.get_object_pose(object_name)

        # Compute relative pose
        attach_pose = link_pose.inverse().multiply(obj_pose)

        self.logger.debug(f"Creating attachment for {object_name} to {link_name}")
        self.logger.debug(f"Link pose: {link_pose.position}")
        self.logger.debug(f"Object pose (ACTUAL): {obj_pose.position}")
        self.logger.debug(f"Computed relative pose: {attach_pose.position}")

        return Attachment(attach_pose, link_name)

    def update_world(self, object_pose_overrides: Mapping[str, torch.Tensor] | None = None) -> None:
        """Synchronize collision world with current Isaac Lab scene state.

        Updates all dynamic object poses in cuRobo's collision world to match their current
        positions in Isaac Lab. This ensures collision checking uses accurate object positions
        after simulation steps, resets, or manual object movements. Static world geometry
        is loaded once during initialization and not updated here for performance.

        The method validates that the set of objects hasn't changed at runtime, as cuRobo
        requires world model reinitialization when objects are added or removed.

        Args:
            object_pose_overrides: Optional environment-relative dynamic-object poses [m] used by
                offline path accumulation instead of their live simulator poses.

        Raises:
            RuntimeError: If the set of objects has changed at runtime.
        """

        # Refresh the env-relative -> robot base frame transform before syncing geometry, so
        # object, goal, and planned poses are reconciled against the current base pose.
        self._refresh_base_frame_transform()
        self._static_world_activate()

        # Establish validation baseline on first call, validate on subsequent calls
        if self._expected_objects is None:
            self._expected_objects = set(self._get_world_object_names())
            self.logger.debug(f"Established object validation baseline: {len(self._expected_objects)} objects")
        else:
            # Subsequent calls: validate no changes
            current_objects = set(self._get_world_object_names())
            if current_objects != self._expected_objects:
                added = current_objects - self._expected_objects
                removed = self._expected_objects - current_objects

                error_msg = "World objects changed at runtime!\n"
                if added:
                    error_msg += f"Added: {added}\n"
                if removed:
                    error_msg += f"Removed: {removed}\n"
                error_msg += "cuRobo world model must be reinitialized."

                # Invalidate cached mappings since object set changed
                self._cached_object_mappings = None

                raise RuntimeError(error_msg)

        # Sync object poses with Isaac Lab
        self._sync_object_poses_with_isaaclab(object_pose_overrides)

        if self.visualize_spheres:
            self._update_sphere_visualization(force_update=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def world_visualization_refresh(self) -> None:
        """Refresh the Rerun world scene from this planner's current synchronized world."""

        if self.plan_visualizer is None:
            return
        world_scene = None
        try:
            self.update_world()
            world_scene = WorldConfig.get_scene_graph(self.motion_gen.world_coll_checker.world_model)
        except Exception as error:
            self.logger.warning(f"Failed to refresh the Rerun world scene: {error}")
        self.plan_visualizer.world_scene_set(world_scene)

    def _get_world_object_names(self) -> list[str]:
        """Extract all object names from cuRobo's collision world model.

        Iterates through all supported primitive types (mesh, cuboid, sphere, etc.) in the
        collision world and collects their names. This is used for world validation to detect
        when objects are added or removed at runtime.

        Returns:
            List of all object names currently in the collision world model
        """
        try:
            world_model = self.motion_gen.world_coll_checker.world_model

            # Handle case where world_model might be a list
            if isinstance(world_model, list):
                if len(world_model) <= self.env_id:
                    return []
                world_model = world_model[self.env_id]

            object_names = []

            # Get all primitive object names using the defined primitive types
            for primitive_type in self.primitive_types:
                if hasattr(world_model, primitive_type) and getattr(world_model, primitive_type):
                    primitive_list = getattr(world_model, primitive_type)
                    for primitive in primitive_list:
                        if primitive.name:
                            object_names.append(str(primitive.name))

            return object_names

        except Exception as e:
            self.logger.debug(f"ERROR getting world object names: {e}")
            return []

    def _sync_object_poses_with_isaaclab(
        self,
        object_pose_overrides: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Synchronize cuRobo collision world with Isaac Lab object positions.

        Updates all dynamic object poses in cuRobo's world model to match their current
        positions in Isaac Lab. This ensures accurate collision checking after simulation
        steps or manual object movements. Static objects (bins, tables, walls) are skipped
        for performance as they shouldn't move during simulation.

        The method updates both the world model and the collision checker to ensure
        consistency across all cuRobo components.
        """
        # Object poses enter in the Datastream's env-relative frame and are converted to the
        # planning-root frame before updating cuRobo's collision world.
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        object_poses = self.datastream.get_object_poses(env_ids=[self.env_id])
        pose_overrides = dict(object_pose_overrides or {})
        unknown_overrides = set(pose_overrides) - set(object_mappings)
        assert not unknown_overrides, f"object-pose overrides reference unknown obstacles: {sorted(unknown_overrides)}"
        static_objects = getattr(self.config, "static_objects", [])

        def _is_static(name: str) -> bool:
            return any(static_name in name.lower() for static_name in static_objects)

        updates: list[tuple[str, str, list[float]]] = []
        for object_name, object_path in object_mappings.items():
            if _is_static(object_name):
                continue
            if object_name in pose_overrides:
                pose_env = pose_overrides[object_name]
            elif object_name in object_poses:
                pose_env = object_poses[object_name][0]
            else:
                continue
            pose_list = self._object_pose_to_curobo_list(self._pose_env_to_base(pose_env))
            updates.append((object_name, object_path, pose_list))

        updated_count = 0
        for object_name, object_path, pose_list in updates:
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated_count += 1

        self.logger.debug(f"SYNC: Updated {updated_count} object poses in cuRobo world model")

        # Sync object poses with collision checker. This preserves static mesh objects unlike
        # load_collision_model which rebuilds everything.
        if updated_count > 0:
            for _, object_path, pose_list in updates:
                # pose_list quaternion is already cuRobo (w, x, y, z) order.
                curobo_pose = self._make_pose(position=pose_list[:3], quaternion=pose_list[3:], quat_is_xyzw=False)
                self.motion_gen.world_coll_checker.update_obstacle_pose(  # type: ignore
                    object_path, curobo_pose, update_cpu_reference=True
                )

            self.logger.debug(f"Updated {updated_count} object poses in collision checker")

    def _object_pose_to_curobo_list(self, pose_mat: torch.Tensor) -> list[float]:
        """Decompose a 4x4 pose into cuRobo's ``[x, y, z, qw, qx, qy, qz]`` list.

        Our :mod:`pose_math` uses the ``(x, y, z, w)`` quaternion convention; cuRobo wants
        ``(w, x, y, z)``, so we reorder here.
        """
        pos, rot = PoseUtils.unmake_pose(pose_mat)
        quat_xyzw = PoseUtils.quat_from_matrix(rot)
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat_xyzw[3]),  # w
            float(quat_xyzw[0]),  # x
            float(quat_xyzw[1]),  # y
            float(quat_xyzw[2]),  # z
        ]

    def _get_object_mappings(self) -> dict[str, str]:
        """Get object mappings with caching for performance optimization.

        Returns cached mappings if available, otherwise computes and caches them.
        Cache is invalidated when the object set changes.

        Returns:
            Dictionary mapping Isaac Lab object names to their corresponding USD paths
        """
        if self._cached_object_mappings is None:
            world_model = self.motion_gen.world_coll_checker.world_model
            object_names = list(self.datastream.get_object_poses(env_ids=[self.env_id]).keys())
            self._cached_object_mappings = self._discover_object_mappings(world_model, object_names)
            self.logger.debug(f"Computed and cached object mappings: {len(self._cached_object_mappings)} objects")

        return self._cached_object_mappings

    def _discover_object_mappings(self, world_model, object_names: list[str]) -> dict[str, str]:
        """Build mapping between scene object names and cuRobo world paths.

        Automatically discovers the correspondence between the live scene's object names
        (from the Datastream) and their full USD paths in cuRobo's world model. This mapping
        is essential for pose synchronization and attachment operations, as cuRobo uses full
        USD paths while the scene uses short object names.

        Args:
            world_model: cuRobo's collision world model containing primitive objects
            object_names: Live scene object names (keys of :meth:`Datastream.get_object_poses`)

        Returns:
            Dictionary mapping scene object names to their corresponding USD paths
        """
        mappings = {}
        world_object_paths = []

        # Collect all primitive objects from cuRobo world model
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name:
                    world_object_paths.append(str(primitive.name))
        world_object_paths = sorted(set(world_object_paths))

        for object_name in object_names:
            root_path = self.datastream.get_object_prim_path(object_name, self.env_id)
            matches = tuple(
                path for path in world_object_paths if path == root_path or path.startswith(f"{root_path}/")
            )
            if len(matches) == 1:
                mappings[object_name] = matches[0]
                self.logger.debug(f"MAPPING: {object_name} -> {matches[0]}")
            elif matches:
                self.logger.debug(
                    f"WARNING: Object {object_name} has {len(matches)} cuRobo obstacles; "
                    "single-obstacle attachment mapping is unavailable"
                )
            else:
                self.logger.debug(f"WARNING: Could not find world path for {object_name}")

        return mappings

    def _update_object_in_world_model(
        self, world_model, object_name: str, object_path: str, pose_list: list[float]
    ) -> bool:
        """Update a single object's pose in cuRobo's collision world model.

        Searches every primitive type for the exact resolved cuRobo obstacle path and updates
        its pose.

        Args:
            world_model: cuRobo's collision world model
            object_name: Short object name from Isaac Lab (e.g., "cube_1")
            object_path: Full USD path for the object in cuRobo world
            pose_list: New pose as [x, y, z, w, x, y, z] list in cuRobo format

        Returns:
            True if object was found and successfully updated, False otherwise
        """
        # Handle case where world_model might be a list
        if isinstance(world_model, list):
            if len(world_model) > self.env_id:
                world_model = world_model[self.env_id]
            else:
                return False

        # Update all primitive types
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name:
                    primitive_name = str(primitive.name)
                    if object_path == primitive_name:
                        primitive.pose = pose_list
                        self.logger.debug(f"Updated {primitive_type} {object_name} pose")
                        return True

        self.logger.debug(f"WARNING: Object {object_name} not found in world model")
        return False

    def object_collision_spheres_fit(
        self,
        object_name: str,
        env_id: int,
        start_joint_positions: torch.Tensor,
        expected_tool_from_object: torch.Tensor,
        object_collision_spheres: ObjectCollisionSpheres | None = None,
    ) -> ObjectCollisionSpheres:
        """Fit or bind the exact object spheres used by carried-object planning.

        The query applies the virtual grasp pose in the planner world, attaches at the supplied
        robot state, captures or reuses object-frame geometry, and detaches before returning.
        Cleanup failures propagate.

        Args:
            object_name: Exact logical scene-object name.
            env_id: Environment index used by the planner.
            start_joint_positions: Arm-local attachment state in planner joint order [rad, m].
            expected_tool_from_object: Expected rigid object pose in the physical tool frame.
            object_collision_spheres: Optional sphere geometry captured by another planner in this episode.

        Returns:
            Immutable object-frame spheres and their fit details.
        """

        assert self.config.planning_enabled, "object sphere fitting requires an ordinary planning context"
        assert env_id == self.env_id, "object sphere fitting must use the planner's environment"
        assert isinstance(expected_tool_from_object, torch.Tensor) and expected_tool_from_object.shape == (4, 4)
        tool_pose = self.joint_positions_to_ee_poses(start_joint_positions.unsqueeze(0))[0]
        object_pose = tool_pose @ expected_tool_from_object.to(device=tool_pose.device, dtype=tool_pose.dtype)
        self.update_world({object_name: object_pose})
        object_path = self._get_object_mappings().get(object_name)
        if object_path is None:
            raise KeyError(f"object {object_name!r} has no exact cuRobo obstacle mapping")
        if object_collision_spheres is not None:
            assert object_collision_spheres.object_name == object_name
            assert object_collision_spheres.obstacle_path == object_path
            cached = self._object_collision_spheres.setdefault(object_name, object_collision_spheres)
            assert cached is object_collision_spheres, "planner already owns different geometry for this object"
        try:
            if not self._attach_object(object_name, object_path, start_joint_positions):
                raise RuntimeError(f"cuRobo failed to fit attachment spheres for {object_name!r}")
            attachment = self.attached_objects[object_name]
            captured_transform = torch.as_tensor(
                attachment.pose.get_matrix(),
                device=expected_tool_from_object.device,
                dtype=expected_tool_from_object.dtype,
            ).reshape(4, 4)
            assert torch.allclose(
                captured_transform, expected_tool_from_object, atol=1.0e-4, rtol=0.0
            ), f"attachment transform for {object_name!r} does not match its planned attachment event"
            return self._object_collision_spheres[object_name]
        finally:
            self._detach_objects()

    def object_collision_spheres_reset(self) -> None:
        """Clear fitted object geometry at an episode boundary."""

        assert not self.attached_objects, "cannot clear object geometry while an object is attached"
        self._object_collision_spheres.clear()

    def _object_collision_spheres_capture(
        self,
        object_name: str,
        object_path: str,
        attachment: Attachment,
    ) -> ObjectCollisionSpheres:
        """Capture active native attachment slots as immutable object-frame spheres."""

        link_name = self.config.attached_object_link_name
        slots = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name).detach().clone()
        assert slots.ndim == 2 and slots.shape[1] == 4, "native attachment slots must have shape (S, 4)"
        active = slots[:, 3] > 0.0
        assert bool(active.any()), f"native attachment fit for {object_name!r} produced no active spheres"
        active_slots = slots[active]
        object_centers = attachment.pose.inverse().transform_points(active_slots[:, :3])
        spheres = tuple(
            CollisionSphere(
                center=tuple(float(value) for value in center),
                radius=float(radius),
            )
            for center, radius in zip(
                object_centers.detach().cpu().tolist(),
                active_slots[:, 3].detach().cpu().tolist(),
                strict=True,
            )
        )
        transform = attachment.pose.get_matrix().detach().cpu().reshape(4, 4).tolist()
        result = ObjectCollisionSpheres(
            object_name=object_name,
            obstacle_path=object_path,
            spheres=spheres,
            fit_type=SphereFitType.SAMPLE_SURFACE.value,
            surface_sphere_radius_m=self.config.surface_sphere_radius,
            slot_capacity=len(slots),
            tool_frame=self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"],
            attachment_link=link_name,
            captured_tool_from_object=tuple(tuple(float(value) for value in row) for row in transform),
        )
        self._object_collision_spheres[object_name] = result
        return result

    def _object_collision_spheres_attach(
        self,
        geometry: ObjectCollisionSpheres,
        attachment: Attachment,
    ) -> None:
        """Reload cached object-frame spheres into the native virtual attachment slots."""

        link_name = self.config.attached_object_link_name
        assert geometry.obstacle_path == self._get_object_mappings().get(geometry.object_name)
        assert geometry.attachment_link == link_name
        assert geometry.fit_type == SphereFitType.SAMPLE_SURFACE.value
        assert math.isclose(geometry.surface_sphere_radius_m, self.config.surface_sphere_radius)
        current_slots = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name)
        assert current_slots.shape == (geometry.slot_capacity, 4), "native attachment-slot capacity changed"
        object_centers = torch.as_tensor(
            [sphere.center for sphere in geometry.spheres],
            device=current_slots.device,
            dtype=current_slots.dtype,
        )
        tool_centers = attachment.pose.transform_points(object_centers)
        sphere_tensor = torch.zeros_like(current_slots)
        sphere_tensor[:, 3] = -10.0
        sphere_tensor[: len(geometry.spheres), :3] = tool_centers
        sphere_tensor[: len(geometry.spheres), 3] = torch.as_tensor(
            [sphere.radius for sphere in geometry.spheres],
            device=current_slots.device,
            dtype=current_slots.dtype,
        )
        self.motion_gen.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)
        self.motion_gen.world_coll_checker.enable_obstacle(geometry.obstacle_path, enable=False)

    def _attach_object(
        self,
        object_name: str,
        object_path: str,
        start_joint_positions: torch.Tensor | None = None,
    ) -> bool:
        """Attach an object to the robot for manipulation planning.

        Establishes an attachment between the specified object and the robot's end-effector
        or configured attachment link. This enables the robot to carry the object during
        motion planning while maintaining proper collision checking. The object's collision
        geometry is disabled in the world model since it's now part of the robot.

        Args:
            object_name: Short name of the object to attach (e.g., "cube_2")
            object_path: Full USD path for the object in cuRobo world model
            start_joint_positions: Optional arm-local attachment state for offline planning.

        Returns:
            True if attachment succeeded, False if attachment failed
        """
        self.logger.debug(f"Attaching {object_name} at path {object_path}")
        try:
            current_joint_state = self._get_current_joint_state_for_curobo(start_joint_positions)
            attachment = self.create_attachment(
                object_name,
                self.config.attached_object_link_name,
                current_joint_state,
            )
            cached_geometry = self._object_collision_spheres.get(object_name)
            if cached_geometry is None:
                success = self.motion_gen.attach_objects_to_robot(
                    joint_state=current_joint_state,
                    object_names=[object_path],
                    link_name=self.config.attached_object_link_name,
                    surface_sphere_radius=self.config.surface_sphere_radius,
                    sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
                    world_objects_pose_offset=None,
                )
                if success:
                    self._object_collision_spheres_capture(object_name, object_path, attachment)
            else:
                assert cached_geometry.obstacle_path == object_path, "cached object obstacle path changed"
                self._object_collision_spheres_attach(cached_geometry, attachment)
                success = True
        except Exception as error:
            self.logger.error(f"Failed to attach {object_name}: {error}")
            success = False

        if success:
            self.attached_objects[object_name] = attachment
            self.logger.debug(f"Successfully attached {object_name}")
            self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

            # Force sphere visualization update
            if self.visualize_spheres:
                try:
                    self._update_sphere_visualization(force_update=True)
                except Exception as error:
                    self.logger.warning(f"Failed to visualize attachment for {object_name}: {error}")

            return True

        self.logger.error(f"cuRobo attach_objects_to_robot failed for {object_name}")
        self.attached_objects.pop(object_name, None)
        try:
            self.motion_gen.detach_object_from_robot(link_name=self.config.attached_object_link_name)
        except Exception as error:
            self.logger.error(f"Failed to clear attachment slots for {object_name}: {error}")
        try:
            self._set_active_links([self.config.attached_object_link_name], active=True)
        except Exception as error:
            self.logger.error(f"Failed to restore attachment spheres for {object_name}: {error}")
        try:
            self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True)
        except Exception as error:
            self.logger.error(f"Failed to restore world obstacle for {object_name}: {error}")
        return False

    def _detach_objects(self, link_names: set[str] | None = None) -> bool:
        """Detach objects from robot and restore collision checking.

        Removes object attachments from specified links and re-enables collision checking
        for both the objects and the parent links. This is necessary when placing objects
        or changing grasps. All attached objects are detached if no specific links are provided.

        Args:
            link_names: Set of parent link names to detach objects from, detaches all if None

        Returns:
            True if detachment operations completed successfully, False otherwise
        """
        if link_names is None:
            link_names = self.attachment_links

        self.logger.debug(f"Detaching objects from links: {link_names}")
        self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

        # Get cached object mappings to find the USD path for re-enabling
        object_mappings = self._get_object_mappings()

        detached_info = []
        detached_links = set()
        for object_name, attachment in list(self.attached_objects.items()):
            if attachment.parent not in link_names:
                continue

            # Find object path and re-enable it in the world
            object_path = object_mappings.get(object_name)
            if object_path:
                self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True)  # type: ignore
                self.logger.debug(f"Re-enabled obstacle {object_path}")

            # Collect the link that will need re-enabling
            detached_links.add(attachment.parent)

            # Remove from attached objects and log info
            del self.attached_objects[object_name]
            detached_info.append((object_name, attachment.parent))

        if detached_info:
            for obj_name, parent_link in detached_info:
                self.logger.debug(f"Detached {obj_name} from {parent_link}")

        # Re-enable collision checking for the attachment links (following the planning pattern)
        if detached_links:
            self._set_active_links(list(detached_links), active=True)
            self.logger.debug(f"Re-enabled collision for attachment links: {detached_links}")

        # Call cuRobo's detach for each link
        for link_name in link_names:
            self.motion_gen.detach_object_from_robot(link_name=link_name)
            self.logger.debug(f"Called cuRobo detach for link {link_name}")

        return True

    def get_attached_objects(self) -> list[str]:
        """Get list of currently attached object names.

        Returns the short names of all objects currently attached to the robot.
        These names correspond to Isaac Lab scene object names, not full USD paths.

        Returns:
            List of attached object names (e.g., ["cube_1", "cube_2"])"""
        return list(self.attached_objects.keys())

    def has_attached_objects(self) -> bool:
        """Check if any objects are currently attached to the robot.

        Useful for determining gripper state and collision checking configuration
        before planning motions.

        Returns:
            True if one or more objects are attached, False if no attachments exist
        """
        return len(self.attached_objects) != 0

    # =====================================================================================
    # JOINT STATE AND KINEMATICS
    # =====================================================================================

    def _get_current_joint_state_for_curobo(
        self,
        start_joint_positions: torch.Tensor | None = None,
    ) -> JointState:
        """Read one state and select the optimizer's joint order.

        Args:
            start_joint_positions: Optional state shaped ``(J,)`` or ``(1, J)`` in
                :attr:`active_joint_names` order [rad, m]. ``None`` reads the live articulation.

        Returns:
            CuRobo planning state with zero velocity and acceleration. Separately commanded
            auxiliary joints remain outside the optimizer's state vector.
        """
        if start_joint_positions is None:
            joint_names = list(self.active_joint_names)
            joint_pos_raw = self.datastream.get_robot_joint_positions(env_ids=[self.env_id], joint_names=joint_names)
        else:
            # Offline accumulation passes one arm-local state in this planner's native order.
            joint_pos_raw = start_joint_positions
            if joint_pos_raw.ndim == 1:
                joint_pos_raw = joint_pos_raw.unsqueeze(0)
            joint_names = list(self.active_joint_names)
            assert joint_pos_raw.ndim == 2 and joint_pos_raw.shape[0] == 1
            assert joint_pos_raw.shape[1] == len(joint_names), (
                f"planning group {self.config.planning_group_name!r} expects {len(joint_names)} "
                f"joint positions, got {joint_pos_raw.shape[1]}"
            )
        joint_vel_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_acc_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)

        # Move to cuRobo device
        joint_pos: torch.Tensor = self._to_curobo_device(joint_pos_raw)
        joint_vel: torch.Tensor = self._to_curobo_device(joint_vel_raw)
        joint_acc: torch.Tensor = self._to_curobo_device(joint_acc_raw)

        cu_js: JointState = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            joint_names=joint_names,
            tensor_args=self.tensor_args,
        ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        position_limits = self.motion_gen.kinematics.get_joint_limits().position
        lower, upper = position_limits[0], position_limits[1]
        margin = 1e-4
        assert bool(torch.all(upper - lower > 2.0 * margin)), "joint limits are narrower than the safety margin"
        position = torch.clamp(cu_js.position, lower + margin, upper - margin)
        return JointState(
            position=position,
            velocity=torch.zeros_like(position),
            acceleration=torch.zeros_like(position),
            joint_names=cu_js.joint_names,
            tensor_args=self.tensor_args,
        )

    def _auxiliary_joint_positions_update(self, joint_positions: torch.Tensor | None) -> None:
        """Hold separately commanded joints fixed while the optimizer moves the arm.

        Args:
            joint_positions: One state in :attr:`active_joint_names` order [rad, m].
                ``None`` reads the live articulation.
        """
        if not self._auxiliary_joint_indices:
            return
        positions = (
            self.datastream.get_robot_joint_positions(env_ids=[self.env_id], joint_names=self.active_joint_names)
            if joint_positions is None
            else joint_positions.reshape(1, -1)
        )
        assert positions.shape == (1, len(self.active_joint_names)) and bool(torch.isfinite(positions).all())
        auxiliary = {name: float(positions[0, index].item()) for name, index in self._auxiliary_joint_indices.items()}
        locked = self.robot_cfg["kinematics"]["lock_joints"]
        if any(locked[name] != value for name, value in auxiliary.items()):
            spheres = {
                link: self.motion_gen.kinematics.kinematics_config.get_link_spheres(link).clone()
                for link in self.attachment_links
            }
            self.motion_gen.update_locked_joints({**locked, **auxiliary}, self.robot_cfg)
            self._joint_limits_sync(self.motion_gen.kinematics.get_joint_limits())
            for link, geometry in spheres.items():
                self.motion_gen.kinematics.kinematics_config.update_link_spheres(link, geometry)

    def _planning_joint_state_expand(
        self, state: JointState, auxiliary_joint_positions: torch.Tensor | None = None
    ) -> JointState:
        """Expand a plan using the current auxiliary-joint targets.

        Args:
            state: Joint state in the optimizer's joint order [rad, m].
            auxiliary_joint_positions: Reference state in :attr:`active_joint_names` order [rad, m].
                ``None`` uses the auxiliary positions held by the planning model.

        Returns:
            Full joint state with the auxiliary positions used for this planning call.
        """
        result = self.motion_gen.get_full_js(state).clone()
        reference = (
            self._to_curobo_device(auxiliary_joint_positions).reshape(-1)
            if auxiliary_joint_positions is not None
            else None
        )
        for name, index in self._auxiliary_joint_indices.items():
            result.position[..., result.joint_names.index(name)] = (
                self.robot_cfg["kinematics"]["lock_joints"][name] if reference is None else reference[index]
            )
        return result

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        """Compute end-effector pose from joint configuration.

        Uses cuRobo's forward kinematics to calculate the end-effector pose
        at the specified joint configuration. Handles device conversion to ensure
        compatibility with cuRobo's CUDA-based computations.

        Args:
            joint_state: Robot joint configuration to compute end-effector pose from

        Returns:
            End-effector pose in world coordinates
        """
        # Ensure joint state is on CUDA device for cuRobo
        if isinstance(joint_state.position, torch.Tensor):
            cuda_position = self._to_curobo_device(joint_state.position)
        else:
            cuda_position = self._to_curobo_device(torch.tensor(joint_state.position))

        # Create new joint state with CUDA tensors
        cuda_joint_state = JointState(
            position=cuda_position,
            velocity=(
                self._to_curobo_device(joint_state.velocity.detach().clone())
                if joint_state.velocity is not None
                else torch.zeros_like(cuda_position)
            ),
            acceleration=(
                self._to_curobo_device(joint_state.acceleration.detach().clone())
                if joint_state.acceleration is not None
                else torch.zeros_like(cuda_position)
            ),
            joint_names=joint_state.joint_names,
            tensor_args=self.tensor_args,
        )

        kin_state: Any = self.motion_gen.rollout_fn.compute_kinematics(cuda_joint_state)
        return kin_state.ee_pose

    def solve_ik(
        self,
        target_pose: torch.Tensor,
        seed_joint_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Solve one collision-feasible arm-local IK problem.

        Args:
            target_pose: End-effector pose as a 4x4 matrix in the env-relative frame.
            seed_joint_positions: Arm-local seed in :attr:`active_joint_names` order. ``None``
                uses the live articulation state.

        Returns:
            Solved arm-local joint positions on the environment device, or ``None`` when IK fails.
        """
        return self._solve_ik(target_pose, seed_joint_positions, require_feasible=True)

    def solve_ik_kinematic(
        self,
        target_pose: torch.Tensor,
        seed_joint_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return the nearest-seed pose-validated arm-local IK candidate for offline scheduling.

        Demonstrated skill poses are replayed as Cartesian controller targets. Their cuRobo joint
        candidates are used to approximate robot geometry during inter-arm scheduling, so world
        contact may make a solver-unsuccessful candidate useful. Robot self collision remains
        forbidden, and explicit forward kinematics must match the target within configured tolerances.

        Args:
            target_pose: End-effector pose as a 4x4 matrix in the env-relative frame.
            seed_joint_positions: Arm-local seed in :attr:`active_joint_names` order. ``None``
                uses the live articulation state.

        Returns:
            Pose-validated arm-local joint positions, or ``None`` when no candidate passes FK.
        """
        return self._solve_ik(target_pose, seed_joint_positions, require_feasible=False)

    @contextmanager
    def _ik_collision_costs_disabled(self) -> Iterator[None]:
        """Disable warmed IK collision rollouts and restore their exact prior states."""

        attributes = (
            "primitive_collision_cost",
            "primitive_collision_constraint",
            "robot_self_collision_cost",
            "robot_self_collision_constraint",
        )
        costs = {
            id(cost): (cost, bool(cost.enabled))
            for rollout in self.motion_gen.ik_solver.get_all_rollout_instances()
            for attribute in attributes
            if (cost := getattr(rollout, attribute, None)) is not None
        }
        try:
            for cost, _ in costs.values():
                cost.disable_cost()
            yield
        finally:
            for cost, was_enabled in costs.values():
                (cost.enable_cost if was_enabled else cost.disable_cost)()

    def joint_positions_to_named_positions(self, joint_positions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expand one active joint configuration, including its dependent mimic joints.

        Args:
            joint_positions: Active joint positions in planner order [rad, m].

        Returns:
            Named scalar positions on the environment device.
        """
        assert joint_positions.shape == (len(self.active_joint_names),)
        state = JointState.from_position(
            position=self._to_curobo_device(joint_positions), joint_names=list(self.active_joint_names)
        )
        expanded = self._kinematics.get_mimic_js(state)
        if expanded is not None:
            state = expanded
        positions = self._to_env_device(state.position)
        return dict(zip(state.joint_names, positions, strict=True))

    def _solve_ik(
        self,
        target_pose: torch.Tensor,
        seed_joint_positions: torch.Tensor | None,
        *,
        require_feasible: bool,
    ) -> torch.Tensor | None:
        """Solve a tool pose and validate candidates with the state-query model.

        Args:
            target_pose: Environment-relative tool transform [m].
            seed_joint_positions: Reference state in :attr:`active_joint_names` order [rad, m].
                ``None`` reads the live articulation.
            require_feasible: Keep native collision constraints enabled during IK.

        Returns:
            Accepted joint state, or ``None`` when no candidate passes the configured checks.
        """
        assert self.config.planning_enabled, "IK is disabled for this collision-only backend"
        assert target_pose.shape == (4, 4), f"target pose must have shape (4, 4), got {target_pose.shape}"
        if seed_joint_positions is None:
            seed_joint_positions = self.datastream.get_robot_joint_positions(
                env_ids=[self.env_id], joint_names=self.active_joint_names
            )[0]
        # Kinematic sampling checks auxiliary geometry in the query model, not the native IK costs.
        if require_feasible:
            self._auxiliary_joint_positions_update(seed_joint_positions)
        seed_state = self._get_current_joint_state_for_curobo(seed_joint_positions)
        target_pose_base = self._to_curobo_device(self._pose_env_to_base(target_pose))
        target_position, target_rotation = PoseUtils.unmake_pose(target_pose_base)
        goal_pose = self._make_pose(
            position=target_position,
            quaternion=PoseUtils.quat_from_matrix(target_rotation),
        )
        candidate_count = self.config.ik_candidate_count
        assert type(candidate_count) is int and candidate_count >= 1, "ik_candidate_count must be positive"
        solver_candidate_count = self.motion_gen.ik_solver.num_seeds
        assert (
            candidate_count <= solver_candidate_count
        ), "ik_candidate_count cannot exceed the warmed-up cuRobo IK seed count"
        returned_candidate_count = solver_candidate_count if self.config.debug_planner else candidate_count

        try:
            collision_scope = nullcontext() if require_feasible else self._ik_collision_costs_disabled()
            with collision_scope:
                result = self.motion_gen.ik_solver.solve_single(
                    goal_pose,
                    retract_config=seed_state.position,
                    seed_config=seed_state.position.unsqueeze(0),
                    return_seeds=returned_candidate_count,
                )
        except Exception as error:
            self.logger.debug(f"IK solve failed: {error}")
            return None

        solution_state = JointState(
            position=result.solution.reshape(-1, len(self.motion_gen.kinematics.joint_names)),
            joint_names=list(self.motion_gen.kinematics.joint_names),
            tensor_args=self.tensor_args,
        )
        solutions = (
            self._planning_joint_state_expand(solution_state, seed_joint_positions)
            .get_ordered_joint_state(self.active_joint_names)
            .position
        )
        successes = result.success.reshape(-1)
        assert (
            len(solutions) == len(successes) == returned_candidate_count
        ), "cuRobo returned an unexpected IK candidate shape"
        solutions = self._to_env_device(solutions).detach()
        successes = successes.to(device=solutions.device)
        try:
            actual_poses = self.joint_positions_to_ee_poses(solutions)
        except Exception as error:
            self.logger.debug(f"IK candidate FK failed: {error}")
            return None
        target_poses = target_pose.to(device=actual_poses.device, dtype=actual_poses.dtype).expand_as(actual_poses)
        position_error = torch.linalg.vector_norm(actual_poses[:, :3, 3] - target_poses[:, :3, 3], dim=-1)
        relative_rotation = actual_poses[:, :3, :3].transpose(-1, -2) @ target_poses[:, :3, :3]
        cosine = torch.clamp((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5, -1.0, 1.0)
        rotation_error = torch.acos(cosine)
        accepted = (
            torch.isfinite(solutions).all(dim=-1)
            & torch.isfinite(actual_poses).all(dim=(-2, -1))
            & (position_error <= self.config.position_threshold)
            & (rotation_error <= self.config.rotation_threshold)
        )
        if require_feasible:
            accepted &= successes
        else:
            candidate_indices = torch.nonzero(accepted[:candidate_count], as_tuple=False).reshape(-1)
            if len(candidate_indices):
                collision_free = self.joint_positions_self_collision_free(
                    solutions.index_select(0, candidate_indices),
                    0.0,
                )
                accepted[candidate_indices] &= collision_free
        seed_positions = self._to_env_device(
            self._planning_joint_state_expand(seed_state, seed_joint_positions)
            .get_ordered_joint_state(self.active_joint_names)
            .position
        ).reshape(1, -1)
        auxiliary_names = set(self.config.initial_auxiliary_joint_positions)
        pose_joint_indices = [
            index for index, name in enumerate(self.active_joint_names) if name not in auxiliary_names
        ]
        assert pose_joint_indices, "IK planner must contain at least one non-auxiliary joint"
        seed_distance = torch.sum(
            torch.square(solutions[:, pose_joint_indices] - seed_positions[:, pose_joint_indices]),
            dim=-1,
        )
        accepted_indices = torch.nonzero(accepted[:candidate_count], as_tuple=False).reshape(-1)
        if len(accepted_indices) == 0:
            if returned_candidate_count > candidate_count:
                position_limits = self._to_env_device(self._kinematics.get_joint_limits().position)
                joint_slack, limiting_joint = torch.min(
                    torch.minimum(
                        solutions[:, pose_joint_indices] - position_limits[0, pose_joint_indices],
                        position_limits[1, pose_joint_indices] - solutions[:, pose_joint_indices],
                    ),
                    dim=-1,
                )
                audit_lines = [
                    f"[{index:02d}] eligible={index < candidate_count} "
                    f"solver_success={bool(successes[index].item())} "
                    f"position_error={float(position_error[index].item()):.6f}m "
                    f"rotation_error={float(rotation_error[index].item()):.6f}rad "
                    f"pose_seed_distance={float(torch.sqrt(seed_distance[index]).item()):.6f} "
                    f"pose_joint_slack={float(joint_slack[index].item()):.6f} "
                    f"limiting_joint={self.active_joint_names[pose_joint_indices[int(limiting_joint[index].item())]]}"
                    for index in range(returned_candidate_count)
                ]
                self.logger.debug(
                    "IK candidate audit for group %r after configured prefix %d failed:\n%s",
                    self.config.planning_group_name,
                    candidate_count,
                    "\n".join(audit_lines),
                )
            best = int(
                torch.argmin(
                    torch.nan_to_num(position_error[:candidate_count] + rotation_error[:candidate_count], nan=torch.inf)
                ).item()
            )
            self.logger.debug(
                "IK candidates for group %r failed FK validation: position_error=%.6f m, "
                "rotation_error=%.6f rad, solver_success=%s",
                self.config.planning_group_name,
                float(position_error[best].item()),
                float(rotation_error[best].item()),
                bool(successes[best].item()),
            )
            return None

        selected = int(accepted_indices[torch.argmin(seed_distance[accepted_indices])].item())
        if not bool(successes[selected].item()):
            self.logger.debug(
                "IK candidate %d for group %r passes explicit FK but is solver-unsuccessful",
                selected,
                self.config.planning_group_name,
            )
        return solutions[selected].clone()

    # =====================================================================================
    # PLANNING CORE METHODS
    # =====================================================================================

    def _make_pose(
        self,
        position: torch.Tensor | np.ndarray | list[float] | None = None,
        quaternion: torch.Tensor | np.ndarray | list[float] | None = None,
        *,
        name: str | None = None,
        normalize_rotation: bool = False,
        quat_is_xyzw: bool = True,
    ) -> Pose:
        """Create a cuRobo Pose with sensible defaults and device/dtype alignment.

        Auto-populates missing fields with identity values and ensures tensors are
        on the cuRobo device with the correct dtype. Handles quaternion format conversion
        from Isaac Lab's (x, y, z, w) to cuRobo's (w, x, y, z) format when needed.

        Args:
            position: Optional position as Tensor/ndarray/list. Defaults to [0, 0, 0].
            quaternion: Optional quaternion as Tensor/ndarray/list. Defaults to identity quaternion.
            name: Optional name of the link that this pose represents.
            normalize_rotation: Whether to normalize the quaternion inside Pose.
            quat_is_xyzw: If True, quaternion is in Isaac Lab format (x, y, z, w) and will be
                converted to cuRobo format. If False, quaternion is already in cuRobo (w, x, y, z) format.

        Returns:
            Pose: A cuRobo Pose on the configured cuRobo device and dtype.
        """
        if position is None:
            position = torch.tensor([0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        if quaternion is None:
            quaternion_wxyz = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device
            )
        else:
            if not isinstance(quaternion, torch.Tensor):
                quaternion = torch.tensor(quaternion, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
            else:
                quaternion = self._to_curobo_device(quaternion).clone()

            if quat_is_xyzw:
                quaternion_wxyz = torch.roll(quaternion, shifts=1, dims=-1)
            else:
                quaternion_wxyz = quaternion

        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            position = self._to_curobo_device(position).clone()

        return Pose(position=position, quaternion=quaternion_wxyz, name=name, normalize_rotation=normalize_rotation)

    def _set_active_links(self, links: list[str], active: bool) -> None:
        """Configure collision checking for specific robot links.

        Enables or disables collision sphere checking for the specified links.
        This is essential for contact scenarios where certain links (like fingers
        or attachment points) need collision checking disabled to allow contact
        with objects being grasped.

        Args:
            links: List of link names to enable or disable collision checking for
            active: True to enable collision checking, False to disable
        """
        for link in links:
            if active:
                self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link)
            else:
                self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link)

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        start_joint_positions: torch.Tensor | None = None,
        retreat_distance: float | None = None,
        object_pose_overrides: Mapping[str, torch.Tensor] | None = None,
    ) -> bool:
        """Plan collision-free motion to target pose.

        Plans a trajectory from the current robot configuration to the specified target pose.
        The method assumes that world updates and locked joint configurations have already
        been handled. Supports optional linear retiming for consistent execution speeds.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            step_size: Step size for linear retiming, enables retiming if provided
            enable_retiming: Whether to enable linear retiming, auto-detected from step_size if None
            start_joint_positions: Optional arm-local planning state in the configured cuRobo
                joint order. ``None`` reads the live articulation state.
            retreat_distance: Optional per-call start-retreat override [m].
            object_pose_overrides: Optional environment-relative dynamic-object poses [m] retained
                for plan visualization during offline accumulation.

        Returns:
            True if planning succeeded and a valid trajectory was found, False otherwise
        """
        assert self.config.planning_enabled, "motion planning is disabled for this collision-only backend"
        self._failed_joint_state = None
        self._auxiliary_joint_positions_update(start_joint_positions)
        if enable_retiming is None:
            enable_retiming = step_size is not None

        # The caller supplies the target in the env-relative frame; cuRobo plans in the robot
        # base frame, so express it there before handing it to the solver.
        target_pose_base = self._pose_env_to_base(target_pose)

        # Ensure target pose is on cuRobo device (CUDA) for device isolation
        target_pose_cuda = self._to_curobo_device(target_pose_base)

        target_pos: torch.Tensor
        target_rot: torch.Tensor
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
        target_curobo_pose: Pose = self._make_pose(
            position=target_pos,
            quaternion=PoseUtils.quat_from_matrix(target_rot),
        )

        start_state = self._get_current_joint_state_for_curobo(start_joint_positions)

        self.logger.debug(f"Retiming enabled: {enable_retiming}, Step size: {step_size}")

        success: bool = self._plan_to_contact(
            start_state=start_state,
            goal_pose=target_curobo_pose,
            retreat_distance=self.config.retreat_distance if retreat_distance is None else retreat_distance,
            approach_distance=self.config.approach_distance,
            retime_plan=enable_retiming,
            step_size=step_size,
            contact=False,
        )

        # Visualize the successful trajectory, or at least retain the failed target and start state.
        if self.visualize_plan:
            self.plan_visualizer.robot_name = self.config.robot_name
            self.plan_visualizer.set_motion_generator_reference(self.motion_gen)
            # Get current spheres for visualization
            self._sync_object_poses_with_isaaclab(object_pose_overrides)
            cu_js = self._failed_joint_state if not success and self._failed_joint_state is not None else start_state
            cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
            sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)[0]

            # Split spheres into robot and attached object spheres
            robot_spheres = []
            attached_spheres = []
            robot_link_count = 0

            # Count robot link spheres
            robot_links = [
                link
                for link in self.robot_cfg["kinematics"]["collision_link_names"]
                if link != self.config.attached_object_link_name
            ]
            for link_name in robot_links:
                link_spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    robot_link_count += int(torch.sum(link_spheres[:, 3] > 0).item())

            # Split spheres
            for i, sphere in enumerate(sphere_list):
                if i < robot_link_count:
                    robot_spheres.append(sphere)
                else:
                    attached_spheres.append(sphere)

            try:
                world_scene = WorldConfig.get_scene_graph(self.motion_gen.world_coll_checker.world_model)
            except Exception:
                world_scene = None

            if not success or self._current_plan is None:
                self.plan_visualizer.visualize_failure(
                    target_pose=target_pose,
                    robot_spheres=robot_spheres,
                    attached_spheres=attached_spheres,
                    world_scene=world_scene,
                )
                return success

            viz_plan = self._current_plan.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            # Compute end-effector positions for visualization
            ee_positions_list = []
            try:
                for i in range(len(viz_plan.position)):
                    js: JointState = viz_plan[i]
                    kin = self.motion_gen.compute_kinematics(js)
                    ee_pos = kin.ee_position if hasattr(kin, "ee_position") else kin.ee_pose.position
                    ee_positions_list.append(ee_pos.cpu().numpy().squeeze())

                self.logger.debug(
                    f"Link names from kinematics: {kin.link_names if len(ee_positions_list) > 0 else 'No EE positions'}"
                )

            except Exception as e:
                self.logger.debug(f"Failed to compute EE positions for visualization: {e}")
                ee_positions_list = None

            # Visualize plan
            self.plan_visualizer.visualize_plan(
                plan=viz_plan,
                target_pose=target_pose,
                robot_spheres=robot_spheres,
                attached_spheres=attached_spheres,
                ee_positions=np.array(ee_positions_list) if ee_positions_list else None,
                world_scene=world_scene,
            )
            # Animate EE positions over the timeline for playback
            if ee_positions_list:
                self.plan_visualizer.animate_plan(
                    np.array(ee_positions_list),
                    timeline="planning",
                )

            # Animate spheres along the path for collision visualization
            self.plan_visualizer.animate_spheres_along_path(
                plan=viz_plan,
                robot_spheres_at_start=robot_spheres,
                attached_spheres_at_start=attached_spheres,
                timeline="planning",
                interpolation_steps=1,
            )

        return success

    def _plan_to_contact_pose(
        self,
        start_state: JointState,
        goal_pose: Pose,
        contact: bool = True,
    ) -> bool:
        """Plan motion with configurable collision checking for contact scenarios.

        Plans a trajectory while optionally disabling collision checking for hand links and
        attached objects. This is crucial for grasping and placing operations where contact
        is expected and collision checking would prevent successful planning.

        Args:
            start_state: Starting joint configuration for planning
            goal_pose: Target pose to reach in cuRobo coordinate frame
            contact: True to disable hand/attached object collisions for contact planning
            retime_plan: Whether to apply linear retiming to the resulting trajectory
            step_size: Step size for retiming if retime_plan is True

        Returns:
            True if planning succeeded, False if no valid trajectory found
        """
        # Use configured hand link names instead of hardcoded ones
        disable_link_names: list[str] = self.config.hand_link_names.copy()
        link_spheres: dict[str, torch.Tensor] = {}

        # Count spheres before planning
        sphere_counts_before = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres before - Total: {sphere_counts_before['total']}, Robot:"
            f" {sphere_counts_before['robot_links']}, Attached: {sphere_counts_before['attached_objects']}"
        )

        if contact:
            # Store current spheres for the attached link so we can restore later
            attached_links: list[str] = list(self.attachment_links)
            for attached_link in attached_links:
                link_spheres[attached_link] = self.motion_gen.kinematics.kinematics_config.get_link_spheres(
                    attached_link
                ).clone()

            self.logger.debug(f"Attached link: {attached_links}")
            # Disable all specified links for contact planning
            self.logger.debug(f"Disable link names: {disable_link_names}")
            self._set_active_links(disable_link_names + attached_links, active=False)
        else:
            self.logger.debug(f"Disable link names: {disable_link_names}")

        # Count spheres after link disabling
        sphere_counts_after_disable = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres after disable - Total:"
            f" {sphere_counts_after_disable['total']}, Robot: {sphere_counts_after_disable['robot_links']},"
            f" Attached: {sphere_counts_after_disable['attached_objects']}"
        )

        planning_success = False
        try:
            try:
                result: Any = self.motion_gen.plan_single(start_state, goal_pose, self.plan_config)
            finally:
                # cuRobo can return early from a world-colliding start without restoring this flag.
                self.motion_gen.rollout_fn.primitive_collision_constraint.enable_cost()
                self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()

            if result.success.item():
                # The optimizer's sample period can differ from the controller's fixed period.
                self._current_plan = result.get_interpolated_plan()
                self.logger.debug(f"Using interpolated plan with {len(self._current_plan.position)} waypoints")

                self._current_plan = self._planning_joint_state_expand(self._current_plan)
                common_js_names: list[str] = [
                    x for x in self.datastream.get_robot_joint_names() if x in self._current_plan.joint_names
                ]
                self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
                self._plan_index = 0

                planning_success = True
                self.logger.debug(f"Contact planning succeeded with {len(self._current_plan.position)} waypoints")
            else:
                self.logger.warning(f"Planning failed for group {self.config.planning_group_name!r}: {result.status}")
                if self.config.debug_planner:
                    self._log_closest_ik_candidate(result)
                    if result.status in {
                        MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
                        MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                    }:
                        self._log_start_collision(
                            start_state,
                            self_collision=result.status == MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
                        )

        except Exception as e:
            self.logger.warning(f"Planning errored for group {self.config.planning_group_name!r}: {e}")

        # Always restore sphere state after planning, regardless of success
        if contact:
            self._set_active_links(disable_link_names, active=True)
            for attached_link, spheres in link_spheres.items():
                self.motion_gen.kinematics.kinematics_config.update_link_spheres(attached_link, spheres)
        return planning_success

    def _log_closest_ik_candidate(self, result: Any) -> None:
        """Retain and report the closest IK candidate returned by a failed plan stage."""

        if not isinstance(result.debug_info, dict) or "ik_result" not in result.debug_info:
            return
        ik_result = result.debug_info["ik_result"]
        solver = self.motion_gen.ik_solver
        positions = ik_result.solution.reshape(-1, solver.dof)
        position_error = ik_result.position_error.reshape(-1)
        rotation_error = ik_result.rotation_error.reshape(-1)
        best = int(torch.argmin(position_error + rotation_error).item())
        self._failed_joint_state = JointState.from_position(
            positions[best].unsqueeze(0).clone(),
            joint_names=solver.joint_names,
        )
        self.logger.debug(
            "Closest IK candidate for group %r: position_error=%.6f m, rotation_error=%.6f rad",
            self.config.planning_group_name,
            float(position_error[best].item()),
            float(rotation_error[best].item()),
        )
        candidate = self._to_env_device(
            self._planning_joint_state_expand(self._failed_joint_state)
            .get_ordered_joint_state(self.active_joint_names)
            .position
        )
        for margin in dict.fromkeys((0.0, self.config.collision_activation_distance)):
            try:
                self_witness = self.joint_positions_self_collision_witness(candidate, margin)
                world_witness = self.joint_positions_world_collision_witness(candidate, margin)
                self.logger.debug(
                    "Closest IK candidate collision at %.6f m margin: self=%s, world=%s",
                    margin,
                    self_witness or "clear",
                    world_witness or "clear",
                )
            except Exception as error:
                self.logger.debug(
                    "Closest IK candidate collision attribution failed at %.6f m margin: %s: %s",
                    margin,
                    type(error).__name__,
                    error,
                )

    def _log_start_collision(self, start_state: JointState, *, self_collision: bool) -> None:
        """Report the exact collision behind a rejected start state."""

        try:
            ordered = self._planning_joint_state_expand(start_state).get_ordered_joint_state(self.active_joint_names)
            margin = self.config.collision_activation_distance
            positions = self._to_env_device(ordered.position)
            witness = (
                self.joint_positions_self_collision_witness(positions, margin)
                if self_collision
                else self.joint_positions_world_collision_witness(positions, margin)
            )
            kind = "self" if self_collision else "world"
            self.logger.debug(
                "Start-state %s-collision witness at %.6f m activation: %s",
                kind,
                margin,
                witness or "unattributed",
            )
        except Exception as error:
            self.logger.debug("Start-state collision attribution failed: %s: %s", type(error).__name__, error)

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Pose,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
    ) -> bool:
        """Execute multi-phase contact planning with approach and retreat phases.

        Implements a planning strategy for manipulation tasks that require approach and contact handling.
        Plans multiple trajectory segments with different collision checking configurations.

        Args:
            start_state: Starting joint state for planning
            goal_pose: Target pose to reach
            retreat_distance: Distance to retreat before transition to contact
            approach_distance: Distance to approach before final pose
            contact: Whether to enable contact planning mode
            retime_plan: Whether to retime the resulting plan
            step_size: Step size for retiming (only used if retime_plan is True)

        Returns:
            True if all planning phases succeeded, False if any phase failed
        """
        self.logger.debug(f"Multi-phase planning: retreat={retreat_distance}, approach={approach_distance}")

        target_poses: list[Pose] = []
        contacts: list[bool] = []
        direction = self.config.approach_direction
        in_world_frame = self.config.approach_retreat_frame == "world"
        world_sign = -1.0 if in_world_frame else 1.0

        if retreat_distance is not None and retreat_distance > 0:
            ee_pose: Pose = self.get_ee_pose(start_state)
            if in_world_frame:
                position = ee_pose.position.squeeze()
                offset = torch.tensor(
                    [world_sign * component * retreat_distance for component in direction],
                    device=position.device,
                    dtype=position.dtype,
                )
                retreat_pose = self._make_pose(
                    position=position + offset,
                    quaternion=ee_pose.quaternion,
                    quat_is_xyzw=False,
                )
            else:
                retreat_pose = ee_pose.multiply(
                    self._make_pose(position=[component * retreat_distance for component in direction])
                )
            target_poses.append(retreat_pose)
            contacts.append(True)
        contacts.append(contact)
        if approach_distance is not None and approach_distance > 0:
            if in_world_frame:
                position = goal_pose.position.squeeze()
                offset = torch.tensor(
                    [world_sign * component * approach_distance for component in direction],
                    device=position.device,
                    dtype=position.dtype,
                )
                approach_pose = self._make_pose(
                    position=position + offset,
                    quaternion=goal_pose.quaternion,
                    quat_is_xyzw=False,
                )
            else:
                approach_pose = goal_pose.multiply(
                    self._make_pose(position=[component * approach_distance for component in direction])
                )
            target_poses.append(approach_pose)
            contacts.append(True)

        target_poses.append(goal_pose)

        current_state: JointState = start_state
        full_plan: JointState | None = None

        for i, (target_pose, contact_flag) in enumerate(zip(target_poses, contacts)):
            self.logger.debug(
                f"Planning phase {i + 1} of {len(target_poses)}: contact={contact_flag} (collision"
                f" {'disabled' if contact_flag else 'enabled'})"
            )

            success: bool = self._plan_to_contact_pose(
                start_state=current_state,
                goal_pose=target_pose,
                contact=contact_flag,
            )

            if not success:
                self.logger.debug(f"Phase {i + 1} planning failed")
                return False

            if full_plan is None:
                full_plan = self._current_plan
            else:
                full_plan = full_plan.stack(self._current_plan)

            last_waypoint: torch.Tensor = self._current_plan.position[-1]
            current_state = JointState(
                position=last_waypoint.unsqueeze(0),
                velocity=torch.zeros_like(last_waypoint.unsqueeze(0)),
                acceleration=torch.zeros_like(last_waypoint.unsqueeze(0)),
                joint_names=self._current_plan.joint_names,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        self._current_plan = full_plan
        self._plan_index = 0

        if retime_plan and step_size is not None:
            original_length: int = len(self._current_plan.position)
            self._current_plan = self._linearly_retime_plan(step_size=step_size, plan=self._current_plan)
            self.logger.debug(
                f"Retimed complete plan from {original_length} to {len(self._current_plan.position)} waypoints"
            )

        self.logger.debug(f"Multi-phase planning succeeded with {len(self._current_plan.position)} total waypoints")

        return True

    def _linearly_retime_plan(
        self,
        step_size: float = 0.01,
        plan: JointState | None = None,
    ) -> JointState | None:
        """Apply linear retiming to trajectory for consistent execution speed.

        Resamples the trajectory with uniform spacing between waypoints to ensure
        consistent motion speed during execution.

        Args:
            step_size: Desired spacing between waypoints in joint space
            plan: Trajectory to retime, uses current plan if None

        Returns:
            Retimed trajectory with uniform waypoint spacing, or None if plan is invalid
        """
        if plan is None:
            plan = self._current_plan

        if plan is None or len(plan.position) == 0:
            return plan

        path = plan.position

        if len(path) <= 1:
            return plan
        assert path.ndim == 2, f"joint path must have shape (T, J), got {tuple(path.shape)}"

        deltas = path[1:] - path[:-1]
        distances = torch.linalg.norm(deltas, dim=-1)
        keep = torch.cat((torch.ones(1, dtype=torch.bool, device=path.device), distances > 1e-6))
        waypoints = path[keep]
        if len(waypoints) <= 1:
            return plan

        if len(waypoints) > 1:
            deltas = waypoints[1:] - waypoints[:-1]
            distances = torch.linalg.norm(deltas, dim=-1)
            cum_distances = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])

        if len(waypoints) < 2 or cum_distances[-1] < 1e-6:
            return plan

        total_distance = cum_distances[-1]
        num_steps = int(torch.ceil(total_distance / step_size).item()) + 1

        # Create linearly spaced distances
        sampled_distances = torch.linspace(cum_distances[0], cum_distances[-1], num_steps, device=cum_distances.device)

        # Linear interpolation
        indices = torch.searchsorted(cum_distances, sampled_distances)
        indices = torch.clamp(indices, 1, len(cum_distances) - 1)

        # Get interpolation weights
        weights = (sampled_distances - cum_distances[indices - 1]) / (
            cum_distances[indices] - cum_distances[indices - 1]
        )
        weights = weights.unsqueeze(-1)

        # Interpolate waypoints
        sampled_waypoints = (1 - weights) * waypoints[indices - 1] + weights * waypoints[indices]

        self.logger.debug(
            f"Retiming: {len(path)} to {len(sampled_waypoints)} waypoints, "
            f"Distance: {total_distance:.3f}, Step size: {step_size}"
        )

        retimed_plan = JointState(
            position=sampled_waypoints,
            velocity=torch.zeros(
                (len(sampled_waypoints), plan.velocity.shape[-1]),
                device=plan.velocity.device,
                dtype=plan.velocity.dtype,
            ),
            acceleration=torch.zeros(
                (len(sampled_waypoints), plan.acceleration.shape[-1]),
                device=plan.acceleration.device,
                dtype=plan.acceleration.dtype,
            ),
            joint_names=plan.joint_names,
        )

        return retimed_plan

    def has_next_waypoint(self) -> bool:
        """Check if more waypoints remain in the current trajectory.

        Returns:
            True if there are unprocessed waypoints, False if trajectory is complete or empty
        """
        return self._current_plan is not None and self._plan_index < len(self._current_plan.position)

    def get_next_waypoint_ee_pose(self) -> Pose:
        """Get end-effector pose for the next waypoint in the trajectory.

        Advances the trajectory execution index and computes the end-effector pose
        for the next waypoint using forward kinematics. The optimized trajectory is
        first projected into :attr:`active_joint_names` order because its storage order
        is not a kinematics API contract for projected humanoid chains.

        Returns:
            End-effector pose for the next waypoint in world coordinates

        Raises:
            IndexError: If no more waypoints remain in the trajectory
        """
        if not self.has_next_waypoint():
            raise IndexError("No more waypoints in the plan.")

        joint_positions = self.current_joint_position_path
        assert joint_positions is not None
        next_joint_positions = joint_positions[self._plan_index : self._plan_index + 1]
        self._plan_index += 1

        ee_pose_env = self.joint_positions_to_ee_poses(next_joint_positions)[0]
        env_pos, env_rot = PoseUtils.unmake_pose(ee_pose_env)
        return self._make_pose(position=env_pos, quaternion=PoseUtils.quat_from_matrix(env_rot))

    def reset_plan(self) -> None:
        """Reset trajectory execution state.

        Clears the current trajectory and resets the execution index to zero.
        This prepares the planner for a new planning operation.
        """
        self._plan_index = 0
        self._current_plan = None
        if self.visualize_plan and hasattr(self, "plan_visualizer"):
            self.plan_visualizer.mark_idle()

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Extract all end-effector poses from current trajectory.

        Computes end-effector poses for all waypoints without affecting the execution state.
        The plan is first ordered by :attr:`active_joint_names`; cuRobo's optimized trajectory
        storage order is not a kinematics API contract and can differ for projected humanoid
        chains. Optionally repeats the final pose when configured for stable goal reaching.

        Returns:
            List of end-effector poses as 4x4 transformation matrices, with optional repetition
        """
        joint_positions = self.current_joint_position_path
        if joint_positions is None:
            return []

        planned_poses = list(self.joint_positions_to_ee_poses(joint_positions).unbind(0))

        if self.n_repeat is not None and self.n_repeat > 0 and planned_poses:
            self.logger.info(f"Repeating final pose {self.n_repeat} times")
            planned_poses.extend(planned_poses[-1].clone() for _ in range(self.n_repeat))

        return planned_poses

    # =====================================================================================
    # VISUALIZATION METHODS
    # =====================================================================================

    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:
        """Update sphere visualization for the robot at specific joint positions.

        Args:
            joint_positions: Joint configuration to visualize collision spheres at
        """
        if not self.visualize_spheres:
            return

        self.frame_counter += 1
        if self.frame_counter % self.sphere_update_freq != 0:
            return

        # Debug-visualization only (gated by ``visualize_spheres``, off by default). This path
        # briefly *mutates* joint targets to render collision spheres, so it uses the raw
        # ``self.robot`` handle directly — the Datastream is a read interface and intentionally does
        # not expose actuation. Not part of the planning pipeline.
        original_joints: torch.Tensor = wp.to_torch(self.robot.data.joint_pos)[self.env_id].clone()

        try:
            # Ensure joint positions are on environment device for robot commands
            env_joint_positions = (
                self._to_env_device(joint_positions)
                if joint_positions.device != self.datastream.device
                else joint_positions
            )
            self.robot.set_joint_position_target(env_joint_positions.view(1, -1), env_ids=[self.env_id])
            self._update_sphere_visualization(force_update=False)
        finally:
            self.robot.set_joint_position_target(original_joints.unsqueeze(0), env_ids=[self.env_id])

    def _update_sphere_visualization(self, force_update: bool = True) -> None:
        """Update visual representation of robot collision spheres in USD stage.

        Creates or updates sphere primitives in the USD stage to show the robot's
        collision model. Different colors are used for robot links (green) and
        attached objects (orange) to help distinguish collision boundaries.

        Args:
            force_update: True to recreate all spheres, False to update existing positions only
        """
        # Get current sphere data
        cu_js = self._get_current_joint_state_for_curobo()
        sphere_position = self._to_curobo_device(
            cu_js.position if isinstance(cu_js.position, torch.Tensor) else torch.tensor(cu_js.position)
        )
        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]
        robot_link_count = self._get_robot_link_sphere_count()

        # Remove existing spheres if force update or first time
        if (self.spheres is None or force_update) and self.spheres is not None:
            self._remove_existing_spheres()

        # Initialize sphere list if needed
        if self.spheres is None or force_update:
            self.spheres = []

        # Create or update all spheres
        for sphere_idx, sphere in enumerate(sphere_list):
            if not self._is_valid_sphere(sphere):
                continue

            sphere_config = self._create_sphere_config(sphere_idx, sphere, robot_link_count)
            prim_path = f"/curobo/robot_sphere_{sphere_idx}"

            # Remove old sphere if updating
            if not (self.spheres is None or force_update):
                if sphere_idx < len(self.spheres) and self.usd_helper.stage.GetPrimAtPath(prim_path).IsValid():
                    self.usd_helper.stage.RemovePrim(prim_path)

            # Spawn sphere
            spawn_mesh_sphere(prim_path=prim_path, translation=sphere_config["position"], cfg=sphere_config["cfg"])

            # Store reference if creating new
            if self.spheres is None or force_update or sphere_idx >= len(self.spheres):
                self.spheres.append((prim_path, float(sphere.radius)))

    def _get_robot_link_sphere_count(self) -> int:
        """Calculate total number of collision spheres for robot links excluding attached objects.

        Iterates through all robot collision links (excluding the attached object link) and
        counts the active collision spheres for each link. This count is used to determine
        which spheres in the visualization represent robot links vs attached objects.

        Returns:
            Total number of active collision spheres for robot links only
        """
        sphere_config = self.motion_gen.kinematics.kinematics_config
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]
        return sum(
            int(torch.sum(sphere_config.get_link_spheres(link_name)[:, 3] > 0).item()) for link_name in robot_links
        )

    def _remove_existing_spheres(self) -> None:
        """Remove all existing sphere visualization primitives from the USD stage.

        Iterates through all stored sphere references and removes their corresponding
        USD primitives from the stage. This is used during force updates or when
        recreating the sphere visualization from scratch.
        """
        stage = self.usd_helper.stage
        for prim_path, _ in self.spheres:
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)

    def _is_valid_sphere(self, sphere) -> bool:
        """Validate sphere data for visualization rendering.

        Checks if a sphere has valid position coordinates (no NaN values) and a positive
        radius. Invalid spheres are skipped during visualization to prevent rendering errors.

        Args:
            sphere: Sphere object containing position and radius data

        Returns:
            True if sphere has valid position and positive radius, False otherwise
        """
        pos_tensor = torch.tensor(sphere.position, dtype=torch.float32)
        return not torch.isnan(pos_tensor).any() and sphere.radius > 0

    def _create_sphere_config(self, sphere_idx: int, sphere, robot_link_count: int) -> dict:
        """Create sphere configuration with position and visual properties for USD rendering.

        Determines sphere type (robot link vs attached object), calculates world position,
        and creates the appropriate visual configuration including colors and materials.
        Robot link spheres are green with lower opacity, while attached object spheres
        are orange with higher opacity for better distinction.

        Args:
            sphere_idx: Index of the sphere in the sphere list
            sphere: Sphere object containing position and radius data
            robot_link_count: Total number of robot link spheres (for type determination)

        Returns:
            Dictionary containing 'position' (world coordinates) and 'cfg' (MeshSphereCfg)
        """

        is_attached = sphere_idx >= robot_link_count
        color = (1.0, 0.5, 0.0) if is_attached else (0.0, 1.0, 0.0)
        opacity = 0.9 if is_attached else 0.5

        # Calculate position in world frame (do not use env_origin)
        root_translation = wp.to_torch(self.robot.data.root_pos_w)[self.env_id, :3].detach().cpu().numpy()
        position = sphere.position.cpu().numpy() if hasattr(sphere.position, "cpu") else sphere.position
        if not is_attached:
            position = position + root_translation

        return {
            "position": position,
            "cfg": MeshSphereCfg(
                radius=float(sphere.radius),
                visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, emissive_color=color),
            ),
        }

    def _is_sphere_attached_object(self, sphere_index: int, sphere_config: Any) -> bool:
        """Check if a sphere belongs to attached_object link.

        Args:
            sphere_index: Index of the sphere to check
            sphere_config: Sphere configuration object

        Returns:
            True if sphere belongs to an attached object, False if it's a robot link sphere
        """
        # Get total number of robot link spheres (excluding attached_object)
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]

        total_robot_spheres = 0
        for link_name in robot_links:
            try:
                link_spheres = sphere_config.get_link_spheres(link_name)
                active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                total_robot_spheres += int(active_spheres)
            except Exception:
                continue

        # If sphere_index >= total_robot_spheres, it's an attached object sphere
        is_attached = sphere_index >= total_robot_spheres

        if sphere_index < 5:  # Debug first few spheres
            self.logger.debug(
                f"SPHERE {sphere_index}: total_robot_spheres={total_robot_spheres}, is_attached={is_attached}"
            )

        return is_attached

    # =====================================================================================
    # HIGH-LEVEL PLANNING INTERFACE
    # =====================================================================================

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        start_joint_positions: torch.Tensor | None = None,
        force_object_attachment: bool = False,
        retreat_distance: float | None = None,
        object_pose_overrides: Mapping[str, torch.Tensor] | None = None,
    ) -> bool:
        """Complete planning pipeline with world updates and object attachment handling.

        Provides a high-level interface that handles the complete planning workflow:
        world synchronization, object attachment/detachment, gripper configuration,
        and motion planning.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            expected_attached_object: Name of object that should be attached, None for no attachment
            env_id: Environment ID for multi-environment setups
            step_size: Step size for linear retiming if retiming is enabled
            enable_retiming: Whether to enable linear retiming of trajectory
            start_joint_positions: Optional arm-local state for offline path accumulation.
            force_object_attachment: Attach the expected object at ``start_joint_positions``
                without consulting the live hand state. Supply its virtual grasp pose through
                ``object_pose_overrides``.
            retreat_distance: Optional per-call start-retreat override [m].
            object_pose_overrides: Optional environment-relative dynamic-object poses [m] used
                instead of live poses during offline path accumulation.

        Returns:
            True if complete planning pipeline succeeded, False if any step failed
        """
        assert self.config.planning_enabled, "motion planning is disabled for this collision-only backend"
        # Always reset the plan before starting a new one to ensure a clean state
        self.reset_plan()

        self.logger.debug("=== MOTION PLANNING DEBUG ===")
        self.logger.debug(f"Expected attached object: {expected_attached_object}")

        self.update_world(object_pose_overrides)
        gripper_closed = expected_attached_object is not None
        self._set_gripper_state(gripper_closed)
        current_attached = self.get_attached_objects()
        gripper_pos = self.datastream.get_robot_joint_positions(env_ids=[env_id])[0, -2:]

        self.logger.debug(f"Current attached objects: {current_attached}")

        # Attach object if expected but not currently attached
        if expected_attached_object and expected_attached_object not in current_attached:
            self.logger.debug(f"Need to attach {expected_attached_object}")

            object_mappings = self._get_object_mappings()

            self.logger.debug(f"Object mappings found: {list(object_mappings.keys())}")

            if expected_attached_object in object_mappings:
                expected_path = object_mappings[expected_attached_object]

                self.logger.debug(f"Object path: {expected_path}")

                # Debug object pose (read through the Datastream, env-relative frame).
                object_poses = self.datastream.get_object_poses(env_ids=[env_id])
                if expected_attached_object in object_poses:
                    obj_pos = object_poses[expected_attached_object][0][:3, 3]
                    self.logger.debug(f"Object position (env-relative): {obj_pos}")
                    self.logger.debug(
                        f"Gripper positions: {gripper_pos}; open val: {self.config.grasp_gripper_open_val}"
                    )

                is_grasped = force_object_attachment or self._check_object_grasped(
                    gripper_pos, expected_attached_object
                )

                self.logger.debug(f"Is grasped check result: {is_grasped}")

                if is_grasped:
                    attached = self._attach_object(
                        expected_attached_object,
                        expected_path,
                        start_joint_positions=start_joint_positions,
                    )
                    if force_object_attachment and not attached:
                        if self.has_attached_objects():
                            self._detach_objects()
                        return False
                    self.logger.debug(f"Attached {expected_attached_object}")
                else:
                    self.logger.debug(
                        "Object not detected as grasped - attachment skipped"
                    )  # This will cause collision with ghost object!
            else:
                self.logger.debug(f"Object {expected_attached_object} not found in world mappings")
                if force_object_attachment:
                    if self.has_attached_objects():
                        self._detach_objects()
                    return False

        # Detach objects if no object should be attached (i.e., placing/releasing)
        if expected_attached_object is None and current_attached:
            self.logger.debug("Detaching all objects as no object expected to be attached")
            self._detach_objects()

        self.logger.debug(f"Planning motion with attached objects: {self.get_attached_objects()}")

        try:
            plan_success = self.plan_motion(
                target_pose,
                step_size,
                enable_retiming,
                start_joint_positions=start_joint_positions,
                retreat_distance=retreat_distance,
                object_pose_overrides=object_pose_overrides,
            )
            self.logger.debug(f"Planning result: {plan_success}")
            self.logger.debug("=== END POST-GRASP DEBUG ===")
            return plan_success
        finally:
            self._detach_objects()

    # =====================================================================================
    # UTILITY METHODS
    # =====================================================================================

    def _check_object_grasped(self, gripper_pos: torch.Tensor, object_name: str) -> bool:
        """Check if a specific object is currently grasped by the robot.

        Uses gripper position to determine if an object is grasped.

        Args:
            gripper_pos: Gripper position tensor
            object_name: Name of object to check (e.g., "cube_1")

        Returns:
            True if object is detected as grasped
        """
        gripper_open_val = self.config.grasp_gripper_open_val
        object_grasped = gripper_pos[0].item() < gripper_open_val

        self.logger.info(
            f"Object {object_name} is grasped: {object_grasped}"
            if object_grasped
            else f"Object {object_name} is not grasped"
        )

        return object_grasped

    def _set_gripper_state(self, has_attached_objects: bool) -> None:
        """Configure gripper joint positions based on object attachment status.

        Sets the gripper to closed position when objects are attached and open position
        when no objects are attached. This ensures proper collision checking and planning
        with the correct gripper configuration.

        Args:
            has_attached_objects: True if robot currently has attached objects requiring closed gripper
        """
        if has_attached_objects:
            # Closed gripper for grasping
            locked_joints = self.config.gripper_closed_positions
        else:
            # Open gripper for manipulation
            locked_joints = self.config.gripper_open_positions

        if not locked_joints:
            return
        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)

    def _count_active_spheres(self) -> dict[str, int]:
        """Count active collision spheres by category for debugging.

        Analyzes the current collision sphere configuration to provide detailed
        statistics about robot links vs attached object spheres. This is helpful
        for debugging collision checking issues and attachment problems.

        Returns:
            Dictionary containing sphere counts by category (total, robot_links, attached_objects)
        """
        cu_js = self._get_current_joint_state_for_curobo()

        # Ensure position tensor is on CUDA for cuRobo
        if isinstance(cu_js.position, torch.Tensor):
            sphere_position = self._to_curobo_device(cu_js.position)
        else:
            # Convert list to tensor and move to CUDA
            sphere_position = self._to_curobo_device(torch.tensor(cu_js.position))

        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]

        # Get sphere configuration
        sphere_config = self.motion_gen.kinematics.kinematics_config

        # Count robot link spheres (excluding attached_object)
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]
        robot_sphere_count = 0
        for link_name in robot_links:
            if hasattr(sphere_config, "get_link_spheres"):
                link_spheres = sphere_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                    robot_sphere_count += int(active_spheres)

        # Count attached object spheres by checking actual sphere list
        attached_sphere_count = 0

        # Handle sphere_list as either a list or single Sphere object
        total_spheres = len(list(sphere_list))

        # Any spheres beyond robot_sphere_count are attached object spheres
        attached_sphere_count = max(0, total_spheres - robot_sphere_count)

        self.logger.debug(
            f"SPHERE COUNT: Total={total_spheres}, Robot={robot_sphere_count},Attached={attached_sphere_count}"
        )

        return {
            "total": total_spheres,
            "robot_links": robot_sphere_count,
            "attached_objects": attached_sphere_count,
        }
