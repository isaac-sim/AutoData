# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Defer Isaac Lab imports to type-checking time so this module can be imported in
    # sim-free contexts (e.g. unit tests that exercise only config or version detection).
    from isaac_autodata_interfaces.datastream.datastream import Datastream


class MotionPlannerBase(ABC):
    """Abstract base class for motion planners.

    This class defines the public interface that all motion planners must implement.
    It focuses on the essential functionality that users interact with, while leaving
    implementation details to specific planner backends.

    Planners read all world state (robot joint configuration, obstacle poses, the collision
    geometry source) through the :class:`Datastream` interface rather than touching the env or
    robot articulation directly, so a backend stays agnostic to the simulator wiring.

    The core workflow is:
    1. Initialize planner with a Datastream and an env index.
    2. Call update_world_and_plan_motion() to plan to a target.
    3. Execute plan using has_next_waypoint() and get_next_waypoint_ee_pose().

    Example:
        >>> from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
        >>> from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        >>> config = CuroboPlannerCfg.franka_config()
        >>> planner = CuroboPlanner(datastream, config, env_id=0)
        >>> success = planner.update_world_and_plan_motion(target_pose)
        >>> if success:
        >>>     while planner.has_next_waypoint():
        >>>         action = planner.get_next_waypoint_ee_pose()
    """

    def __init__(
        self,
        datastream: Datastream,
        env_id: int = 0,
        debug: bool = False,
        robot_asset_name: str = "robot",
        **kwargs,
    ) -> None:
        """Initialize the motion planner.

        Args:
            datastream: Composed read interface over the env, task descriptor, embodiment adapter,
                and source pool. The sole entry point for world state.
            env_id: Environment ID (0 to num_envs-1).
            debug: Whether to print detailed debugging information.
            robot_asset_name: Scene key of the robot articulation, used to derive the raw-handle
                escape hatches below.
            **kwargs: Additional planner-specific arguments.
        """
        self.datastream = datastream
        # Raw-handle escape hatches for backends not yet fully migrated to Datastream reads
        # (the v1 cuRobo backend still extracts geometry / mutates joints via these). New
        # backends read exclusively through ``self.datastream`` and ignore these.
        self.env = datastream.get_env()
        self.robot = self.env.scene[robot_asset_name]
        self.env_id = env_id
        self.debug = debug

    @abstractmethod
    def update_world_and_plan_motion(self, target_pose: torch.Tensor, **kwargs: Any) -> bool:
        """Update collision world and plan motion to target pose.

        This is the main entry point for motion planning. It should:
        1. Update the planner's internal world representation
        2. Plan a collision-free path to the target pose
        3. Store the plan internally for execution

        Args:
            target_pose: Target pose to plan motion to (4x4 transformation matrix)
            **kwargs: Planner-specific arguments (e.g., retiming, contact planning)

        Returns:
            bool: True if planning succeeded, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def has_next_waypoint(self) -> bool:
        """Check if there are more waypoints in current plan.

        Returns:
            bool: True if there are more waypoints, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_waypoint_ee_pose(self) -> Any:
        """Get next waypoint's end-effector pose from current plan.

        This method should only be called after checking has_next_waypoint().

        Returns:
            Any: End-effector pose for the next waypoint in the plan.
        """
        raise NotImplementedError

    def get_planned_poses(self) -> list[Any]:
        """Get all planned poses from current plan.

        Returns:
            list[Any]: List of planned poses.

        Note:
            Default implementation iterates through waypoints.
            Child classes can override for a more efficient implementation.
        """
        planned_poses = []
        # Create a copy of the planner state to not affect the original plan execution
        # This is a placeholder and may need to be implemented by child classes
        # if they manage complex internal state.
        # For now, we assume the planner can be reset and we can iterate through the plan.
        # A more robust solution might involve a dedicated method to get the full plan.
        self.reset_plan()
        while self.has_next_waypoint():
            pose = self.get_next_waypoint_ee_pose()
            planned_poses.append(pose)
        return planned_poses

    @abstractmethod
    def reset_plan(self) -> None:
        """Reset the current plan and execution state.

        This should clear any stored plan and reset the execution index or iterator.
        """
        raise NotImplementedError

    def get_planner_info(self) -> dict[str, Any]:
        """Get information about the planner.

        Returns:
            dict: Information about the planner (name, version, capabilities, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "env_id": self.env_id,
            "debug": self.debug,
        }
