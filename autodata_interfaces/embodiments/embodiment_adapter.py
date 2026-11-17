# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embodiment Adapter ABC — the *transforms* interface.

Wraps an embodiment (typically an Arena ``EmbodimentBase``) and exposes the
pose ↔ action transformation surface plus EEF metadata. This is the only place
in the framework that knows the kinematics of a specific robot. It never
inspects task semantics or scene contents — pure embodiment-side.

Implementations populate the embodiment-side fields of :class:`StaticInfo` and
:class:`CoreStep`.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autodata_interfaces.embodiments.embodiment_types import ActionTargetFrame, NamedJointTrajectory
from autodata_utils import pose_math
from autodata_utils.tensor_utils import as_torch


class EmbodimentAdapter(ABC):
    """Abstract base class for embodiment adapters.

    Concrete subclasses bridge a particular robot abstraction (Arena
    ``EmbodimentBase``, a custom URDF wrapper, etc.) to the Datastream's
    embodiment-side fields.

    Attributes:
        env: Live env handle, bound post-construction via :meth:`bind_env`. ``None`` until bound.
        robot_asset_name: Scene key of the robot articulation this embodiment drives. Used by
            joint-state queries. Subclasses may override (e.g. expose it as a config field).
        raw_config: The embodiment's full parsed YAML/dict config, attached once by
            :func:`~.factory.embodiment_adapter_from_dict` (``None`` for an adapter built directly,
            not through the factory). Lets a consumer read a section the typed adapter fields don't
            model -- e.g. retargeting's ``hand_open``/``hand_close`` hand postures and ``eef_offset``/
            ``eef_rotation``/``eefs.<eef>.offset`` grasp-frame alignment (see
            ``autodata_retargeting.gripper_retargeting.hand_postures_from_adapter`` and
            ``autodata_retargeting.eef_offset.eef_offset_from_adapter``) -- without re-opening and
            re-parsing the same YAML file a second time. Note ``eef_offset`` on
            :class:`~.single_arm_embodiment_adapter.SingleArmEmbodimentAdapter` is this exact
            control-link -> observed-EEF translation read back out of ``raw_config`` by retargeting,
            not a second, unrelated value of the same name.
    """

    env: Any = None
    robot_asset_name: str = "robot"
    raw_config: dict[str, Any] | None = None

    def metric_context(self) -> dict[str, Any]:
        """Describe this embodiment for the dataset metrics; the description is saved with every dataset it records.

        The base returns the optional ``hardware_limits`` section of the embodiment config: datasheet joint limits
        keyed by joint-name pattern. Subclasses add ``pose_action``, which says for each end effector which part of
        the action vector commands its pose (``action_slice``, a half-open ``[start, end)`` index range), the
        value the adapter clips those entries to (``bound``, or None when they are not clipped), and whether they
        command an absolute target pose or a change of pose (``absolute``).
        """
        hardware_limits = (self.raw_config or {}).get("hardware_limits")
        return {"hardware_limits": hardware_limits} if hardware_limits else {}

    def get_navigation_eef_name(self) -> str | None:
        """Return the configured navigation channel without requiring a live environment."""

    def get_navigation_state(self, env_id: int) -> tuple[str, bool, bool] | None:
        """Return a navigation channel and its active/goal flags, or None when unavailable."""

    def get_reference_frame_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor | None:
        """Return the controller frame in environment coordinates, or None for environment coordinates.

        Args:
            env_ids: Environment indices to query. None selects all environments.
        """

    def bind_env(self, env: Any) -> None:
        """Attach the env after construction.

        Asserts the env was not previously bound — call exactly once.
        """

        assert self.env is None, "env already bound"
        self.env = env

    # ------------------------------------------------------------------
    # Joint-state queries (robot-kinematics state)
    # ------------------------------------------------------------------
    # The embodiment adapter is the framework's single owner of a specific robot's kinematics,
    # so consumers that need the raw joint configuration (e.g. a motion planner's start state)
    # read it here rather than reaching into the env's articulation directly.

    def get_joint_positions(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        joint_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Read joint positions [rad, m], optionally in an explicit name order.

        Args:
            env_ids: Environment indices; None selects all environments.
            joint_names: Ordered subset of articulation joints. None preserves native order.

        Returns:
            Positions shaped ``(num_selected_envs, num_selected_joints)``.
        """

        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot = self.env.scene[self.robot_asset_name]
        positions = as_torch(robot.data.joint_pos)[index]
        if joint_names is None:
            return positions
        by_name = {name: index for index, name in enumerate(self.get_joint_names())}
        missing = set(joint_names) - set(by_name)
        assert not missing, f"joints are absent from the articulation: {sorted(missing)}"
        assert len(joint_names) == len(set(joint_names)), "requested joint names must be unique"
        return positions[:, [by_name[name] for name in joint_names]]

    def get_joint_position_limits(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read the robot's live lower and upper joint position limits [rad].

        Returns a tensor of shape ``(len(env_ids), num_dof, 2)`` in the articulation's native
        joint order (see :meth:`get_joint_names`).
        """

        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot = self.env.scene[self.robot_asset_name]
        return as_torch(robot.data.joint_pos_limits)[index]

    def get_joint_velocity_limits(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read live maximum joint speeds [rad/s, m/s] in articulation order.

        Args:
            env_ids: Environment indices; None selects all environments.

        Returns:
            Maximum speeds shaped ``(num_selected_envs, num_joints)``.
        """
        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        return as_torch(self.env.scene[self.robot_asset_name].data.joint_vel_limits)[index]

    def get_joint_names(self) -> list[str]:
        """Return articulation joint names in native order."""

        assert self.env is not None, "Call bind_env(env) before reading state."
        return list(self.env.scene[self.robot_asset_name].data.joint_names)

    def get_planning_model_path(self) -> Path | None:
        """Return the live controller's local robot model, when it exposes one."""

    def get_planning_root_pose(
        self,
        root_pose_source_frame_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Read a named articulation-link pose in each environment frame.

        Args:
            root_pose_source_frame_name: Robot frame used as the planning root.
            env_ids: Environment indices, or None for all environments.

        Returns:
            Environment-relative root transforms with shape (N, 4, 4) [m].
        """

        assert self.env is not None, "Call bind_env(env) before reading state."
        robot = self.env.scene[self.robot_asset_name]
        body_names = tuple(robot.data.body_names)
        assert root_pose_source_frame_name in body_names, f"unknown planning-root frame {root_pose_source_frame_name!r}"
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        body_index = body_names.index(root_pose_source_frame_name)
        body_state = as_torch(robot.data.body_link_state_w)[index, body_index, :7]
        position = body_state[:, :3] - self.env.scene.env_origins[index]
        return pose_math.make_pose(position, pose_math.matrix_from_quat(body_state[:, 3:7]))

    @abstractmethod
    def get_eef_names(self) -> tuple[str, ...]:
        """Return the ordered names of all end-effectors this embodiment exposes.

        Returns:
            Tuple of EEF names. Stable across calls; never empty.
        """

    def get_eef_joint_names(self, eef_name: str) -> tuple[str, ...]:
        """Return positional passthrough joints owned by an end effector.

        Args:
            eef_name: End effector to query.

        Returns:
            Joint names in action order, or an empty tuple when unavailable.
        """

        assert eef_name in self.get_eef_names(), f"unknown end-effector {eef_name!r}"
        return ()

    def get_eef_passthrough_preset(self, eef_name: str, preset_name: str) -> tuple[float, ...]:
        """Return a positional passthrough preset for an end effector.

        Args:
            eef_name: End effector to query.
            preset_name: Configured posture name.

        Returns:
            Joint positions in the end effector's passthrough-action order [rad, m].

        Raises:
            KeyError: The embodiment does not define the requested preset.
        """

        assert eef_name in self.get_eef_names(), f"unknown end-effector {eef_name!r}"
        assert isinstance(preset_name, str) and preset_name, "preset_name must be a non-empty string"
        raise KeyError(f"end-effector {eef_name!r} has no passthrough preset {preset_name!r}")

    def get_action_target_frames(self) -> tuple[ActionTargetFrame, ...]:
        """Return exact controlled frames in pose-action block order, when available."""

        return ()

    def solve_joint_trajectory(
        self,
        target_eef_poses: Mapping[str, torch.Tensor],
        joint_seed_trajectories: Mapping[str, NamedJointTrajectory],
        initial_joint_positions: torch.Tensor,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        joint_state_is_valid: Callable[[tuple[str, ...], torch.Tensor, int], bool] | None = None,
        env_id: int = 0,
    ) -> NamedJointTrajectory | None:
        """Solve joint positions for synchronized end-effector targets using the controller's kinematics.

        This optional capability is used when independently accumulated planning groups share
        joints. Returning ``None`` declares that the embodiment has no shared-joint solver.

        Args:
            target_eef_poses: Per-EEF target poses with shape ``(T, 4, 4)`` [m].
            joint_seed_trajectories: Per-EEF named planner proposals with ``T`` samples.
            initial_joint_positions: Initial articulation state in :meth:`get_joint_names` order
                [rad, m].
            position_tolerance_m: Required EEF position convergence [m].
            rotation_tolerance_rad: Required EEF rotation convergence [rad].
            joint_state_is_valid: Optional collision or feasibility predicate for one computed joint sample.
            env_id: Environment whose controller solves the target poses.

        Returns:
            Controller-owned named joint trajectory, or ``None`` when unsupported.
        """

        del target_eef_poses, joint_seed_trajectories, initial_joint_positions
        del position_tolerance_m, rotation_tolerance_rad, joint_state_is_valid, env_id

    def set_joint_posture_target(
        self,
        joint_names: tuple[str, ...],
        joint_positions: torch.Tensor,
        env_id: int = 0,
    ) -> None:
        """Set an optional controller posture reference for the next action step.

        Args:
            joint_names: Names aligned with ``joint_positions``.
            joint_positions: One named posture target [rad, m].
            env_id: Environment controller to update.
        """

        del joint_names, joint_positions, env_id

    def reset_joint_posture_target(self, env_id: int = 0) -> None:
        """Restore the controller's default posture reference after guided replay.

        Args:
            env_id: Environment controller to update.
        """

        del env_id

    def get_passthrough_actions(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return current command-compatible passthrough actions when recoverable from live state.

        Args:
            env_ids: Environment indices, or None for all environments.

        Returns:
            Batched tensors with shape (N, channel_width) keyed by action channel.
            None means that at least one required channel cannot be recovered.
        """

        del env_ids

    @abstractmethod
    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Read current end-effector poses [m, m, m] from the live env.

        Args:
            env_ids: Environment indices to query. If ``None``, all envs are
                considered.

        Returns:
            Map ``eef_name → pose tensor`` of shape ``(len(env_ids), 4, 4)``.
        """

    @abstractmethod
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse: convert an env action into per-EEF target poses [m, m, m].

        Used to infer the target controller pose sequence from a recorded
        demonstration's action stream.

        Args:
            action: Env action tensor of shape ``(num_envs, action_dim)``.

        Returns:
            Map ``eef_name → target pose tensor`` of shape ``(num_envs, 4, 4)``.
        """

    @abstractmethod
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Forward: convert per-EEF target poses + passthrough actions into an env action.

        Passthrough actions are the non-pose action dimensions copied verbatim from the
        source demo (e.g. eef grippers/hands, mobile base command, etc.).

        Args:
            target_eef_pose_dict: Map ``eef_name → target pose [m, m, m]`` of
                shape ``(4, 4)``.
            passthrough_action_dict: Map ``channel_name → passthrough action tensor``.
                Channel names are the eef names plus any non-eef passthrough channels the embodiment
                declares.
            action_noise_dict: Map ``eef_name → action noise scale``. ``None``
                means no noise.
            env_id: Environment index to compute the action for.

        Returns:
            Action tensor compatible with ``env.step()``, shape ``(action_dim,)``.
        """

    @abstractmethod
    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract passthrough-action slices from a sequence of env actions.

        Inverse of the passthrough half of :meth:`target_eef_pose_to_action`. Pulls every
        passthrough channel out of the action so the generator can replay them verbatim.

        Args:
            actions: Action tensor of shape ``(num_envs, num_steps, action_dim)``.

        Returns:
            Map ``channel_name → passthrough action tensor``.
        """

    def passthrough_actions_to_joint_positions(
        self,
        passthrough_action_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map positional passthrough channels to named joints when supported."""

        return {}

    def joint_positions_to_passthrough_actions(
        self,
        joint_position_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Project named planned joints into positional passthrough channels when supported."""

        del joint_position_dict
        return {name: value.clone() for name, value in passthrough_action_dict.items()}
