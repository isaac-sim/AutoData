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
from collections.abc import Sequence
from typing import Any

from isaac_autodata_utils.tensor_utils import as_torch


class EmbodimentAdapter(ABC):
    """Abstract base class for embodiment adapters.

    Concrete subclasses bridge a particular robot abstraction (Arena
    ``EmbodimentBase``, a custom URDF wrapper, etc.) to the Datastream's
    embodiment-side fields.

    Attributes:
        env: Live env handle, bound post-construction via :meth:`bind_env`. ``None`` until bound.
        robot_asset_name: Scene key of the robot articulation this embodiment drives. Used by
            joint-state queries. Subclasses may override (e.g. expose it as a config field).
    """

    env: Any = None
    robot_asset_name: str = "robot"

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

    def get_joint_positions(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read the robot's current joint positions [rad] from the live env.

        Returns a tensor of shape ``(len(env_ids), num_dof)`` in the articulation's native joint
        order (see :meth:`get_joint_names`).
        """

        assert self.env is not None, "Call bind_env(env) before reading state."
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot = self.env.scene[self.robot_asset_name]
        return as_torch(robot.data.joint_pos)[index]

    def get_joint_names(self) -> list[str]:
        """Return the robot articulation's joint names, ordered to match :meth:`get_joint_positions`."""

        assert self.env is not None, "Call bind_env(env) before reading state."
        return list(self.env.scene[self.robot_asset_name].data.joint_names)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor] | None:
        """Optionally return object poses in the controller frame.

        Most adapters leave scene-object reads to :class:`Datastream`. Mobile embodiments whose
        controller frame differs from the environment origin override this method so object and
        end-effector transforms cannot silently use different frames.
        """

        del env_ids

    @abstractmethod
    def get_eef_names(self) -> tuple[str, ...]:
        """Return the ordered names of all end-effectors this embodiment exposes.

        Returns:
            Tuple of EEF names. Stable across calls; never empty.
        """

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
