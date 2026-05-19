# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
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

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class EmbodimentAdapter(ABC):
    """Abstract base class for embodiment adapters.

    Concrete subclasses bridge a particular robot abstraction (Arena
    ``EmbodimentBase``, a custom URDF wrapper, etc.) to the Datastream's
    embodiment-side fields. See
    :meth:`isaac_autodata_core.coordinator.Coordinator` for how
    instances are composed with :class:`TaskDescriptor` and :class:`SceneProbe`.
    """

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
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Forward: convert per-EEF target poses + gripper actions into an env action.

        Args:
            target_eef_pose_dict: Map ``eef_name → target pose [m, m, m]`` of
                shape ``(4, 4)``.
            gripper_action_dict: Map ``eef_name → gripper action tensor``.
            action_noise_dict: Map ``eef_name → action noise scale``. ``None``
                means no noise.
            env_id: Environment index to compute the action for.

        Returns:
            Action tensor compatible with ``env.step()``, shape ``(action_dim,)``.
        """

    @abstractmethod
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper-actuation slices from a sequence of env actions.

        Args:
            actions: Action tensor of shape ``(num_envs, num_steps, action_dim)``.

        Returns:
            Map ``eef_name → gripper action tensor``.
        """
