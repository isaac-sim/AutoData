# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Waypoint and trajectory primitives consumed by :class:`DataGenerator`.

* :class:`Waypoint` — one target pose + gripper action.
* :class:`WaypointSequence` — ordered list of :class:`Waypoint`.
* :class:`WaypointTrajectory` — list of :class:`WaypointSequence`, supports merge with interpolation.
* :class:`MultiWaypoint` — per-EEF :class:`Waypoint` map; ``execute`` issues one env step.
"""

from __future__ import annotations

import asyncio
import torch
from copy import deepcopy
from typing import TYPE_CHECKING

import isaaclab.utils.math as PoseUtils
from isaaclab.managers import TerminationTermCfg

if TYPE_CHECKING:
    from isaac_autodata_interfaces.datastream.datastream import Datastream


class Waypoint:
    """Single 6-DoF target pose with paired gripper action."""

    def __init__(
        self,
        pose: torch.Tensor,
        gripper_action: torch.Tensor,
        noise: float | torch.Tensor | None = None,
        joint_seed: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            pose: 4x4 pose target [m, rad].
            gripper_action: Gripper command of shape ``[D]``.
            noise: Action-noise amplitude applied to arm actions at this tick. Gripper actions are
                not noised.
            joint_seed: Optional joint-space seed; used by IK and by offline scheduling.
        """
        self.pose = pose
        self.gripper_action = gripper_action
        self.noise = noise
        self.joint_seed = joint_seed

    def __str__(self) -> str:
        return f"Waypoint:\n  Pose:\n{self.pose}\n"


class WaypointSequence:
    """Ordered list of :class:`Waypoint`."""

    def __init__(self, sequence: list[Waypoint] | None = None) -> None:
        if sequence is None:
            self.sequence: list[Waypoint] = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(
        cls,
        poses: torch.Tensor,
        gripper_actions: torch.Tensor,
        action_noise: float | torch.Tensor,
        joint_positions: list[torch.Tensor] | None = None,
    ) -> WaypointSequence:
        """Build a sequence from parallel pose, gripper-action, and noise tensors.

        Args:
            poses: ``[T, 4, 4]`` pose matrices [m, rad].
            gripper_actions: ``[T, D]`` gripper commands.
            action_noise: Either a scalar broadcast across time, or a ``[T]`` / ``[T, 1]`` tensor.
            joint_positions: Optional per-step joint seeds; element ``t`` aligns with ``poses[t]``.
        """
        assert isinstance(action_noise, (float, torch.Tensor))

        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * torch.ones((num_timesteps, 1), dtype=torch.float32)
        action_noise = action_noise.reshape(-1, 1)

        sequence: list[Waypoint] = []
        for t in range(num_timesteps):
            joint_seed = None
            if joint_positions is not None and t < len(joint_positions):
                joint_seed = joint_positions[t]
            sequence.append(
                Waypoint(
                    pose=poses[t],
                    gripper_action=gripper_actions[t],
                    noise=action_noise[t, 0],
                    joint_seed=joint_seed,
                )
            )
        return cls(sequence=sequence)

    def get_poses(self) -> list[torch.Tensor]:
        return [w.pose[:2, 3] for w in self.sequence]

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, ind: int) -> Waypoint:
        return self.sequence[ind]

    def __add__(self, other: WaypointSequence) -> WaypointSequence:
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    def __str__(self) -> str:
        return "\n".join(f"Waypoint {i}: {w}" for i, w in enumerate(self.sequence))

    @property
    def last_waypoint(self) -> Waypoint:
        return deepcopy(self.sequence[-1])

    def split(self, ind: int) -> tuple[WaypointSequence, WaypointSequence]:
        """Split at ``ind``; returns ``(prefix, suffix)``."""
        return WaypointSequence(sequence=self.sequence[:ind]), WaypointSequence(sequence=self.sequence[ind:])


class WaypointTrajectory:
    """Ordered list of :class:`WaypointSequence` representing a full 6-DoF trajectory."""

    def __init__(self) -> None:
        self.waypoint_sequences: list[WaypointSequence] = []

    def __len__(self) -> int:
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind: int) -> Waypoint:
        assert len(self.waypoint_sequences) > 0
        assert 0 <= ind < len(self)

        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if start_ind <= ind < end_ind:
                break
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self) -> Waypoint:
        return self.waypoint_sequences[-1].last_waypoint

    def get_poses(self) -> list[torch.Tensor]:
        return [w.pose[:2, 3] for seq in self.waypoint_sequences for w in seq]

    def add_waypoint_sequence(self, sequence: WaypointSequence) -> None:
        """Append ``sequence`` verbatim (no interpolation)."""
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose: torch.Tensor,
        gripper_action: torch.Tensor,
        num_steps: int,
        skip_interpolation: bool = False,
        action_noise: float = 0.0,
    ) -> None:
        """Append a ``num_steps``-long sequence reaching ``pose``.

        With ``skip_interpolation=False`` (default), the segment linearly interpolates from the
        trajectory's last pose to ``pose``. With ``skip_interpolation=True``, ``pose`` is repeated.

        Args:
            pose: 4x4 target pose [m, rad].
            gripper_action: Gripper command broadcast across the segment.
            num_steps: Number of action steps in the segment.
            skip_interpolation: If True, hold ``pose`` constant; otherwise interpolate.
            action_noise: Gaussian noise scale applied during execution.
        """
        if len(self.waypoint_sequences) == 0:
            assert skip_interpolation, "cannot interpolate from an empty trajectory"

        if skip_interpolation:
            assert num_steps is not None
            poses = pose.unsqueeze(0).repeat((num_steps, 1, 1))
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps, 1))
        else:
            last_waypoint = self.last_waypoint
            poses, num_steps_2 = PoseUtils.interpolate_poses(
                pose_1=last_waypoint.pose,
                pose_2=pose,
                num_steps=num_steps,
            )
            assert num_steps == num_steps_2
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps + 2, 1))
            # Drop the first interpolated sample — it duplicates the trajectory's current endpoint.
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self) -> WaypointSequence:
        """Remove and return the first waypoint; drop the holding sequence if it becomes empty."""
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other: WaypointTrajectory,
        num_steps_interp: int | None = None,
        num_steps_fixed: int | None = None,
        action_noise: float = 0.0,
    ) -> None:
        """Append ``other`` to ``self``, optionally with a bridging interp + fixed segment.

        Args:
            other: Trajectory to splice on.
            num_steps_interp: If > 0, prepend an interpolation segment from ``self``'s last pose
                to ``other``'s first pose.
            num_steps_fixed: If > 0, hold ``other``'s first pose constant for this many steps
                before the rest of ``other``.
            action_noise: Noise scale applied to the bridging segment.
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = need_interp or need_fixed

        if use_interpolation_segment:
            other_first = other.pop_first()
            target_for_interpolation = other_first[0]

            if need_interp:
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                )

            if need_fixed:
                # +1 compensates for the waypoint popped above when no interp segment was added.
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                )

            # Preserve the original noise from the popped waypoint.
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        self.waypoint_sequences += other.waypoint_sequences

    def get_full_sequence(self) -> WaypointSequence:
        """Flatten into a single :class:`WaypointSequence`."""
        return WaypointSequence(sequence=[w for seq in self.waypoint_sequences for w in seq.sequence])


class MultiWaypoint:
    """Per-EEF :class:`Waypoint` map executed as one env tick."""

    def __init__(self, waypoints: dict[str, Waypoint]) -> None:
        self.waypoints = waypoints

    async def execute(
        self,
        datastream: Datastream,
        success_term: TerminationTermCfg,
        env_id: int = 0,
        env_action_queue: asyncio.Queue | None = None,
    ) -> bool:
        """Issue one env step from the assembled multi-EEF action.

        Action encoding is routed through the embodiment adapter (via Datastream). Env-side
        controller ops (``env.step``, ``env.scene.get_state``) are reached through the Datastream
        ``get_env()`` escape hatch — by design, they stay on env.

        Args:
            datastream: Composed datastream; provides action encoding and env access.
            success_term: Termination term used to check task success after the step.
            env_id: Vectorized env index.
            env_action_queue: If given, action is enqueued for the simulator-side loop instead of
                stepped here.

        Returns:
            Whether the task succeeded after this step.
        """
        env = datastream.get_env()

        target_eef_pose_dict = {name: w.pose for name, w in self.waypoints.items()}
        gripper_action_dict = {name: w.gripper_action for name, w in self.waypoints.items()}
        action_noise_dict = {name: w.noise for name, w in self.waypoints.items()}

        play_action = datastream.target_eef_pose_to_action(
            target_eef_pose_dict=target_eef_pose_dict,
            gripper_action_dict=gripper_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

        if play_action.dim() == 1:
            play_action = play_action.unsqueeze(0)

        if env_action_queue is None:
            env.step(play_action)
        else:
            await env_action_queue.put((env_id, play_action[0]))
            await env_action_queue.join()

        # Envs without a success termination cannot be scored; report not-succeeded.
        if success_term is None:
            return False
        return bool(success_term.func(env, **success_term.params)[env_id])
