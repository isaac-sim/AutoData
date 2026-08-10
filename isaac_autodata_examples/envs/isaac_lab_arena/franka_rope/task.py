# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena task definition for grasping and folding a deformable rope."""

from __future__ import annotations

import math
from typing import Never

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from . import mdp


@configclass
class FrankaRopeTaskObservationsCfg:
    """Automatic SoftMimicGen subtask signals."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Signals consumed by AutoData annotation."""

        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaRopeTaskEventsCfg:
    """Rope randomization and endpoint tracking events."""

    reset_rope_end_tracking = EventTerm(
        func=mdp.reset_rope_end_tracking,
        mode="reset",
    )

    reset_object_position = EventTerm(
        func=mdp.reset_rope_nodal_state,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.1),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "partial_pose_range": {
                "x": (0.0, 0.1),
                "y": (0.0, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi / 6.0, math.pi / 6.0),
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class FrankaRopeTaskTerminationsCfg:
    """Rope success, failure, and time-limit terms."""

    time_out = DoneTerm(func=mdp_isaac_lab.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.rope_below_minimum,
        params={
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    success = DoneTerm(func=mdp.rope_ends_close_tracked)


@register_task
class FrankaRopeTask(TaskBase):
    """Bring the two ends of a deformable rope together with a Franka arm."""

    def __init__(self, episode_length_s: float = 10.0) -> None:
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Grasp and manipulate the deformable rope until its ends meet.",
        )
        self.observation_cfg = FrankaRopeTaskObservationsCfg()
        self.events_cfg = FrankaRopeTaskEventsCfg()
        self.termination_cfg = FrankaRopeTaskTerminationsCfg()

    def get_scene_cfg(self) -> None:
        """Return no task scene additions; the environment owns all assets."""

    def get_observation_cfg(self) -> FrankaRopeTaskObservationsCfg:
        """Return automatic subtask-signal observations."""

        return self.observation_cfg

    def get_termination_cfg(self) -> FrankaRopeTaskTerminationsCfg:
        """Return task termination terms."""

        return self.termination_cfg

    def get_events_cfg(self) -> FrankaRopeTaskEventsCfg:
        """Return deformable reset events."""

        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Never:
        """Reject Arena Mimic mode; SoftMimicGen is provided by AutoData."""

        del arm_mode
        raise NotImplementedError("FrankaRopeTask uses AutoData SoftMimicGen, not Arena Mimic mode.")

    def get_metrics(self) -> list[MetricBase]:
        """Return success-rate evaluation."""

        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return the rope workspace camera framing."""

        return ViewerCfg(
            eye=(1.0, 1.0, 0.5),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
        )
