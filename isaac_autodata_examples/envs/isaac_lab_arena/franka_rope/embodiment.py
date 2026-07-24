# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena Franka embodiment configured for deformable rope manipulation."""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset, register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

from . import mdp

_FRANKA_ROPE_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 2.43,
    "panda_finger_joint.*": 0.04,
}

_FRANKA_ROPE_INIT_JOINT_POSE = [
    0.0444,
    -0.1894,
    -0.1107,
    -2.5148,
    0.0044,
    2.3775,
    2.43,
    0.04,
    0.04,
]


@configclass
class FrankaRopeCameraCfg(ArenaCameraCfg):
    """Wrist and fixed-view cameras matching the Isaac Lab rope environment."""

    robot0_eye_in_hand_image = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/robot0_eye_in_hand_image",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(0.03701, 0.03701, -0.70614, -0.70614),
            convention="ros",
        ),
    )

    agentview_image = CameraCfg(
        prim_path="{ENV_REGEX_NS}/agentview_image",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.5, 0.4),
            rot=(0.0, 0.38268, 0.92388, 0.0),
            convention="opengl",
        ),
    )


@configclass
class FrankaRopeObservationsCfg:
    """Arena-standard Franka observations plus deformable rope nodal state."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations recorded into demonstration datasets."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        object_nodal_pos = ObsTerm(func=mdp.object_nodal_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaRopeEventsCfg:
    """Reset the Franka exactly as the equivalent Isaac Lab rope environment."""

    reset_all = EventTerm(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")
    randomize_franka_joint_state = EventTerm(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )


@register_asset
class FrankaRopeIKEmbodiment(FrankaIKEmbodiment):
    """Arena Franka IK embodiment with rope-specific pose, observations, and cameras."""

    name = "franka_rope_ik"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ) -> None:
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=list(_FRANKA_ROPE_INIT_JOINT_POSE),
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
        )
        self.scene_config.robot.init_state.joint_pos = dict(_FRANKA_ROPE_INIT_JOINT_POS)
        self.observation_config = FrankaRopeObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms
        self.camera_config = FrankaRopeCameraCfg()
        self.event_config = FrankaRopeEventsCfg()
        self.reward_config = None


@register_retargeter
class FrankaRopeKeyboardRetargeter(RetargetterBase):
    """Use the standard relative-pose keyboard pipeline for the rope embodiment."""

    device = "keyboard"
    embodiment = FrankaRopeIKEmbodiment.name

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return no custom pipeline; Isaac Lab handles relative-pose keyboard input."""

        del embodiment


@register_retargeter
class FrankaRopeSpaceMouseRetargeter(RetargetterBase):
    """Use the standard relative-pose SpaceMouse pipeline for the rope embodiment."""

    device = "spacemouse"
    embodiment = FrankaRopeIKEmbodiment.name

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return no custom pipeline; Isaac Lab handles relative-pose SpaceMouse input."""

        del embodiment
