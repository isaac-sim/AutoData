# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka rope manipulation environment for Isaac Lab 3."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp

_ROPE_USD_PATH = Path(__file__).resolve().parent / "assets" / "Rope.usd"

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


@configclass
class FrankaRopeSceneCfg(InteractiveSceneCfg):
    """Scene containing a Franka robot, deformable rope, table, cameras, and lighting."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=_FRANKA_ROPE_INIT_JOINT_POS),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )

    object = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(usd_path=str(_ROPE_USD_PATH)),
        debug_vis=False,
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

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
class ActionsCfg:
    """Relative end-effector pose and binary gripper actions."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation groups used for recording and later subtask annotation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Robot, rope, action, and image observations."""

        actions = ObsTerm(func=base_mdp.last_action)
        robot0_joint_pos_rel = ObsTerm(func=base_mdp.joint_pos_rel)
        robot0_joint_vel_rel = ObsTerm(func=base_mdp.joint_vel_rel)
        robot0_eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        robot0_eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        robot0_gripper_qpos = ObsTerm(func=mdp.gripper_pos)
        object_nodal_pos = ObsTerm(func=mdp.object_nodal_pos)
        agentview_image = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("agentview_image"), "data_type": "rgb", "normalize": False},
        )
        robot0_eye_in_hand_image = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot0_eye_in_hand_image"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Signals that will be consumed by the later annotation workflow."""

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

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Scene, robot, and deformable reset events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_franka_joint_state = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    reset_rope_end_tracking = EventTerm(func=mdp.reset_rope_end_tracking, mode="reset")

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
class TerminationsCfg:
    """Episode completion and failure conditions."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.rope_below_minimum,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(func=mdp.rope_ends_close_tracked)


@configclass
class FrankaRopeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for teleoperated Franka rope manipulation."""

    settling_steps: int = 10
    """Number of control iterations used to settle the deformable rope after reset."""

    scene: FrankaRopeSceneCfg = FrankaRopeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    teleop_devices: DevicesCfg = DevicesCfg(
        devices={
            "keyboard": Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.2),
            "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5),
        }
    )

    def __post_init__(self) -> None:
        """Configure simulation timing, rendering, and the default viewer."""

        self.decimation = 5
        self.episode_length_s = 10.0
        self.seed = 7
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.render.antialiasing_mode = "DLSS"
        self.num_rerenders_on_reset = 1
        self.viewer.eye = (1.0, 1.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
