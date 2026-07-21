# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apple Vision Pro teleoperation compatibility for the Isaac Lab G1 task."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

_G1_TASK_NAME = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
_G1_CONFIG_MODULE = "isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg"
_PATCH_MARKER = "_isaac_autodata_g1_avp_teleop_patch_installed"
_IMPORT_WRAPPER_MARKER = "_isaac_autodata_g1_avp_import_wrapper"


def _cli_option_value(arguments: list[str], option: str) -> str | None:
    """Return a command-line option value from split or ``--option=value`` syntax."""
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", maxsplit=1)[1]
    return None


def _is_g1_avp_recording(arguments: list[str]) -> bool:
    """Return whether the current command requests AVP teleoperation for the G1 task."""
    task_name = _cli_option_value(arguments, "--task")
    cloudxr_environment = _cli_option_value(arguments, "--cloudxr_env")
    return task_name == _G1_TASK_NAME and cloudxr_environment is not None and cloudxr_environment.lower() == "avp"


def _build_g1_avp_locomanipulation_pipeline() -> OutputCombiner:
    """Build the G1 locomanipulation action pipeline using AVP hand tracking.

    Apple Vision Pro supplies wrist and finger poses but has no motion-controller buttons or thumbsticks. The upper
    body therefore follows both tracked hands while the lower body receives a fixed standing command.
    """
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand import (
        g1_dex_retargeting_utils as dex_utils,
    )
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        LocomotionFixedRootCmdRetargeter,
        LocomotionFixedRootCmdRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=90.0,
            target_offset_yaw=0.0,
        ),
        name="left_ee_pose",
    ).connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=180.0,
            target_offset_pitch=-90.0,
            target_offset_yaw=0.0,
        ),
        name="right_ee_pose",
    ).connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    left_hand_joint_names = [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
    ]
    right_hand_joint_names = [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ]
    operator_to_mano = (0, 0, 1, 1, 0, 0, 0, 1, 0)
    config_directory = Path(dex_utils.__file__).parent / "data" / "configs" / "dex-retargeting"
    left_hand_urdf = retrieve_file_path(
        f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_left_hand.urdf"
    )
    right_hand_urdf = retrieve_file_path(
        f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_right_hand.urdf"
    )

    left_hand = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(config_directory / "g1_hand_left_dexpilot.yml"),
            hand_urdf=left_hand_urdf,
            hand_joint_names=left_hand_joint_names,
            handtracking_to_baselink_frame_transform=operator_to_mano,
            hand_side="left",
        ),
        name="left_hand",
    ).connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})
    right_hand = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(config_directory / "g1_hand_right_dexpilot.yml"),
            hand_urdf=right_hand_urdf,
            hand_joint_names=right_hand_joint_names,
            handtracking_to_baselink_frame_transform=operator_to_mano,
            hand_side="right",
        ),
        name="right_hand",
    ).connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    locomotion = LocomotionFixedRootCmdRetargeter(
        LocomotionFixedRootCmdRetargeterConfig(hip_height=0.72), name="locomotion"
    )

    left_ee_elements = [
        "l_pos_x",
        "l_pos_y",
        "l_pos_z",
        "l_quat_x",
        "l_quat_y",
        "l_quat_z",
        "l_quat_w",
    ]
    right_ee_elements = [
        "r_pos_x",
        "r_pos_y",
        "r_pos_z",
        "r_quat_x",
        "r_quat_y",
        "r_quat_z",
        "r_quat_w",
    ]
    locomotion_elements = ["loco_vel_x", "loco_vel_y", "loco_rot_vel_z", "loco_hip_height"]
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            "left_hand_index_0_joint",
            "left_hand_middle_0_joint",
            "left_hand_thumb_0_joint",
            "right_hand_index_0_joint",
            "right_hand_middle_0_joint",
            "right_hand_thumb_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_1_joint",
            "left_hand_thumb_1_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_1_joint",
            "right_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "right_hand_thumb_2_joint",
        ]
        + locomotion_elements
    )
    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
            "locomotion": locomotion_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
            "locomotion": "array",
        },
    ).connect({
        "left_ee_pose": left_se3.output("ee_pose"),
        "right_ee_pose": right_se3.output("ee_pose"),
        "left_hand_joints": left_hand.output("hand_joints"),
        "right_hand_joints": right_hand.output("hand_joints"),
        "locomotion": locomotion.output("root_command"),
    })
    return OutputCombiner({"action": reorderer.output("output")})


def _patch_g1_config_module(module: ModuleType) -> None:
    """Replace the controller-only G1 pipeline builder with the AVP builder."""
    if getattr(module, _PATCH_MARKER, False):
        return
    setattr(module, "_build_g1_locomanipulation_pipeline", _build_g1_avp_locomanipulation_pipeline)
    setattr(module, _PATCH_MARKER, True)


def install_g1_avp_teleop_patch_if_requested(arguments: list[str] | None = None) -> None:
    """Install the G1 AVP override only for the documented G1 recording command."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if not _is_g1_avp_recording(arguments):
        return

    loaded_module = sys.modules.get(_G1_CONFIG_MODULE)
    if loaded_module is not None:
        _patch_g1_config_module(loaded_module)
        return

    original_import = builtins.__import__
    if getattr(original_import, _IMPORT_WRAPPER_MARKER, False):
        return

    def _import_and_patch(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        imported = original_import(name, globals, locals, fromlist, level)
        config_module = sys.modules.get(_G1_CONFIG_MODULE)
        module_spec = getattr(config_module, "__spec__", None)
        if config_module is not None and not getattr(module_spec, "_initializing", False):
            _patch_g1_config_module(config_module)
            if builtins.__import__ is _import_and_patch:
                builtins.__import__ = original_import
        return imported

    setattr(_import_and_patch, _IMPORT_WRAPPER_MARKER, True)
    builtins.__import__ = _import_and_patch
