# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one URDF for the per-environment Pink IK controllers."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.controllers import utils as controller_utils
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PinkInverseKinematicsActionSharedUrdf(PinkInverseKinematicsAction):
    """Reuse the converted URDF while retaining independent IK state for each environment."""

    def __init__(self, cfg: PinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        """Convert USD once before the base action creates its per-environment controllers.

        Args:
            cfg: Pink action configuration.
            env: Environment containing the controlled articulations.
        """
        controller = cfg.controller
        if controller.urdf_path is None and controller.usd_path is not None:
            cfg = cfg.copy()
            controller = cfg.controller
            controller.urdf_path, _ = controller_utils.convert_usd_to_urdf(
                controller.usd_path,
                controller.urdf_output_dir or tempfile.mkdtemp(prefix="autodata-pink-"),
                force_conversion=True,
            )
            # Lab's explicit-URDF path attempts to download mesh_path as a file, although
            # the exporter returns a directory. Resolve the exported relative mesh names
            # in the generated URDF so Pinocchio needs no additional package directory.
            urdf_path = Path(controller.urdf_path)
            tree = ET.parse(urdf_path)
            for mesh in tree.iter("mesh"):
                mesh.set("filename", str((urdf_path.parent / mesh.attrib["filename"]).resolve()))
            tree.write(urdf_path, encoding="unicode")
            controller.mesh_path = None
        super().__init__(cfg, env)
