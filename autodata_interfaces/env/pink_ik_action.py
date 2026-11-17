# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one URDF for the per-environment Pink IK controllers."""

from __future__ import annotations

import math
import numpy as np
import tempfile
import torch
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab.controllers import utils as controller_utils
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory

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
            _urdf_mimic_joints_sync(tree.getroot(), controller.usd_path)
            for mesh in tree.iter("mesh"):
                mesh.set("filename", str((urdf_path.parent / mesh.attrib["filename"]).resolve()))
            tree.write(urdf_path, encoding="unicode")
            controller.mesh_path = None
        super().__init__(cfg, env)

    def _initialize_joint_info(self) -> None:
        """Keep passthrough action order independent of the physics backend's joint order."""
        super()._initialize_joint_info()
        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(
            self.cfg.hand_joint_names, preserve_order=True
        )
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def get_action_target_frame_names(self) -> tuple[str, ...]:
        """Return the controlled frame names in action order.

        Returns:
            Frame names shared by the per-environment Pink controllers.
        """

        controllers = tuple(getattr(self, "_ik_controllers", ()))
        assert controllers, "planning action term exposes no live IK controllers"

        frame_count = self._num_frame_tasks
        assert frame_count > 0, "planning action term must expose at least one frame task"
        frame_orders: list[tuple[str, ...]] = []
        for controller in controllers:
            tasks = tuple(controller.cfg.variable_input_tasks)
            assert len(tasks) >= frame_count, "planning controller has too few variable-input tasks"
            frame_names = tuple(getattr(task, "frame", None) for task in tasks[:frame_count])
            assert all(
                isinstance(name, str) and name for name in frame_names
            ), "planning controller frame tasks must lead variable_input_tasks in action order"
            assert not any(
                getattr(task, "frame", None) for task in tasks[frame_count:]
            ), "planning controller contains an unexpected extra frame task"
            frame_orders.append(frame_names)

        assert all(
            order == frame_orders[0] for order in frame_orders[1:]
        ), "planning controllers disagree on action-target frame order"
        return frame_orders[0]

    def get_base_link_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return controller-base poses for the selected environments.

        Args:
            env_ids: Environment indices, or None for all environments.

        Returns:
            Controller-base transforms with shape (N, 4, 4); translation is in [m].
        """
        poses = self._get_base_link_frame_transform().detach().clone()
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        return poses[index]

    def solve_joint_trajectory(
        self,
        target_frame_poses: Mapping[str, torch.Tensor],
        joint_seed_trajectories: Mapping[str, NamedJointTrajectory],
        initial_state: NamedJointTrajectory,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        joint_state_is_valid: Callable[[tuple[str, ...], torch.Tensor, int], bool] | None = None,
        env_id: int = 0,
    ) -> NamedJointTrajectory:
        """Solve joint positions for synchronized frame targets using the controller's kinematics.

        Named planner proposals provide the posture reference and a fallback kinematic seed.
        Values proposed by more than one group (the waist on a humanoid) are averaged. The prior
        coupled state remains the preferred seed for continuity. After the live controller horizon,
        Pink solves only the original frame tasks so execution smoothing cannot mask feasibility.

        Args:
            target_frame_poses: Ordered frame targets with shape (T, 4, 4) [m].
            joint_seed_trajectories: Named posture proposals per frame, with T samples.
            initial_state: One named initial articulation state [rad, m].
            position_tolerance_m: Required frame-position convergence [m].
            rotation_tolerance_rad: Required frame-rotation convergence [rad].
            joint_state_is_valid: Optional feasibility predicate for each named joint sample.
            env_id: Controller to use.

        Returns:
            Named controlled-joint trajectory.
        """

        frame_names = tuple(target_frame_poses)
        assert len(frame_names) == self._num_frame_tasks, "targets must match the action's frame count"
        assert set(joint_seed_trajectories) == set(frame_names)
        sample_counts = {len(poses) for poses in target_frame_poses.values()}
        sample_counts.update(len(trajectory.positions) for trajectory in joint_seed_trajectories.values())
        assert len(sample_counts) == 1, "EEF targets and joint seeds must have one common length"
        sample_count = sample_counts.pop()
        assert all(poses.shape == (sample_count, 4, 4) for poses in target_frame_poses.values())

        live_names = tuple(self._asset.joint_names)
        assert initial_state.joint_names == live_names, "initial state must match the action's asset joint order"
        assert len(initial_state.positions) == 1, "initial state must contain exactly one sample"
        initial_joint_positions = initial_state.positions[0]
        assert bool(torch.isfinite(initial_joint_positions).all())
        assert math.isfinite(position_tolerance_m) and position_tolerance_m > 0.0
        assert math.isfinite(rotation_tolerance_rad) and rotation_tolerance_rad > 0.0
        live_index = {name: index for index, name in enumerate(live_names)}
        posture = initial_joint_positions.repeat(sample_count, 1)
        proposals: dict[str, list[torch.Tensor]] = {}
        for trajectory in joint_seed_trajectories.values():
            missing = set(trajectory.joint_names) - set(live_names)
            assert not missing, f"joint-seed trajectory contains unknown joints: {sorted(missing)}"
            for index, joint_name in enumerate(trajectory.joint_names):
                proposals.setdefault(joint_name, []).append(
                    trajectory.positions[:, index].to(device=posture.device, dtype=posture.dtype)
                )
        for joint_name, values in proposals.items():
            posture[:, live_index[joint_name]] = torch.stack(values).mean(dim=0)
        controller = self._planning_controller(env_id)
        controlled_names = tuple(self._isaaclab_controlled_joint_names)
        controlled_ids = tuple(self._isaaclab_controlled_joint_ids)
        controller_substeps = int(self._env.cfg.decimation)
        assert tuple(controller.cfg.joint_names) == controlled_names
        assert len(controlled_ids) == len(controlled_names)
        assert controller_substeps >= 1

        frame_tasks = tuple(task for task in controller.cfg.variable_input_tasks if hasattr(task, "frame"))
        assert frame_names == tuple(task.frame for task in frame_tasks)
        saved_frame_targets = tuple(self._pink_frame_target(task).copy() for task in frame_tasks)
        posture_tasks = tuple(task for task in controller.cfg.variable_input_tasks if hasattr(task, "target_q"))
        saved_posture_targets = tuple(
            task.target_q.copy() if task.target_q is not None else None for task in posture_tasks
        )

        assert self.cfg.controller.base_link_name == controller.cfg.base_link_name, "controller base-frame mismatch"
        root_pose = self.get_base_link_pose(env_ids=[env_id])[0]
        root_from_env = torch.linalg.inv(root_pose)
        targets_in_root = {
            eef_name: root_from_env.to(device=poses.device, dtype=poses.dtype) @ poses
            for eef_name, poses in target_frame_poses.items()
        }
        current = initial_joint_positions.detach().cpu().numpy().copy()
        solutions: list[torch.Tensor] = []
        from pink import solve_ik

        try:
            for sample in range(sample_count):
                for eef_name, task in zip(frame_names, frame_tasks, strict=True):
                    pose = targets_in_root[eef_name][sample].detach().cpu().numpy()
                    target = self._pink_frame_target(task).copy()
                    target.translation = pose[:3, 3]
                    target.rotation = pose[:3, :3]
                    task.set_target(target)
                proposal = posture[sample].detach().cpu().numpy()
                posture_target = proposal[controller.isaac_lab_to_pink_ordering]
                converged = False
                valid = False
                for seed in (current.copy(), proposal):
                    current = seed.copy()
                    controller.update_null_space_joint_targets(posture_target)
                    for iteration in range(100):
                        if iteration < controller_substeps:
                            solution = controller.compute(current, self._sim_dt)
                            assert solution.shape == (len(controlled_names),) and bool(torch.isfinite(solution).all())
                            current[np.asarray(controlled_ids)] = solution.detach().cpu().numpy()
                        else:
                            controller.pink_configuration.update(current[controller.isaac_lab_to_pink_ordering])
                            velocity = solve_ik(
                                controller.pink_configuration,
                                frame_tasks,
                                self._sim_dt,
                                solver="daqp",
                                safety_break=controller.cfg.fail_on_joint_limit_violation,
                            )
                            joint_delta = velocity * self._sim_dt
                            current[np.asarray(controlled_ids)] += joint_delta[
                                controller.pink_to_isaac_lab_controlled_ordering
                            ]
                            solution = torch.as_tensor(
                                current[np.asarray(controlled_ids)],
                                device=posture.device,
                                dtype=posture.dtype,
                            )
                        controller.pink_configuration.update(current[controller.isaac_lab_to_pink_ordering])
                        frame_errors = tuple(task.compute_error(controller.pink_configuration) for task in frame_tasks)
                        converged = all(
                            np.linalg.norm(error[:3]) <= position_tolerance_m
                            and np.linalg.norm(error[3:]) <= rotation_tolerance_rad
                            for error in frame_errors
                        )
                        if converged and iteration + 1 >= controller_substeps:
                            break
                    if converged:
                        valid = joint_state_is_valid is None or joint_state_is_valid(controlled_names, solution, sample)
                        if valid:
                            break
                residuals = ", ".join(
                    f"{eef_name}=({np.linalg.norm(error[:3]):.6f} m, {np.linalg.norm(error[3:]):.6f} rad)"
                    for eef_name, error in zip(frame_names, frame_errors, strict=True)
                )
                configuration = controller.pink_configuration
                limit_margins = np.minimum(
                    configuration.q - configuration.model.lowerPositionLimit,
                    configuration.model.upperPositionLimit - configuration.q,
                )
                nearest_limits = ", ".join(
                    f"{configuration.controlled_joint_names_pinocchio_order[index]}={limit_margins[index]:.6f} rad"
                    for index in np.argsort(limit_margins)[:3]
                )
                assert converged and valid, (
                    f"Pink could not solve a valid synchronized end-effector target at sample {sample}: {residuals}; "
                    f"collision_free={valid}; nearest joint-limit margins: {nearest_limits}"
                )
                solutions.append(solution.to(device=posture.device, dtype=posture.dtype))
        finally:
            for task, target in zip(frame_tasks, saved_frame_targets, strict=True):
                task.set_target(target)
            for task, target in zip(posture_tasks, saved_posture_targets, strict=True):
                if target is not None:
                    task.set_target(target)
        return NamedJointTrajectory(controlled_names, torch.stack(solutions))

    def set_joint_posture_target(
        self,
        joint_names: tuple[str, ...],
        joint_positions: torch.Tensor,
        env_id: int = 0,
    ) -> None:
        """Set a named posture reference for one controller.

        Args:
            joint_names: Names aligned with the position values.
            joint_positions: Posture target with shape (J,) [rad, m].
            env_id: Environment controller to update.
        """

        controller = self._planning_controller(env_id)
        assert joint_positions.shape == (len(joint_names),)
        by_name = dict(zip(joint_names, joint_positions, strict=True))
        pin_names = tuple(controller.pink_configuration.controlled_joint_names_pinocchio_order)
        posture = controller.init_joint_positions.copy()
        for index, name in enumerate(pin_names):
            if name in by_name:
                posture[index] = float(by_name[name].item())
        controller.update_null_space_joint_targets(posture)

    def reset_joint_posture_target(self, env_id: int = 0) -> None:
        """Restore the controller's initial posture target.

        Args:
            env_id: Environment controller to update.
        """

        controller = self._planning_controller(env_id)
        controller.update_null_space_joint_targets(controller.init_joint_positions)

    def _planning_controller(self, env_id: int) -> Any:
        """Return the live IK controller for the selected environment."""
        controllers = tuple(getattr(self, "_ik_controllers", ()))
        assert 0 <= env_id < len(controllers), "env_id has no live Pink controller"
        return controllers[env_id]

    @staticmethod
    def _pink_frame_target(task: Any) -> Any:
        """Return the pose target stored by the Pink frame task."""
        if hasattr(task, "transform_target_to_base"):
            return task.transform_target_to_base
        assert hasattr(task, "transform_target_to_world"), "Pink pose tasks must expose a target transform"
        return task.transform_target_to_world


def _urdf_mimic_joints_sync(root: ET.Element, usd_path: str) -> None:
    """Match exported PhysX mimic coefficients to the source joint equation."""
    from pxr import Usd, UsdGeom

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    mimics = {name: joint for name, joint in joints.items() if joint.find("mimic") is not None}
    if not mimics:
        return
    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"Cannot open controller model {usd_path!r}."
    units = {
        "revolute": math.pi / 180.0,
        "continuous": math.pi / 180.0,
        "prismatic": UsdGeom.GetStageMetersPerUnit(stage),
    }
    for prim in stage.Traverse():
        joint = mimics.get(prim.GetName())
        metadata = prim.GetMetadata("apiSchemas")
        schemas = metadata.GetAppliedItems() if metadata is not None else ()
        if joint is None or "NewtonMimicAPI" in schemas:
            continue
        instances = [name.split(":", 1)[1] for name in schemas if name.startswith("PhysxMimicJointAPI:")]
        if not instances:
            continue
        assert len(instances) == 1, "URDF joints support exactly one mimic degree of freedom."
        prefix = f"physxMimicJoint:{instances[0]}"
        gearing = prim.GetAttribute(f"{prefix}:gearing").Get()
        offset = prim.GetAttribute(f"{prefix}:offset").Get()
        # PhysX: q + gearing*q_ref + offset = 0. URDF: q = multiplier*q_ref + offset.
        mimic = joint.find("mimic")
        joint_unit = units[joint.get("type")]
        reference_unit = units[joints[mimic.get("joint")].get("type")]
        mimic.set("multiplier", str(-float(gearing if gearing is not None else 1.0) * joint_unit / reference_unit))
        mimic.set("offset", str(-float(offset if offset is not None else 0.0) * joint_unit))
