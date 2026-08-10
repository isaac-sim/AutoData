# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection strategies for picking a source subtask segment during generation.

Subclasses self-register via :class:`SelectionStrategyMeta`. Look up by name with
:func:`make_selection_strategy` (the `SubTaskConfig.selection_strategy` field stores the name).
"""

from __future__ import annotations

import abc
import numpy as np
import torch
from typing import Any

import isaaclab.utils.math as PoseUtils

REGISTERED_SELECTION_STRATEGIES: dict[str, type[SelectionStrategy]] = {}


def make_selection_strategy(name: str, *args: Any, **kwargs: Any) -> SelectionStrategy:
    """Construct the selection strategy registered under ``name``."""
    if name not in REGISTERED_SELECTION_STRATEGIES:
        raise KeyError(f"Unknown selection strategy {name!r}. Registered: {sorted(REGISTERED_SELECTION_STRATEGIES)}")
    return REGISTERED_SELECTION_STRATEGIES[name](*args, **kwargs)


def _register_selection_strategy(cls: type) -> None:
    if cls.__name__ == "SelectionStrategy":
        return
    REGISTERED_SELECTION_STRATEGIES[cls.NAME] = cls


class SelectionStrategyMeta(type):
    """Metaclass that auto-registers concrete :class:`SelectionStrategy` subclasses."""

    def __new__(mcs, name: str, bases: tuple, class_dict: dict) -> type:
        cls = super().__new__(mcs, name, bases, class_dict)
        _register_selection_strategy(cls)
        return cls


class SelectionStrategy(metaclass=SelectionStrategyMeta):
    """ABC for source-demo selection strategies."""

    NAME: str

    @abc.abstractmethod
    def select_source_demo(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor | None,
        src_subtask_datagen_infos: list,
        object_nodal_positions: torch.Tensor | None = None,
    ) -> int:
        """Return the index of the source demo whose subtask segment best fits the current scene.

        Args:
            eef_pose: Current 4x4 EEF pose [m, rad].
            object_pose: Current 4x4 pose of the subtask's reference object [m, rad], or None.
            src_subtask_datagen_infos: Per-source-demo :class:`DatagenInfo` slices covering this subtask.
            object_nodal_positions: Current deformable-object nodes shaped ``(N, 3)`` [m], or None.
        """
        raise NotImplementedError


class RandomStrategy(SelectionStrategy):
    """Uniform random pick over source demos."""

    NAME = "random"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        object_nodal_positions=None,
    ) -> int:
        n_src_demo = len(src_subtask_datagen_infos)
        return torch.randint(0, n_src_demo, (1,)).item()


class NearestNeighborObjectStrategy(SelectionStrategy):
    """Pick the source demo whose subtask-start object pose is closest to the current object pose."""

    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor,
        src_subtask_datagen_infos: list,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        nn_k: int = 3,
        object_nodal_positions: torch.Tensor | None = None,
    ) -> int:
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_object_poses = torch.stack(src_object_poses)

        all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
        obj_pos, obj_rot = PoseUtils.unmake_pose(object_pose)

        obj_pos = obj_pos.view(-1, 3)
        obj_rot_T = obj_rot.transpose(0, 1).view(-1, 3, 3)

        pos_dists = torch.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(dim=-1))

        # Arc-cos of rotation similarity; http://www.boris-belousov.net/2016/12/01/quat-dist/
        delta_R = torch.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)
        rot_dists = torch.acos(arc_cos_in)

        dists = pos_weight * pos_dists + rot_weight * rot_dists
        nn_k = min(nn_k, len(dists))
        rand_k = torch.randint(0, nn_k, (1,)).item()
        return torch.argsort(dists)[:nn_k][rand_k]


class NearestNeighborRobotDistanceStrategy(SelectionStrategy):
    """Pick the source demo whose transformed first EEF pose is closest to the current EEF pose."""

    NAME = "nearest_neighbor_robot_distance"

    def select_source_demo(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor,
        src_subtask_datagen_infos: list,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        nn_k: int = 3,
        object_nodal_positions: torch.Tensor | None = None,
    ) -> int:
        src_eef_poses = []
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_eef_poses.append(di.eef_pose[0])
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_eef_poses = torch.stack(src_eef_poses)
        src_object_poses = torch.stack(src_object_poses)

        # Express source EEF poses in source-object frame, then re-express in current object frame
        src_object_poses_inv = PoseUtils.pose_inv(src_object_poses)
        src_eef_poses_in_obj = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses,
            pose_A_in_B=src_object_poses_inv,
        )
        transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses_in_obj,
            pose_A_in_B=object_pose,
        )

        all_transformed_eef_pos, all_transformed_eef_rot = PoseUtils.unmake_pose(transformed_eef_poses)
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)

        eef_pos = eef_pos.view(-1, 3)
        eef_rot_T = eef_rot.transpose(0, 1).view(-1, 3, 3)

        pos_dists = torch.sqrt(((all_transformed_eef_pos - eef_pos) ** 2).sum(dim=-1))

        delta_R = torch.matmul(all_transformed_eef_rot, eef_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)
        rot_dists = torch.acos(arc_cos_in)

        dists = pos_weight * pos_dists + rot_weight * rot_dists
        nn_k = min(nn_k, len(dists))
        rand_k = torch.randint(0, nn_k, (1,)).item()
        return torch.argsort(dists)[:nn_k][rand_k]


class RegistrationCostStrategy(SelectionStrategy):
    """Pick a source segment using deformable-object TPS registration cost."""

    NAME = "registration_cost"

    def select_source_demo(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor | None,
        src_subtask_datagen_infos: list,
        object_nodal_positions: torch.Tensor | None = None,
        bend_coef: float = 0.1,
        rot_reg: float = 1e-3,
        nn_k: int = 3,
    ) -> int:
        """Choose uniformly among the ``nn_k`` source shapes with lowest TPS cost."""

        from isaac_autodata_core.deformable_transforms import nodal_registration_cost

        assert object_nodal_positions is not None, "registration_cost requires current object nodal positions"
        assert nn_k >= 1, f"nn_k must be at least 1, got {nn_k}"

        costs: list[float] = []
        for datagen_info in src_subtask_datagen_infos:
            assert (
                datagen_info.object_nodal_positions is not None
            ), "registration_cost requires source object nodal positions"
            source_objects = list(datagen_info.object_nodal_positions.values())
            assert (
                len(source_objects) == 1
            ), f"registration_cost expects exactly one source object, got {len(source_objects)}"
            source_nodal_positions = source_objects[0][0]
            try:
                cost = nodal_registration_cost(
                    source_nodal_positions,
                    object_nodal_positions,
                    bend_coef=bend_coef,
                    rot_reg=rot_reg,
                )
            except (AssertionError, np.linalg.LinAlgError, ValueError):
                cost = float("inf")
            costs.append(cost)

        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        nn_k = min(nn_k, len(costs))
        rand_k = torch.randint(0, nn_k, (1,)).item()
        return torch.argsort(costs_tensor)[:nn_k][rand_k]
