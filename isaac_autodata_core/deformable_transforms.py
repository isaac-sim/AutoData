# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Non-rigid transforms used by SoftMimicGen."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils


def _load_tps_functions():
    """Load the pinned Rapprentice TPS implementation on first use."""

    try:
        from rapprentice.tps import tps_cost, tps_eval, tps_fit, tps_fit2, tps_grad
    except ImportError as exc:
        raise ImportError(
            "SoftMimicGen requires the 'rapprentice' package. Rebuild the AutoData Docker image "
            "after the SoftMimicGen dependency change."
        ) from exc
    return tps_cost, tps_eval, tps_fit, tps_fit2, tps_grad


def transform_source_data_segment_using_nodal_registration(
    src_eef_poses: torch.Tensor,
    src_obj_nodal_pos: torch.Tensor,
    tgt_obj_nodal_pos: torch.Tensor,
    *,
    use_rotation_transform: bool = True,
    bend_coef: float = 0.1,
    rot_coef: float = 1e-3,
) -> torch.Tensor:
    """Warp an EEF pose sequence using TPS registration between deformable-object nodes.

    Args:
        src_eef_poses: Source EEF poses shaped ``(T, 4, 4)``.
        src_obj_nodal_pos: Source object node positions shaped ``(N, 3)`` [m].
        tgt_obj_nodal_pos: Target object node positions shaped ``(N, 3)`` [m].
        use_rotation_transform: Transform EEF rotations using the local TPS Jacobian.
        bend_coef: TPS bending regularization coefficient.
        rot_coef: TPS affine-rotation regularization coefficient.

    Returns:
        Warped EEF poses shaped ``(T, 4, 4)``.
    """

    assert src_eef_poses.ndim == 3 and src_eef_poses.shape[-2:] == (
        4,
        4,
    ), f"src_eef_poses must have shape (T, 4, 4), got {tuple(src_eef_poses.shape)}"
    assert (
        src_obj_nodal_pos.ndim == 2 and src_obj_nodal_pos.shape[1] == 3
    ), f"source nodal positions must have shape (N, 3), got {tuple(src_obj_nodal_pos.shape)}"
    assert (
        tgt_obj_nodal_pos.ndim == 2 and tgt_obj_nodal_pos.shape[1] == 3
    ), f"target nodal positions must have shape (N, 3), got {tuple(tgt_obj_nodal_pos.shape)}"
    assert src_obj_nodal_pos.shape[0] == tgt_obj_nodal_pos.shape[0], (
        "corresponding-node TPS requires source and target objects to have the same node count: "
        f"{src_obj_nodal_pos.shape[0]} != {tgt_obj_nodal_pos.shape[0]}"
    )

    _, tps_eval, _, tps_fit2, tps_grad = _load_tps_functions()
    src_nodal_np = src_obj_nodal_pos.detach().cpu().numpy()
    tgt_nodal_np = tgt_obj_nodal_pos.detach().cpu().numpy()
    lin_ag, trans_g, w_ng = tps_fit2(
        x_na=src_nodal_np,
        y_ng=tgt_nodal_np,
        bend_coef=bend_coef,
        rot_coef=rot_coef,
    )

    src_eef_pos, src_eef_rot = PoseUtils.unmake_pose(src_eef_poses)
    src_eef_pos_np = src_eef_pos.detach().cpu().numpy()
    transformed_pos_np = tps_eval(
        x_ma=src_eef_pos_np,
        lin_ag=lin_ag,
        trans_g=trans_g,
        w_ng=w_ng,
        x_na=src_nodal_np,
    )

    if use_rotation_transform:
        jacobians = tps_grad(
            x_ma=src_eef_pos_np,
            lin_ag=lin_ag,
            _trans_g=trans_g,
            w_ng=w_ng,
            x_na=src_nodal_np,
        )
        src_eef_rot_np = src_eef_rot.detach().cpu().numpy()
        transformed_rot_np = np.empty_like(src_eef_rot_np)
        for pose_index, jacobian in enumerate(jacobians):
            transformed_rotation = jacobian @ src_eef_rot_np[pose_index]
            u_mat, _, vt_mat = np.linalg.svd(transformed_rotation)
            transformed_rotation = u_mat @ vt_mat
            if np.linalg.det(transformed_rotation) < 0:
                vt_mat[-1, :] *= -1
                transformed_rotation = u_mat @ vt_mat
            transformed_rot_np[pose_index] = transformed_rotation
    else:
        transformed_rot_np = src_eef_rot.detach().cpu().numpy()

    transformed_pos = torch.as_tensor(
        transformed_pos_np,
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype,
    )
    transformed_rot = torch.as_tensor(
        transformed_rot_np,
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype,
    )
    assert torch.isfinite(transformed_pos).all(), "TPS produced non-finite EEF positions"
    assert torch.isfinite(transformed_rot).all(), "TPS produced non-finite EEF rotations"
    return PoseUtils.make_pose(pos=transformed_pos, rot=transformed_rot)


def nodal_registration_cost(
    src_obj_nodal_pos: torch.Tensor,
    tgt_obj_nodal_pos: torch.Tensor,
    *,
    bend_coef: float = 0.1,
    rot_reg: float = 1e-3,
) -> float:
    """Return the TPS fitting cost between two corresponding deformable node sets.

    Args:
        src_obj_nodal_pos: Source object node positions shaped ``(N, 3)`` [m].
        tgt_obj_nodal_pos: Target object node positions shaped ``(N, 3)`` [m].
        bend_coef: TPS bending regularization coefficient.
        rot_reg: TPS affine-rotation regularization coefficient.

    Returns:
        Scalar TPS residual-plus-bending cost.
    """

    assert src_obj_nodal_pos.shape == tgt_obj_nodal_pos.shape, (
        "registration-cost TPS requires matching nodal shapes: "
        f"{tuple(src_obj_nodal_pos.shape)} != {tuple(tgt_obj_nodal_pos.shape)}"
    )
    assert (
        src_obj_nodal_pos.ndim == 2 and src_obj_nodal_pos.shape[1] == 3
    ), f"nodal positions must have shape (N, 3), got {tuple(src_obj_nodal_pos.shape)}"

    tps_cost, _, tps_fit, _, _ = _load_tps_functions()
    src_nodal_np = src_obj_nodal_pos.detach().cpu().numpy()
    tgt_nodal_np = tgt_obj_nodal_pos.detach().cpu().numpy()
    lin_ag, trans_g, w_ng = tps_fit(
        src_nodal_np,
        tgt_nodal_np,
        bend_coef,
        rot_reg,
    )
    return float(
        tps_cost(
            lin_ag,
            trans_g,
            w_ng,
            src_nodal_np,
            tgt_nodal_np,
            bend_coef,
        )
    )
