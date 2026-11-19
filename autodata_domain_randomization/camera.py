# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab event functions for episodic camera intrinsics/extrinsics randomization."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from autodata_utils import pose_math
from autodata_utils.tensor_utils import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import Camera


def randomize_camera(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    focal_length_scale_range: tuple[float, float],
    position_xyz_min: tuple[float, float, float],
    position_xyz_max: tuple[float, float, float],
    rotation_rpy_min_deg: tuple[float, float, float],
    rotation_rpy_max_deg: tuple[float, float, float],
) -> dict[int, dict[str, tuple[float, ...]]]:
    """Perturb one camera's calibration once per environment reset.

    Perturbations use the nominal sensor config, or the initial authored mount for cameras loaded
    from USD, rather than the previous episode's values, so repeated resets never accumulate drift.
    The draws use Torch's process RNG so Isaac Lab's environment seed controls reproducibility.

    Extrinsics are written as the camera prim's local transform w.r.t. the link it is mounted on,
    which is the same quantity :class:`~isaaclab.sensors.CameraCfg.OffsetCfg` sets at spawn; the
    camera therefore keeps tracking its parent link. Translation is sampled in the parent-link
    frame and rotation in the camera optical frame, matching how a mount tolerance and a hand-eye
    calibration residual are respectively specified.

    Only the focal length of the intrinsic matrix is randomized. Calibrated OpenCV pinhole cameras
    retain their measured principal point and separate x/y focal lengths. Ordinary aperture-based
    cameras retain Isaac Lab's square pixels and centered principal point.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environments whose cameras should be re-sampled.
        asset_cfg: Scene entity identifying the camera sensor.
        focal_length_scale_range: Uniform multiplier range applied to the nominal focal length.
        position_xyz_min: Lower bound of the mount-translation offset [m], in the parent-link frame.
        position_xyz_max: Upper bound of the mount-translation offset [m], in the parent-link frame.
        rotation_rpy_min_deg: Lower bound of the rotation offset [deg] about the camera optical
            frame's x (right), y (down), and z (forward) axes.
        rotation_rpy_max_deg: Upper bound of the rotation offset [deg] about the same axes.

    Returns:
        The calibration applied to each randomized environment, keyed by environment index, with
        ``focal_length_px``, ``position_xyz`` [m], and ``rotation_xyzw`` in the sensor's offset
        convention. Isaac Lab's event manager discards event-term return values; this is here so a
        caller driving the term directly -- a dataset re-render, say -- can record what it applied.
    """
    sensor = env.scene[asset_cfg.name]
    selected_env_ids = [int(env_id) for env_id in env_ids.tolist()]
    if not selected_env_ids:
        return {}
    extrinsics = _randomize_extrinsics(
        env,
        selected_env_ids,
        sensor,
        position_xyz_min,
        position_xyz_max,
        rotation_rpy_min_deg,
        rotation_rpy_max_deg,
    )
    if focal_length_scale_range == (1.0, 1.0):
        # Read the cached calibration without rendering or changing the lens model.
        matrices = as_torch(sensor._data.intrinsic_matrices)
        focal_lengths = {env_id: float(matrices[env_id, 0, 0]) for env_id in selected_env_ids}
    else:
        focal_lengths = _randomize_intrinsics(env, selected_env_ids, sensor, focal_length_scale_range)
    if env.cfg.sim.use_fabric:
        _sync_camera_mounts_to_fabric(env, selected_env_ids, sensor)
    return {env_id: {"focal_length_px": (focal_lengths[env_id],), **extrinsics[env_id]} for env_id in selected_env_ids}


def _randomize_extrinsics(
    env: ManagerBasedEnv,
    env_ids: list[int],
    sensor: Camera,
    position_xyz_min: tuple[float, float, float],
    position_xyz_max: tuple[float, float, float],
    rotation_rpy_min_deg: tuple[float, float, float],
    rotation_rpy_max_deg: tuple[float, float, float],
) -> dict[int, dict[str, tuple[float, ...]]]:
    from isaaclab.sim.utils import standardize_xform_ops
    from pxr import Gf, UsdGeom

    offset = sensor.cfg.offset
    nominal_position = torch.tensor(offset.pos, dtype=torch.float64)
    nominal_rotation = pose_math.convert_camera_frame_orientation_convention(
        torch.tensor(offset.rot, dtype=torch.float64), offset.convention, "ros"
    )
    camera_prims = _find_camera_prims_by_env(env, list(range(len(env.scene.env_prim_paths))), sensor)
    nominal_mounts = getattr(sensor, "_autodata_dr_nominal_mounts", {})
    if sensor.cfg.spawn is None and not nominal_mounts:
        # CameraCfg.offset is not applied to cameras loaded from USD.
        nominal_mounts = {
            env_id: Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
            for env_id, prim in camera_prims.items()
        }
        setattr(sensor, "_autodata_dr_nominal_mounts", nominal_mounts)

    # Pin the full local op stack before an inherited source camera is edited.
    selected = set(env_ids)
    for env_id, prim in camera_prims.items():
        if env_id not in selected:
            xform = UsdGeom.Xformable(prim)
            for op in xform.GetOrderedXformOps():
                attribute = op.GetAttr()
                attribute.Set(attribute.Get())
            order = xform.GetXformOpOrderAttr()
            order.Set(order.Get() or [])

    applied = {}
    for env_id in env_ids:
        if sensor.cfg.spawn is None:
            mount = nominal_mounts[env_id]
            nominal_position = torch.tensor(tuple(mount.GetTranslation()), dtype=torch.float64)
            quaternion = mount.GetRotation().GetQuat()
            nominal_rotation = pose_math.convert_camera_frame_orientation_convention(
                torch.tensor((*quaternion.GetImaginary(), quaternion.GetReal()), dtype=torch.float64), "opengl", "ros"
            )
        position_offset = pose_math.sample_uniform_vector(position_xyz_min, position_xyz_max, env.device)
        rotation_offset_deg = pose_math.sample_uniform_vector(rotation_rpy_min_deg, rotation_rpy_max_deg, env.device)
        position = nominal_position + position_offset
        rotation = pose_math.quat_mul(
            nominal_rotation, pose_math.quat_from_euler_xyz(torch.deg2rad(rotation_offset_deg))
        )
        position_xyz = tuple(position.tolist())
        standardize_xform_ops(
            camera_prims[env_id],
            translation=position_xyz,
            orientation=tuple(
                pose_math.convert_camera_frame_orientation_convention(rotation, "ros", "opengl").tolist()
            ),
        )
        applied[env_id] = {
            "position_xyz": position_xyz,
            "rotation_xyzw": tuple(
                pose_math.convert_camera_frame_orientation_convention(rotation, "ros", offset.convention).tolist()
            ),
        }
    return applied


def _randomize_intrinsics(
    env: ManagerBasedEnv,
    env_ids: list[int],
    sensor: Camera,
    focal_length_scale_range: tuple[float, float],
) -> dict[int, float]:
    spawn_cfg = sensor.cfg.spawn
    focal_length = getattr(spawn_cfg, "focal_length", None)
    horizontal_aperture = getattr(spawn_cfg, "horizontal_aperture", None)
    assert focal_length and horizontal_aperture, (
        f"Camera {sensor.cfg.prim_path!r} must be spawned from a pinhole camera config with a focal "
        "length and a horizontal aperture to randomize its intrinsics."
    )

    height, width = sensor.image_shape
    nominal_focal_length_px = width * focal_length / horizontal_aperture
    scales = torch.tensor(
        [pose_math.sample_uniform_scalar(focal_length_scale_range, env.device) for _ in env_ids],
        dtype=torch.float32,
        device=env.device,
    )

    matrices = torch.zeros((len(env_ids), 3, 3), dtype=torch.float32, device=env.device)
    matrices[:, 0, 0] = nominal_focal_length_px * scales
    matrices[:, 1, 1] = nominal_focal_length_px * scales
    matrices[:, 0, 2] = 0.5 * width
    matrices[:, 1, 2] = 0.5 * height
    matrices[:, 2, 2] = 1.0
    distortion = getattr(spawn_cfg, "distortion", None)
    if distortion is not None:
        assert distortion.model == "opencvPinhole", "Camera DR supports OpenCV pinhole calibration only."
        prims = _find_camera_prims_by_env(env, env_ids, sensor)
        for index, env_id in enumerate(env_ids):
            fx, fy, cx, cy = distortion.fx, distortion.fy, distortion.cx, distortion.cy
            scale = float(scales[index])
            matrices[index, 0, 0], matrices[index, 1, 1] = fx * scale, fy * scale
            matrices[index, 0, 2], matrices[index, 1, 2] = cx, cy
            prim = prims[env_id]
            assert (
                prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvPinhole"
            ), "Camera DR requires the renderer to author the configured OpenCV pinhole lens model."
            prefix = "omni:lensdistortion:opencvPinhole:"
            for name, value in (("fx", fx * scale), ("fy", fy * scale), ("cx", cx), ("cy", cy)):
                prim.GetAttribute(prefix + name).Set(value)
            prim.GetAttribute("focalLength").Set(focal_length * scale)
        # Lab's setter skips cameras with an authored OpenCV model. Edit the lens
        # attributes and use its readback to keep the sensor calibration consistent.
        sensor._update_intrinsic_matrices(env_ids)
    else:
        sensor.set_intrinsic_matrices(matrices, focal_length=focal_length, env_ids=env_ids)
    return {env_id: float(matrices[index, 0, 0]) for index, env_id in enumerate(env_ids)}


def _sync_camera_mounts_to_fabric(env: ManagerBasedEnv, env_ids: list[int], sensor: Camera) -> None:
    """Publish local mounts to the live graphics hierarchy, preserving parent tracking.

    USD camera edits alone can leave Fabric's cached mount unchanged. Update the local
    matrix after both pose and lens edits; never derive world poses from USD because
    moving robot links are driven by PhysX/Fabric rather than the USD rest pose.
    """
    import usdrt
    from isaaclab.sim.utils.stage import get_current_stage_id
    from pxr import UsdGeom
    from usdrt import hierarchy

    stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    fabric_hierarchy = hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        stage.GetFabricId(), stage.GetStageIdAsStageId()
    )
    for prim in _find_camera_prims_by_env(env, env_ids, sensor).values():
        local = UsdGeom.Xformable(prim).GetLocalTransformation()
        matrix = usdrt.Gf.Matrix4d(*[value for row in local for value in row])
        fabric_hierarchy.set_local_xform(usdrt.Sdf.Path(str(prim.GetPath())), matrix)
    fabric_hierarchy.update_world_xforms()


def _find_camera_prims_by_env(env: ManagerBasedEnv, env_ids: list[int], sensor: Camera) -> dict[int, Any]:
    """Map each selected environment index to its cloned camera prim."""
    from isaaclab.sim.utils import find_matching_prims
    from pxr import Sdf

    prims = find_matching_prims(sensor.cfg.prim_path, stage=env.scene.stage)
    env_ids_by_path = {Sdf.Path(env.scene.env_prim_paths[env_id]): env_id for env_id in env_ids}
    matches_by_env = {env_id: [] for env_id in env_ids}
    for prim in prims:
        # Match complete ancestor paths, including cameras mounted beneath robot links.
        for ancestor in prim.GetPath().GetParentPath().GetPrefixes():
            env_id = env_ids_by_path.get(ancestor)
            if env_id is not None:
                matches_by_env[env_id].append(prim)

    prims_by_env = {}
    for env_id in env_ids:
        env_path = env.scene.env_prim_paths[env_id]
        matches = matches_by_env[env_id]
        assert (
            len(matches) == 1
        ), f"Expected exactly one camera prim under {env_path!r} for {sensor.cfg.prim_path!r}, found {len(matches)}."
        prims_by_env[env_id] = matches[0]
    return prims_by_env
