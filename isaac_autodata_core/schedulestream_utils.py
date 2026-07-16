"""Standalone helpers for the schedulestream TAMP planner."""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Set

import numpy as np
import trimesh
from isaaclab.sim import find_matching_prims
from pxr import Usd, UsdGeom

from curobo.types import Pose

from curobo._src.util.usd_scene_parser import UsdSceneParser

from schedulestream.applications.custream2.object import GraspConfig, MeshObject
from schedulestream.applications.custream2.utils import (
    multiply_poses,
    position_from_pose,
    simplify_mesh,
    to_matrix,
    to_pose,
)


def get_name_from_path(scene: Any) -> dict:
    """Prim path -> env asset name for the scene's movables (rigid + deformable).

    Deformable assets (e.g. the lift teddy bear) are movable objects too; mapping
    their prim paths names world objects by the env asset name (e.g. "object"),
    not the USD path.
    """
    env_assets = {**dict(scene.rigid_objects), **dict(scene.deformable_objects)}
    return {
        prim.GetPath().pathString: name
        for name, asset in env_assets.items()
        for prim in find_matching_prims(asset.cfg.prim_path)
    }


def destination_from_contact_sensor(scene: Any, contact_sensor_name: str, env_id: int = 0) -> str:
    """Resolve an Arena contact sensor's filtered destination prim into a world object name
    (the success terms identify the destination only through ``filter_prim_paths_expr``)."""
    sensor = scene.sensors[contact_sensor_name]
    [filter_expr] = sensor.cfg.filter_prim_paths_expr
    path = filter_expr.replace(".*", f"{env_id}")
    name_from_path = get_name_from_path(scene)
    for prim_path, name in name_from_path.items():
        if path.startswith(prim_path) or prim_path.startswith(path):
            return name
    raise KeyError(
        f"Contact sensor {contact_sensor_name!r} destination {path!r} matches no parsed world "
        f"object (known: {sorted(set(name_from_path.values()))})"
    )


def prim_relative_pose(
    stage: Any, prim_path: str, reference_prim_path: str, timecode: float = 0.0
) -> Pose:
    """The prim's stage xform re-expressed in the reference prim's frame."""
    time = Usd.TimeCode(timecode)
    matrix = np.array(
        UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(time)
    ).T
    reference_matrix = np.array(
        UsdGeom.Xformable(stage.GetPrimAtPath(reference_prim_path)).ComputeLocalToWorldTransform(time)
    ).T
    return to_pose(np.linalg.inv(reference_matrix) @ matrix)


def is_degenerate_mesh(mesh: trimesh.Trimesh, name: str = "") -> bool:
    """True when the mesh would break sphere fitting (empty, non-finite
    vertices, or too few points for a hull -- NaN into cKDTree)."""
    if (mesh.vertices.size == 0) or (not np.isfinite(mesh.vertices).all()) or (len(mesh.vertices) < 4):
        print(f"[schedulestream] WARNING: skipping obstacle {name!r} with degenerate mesh")
        return True
    return False

# Static obstacles larger than this (max bounding-box extent, metres) or named
# like a floor are dropped from the world: Arena backgrounds (e.g.
# pick_and_place_maple_table) parse the ground plane into a huge mesh that
# swamps collision checking without ever being reachable.
MAX_OBSTACLE_EXTENT = 10.0
FLOOR_NAME_SUBSTRINGS = ("floor", "ground")


def create_objects(
    scene: Any,
    env_id: int = 0,
    floating: Optional[Set[str]] = None,
    verbose: bool = False,
    **simplify_kwargs: Any,
) -> List[Any]:
    """Parse the live composed stage into MeshObjects, merging every prim of
    the same env asset (rigid or deformable) into one object.

    Follows robolab's ``build_objects_from_usd`` (asset-root grouping: verts
    baked into the root prim's frame, object pose set to that root's
    robot-relative xform — the frame the env reports, so pose syncs don't
    shift the mesh) but parses the live composed stage in the robot
    reference frame. Static prim names are de-duplicated with
    numeric suffixes. ``simplify_kwargs`` forward to ``simplify_mesh`` for
    merged assets.

    ``floating`` overrides which assets are movable: when a set is given, an
    asset floats iff its name is in it; when ``None``, rigid objects float
    unless kinematic and deformables always float.
    """
    usd_parser = UsdSceneParser()
    usd_parser.load_stage(scene.stage)
    env_path = scene.env_regex_ns.replace(".*", f"{env_id}")
    robot_path = f"{env_path}/Robot"
    ignore_list = [robot_path, f"{env_path}/target", "/World/defaultGroundPlane", "/curobo"]
    scene_cfg = usd_parser.get_obstacles_from_stage(
        only_paths=[env_path],
        reference_prim_path=robot_path,
        ignore_substring=ignore_list,
        timecode=0,
    )
    assert scene_cfg.objects, f"no obstacles parsed under {env_path}"
    stage = usd_parser.stage

    # Prim path -> env asset name for movables (rigid + deformable).
    name_from_path = get_name_from_path(scene)

    groups: dict[str, list[Any]] = {}
    roots: dict[str, str] = {}
    statics: list[Any] = []
    for obstacle in scene_cfg.objects:
        for prim_path, name in name_from_path.items():
            if obstacle.name.startswith(prim_path):
                groups.setdefault(name, []).append(obstacle)
                roots.setdefault(name, prim_path)
                break
        else:
            statics.append(obstacle)

    # Asset names are reserved; repeated static names get _1, _2, ... suffixes.
    name_counts = Counter(groups.keys())

    objects = []
    # Unmatched prims (background scenery, tables, fixtures) are static
    # collision-only obstacles.
    for obstacle in statics:
        mesh = obstacle.get_trimesh_mesh()
        if is_degenerate_mesh(mesh, obstacle.name):
            continue
        extent = float(max(mesh.extents))
        if (extent > MAX_OBSTACLE_EXTENT) or any(
            part in obstacle.name.lower() for part in FLOOR_NAME_SUBSTRINGS
        ):
            print(
                f"[schedulestream] skipping floor/oversized obstacle {obstacle.name!r}"
                f" (extent {extent:.1f} m)"
            )
            continue
        base = obstacle.name.replace("/", "_").lstrip("_")
        name_counts[base] += 1
        name = base if name_counts[base] == 1 else f"{base}_{name_counts[base] - 1}"
        objects.append(MeshObject(name, mesh, pose=to_pose(obstacle.pose), surface_config=None))

    for name, obstacles in groups.items():
        if len(obstacles) > 1:
            print(f"[schedulestream] merging {name!r} from {len(obstacles)} prims")
        root_pose = prim_relative_pose(stage, roots[name], robot_path)
        root_inv = root_pose.inverse()
        baked = []
        for obstacle in obstacles:
            mesh = obstacle.get_trimesh_mesh().copy()
            if is_degenerate_mesh(mesh, obstacle.name):
                continue
            mesh.apply_transform(to_matrix(multiply_poses(root_inv, to_pose(obstacle.pose))))
            baked.append(mesh)
        if not baked:
            continue
        mesh = baked[0] if len(baked) == 1 else trimesh.util.concatenate(baked)
        mesh = simplify_mesh(mesh, **simplify_kwargs)
        # Movability mirrors _convert_objects: matched rigid objects unless
        # kinematic; matched deformables always.
        if floating is not None:
            is_floating = name in floating
        elif name in scene.rigid_objects:
            spawn = scene.rigid_objects[name].cfg.spawn
            is_floating = not (
                (spawn is not None)
                and (spawn.rigid_props is not None)
                and spawn.rigid_props.kinematic_enabled
            )
        else:
            is_floating = True
        grasp_config = GraspConfig() if is_floating else None
        objects.append(MeshObject(name, mesh, pose=root_pose, grasp_config=grasp_config, surface_config=None))
        if verbose:
            print(
                f"[schedulestream] Object: {name} | Prims: {len(obstacles)} | Floating: {is_floating}"
                f" | Position: {np.round(position_from_pose(root_pose), 2)}"
            )
    return objects
