# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deform the rope's original round visual mesh for Newton's Kit viewport."""

import numpy as np

import warp as wp
from isaaclab.scene_data.deformable_vis_remap import VolumeVisRemap, build_volume_vis_barycentric_remap
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab_contrib.coupling.coupler import NewtonCouplerManager


@wp.kernel(enable_backward=False)
def _skin_rope_points(
    points: wp.fabricarrayarray(dtype=wp.vec3f),
    matrices: wp.fabricarray(dtype=wp.mat44d),
    offsets: wp.fabricarray(dtype=wp.uint32),
    particles: wp.array(dtype=wp.vec3f),
    indices: wp.array2d(dtype=wp.int32),
    weights: wp.array2d(dtype=wp.float32),
):
    instance, vertex = wp.tid()
    offset = int(offsets[instance])
    world = wp.vec3f(0.0)
    for corner in range(4):
        world += weights[vertex, corner] * particles[offset + indices[vertex, corner]]
    world_to_local = wp.inverse(wp.transpose(wp.mat44f(matrices[instance])))
    points[instance][vertex] = wp.transform_point(world_to_local, world)


class NewtonRopeManager(NewtonCouplerManager):
    """Add visual skinning to the standard rigid/deformable Newton coupler.

    The pinned Lab backend copies simulation nodes into visual vertices by index.
    This rope has different simulation and visual topologies. Embed its original
    round mesh in the simulation tetrahedra and update every visual vertex before
    Kit renders. The solver, collision geometry, and particle state are unchanged.
    """

    _rope_vis_remap: VolumeVisRemap | None = None
    _rope_offset_attr = "autodata:rope:particleOffset"

    @classmethod
    def _initialize_rope_skin(cls) -> None:
        import usdrt
        from pxr import Gf, UsdGeom

        assert len(cls._deformable_registry) == 1, "The rope task expects one deformable asset"
        entry = cls._deformable_registry[0]
        stage = get_current_stage()
        paths = find_matching_prim_paths(entry.vis_mesh_prim_path)
        assert paths, "Rope visual mesh is missing"
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(paths[0]))
        simulation = UsdGeom.TetMesh(
            stage.GetPrimAtPath(visual.GetPath().GetParentPath().AppendChild("simulation_mesh"))
        )
        xforms = UsdGeom.XformCache()
        visual_to_sim = (
            xforms.GetLocalToWorldTransform(visual.GetPrim())
            * xforms.GetLocalToWorldTransform(simulation.GetPrim()).GetInverse()
        )
        visual_points = np.asarray(
            [visual_to_sim.Transform(Gf.Vec3d(point)) for point in visual.GetPointsAttr().Get()], dtype=np.float32
        )
        cls._rope_vis_remap = build_volume_vis_barycentric_remap(
            np.asarray(simulation.GetPointsAttr().Get(), dtype=np.float32),
            np.asarray(simulation.GetTetVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1),
            visual_points,
            device=str(cls._device),
        )
        assert cls._rope_vis_remap is not None, "Cannot embed the rope visual mesh in its simulation tetrahedra"
        for path in paths:
            env_index = int(path.split("/envs/env_", 1)[1].split("/", 1)[0])
            prim = cls._usdrt_stage.GetPrimAtPath(path)
            usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
            prim.CreateAttribute(cls._rope_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True).Set(
                entry.particle_offsets[env_index]
            )

    @classmethod
    def pre_render(cls) -> None:
        """Refresh the round skin before Kit renders, including after state restores."""
        super().pre_render()
        if cls._usdrt_stage is None or cls._clone_physics_only or cls._state_0 is None:
            return
        import usdrt

        if cls._rope_vis_remap is None:
            cls._initialize_rope_skin()
        selection = cls._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, cls._rope_offset_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            ],
            device=str(cls._device),
        )
        wp.launch(
            _skin_rope_points,
            dim=(selection.GetCount(), cls._rope_vis_remap.tet_vertex_indices.shape[0]),
            inputs=[
                wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
                wp.fabricarray(data=selection, attrib=cls._rope_offset_attr),
                cls._state_0.particle_q,
                cls._rope_vis_remap.tet_vertex_indices,
                cls._rope_vis_remap.bary_weights,
            ],
            device=cls._device,
        )

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._rope_vis_remap = None
        super()._solver_specific_clear()
