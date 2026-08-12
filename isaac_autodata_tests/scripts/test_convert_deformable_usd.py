# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for deformable USD schema detection and tetrahedral topology conversion."""

from __future__ import annotations

import pytest
from pxr import Gf, Sdf, Usd

from scripts.convert_deformable_usd import (
    DeformableSchemaKind,
    _load_usd_modules,
    _remove_api_schema,
    _surface_faces,
    _tetrahedra_from_flat_indices,
    inspect_stage,
)

_load_usd_modules()


def test_detect_legacy_volume_schema() -> None:
    """A mesh with legacy simulation arrays is classified as legacy volume."""

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root)
    mesh = stage.DefinePrim("/World/Object", "Mesh")
    mesh.CreateAttribute("physxDeformable:simulationPoints", Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    )
    mesh.CreateAttribute("physxDeformable:simulationIndices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])

    inspection = inspect_stage(stage)

    assert inspection.kind is DeformableSchemaKind.LEGACY_VOLUME
    assert inspection.default_prim_path == "/World"
    assert inspection.legacy_source_prim_paths == ("/World/Object",)


def test_remove_unregistered_api_schema() -> None:
    """Obsolete API tokens can be removed without requiring schema registration."""

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World", "Xform")
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["MaterialBindingAPI", "ObsoleteDeformableAPI"]),
    )

    _remove_api_schema(prim, "ObsoleteDeformableAPI")

    assert prim.GetMetadata("apiSchemas").GetAppliedItems() == ["MaterialBindingAPI"]


def test_group_and_validate_tetrahedra() -> None:
    """Flat tetrahedral indices are grouped without changing connectivity."""

    tetrahedra = _tetrahedra_from_flat_indices([0, 1, 2, 3, 1, 2, 3, 4], vertex_count=5, label="test")

    assert tetrahedra == [(0, 1, 2, 3), (1, 2, 3, 4)]


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([0, 1, 2], "divisible by four"),
        ([0, 1, 1, 3], "duplicate vertices"),
        ([0, 1, 2, 4], "references vertex 4"),
    ],
)
def test_reject_invalid_tetrahedra(indices: list[int], message: str) -> None:
    """Malformed legacy topology fails with a targeted diagnostic."""

    with pytest.raises(AssertionError, match=message):
        _tetrahedra_from_flat_indices(indices, vertex_count=4, label="test")


def test_extract_oriented_surface_faces() -> None:
    """A positive-volume tetrahedron produces four outward boundary triangles."""

    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 0.0, 1.0),
    ]

    faces = _surface_faces([(0, 1, 2, 3)], points, "test")

    assert faces == [(0, 1, 3), (0, 2, 1), (0, 3, 2), (1, 2, 3)]


def test_remove_shared_tetrahedral_face() -> None:
    """The shared face of two tetrahedra is excluded from the surface mesh."""

    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 0.0, 1.0),
        Gf.Vec3f(0.0, 0.0, -1.0),
    ]

    faces = _surface_faces([(0, 1, 2, 3), (0, 2, 1, 4)], points, "test")

    assert len(faces) == 6
    assert (0, 1, 2) not in {tuple(sorted(face)) for face in faces}
