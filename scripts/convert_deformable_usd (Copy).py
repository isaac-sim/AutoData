# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert legacy PhysX volumetric deformable USD assets to the current schema.

The converter recognizes three asset families:

* legacy volumetric deformables whose simulation and collision tetrahedra are
  stored in ``physxDeformable:*`` attributes on a render mesh;
* current OmniPhysics volumetric deformables; and
* current OmniPhysics surface deformables.

Only the legacy volumetric representation is rewritten. Current assets are
validated and, when an output path is provided, copied without modification.
Unknown representations fail instead of being converted heuristically.

Usage::

    python scripts/convert_deformable_usd.py \
        --input_usd ./Rope.usd \
        --output_usd ./Rope_sim6.usd \
        --density 1000 \
        --static_friction 0.25 \
        --poissons_ratio 0.45
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
else:
    Gf = None
    Sdf = None
    Usd = None
    UsdGeom = None
    UsdShade = None
    Vt = None


class DeformableSchemaKind(enum.StrEnum):
    """Deformable schema families recognized by the converter."""

    LEGACY_VOLUME = "legacy_volume"
    CURRENT_VOLUME = "current_volume"
    CURRENT_SURFACE = "current_surface"
    UNSUPPORTED = "unsupported"


@dataclasses.dataclass(frozen=True)
class AssetInspection:
    """Summary of the deformable representation authored in a USD stage."""

    kind: DeformableSchemaKind
    default_prim_path: str | None
    body_prim_paths: tuple[str, ...]
    simulation_prim_paths: tuple[str, ...]
    legacy_source_prim_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MaterialOverrides:
    """Optional values used when the legacy material does not author a property."""

    density: float | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    youngs_modulus: float | None = None
    poissons_ratio: float | None = None
    elasticity_damping: float | None = None


_LEGACY_SIMULATION_POINTS = "physxDeformable:simulationPoints"
_LEGACY_SIMULATION_REST_POINTS = "physxDeformable:simulationRestPoints"
_LEGACY_SIMULATION_INDICES = "physxDeformable:simulationIndices"
_LEGACY_COLLISION_POINTS = "physxDeformable:collisionPoints"
_LEGACY_COLLISION_REST_POINTS = "physxDeformable:collisionRestPoints"
_LEGACY_COLLISION_INDICES = "physxDeformable:collisionIndices"

_CURRENT_BODY_SCHEMAS = {
    "OmniPhysicsDeformableBodyAPI",
    "PhysxBaseDeformableBodyAPI",
}


def _load_usd_modules() -> None:
    """Load USD modules after Isaac Sim starts.

    Isaac Sim requires :mod:`pxr` to be imported after :class:`AppLauncher`
    creates the application. Unit tests may call this helper directly because
    they do not launch Kit.
    """

    global Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from pxr import Gf as gf_module
    from pxr import Sdf as sdf_module
    from pxr import Usd as usd_module
    from pxr import UsdGeom as usd_geom_module
    from pxr import UsdShade as usd_shade_module
    from pxr import Vt as vt_module

    Gf = gf_module
    Sdf = sdf_module
    Usd = usd_module
    UsdGeom = usd_geom_module
    UsdShade = usd_shade_module
    Vt = vt_module


def inspect_stage(stage: Usd.Stage) -> AssetInspection:
    """Inspect a USD stage and classify its deformable representation.

    Args:
        stage: Open USD stage to inspect.

    Returns:
        The detected schema family and relevant prim paths.
    """

    legacy_sources: list[str] = []
    volume_simulation_prims: list[str] = []
    surface_simulation_prims: list[str] = []
    body_prims: list[str] = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        schemas = set(prim.GetAppliedSchemas())

        if prim.HasAttribute(_LEGACY_SIMULATION_POINTS) and prim.HasAttribute(_LEGACY_SIMULATION_INDICES):
            legacy_sources.append(path)
        if "OmniPhysicsVolumeDeformableSimAPI" in schemas:
            volume_simulation_prims.append(path)
        if "OmniPhysicsSurfaceDeformableSimAPI" in schemas:
            surface_simulation_prims.append(path)
        if _CURRENT_BODY_SCHEMAS.issubset(schemas):
            body_prims.append(path)

    if legacy_sources:
        kind = DeformableSchemaKind.LEGACY_VOLUME
        simulation_prims: list[str] = []
    elif surface_simulation_prims and not volume_simulation_prims:
        kind = DeformableSchemaKind.CURRENT_SURFACE
        simulation_prims = surface_simulation_prims
    elif volume_simulation_prims and not surface_simulation_prims:
        kind = DeformableSchemaKind.CURRENT_VOLUME
        simulation_prims = volume_simulation_prims
    else:
        kind = DeformableSchemaKind.UNSUPPORTED
        simulation_prims = volume_simulation_prims + surface_simulation_prims

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    return AssetInspection(
        kind=kind,
        default_prim_path=default_prim_path,
        body_prim_paths=tuple(body_prims),
        simulation_prim_paths=tuple(simulation_prims),
        legacy_source_prim_paths=tuple(legacy_sources),
    )


def _tetrahedra_from_flat_indices(
    indices: list[int] | Vt.IntArray,
    vertex_count: int,
    label: str,
) -> list[tuple[int, int, int, int]]:
    """Validate and group a flat tetrahedral index array."""

    flat_indices = [int(index) for index in indices]
    assert flat_indices, f"{label} indices are empty"
    assert len(flat_indices) % 4 == 0, f"{label} index count must be divisible by four"

    tetrahedra: list[tuple[int, int, int, int]] = []
    for offset in range(0, len(flat_indices), 4):
        tet = tuple(flat_indices[offset : offset + 4])
        assert len(set(tet)) == 4, f"{label} tetrahedron {offset // 4} contains duplicate vertices: {tet}"
        assert min(tet) >= 0, f"{label} tetrahedron {offset // 4} contains a negative index: {tet}"
        assert (
            max(tet) < vertex_count
        ), f"{label} tetrahedron {offset // 4} references vertex {max(tet)}, but the mesh has {vertex_count} vertices"
        tetrahedra.append(tet)
    return tetrahedra


def _surface_faces(
    tetrahedra: list[tuple[int, int, int, int]],
    points: list[Gf.Vec3f] | Vt.Vec3fArray,
    label: str,
) -> list[tuple[int, int, int]]:
    """Extract consistently oriented boundary triangles from tetrahedra."""

    face_counts: Counter[tuple[int, int, int]] = Counter()
    oriented_faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for tet_index, (a, b, c, d) in enumerate(tetrahedra):
        point_a = Gf.Vec3d(points[a])
        point_b = Gf.Vec3d(points[b])
        point_c = Gf.Vec3d(points[c])
        point_d = Gf.Vec3d(points[d])
        signed_volume_six = Gf.Dot(point_b - point_a, Gf.Cross(point_c - point_a, point_d - point_a))
        assert abs(signed_volume_six) > 1.0e-15, f"{label} tetrahedron {tet_index} is degenerate"

        faces = [
            (b, c, d),
            (a, d, c),
            (a, b, d),
            (a, c, b),
        ]
        if signed_volume_six < 0.0:
            faces = [(face[0], face[2], face[1]) for face in faces]

        for face in faces:
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces.setdefault(key, face)

    non_manifold_faces = [key for key, count in face_counts.items() if count > 2]
    assert not non_manifold_faces, f"{label} mesh has non-manifold tetrahedral faces: {non_manifold_faces[:5]}"
    boundary_faces = [oriented_faces[key] for key, count in face_counts.items() if count == 1]
    return sorted(boundary_faces)


def _required_array(prim: Usd.Prim, name: str) -> object:
    """Read a required authored array attribute."""

    attribute = prim.GetAttribute(name)
    assert attribute and attribute.HasAuthoredValueOpinion(), f"Missing required attribute {name} on {prim.GetPath()}"
    value = attribute.Get()
    assert value is not None and len(value) > 0, f"Attribute {name} on {prim.GetPath()} is empty"
    return value


def _optional_array(prim: Usd.Prim, name: str, fallback: object) -> object:
    """Read an optional array attribute or return a fallback value."""

    attribute = prim.GetAttribute(name)
    if attribute and attribute.HasAuthoredValueOpinion():
        value = attribute.Get()
        if value is not None and len(value) > 0:
            return value
    return fallback


def _transform_points(
    points: list[Gf.Vec3f] | Vt.Vec3fArray,
    transform: Gf.Matrix4d,
) -> Vt.Vec3fArray:
    """Transform points and return a float-precision USD array."""

    return Vt.Vec3fArray([Gf.Vec3f(transform.Transform(Gf.Vec3d(point))) for point in points])


def _apply_api(prim: Usd.Prim, schema_name: str, instance_name: str | None = None) -> None:
    """Apply a codeless API schema and fail if it is unavailable."""

    applied = prim.ApplyAPI(schema_name, instance_name) if instance_name else prim.ApplyAPI(schema_name)
    assert applied, f"Could not apply {schema_name} to {prim.GetPath()}; launch the script through Isaac Sim 6"


def _remove_api_schema(prim: Usd.Prim, schema_name: str) -> None:
    """Remove an authored API token, including schemas no longer registered."""

    schemas = prim.GetMetadata("apiSchemas")
    if schemas is None:
        return
    applied_schemas = [schema for schema in schemas.GetAppliedItems() if schema != schema_name]
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(applied_schemas))


def _author_pose(prim: Usd.Prim, points: Vt.Vec3fArray) -> None:
    """Author the default deformable bind pose on a prim."""

    _apply_api(prim, "OmniPhysicsDeformablePoseAPI", "default")
    prim.CreateAttribute(
        "deformablePose:default:omniphysics:points",
        Sdf.ValueTypeNames.Point3fArray,
        custom=False,
    ).Set(points)
    prim.CreateAttribute(
        "deformablePose:default:omniphysics:purposes",
        Sdf.ValueTypeNames.TokenArray,
        custom=False,
    ).Set(Vt.TokenArray(["bindPose"]))


def _author_tet_mesh(
    stage: Usd.Stage,
    path: Sdf.Path,
    points: Vt.Vec3fArray,
    tetrahedra: list[tuple[int, int, int, int]],
    surface_faces: list[tuple[int, int, int]],
) -> Usd.Prim:
    """Create an explicit tetrahedral mesh."""

    assert not stage.GetPrimAtPath(path), f"Output prim already exists: {path}"
    mesh = UsdGeom.TetMesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray([Gf.Vec4i(*tet) for tet in tetrahedra]))
    mesh.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray([Gf.Vec3i(*face) for face in surface_faces]))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return mesh.GetPrim()


def _legacy_scalar(prim: Usd.Prim, name: str) -> float | int | bool | None:
    """Read a legacy scalar when it has an authored value."""

    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    return attribute.Get()


def _find_physics_material(stage: Usd.Stage, source_prim: Usd.Prim, body_prim: Usd.Prim) -> Usd.Prim:
    """Find the legacy physics material or create one under the body root."""

    binding = source_prim.GetRelationship("material:binding:physics")
    if binding:
        targets = binding.GetTargets()
        if targets:
            material_prim = stage.GetPrimAtPath(targets[0])
            if material_prim:
                return material_prim

    for prim in stage.Traverse():
        if any(attribute.GetName().startswith("physxDeformableBodyMaterial:") for attribute in prim.GetAttributes()):
            return prim

    material_path = body_prim.GetPath().AppendChild("PhysicsMaterial")
    return UsdShade.Material.Define(stage, material_path).GetPrim()


def _author_material(
    material_prim: Usd.Prim,
    overrides: MaterialOverrides,
) -> list[str]:
    """Migrate legacy material values and report properties left to schema defaults."""

    _apply_api(material_prim, "OmniPhysicsDeformableMaterialAPI")
    _apply_api(material_prim, "PhysxDeformableMaterialAPI")

    mappings: tuple[tuple[str, str, str, Sdf.ValueTypeName, float | None], ...] = (
        (
            "density",
            "physxDeformableBodyMaterial:density",
            "omniphysics:density",
            Sdf.ValueTypeNames.Float,
            overrides.density,
        ),
        (
            "static friction",
            "physxDeformableBodyMaterial:staticFriction",
            "omniphysics:staticFriction",
            Sdf.ValueTypeNames.Float,
            overrides.static_friction,
        ),
        (
            "dynamic friction",
            "physxDeformableBodyMaterial:dynamicFriction",
            "omniphysics:dynamicFriction",
            Sdf.ValueTypeNames.Float,
            overrides.dynamic_friction,
        ),
        (
            "Young's modulus",
            "physxDeformableBodyMaterial:youngsModulus",
            "omniphysics:youngsModulus",
            Sdf.ValueTypeNames.Float,
            overrides.youngs_modulus,
        ),
        (
            "Poisson's ratio",
            "physxDeformableBodyMaterial:poissonsRatio",
            "omniphysics:poissonsRatio",
            Sdf.ValueTypeNames.Float,
            overrides.poissons_ratio,
        ),
        (
            "elasticity damping",
            "physxDeformableBodyMaterial:elasticityDamping",
            "physxDeformableMaterial:elasticityDamping",
            Sdf.ValueTypeNames.Float,
            overrides.elasticity_damping,
        ),
    )

    warnings: list[str] = []
    for label, legacy_name, current_name, value_type, override in mappings:
        legacy_value = _legacy_scalar(material_prim, legacy_name)
        value = override if override is not None else legacy_value
        if value is None:
            warnings.append(f"{label} is not authored; Isaac Sim 6 will use the schema default")
        else:
            material_prim.CreateAttribute(current_name, value_type, custom=False).Set(float(value))

    for attribute in list(material_prim.GetAuthoredAttributes()):
        if attribute.GetName().startswith("physxDeformableBodyMaterial:"):
            material_prim.RemoveProperty(attribute.GetName())
    return warnings


def _copy_current_asset(input_path: Path, output_path: Path, overwrite: bool) -> None:
    """Copy an already-current asset without rewriting its USD layer."""

    if input_path.resolve() == output_path.resolve():
        assert overwrite, "Input and output paths are identical; pass --overwrite to validate in place"
        return
    assert overwrite or not output_path.exists(), f"Output file already exists: {output_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def _convert_legacy_volume(
    input_path: Path,
    output_path: Path,
    inspection: AssetInspection,
    source_prim_path: str | None,
    body_prim_path: str | None,
    overrides: MaterialOverrides,
    overwrite: bool,
) -> list[str]:
    """Convert one legacy volumetric deformable and export a flattened stage."""

    assert overwrite or not output_path.exists(), f"Output file already exists: {output_path}"
    assert (
        input_path.resolve() != output_path.resolve() or overwrite
    ), "Input and output paths are identical; pass --overwrite to replace the source"

    source_stage = Usd.Stage.Open(str(input_path))
    assert source_stage, f"Could not open input USD: {input_path}"
    flattened_layer = source_stage.Flatten()
    stage = Usd.Stage.Open(flattened_layer)
    assert stage, f"Could not create a writable stage for {input_path}"

    candidate_paths = inspection.legacy_source_prim_paths
    if source_prim_path is None:
        assert len(candidate_paths) == 1, (
            "Expected exactly one legacy deformable mesh, found "
            f"{len(candidate_paths)}: {list(candidate_paths)}. Pass --source_prim."
        )
        source_prim_path = candidate_paths[0]
    assert (
        source_prim_path in candidate_paths
    ), f"Source prim {source_prim_path} is not a detected legacy deformable mesh: {list(candidate_paths)}"

    source_prim = stage.GetPrimAtPath(source_prim_path)
    assert source_prim and source_prim.IsA(UsdGeom.Mesh), f"Legacy source prim must be a Mesh: {source_prim_path}"

    if body_prim_path is None:
        body_prim_path = str(source_prim.GetParent().GetPath())
    body_prim = stage.GetPrimAtPath(body_prim_path)
    assert body_prim, f"Body prim does not exist: {body_prim_path}"
    assert source_prim.GetPath().HasPrefix(
        body_prim.GetPath()
    ), f"Source prim {source_prim_path} must be under body prim {body_prim_path}"

    simulation_pose_points = _required_array(source_prim, _LEGACY_SIMULATION_POINTS)
    simulation_rest_points = _optional_array(
        source_prim,
        _LEGACY_SIMULATION_REST_POINTS,
        simulation_pose_points,
    )
    collision_pose_points = _required_array(source_prim, _LEGACY_COLLISION_POINTS)
    collision_rest_points = _optional_array(
        source_prim,
        _LEGACY_COLLISION_REST_POINTS,
        collision_pose_points,
    )

    simulation_tetrahedra = _tetrahedra_from_flat_indices(
        _required_array(source_prim, _LEGACY_SIMULATION_INDICES),
        len(simulation_rest_points),
        "simulation",
    )
    collision_tetrahedra = _tetrahedra_from_flat_indices(
        _required_array(source_prim, _LEGACY_COLLISION_INDICES),
        len(collision_rest_points),
        "collision",
    )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
    body_to_world = xform_cache.GetLocalToWorldTransform(body_prim)
    source_to_body = source_to_world * body_to_world.GetInverse()

    simulation_pose_points = _transform_points(simulation_pose_points, source_to_body)
    simulation_rest_points = _transform_points(simulation_rest_points, source_to_body)
    collision_pose_points = _transform_points(collision_pose_points, source_to_body)
    collision_rest_points = _transform_points(collision_rest_points, source_to_body)

    simulation_faces = _surface_faces(simulation_tetrahedra, simulation_rest_points, "simulation")
    collision_faces = _surface_faces(collision_tetrahedra, collision_rest_points, "collision")

    simulation_prim = _author_tet_mesh(
        stage,
        body_prim.GetPath().AppendChild("simulation_mesh"),
        simulation_rest_points,
        simulation_tetrahedra,
        simulation_faces,
    )
    _apply_api(simulation_prim, "OmniPhysicsVolumeDeformableSimAPI")
    _author_pose(simulation_prim, simulation_pose_points)
    simulation_prim.CreateAttribute(
        "omniphysics:restShapePoints",
        Sdf.ValueTypeNames.Point3fArray,
        custom=False,
    ).Set(simulation_rest_points)
    simulation_prim.CreateAttribute(
        "omniphysics:restTetVtxIndices",
        Sdf.ValueTypeNames.Int4Array,
        custom=False,
    ).Set(Vt.Vec4iArray([Gf.Vec4i(*tet) for tet in simulation_tetrahedra]))

    collision_prim = _author_tet_mesh(
        stage,
        body_prim.GetPath().AppendChild("collision_mesh"),
        collision_rest_points,
        collision_tetrahedra,
        collision_faces,
    )
    _apply_api(collision_prim, "PhysicsCollisionAPI")
    _apply_api(collision_prim, "PhysxCollisionAPI")
    _author_pose(collision_prim, collision_pose_points)

    contact_offset = _legacy_scalar(source_prim, "physxCollision:contactOffset")
    rest_offset = _legacy_scalar(source_prim, "physxCollision:restOffset")
    if contact_offset is not None:
        collision_prim.CreateAttribute(
            "physxCollision:contactOffset",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(float(contact_offset))
    if rest_offset is not None:
        collision_prim.CreateAttribute(
            "physxCollision:restOffset",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(float(rest_offset))

    _apply_api(body_prim, "OmniPhysicsDeformableBodyAPI")
    _apply_api(body_prim, "PhysxBaseDeformableBodyAPI")
    UsdShade.MaterialBindingAPI.Apply(body_prim)

    solver_iterations = _legacy_scalar(source_prim, "physxDeformable:solverPositionIterationCount")
    linear_damping = _legacy_scalar(source_prim, "physxDeformable:vertexVelocityDamping")
    enable_ccd = _legacy_scalar(source_prim, "physxDeformable:enableCCD")
    if solver_iterations is not None:
        body_prim.CreateAttribute(
            "physxDeformableBody:solverPositionIterationCount",
            Sdf.ValueTypeNames.UInt,
            custom=False,
        ).Set(int(solver_iterations))
    if linear_damping is not None:
        body_prim.CreateAttribute(
            "physxDeformableBody:linearDamping",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(float(linear_damping))
    if enable_ccd is not None:
        body_prim.CreateAttribute(
            "physxDeformableBody:enableSpeculativeCCD",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(bool(enable_ccd))

    material_prim = _find_physics_material(stage, source_prim, body_prim)
    warnings = _author_material(material_prim, overrides)
    body_prim.CreateRelationship("material:binding:physics", custom=False).SetTargets([material_prim.GetPath()])

    render_points = _required_array(source_prim, "points")
    _author_pose(source_prim, Vt.Vec3fArray(render_points))
    source_prim.RemoveProperty("normals")
    source_prim.RemoveProperty("material:binding:physics")
    _remove_api_schema(source_prim, "PhysxDeformableBodyAPI")
    _remove_api_schema(source_prim, "PhysxCollisionAPI")
    _remove_api_schema(source_prim, "PhysicsCollisionAPI")
    for attribute in list(source_prim.GetAuthoredAttributes()):
        if attribute.GetName().startswith(("physxDeformable:", "physxCollision:")):
            source_prim.RemoveProperty(attribute.GetName())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert stage.GetRootLayer().Export(str(output_path)), f"Could not export converted USD: {output_path}"
    return warnings


def _validate_current_asset(stage: Usd.Stage, inspection: AssetInspection) -> None:
    """Check the minimum invariants expected by Isaac Lab's deformable view."""

    assert inspection.body_prim_paths, "Current deformable asset has no recognized body prim"
    assert inspection.simulation_prim_paths, "Current deformable asset has no recognized simulation mesh"

    for path in inspection.simulation_prim_paths:
        prim = stage.GetPrimAtPath(path)
        points = prim.GetAttribute("points").Get()
        assert points is not None and len(points) > 0, f"Simulation mesh has no points: {path}"
        pose_points = prim.GetAttribute("deformablePose:default:omniphysics:points").Get()
        assert pose_points is not None and len(pose_points) == len(
            points
        ), f"Simulation pose count does not match point count on {path}"

        if inspection.kind is DeformableSchemaKind.CURRENT_VOLUME:
            tetrahedra = prim.GetAttribute("tetVertexIndices").Get()
            assert tetrahedra is not None and len(tetrahedra) > 0, f"Volume mesh has no tetrahedra: {path}"
        elif inspection.kind is DeformableSchemaKind.CURRENT_SURFACE:
            triangles = prim.GetAttribute("omniphysics:restTriVtxIndices").Get()
            assert triangles is not None and len(triangles) > 0, f"Surface mesh has no triangles: {path}"


def _print_inspection(stage: Usd.Stage, inspection: AssetInspection) -> None:
    """Print a concise human-readable asset report."""

    print(f"Schema kind: {inspection.kind.value}")
    print(f"Default prim: {inspection.default_prim_path or '<none>'}")
    print(f"Body prims: {list(inspection.body_prim_paths)}")
    print(f"Simulation prims: {list(inspection.simulation_prim_paths)}")
    print(f"Legacy source prims: {list(inspection.legacy_source_prim_paths)}")

    for path in inspection.simulation_prim_paths:
        prim = stage.GetPrimAtPath(path)
        points = prim.GetAttribute("points").Get()
        if inspection.kind is DeformableSchemaKind.CURRENT_VOLUME:
            elements = prim.GetAttribute("tetVertexIndices").Get()
            element_label = "tetrahedra"
        else:
            elements = prim.GetAttribute("omniphysics:restTriVtxIndices").Get()
            element_label = "triangles"
        print(f"  {path}: {len(points)} points, {len(elements)} {element_label}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_usd", type=Path, required=True, help="Input deformable USD file.")
    parser.add_argument(
        "--output_usd",
        type=Path,
        default=None,
        help="Output USD file. Required when converting a legacy asset.",
    )
    parser.add_argument(
        "--source_prim",
        type=str,
        default=None,
        help="Legacy render/deformable Mesh prim. Required only when multiple candidates are present.",
    )
    parser.add_argument(
        "--body_prim",
        type=str,
        default=None,
        help="Body root prim. Defaults to the legacy source mesh's parent.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Inspect and validate the input without converting or copying it.",
    )
    parser.add_argument("--density", type=float, default=None, help="Override deformable density [kg/m^3].")
    parser.add_argument("--static_friction", type=float, default=None, help="Override static friction coefficient.")
    parser.add_argument("--dynamic_friction", type=float, default=None, help="Override dynamic friction coefficient.")
    parser.add_argument("--youngs_modulus", type=float, default=None, help="Override Young's modulus [Pa].")
    parser.add_argument("--poissons_ratio", type=float, default=None, help="Override Poisson's ratio.")
    parser.add_argument("--elasticity_damping", type=float, default=None, help="Override elasticity damping.")
    return parser


def main() -> None:
    """Run schema detection, conversion, and output validation."""

    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    input_path = args.input_usd.expanduser().resolve()
    assert input_path.is_file(), f"Input USD does not exist: {input_path}"
    output_path = args.output_usd.expanduser().resolve() if args.output_usd is not None else None

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _load_usd_modules()
        stage = Usd.Stage.Open(str(input_path))
        assert stage, f"Could not open input USD: {input_path}"
        inspection = inspect_stage(stage)
        _print_inspection(stage, inspection)

        if inspection.kind is DeformableSchemaKind.UNSUPPORTED:
            raise AssertionError("Unsupported or mixed deformable schema; no output was written")

        if args.validate_only:
            if inspection.kind is not DeformableSchemaKind.LEGACY_VOLUME:
                _validate_current_asset(stage, inspection)
            print("Validation passed; no output was written.")
            return

        if inspection.kind is DeformableSchemaKind.LEGACY_VOLUME:
            assert output_path is not None, "--output_usd is required for a legacy asset conversion"
            warnings = _convert_legacy_volume(
                input_path=input_path,
                output_path=output_path,
                inspection=inspection,
                source_prim_path=args.source_prim,
                body_prim_path=args.body_prim,
                overrides=MaterialOverrides(
                    density=args.density,
                    static_friction=args.static_friction,
                    dynamic_friction=args.dynamic_friction,
                    youngs_modulus=args.youngs_modulus,
                    poissons_ratio=args.poissons_ratio,
                    elasticity_damping=args.elasticity_damping,
                ),
                overwrite=args.overwrite,
            )
            converted_stage = Usd.Stage.Open(str(output_path))
            assert converted_stage, f"Could not reopen converted USD: {output_path}"
            converted_inspection = inspect_stage(converted_stage)
            assert (
                converted_inspection.kind is DeformableSchemaKind.CURRENT_VOLUME
            ), f"Converted output classified as {converted_inspection.kind.value}"
            _validate_current_asset(converted_stage, converted_inspection)
            print(f"Converted asset written to: {output_path}")
            _print_inspection(converted_stage, converted_inspection)
            for warning in warnings:
                print(f"WARNING: {warning}")
        else:
            _validate_current_asset(stage, inspection)
            print("Asset already uses a current Isaac Sim deformable schema; conversion is not required.")
            if output_path is not None:
                _copy_current_asset(input_path, output_path, args.overwrite)
                print(f"Unmodified compatible asset copied to: {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
