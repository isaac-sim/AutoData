# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic collision-sphere fitting for URDF link geometry."""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import os
import tempfile
import trimesh
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class CollisionSphere:
    """One collision sphere expressed in its owning link frame.

    Args:
        center: Link-relative XYZ center [m].
        radius: Positive sphere radius [m].
    """

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        """Require a finite three-dimensional center and positive finite radius."""

        assert len(self.center) == 3 and all(math.isfinite(value) for value in self.center)
        assert math.isfinite(self.radius) and self.radius > 0.0

    def to_dict(self) -> dict[str, object]:
        """Return the mapping accepted by cuRobo robot configuration files."""

        return {"center": list(self.center), "radius": self.radius}


@dataclass(frozen=True, kw_only=True)
class ObjectCollisionSpheres:
    """Immutable object-frame spheres captured from one native attachment fit.

    These spheres preserve the exact approximation used by cuRobo for carried-object planning. The fit
    does not claim conservative coverage of the source geometry.

    Args:
        object_name: Exact logical scene-object name.
        obstacle_path: Exact obstacle path resolved in cuRobo's world.
        spheres: Active spheres expressed in the object frame.
        fit_type: Native sphere-fit algorithm name.
        surface_sphere_radius_m: Surface-sample radius [m].
        slot_capacity: Number of reserved native attachment slots.
        tool_frame: Physical tool frame used for capture.
        attachment_link: Identity-fixed virtual link owning native attachment slots.
        captured_tool_from_object: Capture-time rigid transform from object to tool.
    """

    object_name: str
    obstacle_path: str
    spheres: tuple[CollisionSphere, ...]
    fit_type: str
    surface_sphere_radius_m: float
    slot_capacity: int
    tool_frame: str
    attachment_link: str
    captured_tool_from_object: tuple[tuple[float, float, float, float], ...]
    geometry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the captured fit and rigid transform, then hash its object-frame geometry."""

        for value_name in ("object_name", "obstacle_path", "fit_type", "tool_frame", "attachment_link"):
            value = getattr(self, value_name)
            assert (
                isinstance(value, str) and value and value == value.strip()
            ), f"{value_name} must be non-empty and trimmed"
        spheres = tuple(self.spheres)
        assert spheres and all(isinstance(sphere, CollisionSphere) for sphere in spheres)
        assert math.isfinite(self.surface_sphere_radius_m) and self.surface_sphere_radius_m > 0.0
        assert type(self.slot_capacity) is int and self.slot_capacity >= len(spheres)

        transform = tuple(tuple(float(value) for value in row) for row in self.captured_tool_from_object)
        transform_array = np.asarray(transform)
        assert transform_array.shape == (4, 4) and np.isfinite(transform_array).all()
        assert np.allclose(transform_array[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-5, rtol=0.0)
        rotation = transform_array[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4, rtol=0.0)
        assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-4)

        digest_payload = {
            "schema": "isaac-autodata.object-collision-spheres.v1",
            "obstacle_path": self.obstacle_path,
            "fit_type": self.fit_type,
            "surface_sphere_radius_m": self.surface_sphere_radius_m,
            "slot_capacity": self.slot_capacity,
            "spheres": [sphere.to_dict() for sphere in spheres],
        }
        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        object.__setattr__(self, "spheres", spheres)
        object.__setattr__(self, "captured_tool_from_object", transform)
        object.__setattr__(self, "geometry_digest", digest)


@dataclass(frozen=True, kw_only=True)
class CollisionSphereFitConfig:
    """Sampling limits and acceptance thresholds for deterministic per-link fitting.

    Args:
        max_spheres_per_link: Maximum selected spheres for one link.
        surface_samples_per_geometry: Surface samples drawn from each source mesh.
        candidates_per_geometry: Sampled points used to construct tangent-sphere candidates.
        inflation_m: Radius added to each candidate sphere [m].
        min_coverage_ratio: Required fraction of sampled surface points covered by selected spheres.
        max_uncovered_distance_m: Maximum sampled point-to-sphere gap [m].
        convexify_non_watertight: Whether non-watertight meshes use a convex hull for tangent fitting.
    """

    max_spheres_per_link: int = 64
    surface_samples_per_geometry: int = 1_000
    candidates_per_geometry: int = 200
    inflation_m: float = 0.01
    min_coverage_ratio: float = 0.95
    max_uncovered_distance_m: float = 0.01
    convexify_non_watertight: bool = True

    def __post_init__(self) -> None:
        """Check sampling budgets and nonnegative fit tolerances."""

        assert self.max_spheres_per_link > 0
        assert self.surface_samples_per_geometry >= self.max_spheres_per_link
        assert self.candidates_per_geometry >= self.max_spheres_per_link
        assert self.surface_samples_per_geometry >= self.candidates_per_geometry
        assert self.inflation_m >= 0.0
        assert 0.0 < self.min_coverage_ratio <= 1.0
        assert self.max_uncovered_distance_m >= 0.0


@dataclass(frozen=True)
class CollisionSphereSet:
    """Content-addressed collision spheres grouped by URDF link name.

    Args:
        cache_key: SHA-256 identity of selected geometry and fitting settings.
        spheres_by_link: Nonempty sphere sets expressed in their owning link frames [m].
    """

    cache_key: str
    spheres_by_link: Mapping[str, tuple[CollisionSphere, ...]]

    def __post_init__(self) -> None:
        """Validate sphere sets and freeze a link-name-sorted copy of their mapping."""

        assert len(self.cache_key) == 64
        normalized = {name: tuple(spheres) for name, spheres in sorted(self.spheres_by_link.items())}
        assert normalized and all(name and spheres for name, spheres in normalized.items())
        assert all(isinstance(sphere, CollisionSphere) for spheres in normalized.values() for sphere in spheres)
        object.__setattr__(self, "spheres_by_link", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible cache representation."""

        return {
            "schema": "isaac-autodata.collision-spheres.v1",
            "cache_key": self.cache_key,
            "spheres": {
                name: [sphere.to_dict() for sphere in spheres] for name, spheres in self.spheres_by_link.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CollisionSphereSet:
        """Load and validate a cache representation."""

        assert value.get("schema") == "isaac-autodata.collision-spheres.v1"
        raw_spheres = value.get("spheres")
        assert isinstance(raw_spheres, Mapping)
        spheres_by_link: dict[str, tuple[CollisionSphere, ...]] = {}
        for link_name, entries in raw_spheres.items():
            assert isinstance(link_name, str) and isinstance(entries, Sequence)
            spheres_by_link[link_name] = tuple(
                CollisionSphere(
                    center=tuple(float(item) for item in entry["center"]),
                    radius=float(entry["radius"]),
                )
                for entry in entries
            )
        return cls(cache_key=str(value["cache_key"]), spheres_by_link=spheres_by_link)


def collision_spheres_fit_urdf(
    urdf_path: str | Path,
    *,
    link_names: Sequence[str] | None = None,
    config: CollisionSphereFitConfig | None = None,
    package_roots: Mapping[str, str | Path] | None = None,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> CollisionSphereSet:
    """Fit or load deterministic spheres for selected URDF links.

    Visual geometry is preferred because USD-to-URDF collision primitives are often deliberately
    coarse; collision geometry is the fallback when a link has no visual. Every geometry is
    transformed into its link frame before fitting. Cache identity includes the complete geometry,
    selected links, and fit configuration.

    Args:
        urdf_path: Local URDF file used to resolve link geometry.
        link_names: Links to fit; ``None`` selects every geometry-bearing link.
        config: Sampling and acceptance settings; ``None`` uses default settings.
        package_roots: Local roots for resolving ``package://`` mesh references.
        cache_dir: Cache directory; ``None`` uses the AutoData user cache.
        force_rebuild: Whether to fit again even when a matching cached result exists.

    Returns:
        Link-frame spheres and their geometry/settings checksum. Coverage checks apply to
        sampled points, not a proof of conservative coverage of the complete mesh.

    Raises:
        AssertionError: Selected geometry is missing or sampled fitting thresholds are not met.
        FileNotFoundError: The URDF or a referenced mesh cannot be resolved.
    """

    source_path = Path(urdf_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"URDF must be a local file: {source_path}")
    fit_config = config or CollisionSphereFitConfig()
    link_meshes = _urdf_link_meshes(source_path, package_roots or {})
    selected = tuple(sorted(link_meshes)) if link_names is None else tuple(dict.fromkeys(link_names))
    assert selected, "at least one geometry-bearing link is required"
    unknown = sorted(set(selected) - set(link_meshes))
    assert not unknown, f"selected links have no URDF geometry: {unknown}"

    cache_key = _cache_key(selected, fit_config, link_meshes)
    root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "autodata" / "collision_spheres"
    )
    cache_path = root / f"{cache_key}.json"
    if cache_path.is_file() and not force_rebuild:
        cached = CollisionSphereSet.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
        assert cached.cache_key == cache_key, "collision-sphere cache key mismatch"
        return cached

    spheres_by_link = {link_name: _fit_link(link_meshes[link_name], fit_config, link_name) for link_name in selected}
    result = CollisionSphereSet(cache_key=cache_key, spheres_by_link=spheres_by_link)
    root.mkdir(parents=True, exist_ok=True)
    _json_write_atomic(cache_path, result.to_dict())
    return result


def _fit_link(
    source_meshes: tuple[trimesh.Trimesh, ...],
    config: CollisionSphereFitConfig,
    link_name: str,
) -> tuple[CollisionSphere, ...]:
    """Fit inflated tangent-sphere candidates and require the sampled link-coverage thresholds."""

    surface_parts: list[np.ndarray] = []
    center_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    for geometry_index, source_mesh in enumerate(source_meshes):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(source_mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(source_mesh.faces).copy(),
            process=True,
        )
        assert len(mesh.vertices) and len(mesh.faces), f"geometry for {link_name!r} is empty"
        points = trimesh.sample.sample_surface(
            mesh,
            config.surface_samples_per_geometry,
            seed=geometry_index,
        )[0]
        tangent_mesh = mesh
        if not tangent_mesh.is_watertight:
            assert config.convexify_non_watertight, f"geometry for {link_name!r} is not watertight"
            tangent_mesh = tangent_mesh.convex_hull
        candidates = points[: config.candidates_per_geometry]
        centers, radii = trimesh.proximity.max_tangent_sphere(tangent_mesh, candidates)
        radii = np.asarray(radii, dtype=np.float64)
        keep = np.isfinite(radii) & (radii > 0.0) & (radii <= 0.5 * np.linalg.norm(mesh.extents))
        surface_parts.append(points)
        center_parts.append(np.asarray(centers, dtype=np.float64)[keep])
        radius_parts.append(radii[keep] + config.inflation_m)

    points = np.concatenate(surface_parts)
    centers = np.concatenate(center_parts)
    radii = np.concatenate(radius_parts)
    assert len(centers), f"sphere fitting produced no candidates for {link_name!r}"
    spheres, coverage, max_gap = _greedy_cover(points, centers, radii, config)
    assert (
        coverage + 1.0e-12 >= config.min_coverage_ratio
    ), f"sphere fit for {link_name!r} covers {coverage:.3f}, required {config.min_coverage_ratio:.3f}"
    assert (
        max_gap <= config.max_uncovered_distance_m + 1.0e-12
    ), f"sphere fit for {link_name!r} leaves {max_gap:.4f} m, required <= {config.max_uncovered_distance_m:.4f} m"
    return tuple(spheres)


def _greedy_cover(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    config: CollisionSphereFitConfig,
) -> tuple[list[CollisionSphere], float, float]:
    """Select candidates greedily by newly covered surface samples.

    Args:
        points: Link-frame surface samples with shape ``(N, 3)`` [m].
        centers: Candidate sphere centers with shape ``(S, 3)`` [m].
        radii: Candidate sphere radii with shape ``(S,)`` [m].
        config: Sphere budget and sampled coverage/gap thresholds.

    Returns:
        Selected spheres, covered sample fraction and maximum uncovered surface gap [m].
        Reaching the sphere budget can return a result below the configured thresholds.
    """

    gaps = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :]
    uncovered = np.ones(len(points), dtype=bool)
    available = np.ones(len(centers), dtype=bool)
    min_gaps = np.full(len(points), math.inf)
    selected: list[CollisionSphere] = []
    for _ in range(config.max_spheres_per_link):
        gains = np.count_nonzero(gaps[uncovered] <= 0.0, axis=0)
        gains[~available] = -1
        index = int(np.argmax(gains))
        if gains[index] <= 0:
            break
        sphere = CollisionSphere(center=tuple(float(value) for value in centers[index]), radius=float(radii[index]))
        selected.append(sphere)
        available[index] = False
        min_gaps = np.minimum(min_gaps, gaps[:, index])
        uncovered = min_gaps > 0.0
        coverage = float(np.count_nonzero(~uncovered) / len(points))
        max_gap = float(np.max(np.maximum(min_gaps, 0.0)))
        if coverage >= config.min_coverage_ratio and max_gap <= config.max_uncovered_distance_m:
            return selected, coverage, max_gap
        if not np.any(uncovered):
            break
    coverage = float(np.count_nonzero(~uncovered) / len(points))
    return selected, coverage, float(np.max(np.maximum(min_gaps, 0.0)))


def _urdf_link_meshes(
    urdf_path: Path,
    package_roots: Mapping[str, str | Path],
) -> dict[str, tuple[trimesh.Trimesh, ...]]:
    """Load link-frame meshes, preferring each link's visual geometry over collision geometry."""

    root = ET.fromstring(urdf_path.read_bytes())
    resolved_roots = {name: Path(path).expanduser().resolve() for name, path in package_roots.items()}
    by_link: dict[str, list[trimesh.Trimesh]] = defaultdict(list)
    for link in root.findall("link"):
        link_name = link.get("name")
        assert link_name, "URDF link is missing a name"
        elements = link.findall("visual") or link.findall("collision")
        for element in elements:
            geometry = element.find("geometry")
            assert geometry is not None, f"geometry on {link_name!r} is incomplete"
            mesh = _geometry_mesh(geometry, urdf_path.parent, resolved_roots)
            mesh.apply_transform(_origin_matrix(element.find("origin")))
            by_link[link_name].append(mesh)
    assert by_link, f"URDF has no visual or collision geometry: {urdf_path}"
    return {link_name: tuple(meshes) for link_name, meshes in sorted(by_link.items())}


def _geometry_mesh(
    geometry: ET.Element,
    urdf_root: Path,
    package_roots: Mapping[str, Path],
) -> trimesh.Trimesh:
    """Load scaled mesh geometry or build a supported URDF primitive in local metres."""

    if (element := geometry.find("mesh")) is not None:
        reference = element.get("filename") or element.get("url")
        assert reference, "URDF mesh geometry has no filename"
        path = _mesh_path(reference, urdf_root, package_roots)
        loaded = trimesh.load(path, process=False)
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        assert isinstance(mesh, trimesh.Trimesh) and len(mesh.faces), f"mesh has no faces: {path}"
        scale = _vector(element.get("scale"), (1.0, 1.0, 1.0))
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        return mesh
    if (element := geometry.find("box")) is not None:
        return trimesh.creation.box(extents=_vector(element.get("size")))
    if (element := geometry.find("cylinder")) is not None:
        return trimesh.creation.cylinder(radius=float(element.get("radius")), height=float(element.get("length")))
    if (element := geometry.find("sphere")) is not None:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(element.get("radius")))
    raise ValueError("unsupported URDF collision geometry")


def _mesh_path(reference: str, urdf_root: Path, package_roots: Mapping[str, Path]) -> Path:
    """Resolve package, file or relative mesh references, including adjacent export meshes."""

    if reference.startswith("package://"):
        package_name, separator, relative = reference[len("package://") :].partition("/")
        assert separator and package_name in package_roots, f"no package root for {reference!r}"
        path = package_roots[package_name] / relative
    elif reference.startswith("file://"):
        path = Path(reference[len("file://") :])
    else:
        path = Path(reference)
        if not path.is_absolute():
            path = urdf_root / path
    path = path.expanduser().resolve()
    if not path.is_file():
        adjacent_export = urdf_root.parent / "meshes" / path.name
        if adjacent_export.is_file():
            path = adjacent_export
    if not path.is_file():
        raise FileNotFoundError(f"URDF mesh does not exist: {path}")
    return path


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """Convert optional URDF XYZ [m] and roll/pitch/yaw [rad] into a rigid transform."""

    xyz = (0.0, 0.0, 0.0) if origin is None else _vector(origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = (0.0, 0.0, 0.0) if origin is None else _vector(origin.get("rpy"), (0.0, 0.0, 0.0))
    transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def _vector(value: str | None, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    """Parse three finite URDF values, or use the supplied default when absent."""

    if value is None:
        assert default is not None, "required URDF vector is missing"
        return default
    result = tuple(float(item) for item in value.split())
    assert len(result) == 3 and all(math.isfinite(item) for item in result)
    return result


def _cache_key(
    links: tuple[str, ...],
    config: CollisionSphereFitConfig,
    meshes: Mapping[str, tuple[trimesh.Trimesh, ...]],
) -> str:
    """Hash selected link geometry and fit settings, excluding export paths and joint metadata."""

    # Export directories and joint metadata do not affect link-local sphere fitting.
    digest = hashlib.sha256(b"autodata.collision-sphere-fit.v2")
    digest.update(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode())
    for link_name in links:
        digest.update(json.dumps((link_name, len(meshes[link_name]))).encode())
        for mesh in meshes[link_name]:
            digest.update(json.dumps((mesh.vertices.shape, mesh.faces.shape)).encode())
            digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
            digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _json_write_atomic(path: Path, value: Mapping[str, object]) -> None:
    """Flush a JSON cache entry to a temporary sibling before atomically replacing its target."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
