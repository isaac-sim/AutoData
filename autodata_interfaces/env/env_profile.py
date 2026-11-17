# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative environment-profile schema for base-task variants.

An :class:`EnvironmentProfile` describes viewport, scene, event, and controller changes plus the
motion-planner profile of a task variant. It is overlaid on the parsed env config before
``gym.make`` by :func:`~autodata_interfaces.env.isaaclab_env_interface.apply_env_profile`;
this module holds only the schema and its validation and does not import ``isaaclab``.
"""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_CAMERA_CONVENTIONS = {"opengl", "ros", "world"}
_EVENT_MODES = {"startup", "reset", "interval"}
_VIEWER_ORIGIN_TYPES = {"world", "env", "asset_root", "asset_body"}


@dataclass(kw_only=True)
class RigidObjectAddSpec:
    """A rigid object to add to the scene, spawned from a USD file.

    Entries under ``scene.rigid_objects.add`` map a new scene-asset name to this spec.
    """

    prim_path: str
    """Prim path of the spawned object, e.g. ``"{ENV_REGEX_NS}/<Prim>"``."""
    usd_path: str
    """USD file path; may reference the ``{ISAACLAB_NUCLEUS_DIR}``/``{ISAAC_NUCLEUS_DIR}`` roots."""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial root position in the env frame [m]."""
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Initial root orientation quaternion (x, y, z, w), baked into the spawned USD prim.

    Stage readers (e.g. the motion planner's collision world) see this spawn orientation, not
    reset-event poses; keep it consistent with the asset's reset event.
    """
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale applied to the spawned USD prim per axis."""
    rigid_props: dict[str, Any] = field(default_factory=dict)
    """Values for the spawned asset's ``RigidBodyPropertiesCfg`` fields."""


@dataclass(kw_only=True)
class RigidObjectOverrideSpec:
    """A patch applied to a rigid object that already exists in the scene.

    Entries under ``scene.rigid_objects.override`` map an existing scene-asset name to this spec.
    """

    position: tuple[float, float, float] | None = None
    """Optional replacement initial root position in the env frame [m]."""
    scale: tuple[float, float, float] | None = None
    """Optional replacement spawn scale per axis."""
    rigid_props: dict[str, Any] = field(default_factory=dict)
    """Overrides merged into the asset's spawn ``RigidBodyPropertiesCfg`` fields."""
    physics_material: dict[str, float] | None = None
    """Optional solver-common ``UsdPhysicsRigidBodyMaterialCfg`` spawn properties."""


@dataclass(kw_only=True)
class RigidObjectsSpec:
    """Rigid-object changes relative to the base task's scene.

    ``add`` spawns new objects (:class:`RigidObjectAddSpec`); ``override`` patches existing ones
    (:class:`RigidObjectOverrideSpec`).
    """

    add: dict[str, RigidObjectAddSpec] = field(default_factory=dict)
    override: dict[str, RigidObjectOverrideSpec] = field(default_factory=dict)


@dataclass(kw_only=True)
class CameraOverrideSpec:
    """A pose and optional optical-property patch for an existing scene camera.

    Entries under ``scene.cameras.override`` map an existing scene-camera name to this spec.
    The pose is relative to the camera prim's parent, which is the environment frame for cameras
    spawned directly below ``{ENV_REGEX_NS}``.
    """

    position: tuple[float, float, float]
    """Camera position relative to its parent frame [m]."""
    rotation_xyzw: tuple[float, float, float, float]
    """Unit quaternion ``(x, y, z, w)`` relative to the parent frame."""
    prim_path: str | None = None
    """Optional replacement sensor prim path, used to reparent the existing camera."""
    convention: str = "ros"
    """Isaac Lab camera convention: ``ros``, ``opengl``, or ``world``."""
    clipping_range: tuple[float, float] | None = None
    """Optional near and far clipping distances [m]."""
    focal_length: float | None = None
    """Optional pinhole-camera focal length [mm]."""
    width: int | None = None
    """Optional image width [px]; provided together with :attr:`height`."""
    height: int | None = None
    """Optional image height [px]; provided together with :attr:`width`."""


@dataclass(kw_only=True)
class CamerasSpec:
    """Camera changes relative to the base task."""

    override: dict[str, CameraOverrideSpec] = field(default_factory=dict)


@dataclass(kw_only=True)
class ViewerOverrideSpec:
    """A pose and tracking-frame patch for the environment viewport camera."""

    eye: tuple[float, float, float]
    """Camera position relative to :attr:`origin_type` [m]."""
    lookat: tuple[float, float, float]
    """Camera target relative to :attr:`origin_type` [m]."""
    origin_type: str = "world"
    """Tracking frame: ``world``, ``env``, ``asset_root``, or ``asset_body``."""
    asset_name: str | None = None
    """Scene asset tracked by ``asset_root`` or ``asset_body`` origins."""
    body_name: str | None = None
    """Asset body tracked by an ``asset_body`` origin."""


@dataclass(kw_only=True)
class SceneSpec:
    """Scene changes relative to the base task."""

    rigid_objects: RigidObjectsSpec = field(default_factory=RigidObjectsSpec)
    cameras: CamerasSpec = field(default_factory=CamerasSpec)


@dataclass(kw_only=True)
class EventAddSpec:
    """An event term to add, defined by its function, mode, and parameters.

    Entries under ``events.add`` map a new event-term name to this spec.
    """

    func: str
    """Event function referenced as ``"module.path:function"``; resolved when the profile is applied."""
    mode: str = "reset"
    """Event mode: ``"startup"``, ``"reset"``, or ``"interval"``."""
    params: dict[str, Any] = field(default_factory=dict)
    """Term parameters; ``*_cfg``/``*_cfgs`` entries name scene assets and become
    :class:`~isaaclab.managers.SceneEntityCfg` when the profile is applied."""


@dataclass(kw_only=True)
class EventOverrideSpec:
    """A parameter patch merged into an event term that already exists on the base task.

    Entries under ``events.override`` map an existing event-term name to this spec.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """Parameters merged into the existing term's params; same conventions as
    :attr:`EventAddSpec.params`."""


@dataclass(kw_only=True)
class EventsSpec:
    """Event-term changes relative to the base task's event config.

    ``remove`` disables base-task terms by name, ``override`` patches the parameters of existing
    terms (:class:`EventOverrideSpec`), and ``add`` defines new terms (:class:`EventAddSpec`).
    """

    remove: list[str] = field(default_factory=list)
    override: dict[str, EventOverrideSpec] = field(default_factory=dict)
    add: dict[str, EventAddSpec] = field(default_factory=dict)


@dataclass(kw_only=True)
class PinkIKFrameTaskOverrideSpec:
    """Objective weights for one Pink IK frame task, selected by exact frame name."""

    position_cost: float
    """Position-error cost."""
    orientation_cost: float
    """Orientation-error cost."""
    lm_damping: float
    """Levenberg-Marquardt damping."""
    gain: float
    """Task gain."""


@dataclass(kw_only=True)
class PinkIKDampingTaskOverrideSpec:
    """Objective weight for the unique Pink IK damping task."""

    cost: float
    """Joint-velocity damping cost."""


@dataclass(kw_only=True)
class PinkIKNullSpacePostureTaskOverrideSpec:
    """Objective weights for the unique Pink IK null-space posture task."""

    cost: float
    """Posture regularization cost."""
    lm_damping: float
    """Levenberg-Marquardt damping."""


@dataclass(kw_only=True)
class PinkIKControlOverrideSpec:
    """Task-objective changes for one existing Pink IK action term."""

    action_term: str
    """Attribute name of the action term on the environment actions config."""
    additional_joint_names: tuple[str, ...] = ()
    """Additional articulation joints controlled by the existing Pink action term."""
    frame_tasks: dict[str, PinkIKFrameTaskOverrideSpec] = field(default_factory=dict)
    """Frame name to objective-weight override."""
    damping_task: PinkIKDampingTaskOverrideSpec | None = None
    """Optional override for the unique damping task."""
    null_space_posture_task: PinkIKNullSpacePostureTaskOverrideSpec | None = None
    """Optional override for the unique null-space posture task."""


@dataclass(kw_only=True)
class ControlSpec:
    """Controller changes relative to the base task."""

    pink_ik: PinkIKControlOverrideSpec | None = None


@dataclass(kw_only=True)
class EnvironmentProfile:
    """Declarative overlay for a base task's parsed env config.

    Attributes:
        name: Profile name, recorded in the generation-result sidecar.
        description: Human-readable summary of the variant.
        base_env: Env id the profile is written against; must equal the id used at apply time.
        viewer: Optional viewport pose and tracking-frame override.
        scene: Scene changes (:class:`SceneSpec`).
        events: Event-term changes (:class:`EventsSpec`).
        actions: Parameter overrides for existing action terms.
        control: Controller changes (:class:`ControlSpec`).
        planner: Name of the motion-planner profile tuned for this scene (key into
            :meth:`CuroboPlannerCfg.from_profile`), or None to fall back to task-name matching.
    """

    name: str
    base_env: str
    description: str = ""
    viewer: ViewerOverrideSpec | None = None
    scene: SceneSpec = field(default_factory=SceneSpec)
    events: EventsSpec = field(default_factory=EventsSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    planner: str | None = None
    actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Action-term parameter overrides, applied before the action manager is built."""

    @classmethod
    def from_yaml(cls, path: str) -> EnvironmentProfile:
        """Load and validate an environment profile from a YAML file.

        Args:
            path: Path to the profile YAML file.

        Expected schema::

            name: <str>
            description: <str>                  # optional
            base_env: <env id>
            planner: <planner profile name>     # optional
            actions:                            # optional
              <action_term_name>: {<parameter>: <value>}
            viewer:                             # optional
              eye: [x, y, z]                    # [m], relative to origin
              lookat: [x, y, z]                 # [m], relative to origin
              origin_type: asset_body            # world, env, asset_root, asset_body
              asset_name: robot                  # required for asset origins
              body_name: base_link               # required for asset_body
            scene:                              # optional
              rigid_objects:
                add:
                  <asset_name>:
                    prim_path: "{ENV_REGEX_NS}/<Prim>"
                    usd_path: "{ISAACLAB_NUCLEUS_DIR}/<...>.usd"
                    position: [x, y, z]           # optional
                    rotation: [x, y, z, w]        # optional
                    scale: [x, y, z]              # optional
                    rigid_props: {<field>: <value>}   # optional
                override:
                  <asset_name>:
                    position: [x, y, z]           # optional initial root position [m]
                    scale: [x, y, z]              # optional spawn scale
                    rigid_props: {<field>: <value>}
              cameras:
                override:
                  <camera_name>:
                    position: [x, y, z]            # [m], relative to parent
                    rotation_xyzw: [x, y, z, w]    # unit quaternion
                    prim_path: "{ENV_REGEX_NS}/ValidationCam"  # optional
                    convention: ros                # ros, opengl, or world
                    clipping_range: [near, far]     # optional [m]
                    focal_length: 18.15             # optional [mm]
                    width: 512                      # optional, with height [px]
                    height: 320
            events:                             # optional
              remove: [<term_name>, ...]
              override:
                <term_name>: {params: {...}}
              add:
                <term_name>:
                  func: "<module.path>:<function>"
                  mode: reset                   # optional
                  params: {...}                 # *_cfg/*_cfgs values name scene assets
            control:                            # optional
              pink_ik:
                action_term: gr1_action
                frame_tasks:
                  <exact_frame_name>:
                    position_cost: 30.0
                    orientation_cost: 5.0
                    lm_damping: 50.0
                    gain: 0.08
                damping_task: {cost: 0.1}       # optional
                null_space_posture_task:         # optional
                  cost: 0.1
                  lm_damping: 50.0
        """
        return cls.from_dict(_profile_data_load(Path(path).expanduser().resolve()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentProfile:
        """Build a validated :class:`EnvironmentProfile` from a parsed dict.

        Args:
            data: Parsed profile dict; see :meth:`from_yaml` for the expected schema.
        """
        validate_profile_dict(data)

        objects_data = data.get("scene", {}).get("rigid_objects", {})
        cameras_data = data.get("scene", {}).get("cameras", {})
        scene = SceneSpec(
            rigid_objects=RigidObjectsSpec(
                add={
                    name: RigidObjectAddSpec(**_coerce_vector_fields(spec))
                    for name, spec in objects_data.get("add", {}).items()
                },
                override={
                    name: RigidObjectOverrideSpec(**_coerce_vector_fields(spec))
                    for name, spec in objects_data.get("override", {}).items()
                },
            ),
            cameras=CamerasSpec(
                override={
                    name: CameraOverrideSpec(**_coerce_camera_vector_fields(spec))
                    for name, spec in cameras_data.get("override", {}).items()
                }
            ),
        )

        events_data = data.get("events", {})
        events = EventsSpec(
            remove=list(events_data.get("remove", [])),
            override={name: EventOverrideSpec(**spec) for name, spec in events_data.get("override", {}).items()},
            add={name: EventAddSpec(**spec) for name, spec in events_data.get("add", {}).items()},
        )

        control_data = data.get("control", {})
        pink_ik_data = control_data.get("pink_ik")
        pink_ik = None
        if pink_ik_data is not None:
            damping_data = pink_ik_data.get("damping_task")
            null_space_data = pink_ik_data.get("null_space_posture_task")
            pink_ik = PinkIKControlOverrideSpec(
                action_term=pink_ik_data["action_term"],
                additional_joint_names=tuple(pink_ik_data.get("additional_joint_names", ())),
                frame_tasks={
                    frame_name: PinkIKFrameTaskOverrideSpec(**spec)
                    for frame_name, spec in pink_ik_data.get("frame_tasks", {}).items()
                },
                damping_task=PinkIKDampingTaskOverrideSpec(**damping_data) if damping_data is not None else None,
                null_space_posture_task=(
                    PinkIKNullSpacePostureTaskOverrideSpec(**null_space_data) if null_space_data is not None else None
                ),
            )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base_env=data["base_env"],
            viewer=(
                ViewerOverrideSpec(**_coerce_viewer_vector_fields(data["viewer"]))
                if data.get("viewer") is not None
                else None
            ),
            scene=scene,
            events=events,
            control=ControlSpec(pink_ik=pink_ik),
            planner=data.get("planner"),
            actions=data.get("actions", {}),
        )

    def referenced_asset_names(self) -> set[str]:
        """Scene-asset names referenced by ``*_cfg``/``*_cfgs`` params of event changes."""
        names: set[str] = set()
        for add_spec in self.events.add.values():
            names |= _referenced_asset_names(add_spec.params)
        for override_spec in self.events.override.values():
            names |= _referenced_asset_names(override_spec.params)
        return names


def validate_profile_dict(data: dict[str, Any]) -> None:
    """Check a parsed environment-profile dict against the expected schema.

    See :meth:`EnvironmentProfile.from_yaml` for the schema.
    """
    assert isinstance(data, dict), f"Expected top-level dict, got {type(data).__name__}"

    required_keys = {"name", "base_env"}
    optional_keys = {"description", "planner", "viewer", "scene", "events", "control", "actions"}
    keys = set(data)
    missing = required_keys - keys
    assert not missing, f"Missing required top-level keys: {sorted(missing)}"
    unknown = keys - required_keys - optional_keys
    assert not unknown, f"Unknown top-level keys: {sorted(unknown)}. Allowed: {sorted(required_keys | optional_keys)}"

    assert isinstance(data["name"], str), f"'name' must be a string, got {type(data['name']).__name__}"
    assert isinstance(data["base_env"], str), f"'base_env' must be a string, got {type(data['base_env']).__name__}"
    assert isinstance(
        data.get("description", ""), str
    ), f"'description' must be a string, got {type(data['description']).__name__}"
    planner = data.get("planner")
    assert planner is None or isinstance(planner, str), f"'planner' must be a string, got {type(planner).__name__}"

    _validate_viewer_dict(data.get("viewer"))
    _validate_scene_dict(data.get("scene", {}))
    _validate_events_dict(data.get("events", {}))
    actions = data.get("actions", {})
    assert isinstance(actions, dict), "'actions' must map action term names to parameter overrides."
    assert all(
        isinstance(name, str) and isinstance(params, dict) and all(isinstance(key, str) for key in params)
        for name, params in actions.items()
    ), "Each action override must map a term name to a parameter dict with string keys."
    _validate_control_dict(data.get("control", {}))


def _validate_viewer_dict(viewer: Any) -> None:
    """Validate viewport placement and the frame used for its origin."""
    if viewer is None:
        return
    assert isinstance(viewer, dict), f"'viewer' must be a dict, got {type(viewer).__name__}"
    viewer_keys = {f.name for f in fields(ViewerOverrideSpec)}
    required_keys = {"eye", "lookat"}
    missing = required_keys - set(viewer)
    assert not missing, f"'viewer' missing required keys: {sorted(missing)}"
    unknown = set(viewer) - viewer_keys
    assert not unknown, f"'viewer' has unknown keys {sorted(unknown)}. Allowed: {sorted(viewer_keys)}"

    _validate_numeric_vector(viewer["eye"], length=3, label="viewer.eye")
    _validate_numeric_vector(viewer["lookat"], length=3, label="viewer.lookat")
    origin_type = viewer.get("origin_type", "world")
    assert (
        isinstance(origin_type, str) and origin_type in _VIEWER_ORIGIN_TYPES
    ), f"'viewer.origin_type' must be one of {sorted(_VIEWER_ORIGIN_TYPES)}, got {origin_type!r}"
    asset_name = viewer.get("asset_name")
    body_name = viewer.get("body_name")
    if origin_type in {"world", "env"}:
        assert (
            asset_name is None and body_name is None
        ), f"'viewer.asset_name' and 'viewer.body_name' are invalid for {origin_type!r}"
    elif origin_type == "asset_root":
        assert isinstance(asset_name, str) and asset_name, f"'viewer.asset_name' is required for {origin_type!r}"
        assert body_name is None, "'viewer.body_name' is invalid for 'asset_root'"
    else:
        assert isinstance(asset_name, str) and asset_name, "'viewer.asset_name' is required for 'asset_body'"
        assert isinstance(body_name, str) and body_name, "'viewer.body_name' is required for 'asset_body'"


def _validate_scene_dict(scene: dict[str, Any]) -> None:
    """Validate rigid-object additions, overrides and requested scene removals."""
    assert isinstance(scene, dict), f"'scene' must be a dict, got {type(scene).__name__}"
    scene_keys = {f.name for f in fields(SceneSpec)}
    unknown = set(scene) - scene_keys
    assert not unknown, f"'scene' has unknown keys {sorted(unknown)}. Allowed: {sorted(scene_keys)}"

    rigid_objects = scene.get("rigid_objects", {})
    assert isinstance(rigid_objects, dict), f"'scene.rigid_objects' must be a dict, got {type(rigid_objects).__name__}"
    objects_keys = {f.name for f in fields(RigidObjectsSpec)}
    unknown = set(rigid_objects) - objects_keys
    assert not unknown, f"'scene.rigid_objects' has unknown keys {sorted(unknown)}. Allowed: {sorted(objects_keys)}"

    add = rigid_objects.get("add", {})
    assert isinstance(add, dict), f"'scene.rigid_objects.add' must be a dict, got {type(add).__name__}"
    add_keys = {f.name for f in fields(RigidObjectAddSpec)}
    required_add_keys = {"prim_path", "usd_path"}
    vector_lengths = {"position": 3, "rotation": 4, "scale": 3}
    for name, spec in add.items():
        assert isinstance(spec, dict), f"scene.rigid_objects.add[{name!r}] must be a dict, got {type(spec).__name__}"
        missing = required_add_keys - set(spec)
        assert not missing, f"scene.rigid_objects.add[{name!r}] missing required keys: {sorted(missing)}"
        unknown = set(spec) - add_keys
        assert (
            not unknown
        ), f"scene.rigid_objects.add[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(add_keys)}"
        for vector_key, length in vector_lengths.items():
            if vector_key not in spec:
                continue
            value = spec[vector_key]
            assert (
                isinstance(value, (list, tuple))
                and len(value) == length
                and all(isinstance(v, (int, float)) for v in value)
            ), f"scene.rigid_objects.add[{name!r}].{vector_key} must be a list of {length} numbers, got {value!r}"
        rigid_props = spec.get("rigid_props", {})
        assert isinstance(
            rigid_props, dict
        ), f"scene.rigid_objects.add[{name!r}].rigid_props must be a dict, got {type(rigid_props).__name__}"

    override = rigid_objects.get("override", {})
    assert isinstance(override, dict), f"'scene.rigid_objects.override' must be a dict, got {type(override).__name__}"
    override_keys = {f.name for f in fields(RigidObjectOverrideSpec)}
    for name, spec in override.items():
        assert isinstance(
            spec, dict
        ), f"scene.rigid_objects.override[{name!r}] must be a dict, got {type(spec).__name__}"
        unknown = set(spec) - override_keys
        assert not unknown, (
            f"scene.rigid_objects.override[{name!r}] has unknown keys {sorted(unknown)}. "
            f"Allowed: {sorted(override_keys)}"
        )
        position = spec.get("position")
        if position is not None:
            _validate_numeric_vector(position, length=3, label=f"scene.rigid_objects.override[{name!r}].position")
        scale = spec.get("scale")
        if scale is not None:
            _validate_numeric_vector(scale, length=3, label=f"scene.rigid_objects.override[{name!r}].scale")
            assert all(
                value > 0.0 for value in scale
            ), f"scene.rigid_objects.override[{name!r}].scale entries must be positive"
        rigid_props = spec.get("rigid_props", {})
        assert isinstance(rigid_props, dict), (
            f"scene.rigid_objects.override[{name!r}] must provide a non-empty 'rigid_props' dict "
            f"or a position/scale, got rigid_props={type(rigid_props).__name__}"
        )
        physics_material = spec.get("physics_material")
        assert (
            physics_material is None or isinstance(physics_material, dict) and physics_material
        ), f"scene.rigid_objects.override[{name!r}].physics_material must be a non-empty dict"
        assert position is not None or scale is not None or rigid_props or physics_material, (
            f"scene.rigid_objects.override[{name!r}] must provide a position/scale "
            "or non-empty 'rigid_props' or 'physics_material'"
        )

    cameras = scene.get("cameras", {})
    assert isinstance(cameras, dict), f"'scene.cameras' must be a dict, got {type(cameras).__name__}"
    cameras_keys = {f.name for f in fields(CamerasSpec)}
    unknown = set(cameras) - cameras_keys
    assert not unknown, f"'scene.cameras' has unknown keys {sorted(unknown)}. Allowed: {sorted(cameras_keys)}"

    camera_overrides = cameras.get("override", {})
    assert isinstance(
        camera_overrides, dict
    ), f"'scene.cameras.override' must be a dict, got {type(camera_overrides).__name__}"
    camera_override_keys = {f.name for f in fields(CameraOverrideSpec)}
    required_camera_override_keys = {"position", "rotation_xyzw"}
    for name, spec in camera_overrides.items():
        prefix = f"scene.cameras.override[{name!r}]"
        assert isinstance(spec, dict), f"{prefix} must be a dict, got {type(spec).__name__}"
        missing = required_camera_override_keys - set(spec)
        assert not missing, f"{prefix} missing required keys: {sorted(missing)}"
        unknown = set(spec) - camera_override_keys
        assert not unknown, f"{prefix} has unknown keys {sorted(unknown)}. Allowed: {sorted(camera_override_keys)}"

        _validate_numeric_vector(spec["position"], length=3, label=f"{prefix}.position")
        _validate_numeric_vector(spec["rotation_xyzw"], length=4, label=f"{prefix}.rotation_xyzw")
        quaternion_norm = math.sqrt(sum(component * component for component in spec["rotation_xyzw"]))
        assert math.isclose(
            quaternion_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4
        ), f"{prefix}.rotation_xyzw must be a unit quaternion in (x, y, z, w) order, got norm {quaternion_norm:.6f}"

        prim_path = spec.get("prim_path")
        if prim_path is not None:
            assert isinstance(prim_path, str) and prim_path, f"{prefix}.prim_path must be a non-empty string"
        convention = spec.get("convention", "ros")
        assert (
            convention in _CAMERA_CONVENTIONS
        ), f"{prefix}.convention must be one of {sorted(_CAMERA_CONVENTIONS)}, got {convention!r}"
        clipping_range = spec.get("clipping_range")
        if clipping_range is not None:
            _validate_numeric_vector(clipping_range, length=2, label=f"{prefix}.clipping_range")
            assert (
                0.0 < clipping_range[0] < clipping_range[1]
            ), f"{prefix}.clipping_range must satisfy 0 < near < far, got {clipping_range!r}"
        focal_length = spec.get("focal_length")
        if focal_length is not None:
            assert (
                isinstance(focal_length, (int, float)) and not isinstance(focal_length, bool) and focal_length > 0
            ), f"{prefix}.focal_length must be positive, got {focal_length!r}"
        width = spec.get("width")
        height = spec.get("height")
        assert (width is None) == (height is None), f"{prefix}.width and height must be provided together"
        if width is not None:
            assert (
                isinstance(width, int) and not isinstance(width, bool) and width > 0
            ), f"{prefix}.width must be a positive integer, got {width!r}"
            assert (
                isinstance(height, int) and not isinstance(height, bool) and height > 0
            ), f"{prefix}.height must be a positive integer, got {height!r}"


def _validate_events_dict(events: dict[str, Any]) -> None:
    assert isinstance(events, dict), f"'events' must be a dict, got {type(events).__name__}"
    events_keys = {f.name for f in fields(EventsSpec)}
    unknown = set(events) - events_keys
    assert not unknown, f"'events' has unknown keys {sorted(unknown)}. Allowed: {sorted(events_keys)}"

    remove = events.get("remove", [])
    assert isinstance(remove, list), f"'events.remove' must be a list, got {type(remove).__name__}"
    assert all(isinstance(name, str) for name in remove), "'events.remove' entries must be strings"

    override = events.get("override", {})
    assert isinstance(override, dict), f"'events.override' must be a dict, got {type(override).__name__}"
    override_keys = {f.name for f in fields(EventOverrideSpec)}
    for name, spec in override.items():
        assert isinstance(spec, dict), f"events.override[{name!r}] must be a dict, got {type(spec).__name__}"
        unknown = set(spec) - override_keys
        assert (
            not unknown
        ), f"events.override[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(override_keys)}"
        assert spec.get("params"), f"events.override[{name!r}] must provide a non-empty 'params' dict"

    add = events.get("add", {})
    assert isinstance(add, dict), f"'events.add' must be a dict, got {type(add).__name__}"
    add_keys = {f.name for f in fields(EventAddSpec)}
    for name, spec in add.items():
        assert isinstance(spec, dict), f"events.add[{name!r}] must be a dict, got {type(spec).__name__}"
        assert "func" in spec, f"events.add[{name!r}] missing required key 'func'"
        unknown = set(spec) - add_keys
        assert not unknown, f"events.add[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(add_keys)}"
        assert (
            isinstance(spec["func"], str) and ":" in spec["func"]
        ), f"events.add[{name!r}].func must be a '<module>:<function>' string, got {spec['func']!r}"
        mode = spec.get("mode", "reset")
        assert mode in _EVENT_MODES, f"events.add[{name!r}].mode must be one of {sorted(_EVENT_MODES)}"


def _validate_control_dict(control: dict[str, Any]) -> None:
    """Validate the supported Pink IK objective overrides."""
    assert isinstance(control, dict), f"'control' must be a dict, got {type(control).__name__}"
    control_keys = {f.name for f in fields(ControlSpec)}
    unknown = set(control) - control_keys
    assert not unknown, f"'control' has unknown keys {sorted(unknown)}. Allowed: {sorted(control_keys)}"

    if "pink_ik" not in control:
        return
    pink_ik = control["pink_ik"]
    assert isinstance(pink_ik, dict), f"'control.pink_ik' must be a dict, got {type(pink_ik).__name__}"
    pink_ik_keys = {f.name for f in fields(PinkIKControlOverrideSpec)}
    unknown = set(pink_ik) - pink_ik_keys
    assert not unknown, f"'control.pink_ik' has unknown keys {sorted(unknown)}. Allowed: {sorted(pink_ik_keys)}"
    assert "action_term" in pink_ik, "'control.pink_ik' missing required key 'action_term'"
    assert (
        isinstance(pink_ik["action_term"], str) and pink_ik["action_term"]
    ), "'control.pink_ik.action_term' must be a non-empty string"

    frame_tasks = pink_ik.get("frame_tasks", {})
    assert isinstance(
        frame_tasks, dict
    ), f"'control.pink_ik.frame_tasks' must be a dict, got {type(frame_tasks).__name__}"
    frame_task_keys = {f.name for f in fields(PinkIKFrameTaskOverrideSpec)}
    for frame_name, spec in frame_tasks.items():
        prefix = f"control.pink_ik.frame_tasks[{frame_name!r}]"
        assert isinstance(frame_name, str) and frame_name, "Pink IK frame-task names must be non-empty strings"
        assert isinstance(spec, dict), f"{prefix} must be a dict, got {type(spec).__name__}"
        missing = frame_task_keys - set(spec)
        assert not missing, f"{prefix} missing required keys: {sorted(missing)}"
        unknown = set(spec) - frame_task_keys
        assert not unknown, f"{prefix} has unknown keys {sorted(unknown)}. Allowed: {sorted(frame_task_keys)}"
        for field_name, value in spec.items():
            _validate_nonnegative_number(value, label=f"{prefix}.{field_name}")

    damping_task = pink_ik.get("damping_task")
    if damping_task is not None:
        _validate_control_task_dict(
            damping_task,
            spec_type=PinkIKDampingTaskOverrideSpec,
            label="control.pink_ik.damping_task",
        )

    null_space_task = pink_ik.get("null_space_posture_task")
    if null_space_task is not None:
        _validate_control_task_dict(
            null_space_task,
            spec_type=PinkIKNullSpacePostureTaskOverrideSpec,
            label="control.pink_ik.null_space_posture_task",
        )

    additional_joint_names = pink_ik.get("additional_joint_names", [])
    assert (
        isinstance(additional_joint_names, list)
        and all(isinstance(name, str) and name for name in additional_joint_names)
        and len(additional_joint_names) == len(set(additional_joint_names))
    ), "'control.pink_ik.additional_joint_names' must contain unique non-empty strings"
    assert (
        frame_tasks or damping_task is not None or null_space_task is not None or additional_joint_names
    ), "'control.pink_ik' must override at least one task or add controlled joints"


def _profile_data_load(path: Path, parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one profile and recursively merge an optional relative parent."""

    path = path.resolve()
    assert path.is_file(), f"environment profile does not exist: {path}"
    assert path not in parents, f"cyclic environment-profile inheritance: {path}"
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    assert isinstance(data, dict), f"Expected top-level dict, got {type(data).__name__}"
    parent = data.pop("extends", None)
    if parent is None:
        return data
    assert isinstance(parent, str) and parent, "'extends' must name a relative environment profile"
    base = _profile_data_load(path.parent / parent, (*parents, path))
    return _mapping_merge(base, data)


def _mapping_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and sequence leaves."""

    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _mapping_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_control_task_dict(spec: Any, *, spec_type: type, label: str) -> None:
    """Validate a scalar-only Pink IK auxiliary-task override."""
    assert isinstance(spec, dict), f"'{label}' must be a dict, got {type(spec).__name__}"
    expected_keys = {f.name for f in fields(spec_type)}
    missing = expected_keys - set(spec)
    assert not missing, f"'{label}' missing required keys: {sorted(missing)}"
    unknown = set(spec) - expected_keys
    assert not unknown, f"'{label}' has unknown keys {sorted(unknown)}. Allowed: {sorted(expected_keys)}"
    for field_name, value in spec.items():
        _validate_nonnegative_number(value, label=f"{label}.{field_name}")


def _coerce_vector_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-list vector fields of a rigid-object spec to tuples."""
    return {key: tuple(val) if key in ("position", "rotation", "scale") else val for key, val in spec.items()}


def _coerce_camera_vector_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-list vector fields of a camera override to tuples."""
    vector_fields = {"position", "rotation_xyzw", "clipping_range"}
    return {key: tuple(value) if key in vector_fields and value is not None else value for key, value in spec.items()}


def _coerce_viewer_vector_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-list viewport vectors to tuples."""
    return {key: tuple(value) if key in {"eye", "lookat"} else value for key, value in spec.items()}


def _validate_numeric_vector(value: Any, *, length: int, label: str) -> None:
    """Validate a fixed-length numeric profile vector."""
    assert (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(isinstance(component, (int, float)) and not isinstance(component, bool) for component in value)
    ), f"{label} must be a list of {length} numbers, got {value!r}"


def _validate_nonnegative_number(value: Any, *, label: str) -> None:
    """Validate a finite nonnegative scalar controller parameter."""
    assert (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
    ), f"{label} must be a finite nonnegative number, got {value!r}"


def convert_event_params(params: dict[str, Any], scene_entity_cfg_cls: type) -> dict[str, Any]:
    """Convert asset-name strings in ``*_cfg``/``*_cfgs`` params to scene-entity configs.

    The scene-entity class (:class:`~isaaclab.managers.SceneEntityCfg`) is passed in by the
    applier so this module stays free of ``isaaclab`` imports.
    """
    converted: dict[str, Any] = {}
    for key, value in params.items():
        if key.endswith("_cfg") and isinstance(value, str):
            converted[key] = scene_entity_cfg_cls(value)
        elif key.endswith("_cfgs") and isinstance(value, list):
            assert all(isinstance(name, str) for name in value), f"Event param {key!r} entries must be asset names"
            converted[key] = [scene_entity_cfg_cls(name) for name in value]
        else:
            converted[key] = value
    return converted


def _referenced_asset_names(params: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in params.items():
        if key.endswith("_cfg") and isinstance(value, str):
            names.add(value)
        elif key.endswith("_cfgs") and isinstance(value, list):
            names.update(name for name in value if isinstance(name, str))
    return names
