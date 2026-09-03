# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative environment-profile schema for base-task variants.

An :class:`EnvironmentProfile` describes scene changes, event-term changes, and the
motion-planner profile of a task variant. It is overlaid on the parsed env config before
``gym.make`` by :func:`~autodata_interfaces.env.isaaclab_env_interface.apply_env_profile`;
this module holds only the schema and its validation and does not import ``isaaclab``.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields
from typing import Any

_EVENT_MODES = {"startup", "reset", "interval"}


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

    rigid_props: dict[str, Any] = field(default_factory=dict)
    """Overrides merged into the asset's spawn ``RigidBodyPropertiesCfg`` fields."""


@dataclass(kw_only=True)
class RigidObjectsSpec:
    """Rigid-object changes relative to the base task's scene.

    ``add`` spawns new objects (:class:`RigidObjectAddSpec`); ``override`` patches existing ones
    (:class:`RigidObjectOverrideSpec`).
    """

    add: dict[str, RigidObjectAddSpec] = field(default_factory=dict)
    override: dict[str, RigidObjectOverrideSpec] = field(default_factory=dict)


@dataclass(kw_only=True)
class SceneSpec:
    """Scene changes relative to the base task; currently rigid objects only."""

    rigid_objects: RigidObjectsSpec = field(default_factory=RigidObjectsSpec)


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
class EnvironmentProfile:
    """Declarative overlay for a base task's parsed env config.

    Attributes:
        name: Profile name, recorded in the generation-result sidecar.
        description: Human-readable summary of the variant.
        base_env: Env id the profile is written against; must equal the id used at apply time.
        scene: Scene changes (:class:`SceneSpec`).
        events: Event-term changes (:class:`EventsSpec`).
        planner: Name of the motion-planner profile tuned for this scene (key into
            :meth:`CuroboPlannerCfg.from_profile`), or None to fall back to task-name matching.
    """

    name: str
    base_env: str
    description: str = ""
    scene: SceneSpec = field(default_factory=SceneSpec)
    events: EventsSpec = field(default_factory=EventsSpec)
    planner: str | None = None

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
                    rigid_props: {<field>: <value>}
            events:                             # optional
              remove: [<term_name>, ...]
              override:
                <term_name>: {params: {...}}
              add:
                <term_name>:
                  func: "<module.path>:<function>"
                  mode: reset                   # optional
                  params: {...}                 # *_cfg/*_cfgs values name scene assets
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentProfile:
        """Build a validated :class:`EnvironmentProfile` from a parsed dict.

        Args:
            data: Parsed profile dict; see :meth:`from_yaml` for the expected schema.
        """
        validate_profile_dict(data)

        objects_data = data.get("scene", {}).get("rigid_objects", {})
        scene = SceneSpec(
            rigid_objects=RigidObjectsSpec(
                add={
                    name: RigidObjectAddSpec(**_coerce_vector_fields(spec))
                    for name, spec in objects_data.get("add", {}).items()
                },
                override={
                    name: RigidObjectOverrideSpec(**spec) for name, spec in objects_data.get("override", {}).items()
                },
            )
        )

        events_data = data.get("events", {})
        events = EventsSpec(
            remove=list(events_data.get("remove", [])),
            override={name: EventOverrideSpec(**spec) for name, spec in events_data.get("override", {}).items()},
            add={name: EventAddSpec(**spec) for name, spec in events_data.get("add", {}).items()},
        )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base_env=data["base_env"],
            scene=scene,
            events=events,
            planner=data.get("planner"),
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
    optional_keys = {"description", "planner", "scene", "events"}
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

    _validate_scene_dict(data.get("scene", {}))
    _validate_events_dict(data.get("events", {}))


def _validate_scene_dict(scene: dict[str, Any]) -> None:
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
        rigid_props = spec.get("rigid_props")
        assert (
            isinstance(rigid_props, dict) and rigid_props
        ), f"scene.rigid_objects.override[{name!r}] must provide a non-empty 'rigid_props' dict, got {rigid_props!r}"


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


def _coerce_vector_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-list vector fields of a rigid-object spec to tuples."""
    return {key: tuple(val) if key in ("position", "rotation", "scale") else val for key, val in spec.items()}


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
