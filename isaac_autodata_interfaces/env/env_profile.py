# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative environment-profile schema for base-task variants.

An :class:`EnvironmentProfile` describes scene additions, reset-event changes, and the
motion-planner profile of a task variant. It is overlaid on the parsed env config before
``gym.make`` by :func:`~isaac_autodata_interfaces.env.isaaclab_env_interface.apply_env_profile`;
this module holds only the schema and its validation and does not import ``isaaclab``.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields
from typing import Any

_EVENT_MODES = {"startup", "reset", "interval"}


@dataclass(kw_only=True)
class RigidObjectSpec:
    """A rigid object to add to the scene, spawned from a USD file."""

    prim_path: str
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
    rigid_props: dict[str, Any] = field(default_factory=dict)
    """Overrides for the spawned asset's ``RigidBodyPropertiesCfg`` fields."""


@dataclass(kw_only=True)
class EventTermSpec:
    """An event term to add, with its function referenced as ``"module.path:function"``."""

    func: str
    mode: str = "reset"
    params: dict[str, Any] = field(default_factory=dict)
    """Term parameters; ``*_cfg``/``*_cfgs`` entries name scene assets and become
    :class:`~isaaclab.managers.SceneEntityCfg` when the profile is applied."""


@dataclass(kw_only=True)
class EventOverrideSpec:
    """Parameter patch merged into an existing event term."""

    params: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class SceneSpec:
    """Scene modifications: assets to add and physics patches on existing assets."""

    rigid_objects: dict[str, RigidObjectSpec] = field(default_factory=dict)
    rigid_body_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per existing asset name, ``RigidBodyPropertiesCfg`` field overrides on its spawn config."""


@dataclass(kw_only=True)
class ResetEventsSpec:
    """Event-term changes relative to the base task's event config."""

    remove: list[str] = field(default_factory=list)
    override: dict[str, EventOverrideSpec] = field(default_factory=dict)
    add: dict[str, EventTermSpec] = field(default_factory=dict)


@dataclass(kw_only=True)
class EnvironmentProfile:
    """Declarative overlay for a base task's parsed env config.

    Attributes:
        name: Profile name, recorded in the generation-result sidecar.
        description: Human-readable summary of the variant.
        base_env: Env id the profile is written against; must equal the id used at apply time.
        scene: Scene additions and physics patches.
        reset_events: Event-term removals, overrides, and additions.
        planner: Name of the motion-planner profile tuned for this scene (key into
            :meth:`CuroboPlannerCfg.from_profile`), or None to fall back to task-name matching.
    """

    name: str
    base_env: str
    description: str = ""
    scene: SceneSpec = field(default_factory=SceneSpec)
    reset_events: ResetEventsSpec = field(default_factory=ResetEventsSpec)
    planner: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> EnvironmentProfile:
        """Load and validate an environment profile from a YAML file.

        Expected schema::

            name: <str>
            description: <str>                  # optional
            base_env: <env id>
            planner: <planner profile name>     # optional
            scene:                              # optional
              rigid_objects:
                <asset_name>:
                  prim_path: "{ENV_REGEX_NS}/<Prim>"
                  usd_path: "{ISAACLAB_NUCLEUS_DIR}/<...>.usd"
                  position: [x, y, z]           # optional
                  rotation: [x, y, z, w]        # optional (quaternion, InitialStateCfg order)
                  scale: [x, y, z]              # optional
                  rigid_props: {<field>: <value>}   # optional
              rigid_body_properties:
                <asset_name>: {<field>: <value>}
            reset_events:                       # optional
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
        """Build a validated :class:`EnvironmentProfile` from a parsed dict."""
        validate_profile_dict(data)

        scene_data = data.get("scene", {})
        scene = SceneSpec(
            rigid_objects={
                name: RigidObjectSpec(**_coerce_vector_fields(spec))
                for name, spec in scene_data.get("rigid_objects", {}).items()
            },
            rigid_body_properties=dict(scene_data.get("rigid_body_properties", {})),
        )

        events_data = data.get("reset_events", {})
        reset_events = ResetEventsSpec(
            remove=list(events_data.get("remove", [])),
            override={name: EventOverrideSpec(**spec) for name, spec in events_data.get("override", {}).items()},
            add={name: EventTermSpec(**spec) for name, spec in events_data.get("add", {}).items()},
        )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base_env=data["base_env"],
            scene=scene,
            reset_events=reset_events,
            planner=data.get("planner"),
        )


def validate_profile_dict(data: dict[str, Any]) -> None:
    """Check a parsed environment-profile dict against the expected schema.

    See :meth:`EnvironmentProfile.from_yaml` for the schema.
    """
    assert isinstance(data, dict), f"Expected top-level dict, got {type(data).__name__}"

    required_keys = {"name", "base_env"}
    optional_keys = {"description", "planner", "scene", "reset_events"}
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
    _validate_reset_events_dict(data.get("reset_events", {}))


def _validate_scene_dict(scene: dict[str, Any]) -> None:
    assert isinstance(scene, dict), f"'scene' must be a dict, got {type(scene).__name__}"
    scene_keys = {f.name for f in fields(SceneSpec)}
    unknown = set(scene) - scene_keys
    assert not unknown, f"'scene' has unknown keys {sorted(unknown)}. Allowed: {sorted(scene_keys)}"

    rigid_objects = scene.get("rigid_objects", {})
    assert isinstance(rigid_objects, dict), f"'scene.rigid_objects' must be a dict, got {type(rigid_objects).__name__}"
    object_keys = {f.name for f in fields(RigidObjectSpec)}
    required_object_keys = {"prim_path", "usd_path"}
    for name, spec in rigid_objects.items():
        assert isinstance(spec, dict), f"scene.rigid_objects[{name!r}] must be a dict, got {type(spec).__name__}"
        missing = required_object_keys - set(spec)
        assert not missing, f"scene.rigid_objects[{name!r}] missing required keys: {sorted(missing)}"
        unknown = set(spec) - object_keys
        assert (
            not unknown
        ), f"scene.rigid_objects[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(object_keys)}"

    rigid_body_properties = scene.get("rigid_body_properties", {})
    assert isinstance(
        rigid_body_properties, dict
    ), f"'scene.rigid_body_properties' must be a dict, got {type(rigid_body_properties).__name__}"
    for name, props in rigid_body_properties.items():
        assert isinstance(
            props, dict
        ), f"scene.rigid_body_properties[{name!r}] must be a dict, got {type(props).__name__}"


def _validate_reset_events_dict(events: dict[str, Any]) -> None:
    assert isinstance(events, dict), f"'reset_events' must be a dict, got {type(events).__name__}"
    events_keys = {f.name for f in fields(ResetEventsSpec)}
    unknown = set(events) - events_keys
    assert not unknown, f"'reset_events' has unknown keys {sorted(unknown)}. Allowed: {sorted(events_keys)}"

    remove = events.get("remove", [])
    assert isinstance(remove, list), f"'reset_events.remove' must be a list, got {type(remove).__name__}"
    assert all(isinstance(name, str) for name in remove), "'reset_events.remove' entries must be strings"

    override = events.get("override", {})
    assert isinstance(override, dict), f"'reset_events.override' must be a dict, got {type(override).__name__}"
    override_keys = {f.name for f in fields(EventOverrideSpec)}
    for name, spec in override.items():
        assert isinstance(spec, dict), f"reset_events.override[{name!r}] must be a dict, got {type(spec).__name__}"
        unknown = set(spec) - override_keys
        assert (
            not unknown
        ), f"reset_events.override[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(override_keys)}"
        assert spec.get("params"), f"reset_events.override[{name!r}] must provide a non-empty 'params' dict"

    add = events.get("add", {})
    assert isinstance(add, dict), f"'reset_events.add' must be a dict, got {type(add).__name__}"
    add_keys = {f.name for f in fields(EventTermSpec)}
    for name, spec in add.items():
        assert isinstance(spec, dict), f"reset_events.add[{name!r}] must be a dict, got {type(spec).__name__}"
        assert "func" in spec, f"reset_events.add[{name!r}] missing required key 'func'"
        unknown = set(spec) - add_keys
        assert (
            not unknown
        ), f"reset_events.add[{name!r}] has unknown keys {sorted(unknown)}. Allowed: {sorted(add_keys)}"
        assert (
            isinstance(spec["func"], str) and ":" in spec["func"]
        ), f"reset_events.add[{name!r}].func must be a '<module>:<function>' string, got {spec['func']!r}"
        mode = spec.get("mode", "reset")
        assert mode in _EVENT_MODES, f"reset_events.add[{name!r}].mode must be one of {sorted(_EVENT_MODES)}"


def _coerce_vector_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-list vector fields of a rigid-object spec to tuples."""
    return {key: tuple(val) if key in ("position", "rotation", "scale") else val for key, val in spec.items()}
