# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create ordinary arm-local cuRobo planners from a named-group description."""

from __future__ import annotations

import hashlib
import json
import math
import os
import yaml
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from autodata_interfaces.motion_planners.curobo.collision_spheres import (
    CollisionSphereFitConfig,
    collision_spheres_fit_urdf,
)
from autodata_interfaces.motion_planners.curobo.robot_config import (
    ATTACHED_OBJECT_LINK_NAME,
    curobo_arm_chain,
    curobo_full_robot_config_create,
    curobo_robot_config_create,
    curobo_robot_config_write,
    curobo_robot_model_prepare,
)

if TYPE_CHECKING:
    from autodata_interfaces.datastream.datastream import Datastream
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner


_SCHEMA = "isaac-autodata.curobo-planning-groups.v1"
_CUROBO_SELF_COLLISION_SPHERE_LIMIT = 1024
_OVERLAY_REQUIRED_KEYS = {"schema", "extends", "name"}
_OVERLAY_KEYS = _OVERLAY_REQUIRED_KEYS | {
    "full_state_collision_scope",
    "inactive_joint_names",
    "planner",
    "robot_config",
    "sphere_fit",
}
_ROOT_KEYS = {
    "schema",
    "name",
    "root_link",
    "root_pose_source_frame",
    "full_state_collision_scope",
    "inactive_joint_names",
    "planning_groups",
    "sphere_fit",
    "robot_config",
    "planner",
}
_ROBOT_CONFIG_KEYS = {
    "attachment_sphere_count",
    "self_collision_ignore_depth",
    "max_acceleration",
    "max_jerk",
}
_SPHERE_FIT_KEYS = set(CollisionSphereFitConfig.__dataclass_fields__)
_PLANNER_KEYS = {
    "world_config_file",
    "static_objects",
    "world_ignore_substrings",
    "num_trajopt_seeds",
    "num_graph_seeds",
    "interpolation_dt",
    "max_joint_velocity",
    "collision_cache_size",
    "trajopt_tsteps",
    "collision_activation_distance",
    "approach_distance",
    "retreat_distance",
    "approach_retreat_frame",
    "approach_direction",
    "enable_graph",
    "enable_opt",
    "enable_graph_attempt",
    "max_planning_attempts",
    "enable_finetune_trajopt",
    "time_dilation_factor",
    "surface_sphere_radius",
    "n_repeat",
    "motion_step_size",
    "debug_planner",
    "sphere_update_freq",
    "motion_noise_scale",
    "position_threshold",
    "rotation_threshold",
    "ik_candidate_count",
    "cuda_device",
}


@dataclass(frozen=True)
class CuroboPlanningContext:
    """Arm-local planning handles plus one designated full-state collision context.

    Args:
        group_planners: Arm planners keyed by end-effector name.
        full_state: Combined model used to check scheduled robot states.
    """

    group_planners: Mapping[str, CuroboPlanner]
    full_state: CuroboPlanner

    def __post_init__(self) -> None:
        """Require arm planners and freeze a copy of the end-effector mapping."""

        assert self.group_planners, "group_planners must not be empty"
        object.__setattr__(self, "group_planners", MappingProxyType(dict(self.group_planners)))


def curobo_planning_context_create(
    datastream: Datastream,
    description_path: str | Path,
    *,
    env_id: int = 0,
    visualize_plan: bool = False,
    rerun_output: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> CuroboPlanningContext:
    """Build arm-local planners and one collision-only full-state context.

    Args:
        datastream: Bound environment, embodiment and task interface.
        description_path: Planning-groups YAML file.
        env_id: Environment providing the named initial robot state.
        visualize_plan: Whether environment zero logs paths and spheres to Rerun.
        rerun_output: Optional recording file; ``None`` uses a live viewer when visualization is enabled.
        cache_dir: Optional directory for reusable robot configurations and fitted spheres.

    Returns:
        Per-arm planners and the combined collision model sharing one prepared robot description.
    """

    group_planners, full_state = _curobo_planners_create(
        datastream,
        description_path,
        env_id=env_id,
        visualize_plan=visualize_plan,
        rerun_output=rerun_output,
        cache_dir=cache_dir,
        include_full_state=True,
    )
    assert full_state is not None
    return CuroboPlanningContext(group_planners, full_state)


def curobo_group_planners_create(
    datastream: Datastream,
    description_path: str | Path,
    *,
    env_id: int = 0,
    visualize_plan: bool = False,
    rerun_output: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, CuroboPlanner]:
    """Build arm-local planners without allocating a combined collision model.

    Args:
        datastream: Bound environment, embodiment and task interface.
        description_path: Planning-groups YAML file.
        env_id: Environment providing the named initial robot state.
        visualize_plan: Whether environment zero logs paths and spheres to Rerun.
        rerun_output: Optional recording file; ``None`` uses a live viewer when visualization is enabled.
        cache_dir: Optional directory for reusable robot configurations and fitted spheres.

    Returns:
        Planners keyed by end-effector name.
    """

    group_planners, full_state = _curobo_planners_create(
        datastream,
        description_path,
        env_id=env_id,
        visualize_plan=visualize_plan,
        rerun_output=rerun_output,
        cache_dir=cache_dir,
        include_full_state=False,
    )
    assert full_state is None
    return group_planners


def _curobo_planners_create(
    datastream: Datastream,
    description_path: str | Path,
    *,
    env_id: int,
    visualize_plan: bool,
    rerun_output: str | Path | None,
    cache_dir: str | Path | None,
    include_full_state: bool,
) -> tuple[dict[str, CuroboPlanner], CuroboPlanner | None]:
    """Build planners from one prepared model and one shared sphere set.

    Collision spheres are fitted once over the union of collision-bearing links. Each generated
    robot configuration then selects spheres for its root-to-tool chain plus any
    explicitly commanded auxiliary joints. Other owned joints are locked to the exact named
    live state; unrelated branches are omitted from arm-local configurations.

    Args:
        datastream: Bound AutoData environment, embodiment, and task interface.
        description_path: Planning-groups YAML path.
        env_id: Environment whose named start state initializes the planners.
        visualize_plan: Whether environment zero logs paths and arm-local spheres to Rerun.
        rerun_output: Optional Rerun recording path. When set, no live viewer is spawned.
        cache_dir: Optional cache directory for robot configurations and fitted spheres.
        include_full_state: Whether to allocate a combined collision-only robot model.

    Returns:
        Group planners and the optional designated full-state context.
    """

    assert type(env_id) is int and env_id >= 0, "env_id must be non-negative"
    description = _description_load(description_path)
    _action_target_frames_validate(datastream, description["planning_groups"])
    controller_model_path = datastream.get_robot_planning_model_path()
    assert (
        controller_model_path is not None and controller_model_path.is_file()
    ), "the embodiment must expose its live controller URDF"

    root_link = description["root_link"]
    inactive_joint_names = tuple(description["inactive_joint_names"])
    groups = description["planning_groups"]
    auxiliary_by_group = {
        group_name: tuple(group.get("auxiliary_joint_names", datastream.get_robot_eef_joint_names(group["eef_name"])))
        for group_name, group in groups.items()
    }
    auxiliary_joint_names = tuple(
        dict.fromkeys(name for group_name in groups for name in auxiliary_by_group[group_name])
    )
    model_path = curobo_robot_model_prepare(
        controller_model_path,
        auxiliary_joint_names=auxiliary_joint_names,
    )
    chains = {
        group_name: curobo_arm_chain(
            model_path,
            group["tool_link"],
            root_link=root_link,
            inactive_joint_names=inactive_joint_names,
            auxiliary_joint_names=auxiliary_by_group[group_name],
        )
        for group_name, group in groups.items()
    }
    shared_collision_link_names = (
        set.intersection(*(set(chain.collision_link_names) for chain in chains.values())) if len(chains) > 1 else set()
    )
    collision_links_by_group = {
        group_name: tuple(
            link_name
            for link_name in chain.collision_link_names
            if link_name not in shared_collision_link_names or link_name in chain.moving_link_names
        )
        for group_name, chain in chains.items()
    }
    assert all(collision_links_by_group.values()), "each planning group must own links outside shared ancestry"
    group_collision_link_names = tuple(
        dict.fromkeys(link_name for names in collision_links_by_group.values() for link_name in names)
    )
    assert group_collision_link_names, "planning groups contain no collision-bearing moving links"
    full_state_link_names = (
        group_collision_link_names if description["full_state_collision_scope"] == "planning_groups" else None
    )

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "autodata" / "curobo"
    )
    spheres = collision_spheres_fit_urdf(
        model_path,
        link_names=full_state_link_names if include_full_state else group_collision_link_names,
        config=CollisionSphereFitConfig(**description["sphere_fit"]),
        cache_dir=cache_root / "collision_spheres",
    )
    if include_full_state:
        _full_state_sphere_count_validate(spheres.spheres_by_link)
    start_joint_positions = _named_start_state(datastream, env_id)

    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    planners: dict[str, CuroboPlanner] = {}
    for group_name, group in groups.items():
        tool_link = group["tool_link"]
        initial_auxiliary_positions = _initial_auxiliary_positions(
            group.get("initial_auxiliary_joint_positions", {}), auxiliary_by_group[group_name]
        )
        robot_config = curobo_robot_config_create(
            model_path,
            tool_link,
            start_joint_positions,
            spheres,
            root_link=root_link,
            inactive_joint_names=inactive_joint_names,
            auxiliary_joint_names=auxiliary_by_group[group_name],
            collision_link_names=collision_links_by_group[group_name],
            self_collision_ignore_pairs=group.get("self_collision_ignore_pairs", ()),
            **description["robot_config"],
        )
        robot_config_path = _robot_config_cache_write(robot_config, cache_root / "robot_configs")
        planning_config_path = robot_config_path
        if auxiliary_by_group[group_name]:
            # Auxiliary joints follow their own action channel, not the transit optimizer.
            planning_config = curobo_robot_config_create(
                model_path,
                tool_link,
                {**start_joint_positions, **initial_auxiliary_positions},
                spheres,
                root_link=root_link,
                inactive_joint_names=inactive_joint_names,
                auxiliary_joint_names=auxiliary_by_group[group_name],
                lock_auxiliary_joints=True,
                collision_link_names=collision_links_by_group[group_name],
                self_collision_ignore_pairs=group.get("self_collision_ignore_pairs", ()),
                **description["robot_config"],
            )
            planning_config_path = _robot_config_cache_write(planning_config, cache_root / "robot_configs")
        planner_config = CuroboPlannerCfg(
            robot_config_file=str(planning_config_path),
            kinematic_robot_config_file=str(robot_config_path) if planning_config_path != robot_config_path else None,
            robot_name=f"{description['name']}_{group_name}",
            planning_group_name=group_name,
            ee_link_name=tool_link,
            root_pose_source_frame_name=description["root_pose_source_frame"],
            include_configured_world=True,
            initial_auxiliary_joint_positions=initial_auxiliary_positions,
            hand_link_names=list(chains[group_name].contact_link_names),
            attached_object_link_name=ATTACHED_OBJECT_LINK_NAME,
            extra_collision_spheres={},
            visualize_plan=False,
            visualize_spheres=False,
            **description["planner"],
        )
        eef_name = group["eef_name"]
        planners[eef_name] = CuroboPlanner(datastream=datastream, config=planner_config, env_id=env_id)

    full_state: CuroboPlanner | None = None
    if include_full_state:
        full_robot_options = dict(description["robot_config"])
        full_robot_options.pop("attachment_sphere_count", None)
        tool_links = tuple(group["tool_link"] for group in groups.values())
        active_joint_names = tuple(
            dict.fromkeys(joint_name for chain in chains.values() for joint_name in chain.active_joint_names)
        )
        ignore_pairs = tuple(
            dict.fromkeys(
                tuple(sorted(pair))
                for group in groups.values()
                for pair in group.get("self_collision_ignore_pairs", ())
            )
        )
        full_robot_config = curobo_full_robot_config_create(
            model_path,
            tool_links,
            active_joint_names,
            start_joint_positions,
            spheres,
            root_link=root_link,
            self_collision_ignore_pairs=ignore_pairs,
            **full_robot_options,
        )
        full_config_path = _robot_config_cache_write(full_robot_config, cache_root / "robot_configs")
        full_planner_config = CuroboPlannerCfg(
            robot_config_file=str(full_config_path),
            robot_name=f"{description['name']}_full_state",
            planning_group_name=f"{description['name']}_full_state",
            ee_link_name=tool_links[0],
            root_pose_source_frame_name=description["root_pose_source_frame"],
            include_configured_world=False,
            extra_collision_spheres={},
            visualize_plan=False,
            visualize_spheres=False,
            planning_enabled=False,
            use_native_self_collision=False,
            **description["planner"],
        )
        full_state = CuroboPlanner(
            datastream=datastream,
            config=full_planner_config,
            env_id=env_id,
            collision_world_source=next(iter(planners.values())),
        )

    if visualize_plan and env_id == 0:
        from autodata_interfaces.motion_planners.curobo.plan_visualizer import PlanVisualizer

        root_pose = datastream.get_robot_planning_root_pose(
            description["root_pose_source_frame"],
            env_ids=[env_id],
        )[0]
        visualizer = PlanVisualizer(
            robot_name=description["name"],
            recording_id=f"curobo_plan_{env_id}",
            debug=any(planner.config.debug_planner for planner in planners.values()),
            save_path=str(Path(rerun_output).expanduser().resolve()) if rerun_output is not None else None,
            base_pose_in_env=root_pose,
        )
        for planner in planners.values():
            planner.visualize_plan = True
            planner.plan_visualizer = visualizer
        if full_state is not None:
            full_state.plan_visualizer = visualizer
    return planners, full_state


def _full_state_sphere_count_validate(spheres_by_link: Mapping[str, tuple[Any, ...]]) -> None:
    """Fail before cuRobo can abort on an oversized self-collision model."""

    sphere_count = sum(len(spheres) for spheres in spheres_by_link.values())
    assert sphere_count < _CUROBO_SELF_COLLISION_SPHERE_LIMIT, (
        f"full-state model has {sphere_count} spheres; pinned cuRobo requires fewer than "
        f"{_CUROBO_SELF_COLLISION_SPHERE_LIMIT}"
    )


def _action_target_frames_validate(datastream: Datastream, groups: Mapping[str, Mapping[str, Any]]) -> None:
    """Require task, controller action, and planner tool frames to agree exactly."""

    target_frames = datastream.get_robot_action_target_frames()
    task_eef_names = tuple(datastream.get_eef_names())
    assert target_frames, "group planning requires explicit live action-target frames"
    assert (
        tuple(frame.eef_name for frame in target_frames) == task_eef_names
    ), "action-target frames must cover task EEFs in action order"

    group_by_eef = {group["eef_name"]: group for group in groups.values()}
    assert len(group_by_eef) == len(groups), "planning-group end-effectors must be unique"
    assert set(group_by_eef) == set(task_eef_names), "planning groups must cover every task end-effector"
    for target_frame in target_frames:
        tool_link = group_by_eef[target_frame.eef_name]["tool_link"]
        assert tool_link == target_frame.frame_name, (
            f"planning tool {tool_link!r} does not match {target_frame.eef_name!r} "
            f"action target {target_frame.frame_name!r}"
        )


def _description_load(path: str | Path, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve planning-description overlays and validate supported fields.

    Args:
        path: Base or overlay YAML file; relative overlay parents resolve beside this file.
        _parents: Already visited files used to reject cyclic inheritance.

    Returns:
        Fully merged description with validated planning groups and settings.

    Raises:
        AssertionError: A file is absent, inheritance is cyclic, or fields are unsupported or invalid.
    """

    description_path = Path(path).expanduser().resolve()
    assert description_path.is_file(), f"planning-groups description does not exist: {description_path}"
    assert description_path not in _parents, f"cyclic planning-groups overlay: {description_path}"
    with description_path.open(encoding="utf-8") as stream:
        description = yaml.safe_load(stream)
    assert isinstance(description, dict), "planning-groups description must be a mapping"

    if "extends" in description:
        assert _OVERLAY_REQUIRED_KEYS <= set(description) <= _OVERLAY_KEYS, (
            f"planning-groups overlay fields must include {sorted(_OVERLAY_REQUIRED_KEYS)} "
            f"and may include {sorted(_OVERLAY_KEYS - _OVERLAY_REQUIRED_KEYS)}"
        )
        assert description["schema"] == _SCHEMA, f"unsupported planning-groups schema: {description['schema']!r}"
        assert isinstance(description["extends"], str) and description["extends"], "extends must name a base file"
        assert isinstance(description["name"], str) and description["name"], "name must be a non-empty string"
        _keys_validate(description.get("planner", {}), _PLANNER_KEYS, "planner overlay")
        _keys_validate(description.get("robot_config", {}), _ROBOT_CONFIG_KEYS, "robot config overlay")
        _keys_validate(description.get("sphere_fit", {}), _SPHERE_FIT_KEYS, "sphere-fit overlay")
        base = _description_load(description_path.parent / description["extends"], (*_parents, description_path))
        description = {
            **base,
            "name": description["name"],
            "full_state_collision_scope": description.get(
                "full_state_collision_scope", base["full_state_collision_scope"]
            ),
            "inactive_joint_names": description.get("inactive_joint_names", base["inactive_joint_names"]),
            "planner": {**base["planner"], **description.get("planner", {})},
            "robot_config": {**base["robot_config"], **description.get("robot_config", {})},
            "sphere_fit": {**base["sphere_fit"], **description.get("sphere_fit", {})},
        }

    assert set(description) == _ROOT_KEYS, f"planning-groups fields must be {sorted(_ROOT_KEYS)}"
    assert description["schema"] == _SCHEMA, f"unsupported planning-groups schema: {description['schema']!r}"
    for key in ("name", "root_link", "root_pose_source_frame"):
        assert isinstance(description[key], str) and description[key], f"{key} must be a non-empty string"
    assert description["full_state_collision_scope"] in {
        "complete",
        "planning_groups",
    }, "full_state_collision_scope must be 'complete' or 'planning_groups'"
    inactive = description["inactive_joint_names"]
    assert (
        isinstance(inactive, list)
        and all(isinstance(name, str) and name for name in inactive)
        and len(inactive) == len(set(inactive))
    ), "inactive_joint_names must contain unique non-empty strings"
    groups = description["planning_groups"]
    assert isinstance(groups, dict) and groups, "planning_groups must be a non-empty mapping"
    eef_names: list[str] = []
    for group_name, group in groups.items():
        assert isinstance(group_name, str) and group_name and isinstance(group, dict)
        required = {"eef_name", "tool_link"}
        allowed = required | {
            "auxiliary_joint_names",
            "initial_auxiliary_joint_positions",
            "self_collision_ignore_pairs",
        }
        assert required <= set(group) <= allowed, "planning groups contain unsupported fields"
        assert all(isinstance(group[key], str) and group[key] for key in required)
        auxiliary = group.get("auxiliary_joint_names", [])
        assert (
            isinstance(auxiliary, list)
            and all(isinstance(name, str) and name for name in auxiliary)
            and len(auxiliary) == len(set(auxiliary))
        ), "auxiliary_joint_names must contain unique non-empty strings"
        initial_auxiliary_positions = group.get("initial_auxiliary_joint_positions", {})
        assert isinstance(
            initial_auxiliary_positions, dict | int | float
        ), "initial_auxiliary_joint_positions must be a finite scalar or mapping"
        if isinstance(initial_auxiliary_positions, dict):
            assert not auxiliary or set(initial_auxiliary_positions) <= set(
                auxiliary
            ), "initial_auxiliary_joint_positions must only name explicit auxiliary joints"
            initial_values = initial_auxiliary_positions.values()
        else:
            initial_values = (initial_auxiliary_positions,)
        assert all(
            isinstance(value, int | float) and math.isfinite(value) for value in initial_values
        ), "initial_auxiliary_joint_positions must contain finite numeric values"
        ignore_pairs = group.get("self_collision_ignore_pairs", [])
        assert isinstance(ignore_pairs, list), "self_collision_ignore_pairs must be a list"
        assert all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(name, str) and name for name in pair)
            and pair[0] != pair[1]
            for pair in ignore_pairs
        ), "self_collision_ignore_pairs must contain two distinct non-empty link names"
        eef_names.append(group["eef_name"])
    assert len(eef_names) == len(set(eef_names)), "planning-group end-effectors must be unique"
    _keys_validate(description["sphere_fit"], _SPHERE_FIT_KEYS, "sphere_fit")
    _keys_validate(description["robot_config"], _ROBOT_CONFIG_KEYS, "robot_config")
    _keys_validate(description["planner"], _PLANNER_KEYS, "planner")
    return description


def _keys_validate(value: object, allowed: set[str], owner: str) -> None:
    """Require a settings mapping with only the supported keys for its owner."""

    assert isinstance(value, Mapping), f"{owner} must be a mapping"
    unknown = set(value) - allowed
    assert not unknown, f"{owner} contains unknown fields: {sorted(unknown)}"


def _initial_auxiliary_positions(
    configured: Mapping[str, float] | int | float,
    joint_names: tuple[str, ...],
) -> dict[str, float]:
    """Expand a scalar auxiliary-joint target or validate an explicitly named subset [rad, m]."""

    if isinstance(configured, Mapping):
        unknown = set(configured) - set(joint_names)
        assert not unknown, f"initial auxiliary joints are not owned by the embodiment EEF: {sorted(unknown)}"
        return {name: float(value) for name, value in configured.items()}
    return {name: float(configured) for name in joint_names}


def _named_start_state(datastream: Datastream, env_id: int) -> dict[str, float]:
    """Pair measured joint values [rad, m] with unique names in live articulation order."""

    joint_names = tuple(datastream.get_robot_joint_names())
    assert joint_names and len(joint_names) == len(set(joint_names)), "live joint names must be non-empty and unique"
    positions = datastream.get_robot_joint_positions(env_ids=[env_id])
    assert positions.ndim == 2 and positions.shape == (1, len(joint_names)), "live joint state shape is inconsistent"
    return dict(zip(joint_names, (float(value) for value in positions[0].detach().cpu().tolist()), strict=True))


def _robot_config_cache_write(config: Mapping[str, object], cache_dir: Path) -> Path:
    """Write a finite, content-addressed robot configuration unless already cached."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    output_path = cache_dir / f"{hashlib.sha256(payload).hexdigest()}.yml"
    if not output_path.is_file():
        curobo_robot_config_write(config, output_path)
    return output_path


__all__ = ["CuroboPlanningContext", "curobo_group_planners_create", "curobo_planning_context_create"]
