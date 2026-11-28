# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import yaml
from dataclasses import fields
from typing import Any, get_origin, get_type_hints

from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from autodata_interfaces.tasks.schedule_constraint_spec import (
    NonOverlapConstraintSpec,
    NonPreemptiveConstraintSpec,
    PrecedenceConstraintSpec,
    SchedulerStrategy,
    ScheduleSpec,
    SegmentAnchor,
)
from autodata_interfaces.tasks.segment_spec import (
    DemonstrationSourceWindowSpec,
    MotionGoalSpec,
    SegmentEventPoint,
    SegmentEventSpec,
    SegmentEventType,
    SegmentId,
    SegmentKind,
    SegmentProvider,
    SegmentRobotContactSpec,
    SegmentSpec,
    SegmentWorldContactSpec,
)
from autodata_interfaces.tasks.subtask_constraint_spec import (
    SubtaskConstraint,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from autodata_interfaces.tasks.subtask_spec import ALGO_PARAMS_REGISTRY, Subtask, SubtaskAlgoParams


def validate_task_dict(data: dict[str, Any]) -> None:
    """Check a parsed task config dict against the expected schema.

    Expected schema:

       name: <str>
       description: <str>          # optional
       algo: <str>                 # key into ALGO_PARAMS_REGISTRY
       subtasks:
           <eef_name>:
           - object_ref: <str>            # optional
               description: <str>           # optional
               subtask_start_signal: <str>  # optional
               subtask_term_signal: <str>   # optional
               algo_params:                 # optional; fields match the
               <kwarg>: <value>           #   chosen algo's dataclass
           - ...
           <other_eef>: [...]
    """

    assert isinstance(data, dict), f"Expected top-level dict, got {type(data).__name__}"

    required_keys = {"name", "algo", "subtasks"}
    optional_keys = {
        "attempt_diagnostics",
        "constraints",
        "description",
        "generation_policy",
        "schedule",
    }
    keys = set(data)
    missing = required_keys - keys
    assert not missing, f"Missing required top-level keys: {sorted(missing)}"
    unknown = keys - required_keys - optional_keys
    assert not unknown, f"Unknown top-level keys: {sorted(unknown)}. Allowed: {sorted(required_keys | optional_keys)}"

    assert isinstance(data["name"], str), f"'name' must be a string, got {type(data['name']).__name__}"
    assert isinstance(
        data.get("description", ""), str
    ), f"'description' must be a string, got {type(data['description']).__name__}"
    assert isinstance(data["algo"], str), f"'algo' must be a string, got {type(data['algo']).__name__}"
    assert (
        data["algo"] in ALGO_PARAMS_REGISTRY
    ), f"Unknown algo {data['algo']!r}. Registered: {sorted(ALGO_PARAMS_REGISTRY)}"
    attempt_diagnostics = data.get("attempt_diagnostics")
    assert attempt_diagnostics is None or (
        isinstance(attempt_diagnostics, str)
        and attempt_diagnostics.strip() == attempt_diagnostics
        and attempt_diagnostics
    ), "'attempt_diagnostics' must be a non-empty, trimmed callable path"
    subtasks = data["subtasks"]
    assert isinstance(
        subtasks, dict
    ), f"'subtasks' must be a dict (eef_name -> list of subtasks), got {type(subtasks).__name__}"

    algo_cls = ALGO_PARAMS_REGISTRY[data["algo"]]
    subtask_keys = {f.name for f in fields(Subtask)}
    algo_keys = {f.name for f in fields(algo_cls)}
    segment_keys = {f.name for f in fields(SegmentSpec)}
    event_keys = {f.name for f in fields(SegmentEventSpec)}
    source_window_keys = {f.name for f in fields(DemonstrationSourceWindowSpec)}
    motion_goal_keys = {f.name for f in fields(MotionGoalSpec)}
    robot_contact_keys = {f.name for f in fields(SegmentRobotContactSpec)}
    world_contact_keys = {f.name for f in fields(SegmentWorldContactSpec)}

    for eef_name, eef_subtasks in subtasks.items():
        assert isinstance(
            eef_subtasks, list
        ), f"subtasks[{eef_name!r}] must be a list, got {type(eef_subtasks).__name__}"
        for i, st in enumerate(eef_subtasks):
            assert isinstance(st, dict), f"subtasks[{eef_name!r}][{i}] must be a dict, got {type(st).__name__}"
            st_unknown = set(st) - subtask_keys
            assert (
                not st_unknown
            ), f"subtasks[{eef_name!r}][{i}] has unknown keys {sorted(st_unknown)}. Allowed: {sorted(subtask_keys)}"
            algo_params = st.get("algo_params", {})
            assert isinstance(
                algo_params, dict
            ), f"subtasks[{eef_name!r}][{i}].algo_params must be a dict, got {type(algo_params).__name__}"
            ap_unknown = set(algo_params) - algo_keys
            assert not ap_unknown, (
                f"subtasks[{eef_name!r}][{i}].algo_params has unknown keys "
                f"{sorted(ap_unknown)} for algo {data['algo']!r}. "
                f"Allowed: {sorted(algo_keys)}"
            )
            segments = st.get("segments", [])
            assert isinstance(segments, list), f"subtasks[{eef_name!r}][{i}].segments must be a list"
            for segment in segments:
                assert isinstance(segment, dict), "each segment must be a mapping"
                assert {"name", "kind", "provider"}.issubset(segment), "segments require name, kind, and provider"
                segment_unknown = set(segment) - segment_keys
                assert not segment_unknown, f"segment has unknown keys: {sorted(segment_unknown)}"
                events = segment.get("events", [])
                assert isinstance(events, list), "segment.events must be a list"
                for event in events:
                    assert isinstance(event, dict), "each segment event must be a mapping"
                    assert {"event_type", "object_ref"}.issubset(
                        event
                    ), "segment events require event_type and object_ref"
                    assert (
                        not set(event) - event_keys
                    ), f"segment event has unknown keys: {sorted(set(event) - event_keys)}"
                if "demonstration_source_window" in segment:
                    source_window = segment["demonstration_source_window"]
                    assert isinstance(source_window, dict), "segment.demonstration_source_window must be a mapping"
                    assert set(source_window) == source_window_keys, (
                        "segment.demonstration_source_window requires exactly "
                        "anchor_annotation_key and post_event_sample_count"
                    )
                if "motion_goal" in segment:
                    motion_goal = segment["motion_goal"]
                    assert isinstance(motion_goal, dict), "segment.motion_goal must be a mapping"
                    required_motion_goal_keys = {"position_m", "rotation_xyzw"}
                    assert required_motion_goal_keys <= set(
                        motion_goal
                    ), "segment.motion_goal requires position_m and rotation_xyzw"
                    assert (
                        not set(motion_goal) - motion_goal_keys
                    ), f"segment.motion_goal has unknown keys: {sorted(set(motion_goal) - motion_goal_keys)}"
                robot_contacts = segment.get("robot_contacts", [])
                assert isinstance(robot_contacts, list), "segment.robot_contacts must be a list"
                for contact in robot_contacts:
                    assert isinstance(contact, dict), "each segment robot contact must be a mapping"
                    assert {"object_ref", "robot_link_names"}.issubset(
                        contact
                    ), "segment robot contacts require object_ref and robot_link_names"
                    assert (
                        not set(contact) - robot_contact_keys
                    ), f"segment robot contact has unknown keys: {sorted(set(contact) - robot_contact_keys)}"
                    assert isinstance(
                        contact["robot_link_names"], list
                    ), "segment robot contact robot_link_names must be a list"
                world_contacts = segment.get("world_contacts", [])
                assert isinstance(world_contacts, list), "segment.world_contacts must be a list"
                for contact in world_contacts:
                    assert isinstance(contact, dict), "each segment world contact must be a mapping"
                    assert (
                        set(contact) == world_contact_keys
                    ), "segment world contacts require exactly object_ref and fixture_ref"

    constraints = data.get("constraints", [])
    assert isinstance(constraints, list), f"'constraints' must be a list, got {type(constraints).__name__}"
    constraint_keys = {f.name for f in fields(SubtaskConstraint)}
    for i, c in enumerate(constraints):
        assert isinstance(c, dict), f"constraints[{i}] must be a dict, got {type(c).__name__}"
        assert "constraint_type" in c, f"constraints[{i}] missing required key 'constraint_type'"
        c_unknown = set(c) - constraint_keys
        assert (
            not c_unknown
        ), f"constraints[{i}] has unknown keys {sorted(c_unknown)}. Allowed: {sorted(constraint_keys)}"

    policy = data.get("generation_policy", {})
    assert isinstance(policy, dict), f"'generation_policy' must be a dict, got {type(policy).__name__}"
    policy_keys = {f.name for f in fields(GenerationPolicy)}
    policy_unknown = set(policy) - policy_keys
    assert (
        not policy_unknown
    ), f"'generation_policy' has unknown keys {sorted(policy_unknown)}. Allowed: {sorted(policy_keys)}"

    schedule = data.get("schedule")
    if schedule is not None:
        assert isinstance(schedule, dict), "'schedule' must be a mapping"
        schedule_keys = {
            "strategy",
            "constraints",
            "collision_margin_m",
            "time_limit_s",
            "max_block_size",
            "minimize_max_wait",
        }
        assert not set(schedule) - schedule_keys, f"unknown schedule keys: {sorted(set(schedule) - schedule_keys)}"
        assert isinstance(schedule.get("constraints", []), list), "schedule.constraints must be a list"


def resolve_task_algorithm(algorithm: str | None, task_descriptor_path: str) -> str:
    """Validate a task descriptor YAML and return its ``algo``.

    Imports no simulator modules, so it can run before Isaac Sim launches.

    Args:
        algorithm: Requested algorithm, or ``None`` to use the descriptor's ``algo``.
        task_descriptor_path: Path to the task descriptor YAML.
    """

    with open(task_descriptor_path) as f:
        data = yaml.safe_load(f)
    validate_task_dict(data)
    declared = data["algo"]
    if algorithm not in (None, declared):
        raise ValueError(
            f"Requested algorithm {algorithm!r} does not match algo {declared!r} in {task_descriptor_path}"
        )
    return declared


def _coerce_enum(value: Any, enum_cls: type[enum.IntEnum]) -> enum.IntEnum:
    """Resolve a YAML scalar (enum name string or int) to an ``enum_cls`` member.

    Args:
        value: An ``enum_cls`` member, a member name (case-insensitive), or an int value.
        enum_cls: Target enum type.
    """

    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        assert (
            value.upper() in enum_cls.__members__
        ), f"{value!r} is not a valid {enum_cls.__name__}; choose from {sorted(enum_cls.__members__)}"
        return enum_cls[value.upper()]
    return enum_cls(value)


def _coerce_tuples(kwargs: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    """Coerce list values back to tuples where a dataclass annotates the field as ``tuple``.

    YAML scalar sequences load as Python lists; tuple-typed dataclass fields need tuples.

    Args:
        kwargs: Field-name to value mapping parsed from YAML.
        hints: ``get_type_hints`` of the target dataclass.
    """

    return {
        name: tuple(val) if isinstance(val, list) and get_origin(hints.get(name)) is tuple else val
        for name, val in kwargs.items()
    }


def build_subtask(st_data: dict[str, Any], algo_cls: type[SubtaskAlgoParams]) -> Subtask:
    """Build a :class:`Subtask` from a parsed subtask dict.

    ``algo_cls``'s ``algo_params`` are built via :meth:`~.subtask_spec.SubtaskAlgoParams.from_dict`
    (the registration hook every algo's params class can override for its own nested schema).
    """

    st_data = dict(st_data)  # copy so we don't mutate caller's dict
    algo_kwargs = st_data.pop("algo_params", {})
    segment_data = st_data.pop("segments", [])

    st_kwargs = _coerce_tuples(st_data, get_type_hints(Subtask))

    return Subtask(
        **st_kwargs,
        segments=tuple(build_segment(segment) for segment in segment_data),
        algo_params=algo_cls.from_dict(algo_kwargs),
    )


def build_segment(data: dict[str, Any]) -> SegmentSpec:
    """Build one scheduled trajectory segment."""

    values = dict(data)
    events = values.pop("events", [])
    source_window = values.pop("demonstration_source_window", None)
    motion_goal = values.pop("motion_goal", None)
    robot_contacts = values.pop("robot_contacts", [])
    world_contacts = values.pop("world_contacts", [])
    values["kind"] = SegmentKind(values["kind"])
    values["provider"] = SegmentProvider(values["provider"])
    if "goal_segment_id" in values:
        values["goal_segment_id"] = SegmentId.from_string(values["goal_segment_id"])
    if motion_goal is not None and "object_pose_after_segment_id" in motion_goal:
        motion_goal = dict(motion_goal)
        motion_goal["object_pose_after_segment_id"] = SegmentId.from_string(motion_goal["object_pose_after_segment_id"])
    values = _coerce_tuples(values, get_type_hints(SegmentSpec))
    return SegmentSpec(
        **values,
        events=tuple(build_segment_event(event) for event in events),
        motion_goal=(
            MotionGoalSpec(**_coerce_tuples(motion_goal, get_type_hints(MotionGoalSpec)))
            if motion_goal is not None
            else None
        ),
        demonstration_source_window=(
            DemonstrationSourceWindowSpec(**source_window) if source_window is not None else None
        ),
        robot_contacts=tuple(
            SegmentRobotContactSpec(**{**contact, "robot_link_names": tuple(contact["robot_link_names"])})
            for contact in robot_contacts
        ),
        world_contacts=tuple(SegmentWorldContactSpec(**contact) for contact in world_contacts),
    )


def build_segment_event(data: dict[str, Any]) -> SegmentEventSpec:
    """Build one object-state event."""

    values = dict(data)
    values["event_type"] = SegmentEventType(values["event_type"])
    if "point" in values:
        values["point"] = SegmentEventPoint(values["point"])
    return SegmentEventSpec(**values)


def build_schedule(data: dict[str, Any]) -> ScheduleSpec:
    """Build the compact waypoint-retiming schedule specification."""

    values = dict(data)
    constraints = values.pop("constraints", [])
    if "strategy" in values:
        values["strategy"] = SchedulerStrategy(values["strategy"])
    parsed_constraints = []
    for constraint in constraints:
        assert isinstance(constraint, dict), "each schedule constraint must be a mapping"
        item = dict(constraint)
        constraint_type = item.pop("type", None)
        if constraint_type == "precedence":
            item["before"] = SegmentAnchor.from_string(item["before"])
            item["after"] = SegmentAnchor.from_string(item["after"])
            parsed_constraints.append(PrecedenceConstraintSpec(**item))
        elif constraint_type == "non_overlap":
            assert set(item) == {"segments"}, "non_overlap requires exactly a segments field"
            segments = item["segments"]
            assert isinstance(segments, list) and len(segments) == 2, "non_overlap.segments must contain two IDs"
            parsed_constraints.append(
                NonOverlapConstraintSpec(tuple(SegmentId.from_string(value) for value in segments))
            )
        elif constraint_type == "non_preemptive":
            assert set(item) == {"segments"}, "non_preemptive requires exactly a segments field"
            segments = item["segments"]
            assert isinstance(segments, list) and segments, "non_preemptive.segments must not be empty"
            parsed_constraints.append(
                NonPreemptiveConstraintSpec(tuple(SegmentId.from_string(value) for value in segments))
            )
        else:
            assert False, f"unsupported schedule constraint type: {constraint_type!r}"
    return ScheduleSpec(**values, constraints=tuple(parsed_constraints))


def build_constraint(c_data: dict[str, Any]) -> SubtaskConstraint:
    """Build a :class:`SubtaskConstraint` from a parsed constraint dict.

    Enum fields accept their member name (case-insensitive); the ``(eef, index)`` pairs and other
    tuple fields accept YAML lists.
    """

    c_data = dict(c_data)  # copy so we don't mutate caller's dict
    c_data["constraint_type"] = _coerce_enum(c_data["constraint_type"], SubTaskConstraintType)
    if "coordination_scheme" in c_data:
        c_data["coordination_scheme"] = _coerce_enum(c_data["coordination_scheme"], SubTaskConstraintCoordinationScheme)
    if "eef_subtask_constraint_tuple" in c_data:
        c_data["eef_subtask_constraint_tuple"] = tuple(tuple(pair) for pair in c_data["eef_subtask_constraint_tuple"])
    return SubtaskConstraint(**c_data)
