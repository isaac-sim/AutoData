# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from dataclasses import fields
from typing import Any, get_origin, get_type_hints

from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from isaac_autodata_interfaces.tasks.subtask_constraint_spec import (
    SubtaskConstraint,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaac_autodata_interfaces.tasks.subtask_spec import ALGO_PARAMS_REGISTRY, Subtask, SubtaskAlgoParams


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
    optional_keys = {"description", "constraints", "generation_policy"}
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

    subtasks = data["subtasks"]
    assert isinstance(
        subtasks, dict
    ), f"'subtasks' must be a dict (eef_name -> list of subtasks), got {type(subtasks).__name__}"

    algo_cls = ALGO_PARAMS_REGISTRY[data["algo"]]
    subtask_keys = {f.name for f in fields(Subtask)}
    algo_keys = {f.name for f in fields(algo_cls)}

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
    """Build a :class:`Subtask` from a parsed subtask dict."""

    st_data = dict(st_data)  # copy so we don't mutate caller's dict
    algo_kwargs = st_data.pop("algo_params", {})

    st_kwargs = _coerce_tuples(st_data, get_type_hints(Subtask))
    algo_kwargs = _coerce_tuples(algo_kwargs, get_type_hints(algo_cls))

    return Subtask(**st_kwargs, algo_params=algo_cls(**algo_kwargs))


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
