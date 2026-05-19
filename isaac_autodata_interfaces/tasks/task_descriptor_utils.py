# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import fields
from typing import Any, get_origin, get_type_hints

from isaac_autodata_interfaces.tasks.subtask_spec import (
    ALGO_PARAMS_REGISTRY,
    Subtask,
    SubtaskAlgoParams,
)


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

    assert isinstance(data, dict), (
        f"Expected top-level dict, got {type(data).__name__}"
    )

    required_keys = {"name", "algo", "subtasks"}
    optional_keys = {"description"}
    keys = set(data)
    missing = required_keys - keys
    assert not missing, f"Missing required top-level keys: {sorted(missing)}"
    unknown = keys - required_keys - optional_keys
    assert not unknown, (
        f"Unknown top-level keys: {sorted(unknown)}. "
        f"Allowed: {sorted(required_keys | optional_keys)}"
    )

    assert isinstance(data["name"], str), (
        f"'name' must be a string, got {type(data['name']).__name__}"
    )
    assert isinstance(data.get("description", ""), str), (
        f"'description' must be a string, got {type(data['description']).__name__}"
    )
    assert isinstance(data["algo"], str), (
        f"'algo' must be a string, got {type(data['algo']).__name__}"
    )
    assert data["algo"] in ALGO_PARAMS_REGISTRY, (
        f"Unknown algo {data['algo']!r}. "
        f"Registered: {sorted(ALGO_PARAMS_REGISTRY)}"
    )

    subtasks = data["subtasks"]
    assert isinstance(subtasks, dict), (
        f"'subtasks' must be a dict (eef_name -> list of subtasks), "
        f"got {type(subtasks).__name__}"
    )

    algo_cls = ALGO_PARAMS_REGISTRY[data["algo"]]
    subtask_keys = {f.name for f in fields(Subtask)}
    algo_keys = {f.name for f in fields(algo_cls)}

    for eef_name, eef_subtasks in subtasks.items():
        assert isinstance(eef_subtasks, list), (
            f"subtasks[{eef_name!r}] must be a list, "
            f"got {type(eef_subtasks).__name__}"
        )
        for i, st in enumerate(eef_subtasks):
            assert isinstance(st, dict), (
                f"subtasks[{eef_name!r}][{i}] must be a dict, "
                f"got {type(st).__name__}"
            )
            st_unknown = set(st) - subtask_keys
            assert not st_unknown, (
                f"subtasks[{eef_name!r}][{i}] has unknown keys "
                f"{sorted(st_unknown)}. Allowed: {sorted(subtask_keys)}"
            )
            algo_params = st.get("algo_params", {})
            assert isinstance(algo_params, dict), (
                f"subtasks[{eef_name!r}][{i}].algo_params must be a dict, "
                f"got {type(algo_params).__name__}"
            )
            ap_unknown = set(algo_params) - algo_keys
            assert not ap_unknown, (
                f"subtasks[{eef_name!r}][{i}].algo_params has unknown keys "
                f"{sorted(ap_unknown)} for algo {data['algo']!r}. "
                f"Allowed: {sorted(algo_keys)}"
            )


def build_subtask(
    st_data: dict[str, Any], algo_cls: type[SubtaskAlgoParams]
) -> Subtask:
    """Build a :class:`Subtask` from a parsed subtask dict."""

    st_data = dict(st_data)  # copy so we don't mutate caller's dict
    algo_kwargs = st_data.pop("algo_params", {})

    hints = get_type_hints(algo_cls)
    # Coerces list values back to tuples where the dataclass annotates
    # the field as ``tuple`` (YAML scalar sequences load as Python lists).
    algo_kwargs = {
        name: tuple(val)
        if isinstance(val, list) and get_origin(hints.get(name)) is tuple
        else val
        for name, val in algo_kwargs.items()
    }

    return Subtask(**st_data, algo_params=algo_cls(**algo_kwargs))
