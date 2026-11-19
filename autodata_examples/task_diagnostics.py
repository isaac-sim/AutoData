# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific attempt diagnostics for shipped examples."""

from __future__ import annotations

import inspect
import torch
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import TerminationTermCfg

    from autodata_interfaces.datastream.datastream import Datastream


def _term_parameter(success_term: TerminationTermCfg, name: str) -> Any:
    """Return an effective termination parameter, including callable defaults."""

    if name in success_term.params:
        return success_term.params[name]
    parameter = inspect.signature(success_term.func).parameters.get(name)
    assert (
        parameter is not None and parameter.default is not inspect.Parameter.empty
    ), f"success term {success_term.func.__name__!r} has no value for {name!r}"
    return parameter.default


def _vector_format(vector: torch.Tensor) -> str:
    """Format a vector as comma-separated scalar diagnostic values."""
    return ",".join(f"{float(value):.6f}" for value in vector)


def gr1t2_nut_pour_attempt_diagnostics(
    datastream: Datastream, env_id: int, success_term: TerminationTermCfg
) -> tuple[tuple[str, str], ...]:
    """Report exact terminal components of the GR1T2 nut-pour success predicate.

    The predicate's vertical deltas are deliberately signed to match the Isaac Lab task. Absolute
    object positions are also recorded so one-sided vertical checks cannot hide a dropped object.

    Args:
        datastream: Bound AutoData datastream providing env-relative object poses.
        env_id: Vectorized environment index.
        success_term: Isaac Lab termination term whose effective defaults define the thresholds.

    Returns:
        Stable string diagnostics suitable for an attempt result sidecar.
    """

    object_poses = datastream.get_object_poses(env_ids=[env_id])
    object_parameters = {
        "sorting_scale": "sorting_scale_cfg",
        "sorting_bowl": "sorting_bowl_cfg",
        "sorting_beaker": "sorting_beaker_cfg",
        "factory_nut": "factory_nut_cfg",
        "sorting_bin": "sorting_bin_cfg",
    }
    positions = {
        label: object_poses[_term_parameter(success_term, parameter).name][0, :3, 3]
        for label, parameter in object_parameters.items()
    }
    relations = (
        (
            "nut_in_bowl",
            "nut_to_bowl",
            "factory_nut",
            "sorting_bowl",
            ("max_nut_to_bowl_x", "max_nut_to_bowl_y", "max_nut_to_bowl_z"),
        ),
        (
            "bowl_on_scale",
            "bowl_to_scale",
            "sorting_bowl",
            "sorting_scale",
            ("max_bowl_to_scale_x", "max_bowl_to_scale_y", "max_bowl_to_scale_z"),
        ),
        (
            "beaker_in_bin",
            "beaker_to_bin",
            "sorting_beaker",
            "sorting_bin",
            ("max_beaker_to_bin_x", "max_beaker_to_bin_y", "max_beaker_to_bin_z"),
        ),
    )

    diagnostics: list[tuple[str, str]] = []
    for component, relation, subject, reference, threshold_names in relations:
        displacement = positions[subject] - positions[reference]
        delta = torch.stack((displacement[0].abs(), displacement[1].abs(), displacement[2]))
        limits = torch.tensor(
            [_term_parameter(success_term, name) for name in threshold_names],
            device=delta.device,
            dtype=delta.dtype,
        )
        passed = bool(torch.all(delta < limits).item())
        diagnostics.extend((
            (f"final_{component}", str(passed).lower()),
            (f"final_{relation}_delta_m", _vector_format(delta)),
            (f"final_{relation}_limit_m", _vector_format(limits)),
        ))

    task_success = bool(success_term.func(datastream.get_env(), **success_term.params)[env_id].item())
    diagnostics.append(("final_task_success", str(task_success).lower()))
    diagnostics.extend((f"final_{label}_position_m", _vector_format(position)) for label, position in positions.items())
    return tuple(diagnostics)


__all__ = ["gr1t2_nut_pour_attempt_diagnostics"]
