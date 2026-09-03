# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embodiment factory: YAML/dict to concrete :class:`EmbodimentAdapter`.

The registry maps a YAML ``type:`` discriminator to the concrete adapter
class. :func:`embodiment_adapter_from_yaml` and
:func:`embodiment_adapter_from_dict` dispatch through it. Both accept an
optional ``env`` to bind in one step. New morphology + controller
combinations register themselves by adding to
:data:`EMBODIMENT_TYPE_REGISTRY`.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from autodata_interfaces.embodiments.bimanual_embodiment_adapter import AbsolutePoseWholeBodyBimanualAdapter
from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_interfaces.embodiments.single_arm_embodiment_adapter import DeltaPoseIKSingleArmAdapter

EMBODIMENT_TYPE_REGISTRY: dict[str, type[EmbodimentAdapter]] = {
    "delta_pose_ik_single_arm": DeltaPoseIKSingleArmAdapter,
    "absolute_pose_whole_body_bimanual": AbsolutePoseWholeBodyBimanualAdapter,
}
"""Maps the ``type:`` discriminator in an embodiment YAML to the corresponding
concrete adapter class. New morphology + controller combinations register
themselves here."""


def embodiment_adapter_from_yaml(path: str | Path, env: Any = None) -> EmbodimentAdapter:
    """Load and instantiate an :class:`EmbodimentAdapter` from a YAML file.

    Args:
        path: Path to the YAML config.
        env: Optional live env handle. If provided, the returned adapter is
            pre-bound via :meth:`EmbodimentAdapter.bind_env`.

    Returns:
        Concrete :class:`EmbodimentAdapter` instance.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return embodiment_adapter_from_dict(data, env=env)


def embodiment_adapter_from_dict(data: dict[str, Any], env: Any = None) -> EmbodimentAdapter:
    """Build an :class:`EmbodimentAdapter` from a parsed config dict.

    Validates ``type:`` is present and registered, then delegates the rest
    to the concrete class's ``from_dict`` classmethod.

    Args:
        data: Parsed YAML/JSON config dict.
        env: Optional live env handle. If provided, the returned adapter is
            pre-bound via :meth:`EmbodimentAdapter.bind_env`.

    Returns:
        Concrete :class:`EmbodimentAdapter` instance.
    """
    assert isinstance(data, dict), f"Expected top-level dict, got {type(data).__name__}"
    assert "type" in data, f"Missing required top-level key 'type'. Got: {sorted(data)}"
    type_key = data["type"]
    assert isinstance(type_key, str), f"'type' must be a string, got {type(type_key).__name__}"
    assert (
        type_key in EMBODIMENT_TYPE_REGISTRY
    ), f"Unknown embodiment type {type_key!r}. Registered: {sorted(EMBODIMENT_TYPE_REGISTRY)}"
    cls = EMBODIMENT_TYPE_REGISTRY[type_key]
    payload = {k: v for k, v in data.items() if k != "type"}
    adapter = cls.from_dict(payload)  # type: ignore[attr-defined]
    if env is not None:
        adapter.bind_env(env)
    return adapter
