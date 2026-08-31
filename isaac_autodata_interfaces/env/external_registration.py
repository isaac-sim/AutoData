# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy registration hook for environments owned by another repository."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExternalEnvironmentRegistration:
    """Result returned by an external environment registration callback.

    Args:
        env_name: Gym environment id registered by the callback.
        env_kwargs: Constructor arguments that cannot safely live in Gym's global registry.
    """

    env_name: str
    env_kwargs: dict[str, Any]


def _resolve_callback(path: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = path.partition(":")
    assert (
        separator and module_name and attribute_name
    ), f"External environment callback must use 'module.path:callable', got {path!r}."
    callback = getattr(importlib.import_module(module_name), attribute_name)
    assert callable(callback), f"External environment callback is not callable: {path}"
    return callback


def register_external_environment(
    callback_path: str | None,
    *,
    env_name: str,
    num_envs: int,
    device: str,
    seed: int,
    enable_cameras: bool,
) -> ExternalEnvironmentRegistration:
    """Invoke an optional environment-owner callback after Isaac Sim starts.

    The callback registers the Gym id and may return constructor-only kwargs such as Arena's
    variation recorder. A missing callback preserves the behavior of built-in environments.
    """

    if callback_path is None:
        return ExternalEnvironmentRegistration(env_name=env_name, env_kwargs={})

    result = _resolve_callback(callback_path)(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        seed=seed,
        enable_cameras=enable_cameras,
    )
    assert isinstance(result, Mapping), (
        "External environment callback must return a mapping with 'env_name' and optional "
        f"'env_kwargs', got {type(result).__name__}."
    )
    registered_name = result.get("env_name", env_name)
    env_kwargs = result.get("env_kwargs", {})
    assert isinstance(registered_name, str) and registered_name, "Registered env_name must be non-empty."
    assert isinstance(env_kwargs, dict), "External env_kwargs must be a dict."

    import gymnasium as gym

    assert registered_name in gym.registry, f"Callback did not register Gym environment {registered_name!r}."
    return ExternalEnvironmentRegistration(env_name=registered_name, env_kwargs=dict(env_kwargs))


__all__ = ["ExternalEnvironmentRegistration", "register_external_environment"]
