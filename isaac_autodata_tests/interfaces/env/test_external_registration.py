# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy registration of externally owned environments."""

import gymnasium as gym

from isaac_autodata_interfaces.env.external_registration import register_external_environment

_ENV_NAME = "Isaac-AutoData-External-Registration-Test-v0"
_CALLS = []


def _register(**kwargs):
    _CALLS.append(kwargs)
    if _ENV_NAME not in gym.registry:
        gym.register(_ENV_NAME, entry_point=lambda **unused: unused)
    return {"env_name": _ENV_NAME, "env_kwargs": {"sentinel": 7}}


def test_external_callback_registers_after_launch_and_returns_constructor_kwargs():
    try:
        result = register_external_environment(
            f"{__name__}:_register",
            env_name=_ENV_NAME,
            num_envs=3,
            device="cuda:1",
            seed=9,
            enable_cameras=True,
        )
        assert result.env_name == _ENV_NAME
        assert result.env_kwargs == {"sentinel": 7}
        assert _CALLS[-1] == {
            "env_name": _ENV_NAME,
            "num_envs": 3,
            "device": "cuda:1",
            "seed": 9,
            "enable_cameras": True,
        }
    finally:
        gym.registry.pop(_ENV_NAME, None)


def test_missing_callback_preserves_builtin_environment_behavior():
    result = register_external_environment(
        None,
        env_name="builtin",
        num_envs=1,
        device="cpu",
        seed=1,
        enable_cameras=False,
    )
    assert result.env_name == "builtin"
    assert result.env_kwargs == {}
