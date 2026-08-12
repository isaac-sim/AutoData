# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register AutoData-owned Arena environments for an AutoData run."""

from __future__ import annotations

import argparse
import gymnasium as gym
from functools import partial
from typing import Any

from isaaclab_arena.assets.registries import EnvironmentRegistry

from .franka_rope import FRANKA_ROPE_ARENA_ENV_ID


def is_arena_environment(env_name: str) -> bool:
    """Return whether ``env_name`` is owned by AutoData's Arena package."""

    return env_name == FRANKA_ROPE_ARENA_ENV_ID


def _bind_gym_make_kwargs(env_name: str, env_kwargs: dict[str, Any]) -> None:
    """Bind Arena-only constructor arguments for callers that use plain ``gym.make``.

    Isaac Lab's external-callback API can register an environment but cannot return
    keyword arguments to the later ``gym.make`` call. Bind those arguments into the
    registered entry point instead. Keeping them out of ``EnvSpec.kwargs`` is
    intentional: Gym deep-copies that dictionary, but Arena's variation recorder must
    retain object identity with the variations that notify it.
    """

    env_spec = gym.spec(env_name)
    entry_point = env_spec.entry_point
    assert entry_point is not None, f"Environment {env_name!r} has no Gym entry point."
    if isinstance(entry_point, str):
        from gymnasium.envs.registration import load_env_creator

        entry_point = load_env_creator(entry_point)
    env_spec.entry_point = partial(entry_point, **env_kwargs)


def register_environment_for_run(
    *,
    enable_cameras: bool,
    num_envs: int,
    device: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build and register AutoData's typed Arena environments for the current run.

    Args:
        enable_cameras: Whether to include environment camera sensors.
        num_envs: Number of parallel environments in this run.
        device: Simulation device for this run.
        seed: Environment seed for this run.

    Returns:
        Gym constructor kwargs keyed by registered environment ID.
    """

    assert (
        FRANKA_ROPE_ARENA_ENV_ID not in gym.registry
    ), f"Environment {FRANKA_ROPE_ARENA_ENV_ID!r} is already registered in this process."

    # ArenaEnvBuilder imports simulator modules and must be loaded only after AppLauncher starts Isaac Sim.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    environment_registry = EnvironmentRegistry()
    environment_factory_type = environment_registry.get_component_by_name(FRANKA_ROPE_ARENA_ENV_ID)
    environment_cfg_type = environment_registry.get_environment_cfg_type(environment_factory_type)
    arena_environment = environment_factory_type().build(environment_cfg_type(enable_cameras=enable_cameras))

    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(
            num_envs=num_envs,
            env_spacing=2.5,
            seed=seed,
            solve_relations=False,
            device=device,
        ),
    )
    env_name, _, env_kwargs = builder.build_registered()
    return {env_name: env_kwargs}


def register_environment_from_cli() -> list[str]:
    """Register the Arena rope environment through Isaac Lab's callback API.

    Isaac Lab invokes external registration callbacks without arguments. This adapter
    reads the standard Lab and Arena options already present in ``sys.argv``, builds the
    typed AutoData environment, and binds Arena's additional constructor arguments for
    the script's later plain ``gym.make`` call.

    Returns:
        Command-line arguments not consumed by the environment registration parser.
    """

    from isaaclab.app import AppLauncher
    from isaaclab_arena.cli.isaaclab_arena_cli import (
        add_isaac_lab_cli_args,
        add_isaaclab_arena_cli_args,
        arena_env_builder_cfg_from_argparse,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.cli import add_environment_cli_args, build_environment_from_cli

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default=FRANKA_ROPE_ARENA_ENV_ID)
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)

    environment_factory_type = EnvironmentRegistry().get_component_by_name(FRANKA_ROPE_ARENA_ENV_ID)
    add_environment_cli_args(parser, environment_factory_type)
    parser.set_defaults(env_spacing=2.5, solve_relations=False)
    args, remaining_args = parser.parse_known_args()

    requested_env_name = args.task.split(":")[-1]
    assert (
        requested_env_name == FRANKA_ROPE_ARENA_ENV_ID
    ), f"Arena callback expected {FRANKA_ROPE_ARENA_ENV_ID!r}, got {requested_env_name!r}."

    arena_environment = build_environment_from_cli(environment_factory_type, args)
    builder = ArenaEnvBuilder(arena_environment, arena_env_builder_cfg_from_argparse(args))
    env_name, _, env_kwargs = builder.build_registered()
    _bind_gym_make_kwargs(env_name, env_kwargs)
    return remaining_args
