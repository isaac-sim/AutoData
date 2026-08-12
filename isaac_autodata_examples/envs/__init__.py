# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example simulation environments used by Isaac AutoData workflows."""

import argparse
from typing import Any


def register_environment_for_run(
    *,
    env_name: str,
    enable_cameras: bool,
    num_envs: int,
    device: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Register the AutoData-owned environment selected for the current run.

    Args:
        env_name: Gym ID of the environment selected for this run.
        enable_cameras: Whether Arena environments should include their camera sensors.
        num_envs: Number of parallel environments in this run.
        device: Simulation device for this run.
        seed: Environment seed for this run.

    Returns:
        Gym constructor kwargs keyed by registered environment ID.
    """

    from .isaac_lab_arena import is_arena_environment
    from .isaac_lab_arena import register_environment_for_run as register_arena_environment

    if not is_arena_environment(env_name):
        return {}
    return register_arena_environment(
        enable_cameras=enable_cameras,
        num_envs=num_envs,
        device=device,
        seed=seed,
    )


def register_environments() -> list[str]:
    """External callback used by unmodified Isaac Lab scripts.

    Isaac Lab invokes this function without arguments after starting Isaac Sim. The
    selected ``--task`` determines whether the callback needs to compile an AutoData
    Arena environment or only import AutoData's regular Isaac Lab registrations.

    Returns:
        Command-line arguments not consumed by Arena environment registration.
    """

    from .isaac_lab_arena import is_arena_environment, register_environment_from_cli

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str)
    args, _ = parser.parse_known_args()
    env_name = args.task.split(":")[-1] if args.task is not None else None
    if env_name is not None and not is_arena_environment(env_name):
        return []
    return register_environment_from_cli()
