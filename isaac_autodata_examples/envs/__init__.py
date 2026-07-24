# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Example simulation environments used by Isaac AutoData workflows."""


def register_environments(enable_cameras: bool = True) -> list[str]:
    """Register all AutoData-owned Isaac Lab and Isaac Lab Arena environments.

    Args:
        enable_cameras: Whether Arena environments should include their camera sensors.

    Returns:
        An empty list of callback-specific command-line arguments.
    """

    from .isaac_lab import register_environments as register_isaac_lab_environments
    from .isaac_lab_arena import register_environments as register_isaac_lab_arena_environments

    register_isaac_lab_environments()
    register_isaac_lab_arena_environments(enable_cameras=enable_cameras)
    return []
