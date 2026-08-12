# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Arena environments composed by Isaac AutoData."""

from .registration import is_arena_environment, register_environment_for_run, register_environment_from_cli

__all__ = ["is_arena_environment", "register_environment_for_run", "register_environment_from_cli"]
