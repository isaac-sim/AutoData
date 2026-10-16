# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight CLI validation that can run before simulation startup."""

import argparse


def callable_path(value: str) -> str:
    """Validate a callable path without importing its module.

    Args:
        value: Callable written as module.function or module:function.

    Returns:
        The validated path.
    """
    separator = ":" if ":" in value else "."
    module_name, _, name = value.rpartition(separator)
    if not name.isidentifier() or not all(part.isidentifier() for part in module_name.split(".")):
        raise argparse.ArgumentTypeError(
            f"Expected a full callable path (module.function or module:function), got {value!r}."
        )
    return value
