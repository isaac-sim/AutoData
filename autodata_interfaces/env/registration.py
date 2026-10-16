# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load externally registered environments without coupling generation to their provider."""

import argparse
import sys
from collections.abc import Callable
from typing import Any

from isaaclab.utils.string import string_to_callable

from autodata_utils.cli import callable_path


def load_callable(path: str) -> Callable[..., Any]:
    """Resolve a callable written as ``module:function`` or ``module.function``."""
    path = callable_path(path)
    return string_to_callable(path, separator=":" if ":" in path else ".")


def register_external_environment(args: argparse.Namespace, extra_args: list[str]) -> None:
    """Invoke an Isaac Lab registration callback after the simulation app starts.

    The callback receives environment arguments through ``sys.argv``, as in Isaac Lab's
    native tools. AutoData's descriptor arguments are excluded to avoid collisions with
    provider-specific options. The original arguments are restored even if registration fails.
    Options after ``--`` are forwarded verbatim, allowing provider options to reuse AutoData names.

    Args:
        args: AutoData CLI arguments, including ``external_callback`` and ``env_name``.
        extra_args: Environment-provider arguments left by ``parse_known_args``.
    """
    if args.external_callback is None:
        assert not extra_args, f"Unrecognized arguments: {extra_args}. Use --external_callback for environment options."
        return
    assert args.env_name, "--env_name is required when using --external_callback."
    callback_args = ["--task", args.env_name, "--device", args.device]
    if hasattr(args, "num_envs"):
        callback_args += ["--num_envs", str(args.num_envs)]
    if args.record_images:
        callback_args.append("--enable_cameras")
    if "--" in extra_args:
        separator = extra_args.index("--")
        extra_args = extra_args[:separator] + extra_args[separator + 1 :]
    callback_args.extend(extra_args)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *callback_args]
        remaining = load_callable(args.external_callback)()
        assert not remaining, f"Environment callback did not consume arguments: {remaining}"
    finally:
        sys.argv = original_argv
