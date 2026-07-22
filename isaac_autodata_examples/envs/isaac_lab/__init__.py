# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab environment registrations supplied by Isaac AutoData examples."""


def register_environments() -> list[str]:
    """Register AutoData's example Isaac Lab environments.

    This callback is intended for Isaac Lab tools that expose ``--external_callback``.
    The returned empty list indicates that the registration callback consumes no extra
    command-line arguments.

    Returns:
        An empty list of callback-specific command-line arguments.
    """

    from . import franka_rope  # noqa: F401

    return []
