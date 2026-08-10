# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset coordination between asynchronous generators and the synchronous environment loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvResetRequest:
    """Request an environment reset and wait for data-generation preparation to finish.

    Args:
        env_id: Environment index to reset.
        completion: Future resolved by the environment loop after reset settling and recorder
            initialization are complete.
    """

    env_id: int
    completion: asyncio.Future[None]
