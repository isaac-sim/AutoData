# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor interop helpers shared across the framework."""

from __future__ import annotations

import torch
from typing import Any


def as_torch(arr: Any) -> torch.Tensor:
    """Return ``arr`` as a :class:`torch.Tensor`.

    Articulation-data handles exposed as :class:`warp.array` do not support Python-style indexing
    and are converted to a torch view; inputs that are already tensors are returned unchanged.
    """

    import warp as wp

    return wp.to_torch(arr) if isinstance(arr, wp.array) else arr
