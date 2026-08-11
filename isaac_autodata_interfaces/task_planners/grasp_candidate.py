# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Planner-neutral finite grasp candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from isaac_autodata_core.transform import Matrix4, matrix4

_GRASP_METHODS = frozenset({"analytical", "graspgen"})
_MAX_GRASP_CANDIDATES = 10_000


@dataclass(frozen=True)
class GraspCandidateSet:
    """Finite grasp candidates for one resolved scene object.

    ``link_from_object`` maps coordinates from the object's local planning frame into the
    planner's configured tool-link frame. The backend adapter owns conversion to its native pose
    type.

    Args:
        method: Concrete candidate source, ``analytical`` or ``graspgen``.
        object_id: Resolved semantic ID of the grasped object.
        link_from_object: Unique rigid candidate transforms.
        scores: Optional source scores aligned one-to-one with the transforms.
    """

    method: str
    object_id: str
    link_from_object: tuple[Matrix4, ...]
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.method not in _GRASP_METHODS:
            raise ValueError(f"method must be one of {sorted(_GRASP_METHODS)}")
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if len(self.object_id) > 512 or "\x00" in self.object_id:
            raise ValueError("object_id is invalid")
        if not self.link_from_object:
            raise ValueError("grasp candidate set must not be empty")
        if len(self.link_from_object) > _MAX_GRASP_CANDIDATES:
            raise ValueError(f"grasp candidate count exceeds {_MAX_GRASP_CANDIDATES}")
        transforms = tuple(
            matrix4(transform, f"link_from_object[{index}]") for index, transform in enumerate(self.link_from_object)
        )
        if len(transforms) != len(set(transforms)):
            raise ValueError("grasp candidate transforms must be unique")
        object.__setattr__(self, "link_from_object", transforms)
        if self.scores is None:
            return
        if len(self.scores) != len(transforms):
            raise ValueError("scores must align one-to-one with grasp candidates")
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in self.scores):
            raise ValueError("grasp candidate scores must be numbers")
        scores = tuple(float(score) for score in self.scores)
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("grasp candidate scores must be finite")
        object.__setattr__(self, "scores", scores)

    def __len__(self) -> int:
        """Return the number of materialized candidates."""

        return len(self.link_from_object)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "link_from_object": [[list(row) for row in transform] for transform in self.link_from_object],
            "method": self.method,
            "object_id": self.object_id,
            "scores": None if self.scores is None else list(self.scores),
        }


__all__ = ["GraspCandidateSet"]
