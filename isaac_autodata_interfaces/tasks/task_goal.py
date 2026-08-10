# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Planner-neutral task goals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class GoalPredicate:
    """One relational condition that must hold when a task is complete.

    Args:
        relation: Planner-neutral relation such as ``on`` or ``holding``.
        subject: Semantic ID of the entity constrained by the relation.
        target: Optional semantic ID of the relation target.
    """

    relation: str
    subject: str
    target: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.relation, "relation", maximum=128)
        _require_text(self.subject, "subject")
        if self.target is not None:
            _require_text(self.target, "target")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""

        result = {"relation": self.relation, "subject": self.subject}
        if self.target is not None:
            result["target"] = self.target
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoalPredicate:
        """Construct a predicate from a JSON-compatible mapping.

        Args:
            value: Mapping containing ``relation``, ``subject``, and optional ``target``.

        Returns:
            The validated predicate.
        """

        return cls(relation=value["relation"], subject=value["subject"], target=value.get("target"))


def _require_text(value: str, field_name: str, *, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
