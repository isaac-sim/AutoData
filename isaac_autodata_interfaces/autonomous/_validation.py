# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free helpers for strict nested schema validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue

FieldPath = tuple[str | int, ...]


class IssueCollector:
    """Collect deterministic field-level validation issues before raising them together."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, path: FieldPath, code: str, message: str) -> None:
        """Append an issue at ``path``."""

        self.issues.append(ValidationIssue(path=path, code=code, message=message))

    def check_keys(
        self,
        value: dict[str, Any],
        path: FieldPath,
        *,
        required: Iterable[str],
        optional: Iterable[str] = (),
    ) -> None:
        """Report missing and unknown keys for one mapping."""

        required_keys = set(required)
        allowed_keys = required_keys | set(optional)
        for key in sorted(required_keys - set(value)):
            self.add(path + (key,), "missing_field", "required field is missing")
        for key in sorted(set(value) - allowed_keys):
            self.add(path + (key,), "unknown_field", "field is not allowed by schema v1")

    def raise_if_any(self) -> None:
        """Raise :class:`AutonomousValidationError` when any issues were collected."""

        if self.issues:
            raise AutonomousValidationError(self.issues)


def require_mapping(value: Any, path: FieldPath, issues: IssueCollector) -> dict[str, Any] | None:
    """Return ``value`` as a string-keyed mapping or report an exact-type error."""

    if type(value) is not dict:
        issues.add(path, "invalid_type", f"expected mapping, got {type(value).__name__}")
        return None
    invalid_keys = [key for key in value if type(key) is not str]
    if invalid_keys:
        issues.add(path, "invalid_mapping_key", "mapping keys must be strings")
        return None
    return value


def require_string(
    value: Any,
    path: FieldPath,
    issues: IssueCollector,
    *,
    allow_empty: bool = False,
) -> str | None:
    """Return a string while rejecting non-strings and blank identifiers."""

    if type(value) is not str:
        issues.add(path, "invalid_type", f"expected string, got {type(value).__name__}")
        return None
    if not allow_empty and not value.strip():
        issues.add(path, "empty_value", "value must not be empty or whitespace-only")
        return None
    if "\x00" in value:
        issues.add(path, "invalid_value", "value must not contain a NUL character")
        return None
    return value


def require_bool(value: Any, path: FieldPath, issues: IssueCollector) -> bool | None:
    """Return an exact YAML boolean; integers and truthy strings are rejected."""

    if type(value) is not bool:
        issues.add(path, "invalid_type", f"expected boolean, got {type(value).__name__}")
        return None
    return value


def require_int(
    value: Any,
    path: FieldPath,
    issues: IssueCollector,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Return an exact integer with optional inclusive bounds."""

    if type(value) is not int:
        issues.add(path, "invalid_type", f"expected integer, got {type(value).__name__}")
        return None
    if minimum is not None and value < minimum:
        issues.add(path, "out_of_range", f"value must be at least {minimum}, got {value}")
        return None
    if maximum is not None and value > maximum:
        issues.add(path, "out_of_range", f"value must be at most {maximum}, got {value}")
        return None
    return value


def require_finite_number(
    value: Any,
    path: FieldPath,
    issues: IssueCollector,
    *,
    minimum_exclusive: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Return a finite exact int/float value, excluding booleans."""

    if type(value) not in (int, float):
        issues.add(path, "invalid_type", f"expected number, got {type(value).__name__}")
        return None
    result = float(value)
    if not math.isfinite(result):
        issues.add(path, "non_finite", "value must be finite")
        return None
    if minimum_exclusive is not None and result <= minimum_exclusive:
        issues.add(path, "out_of_range", f"value must be greater than {minimum_exclusive:g}, got {value}")
        return None
    if maximum is not None and result > maximum:
        issues.add(path, "out_of_range", f"value must be at most {maximum:g}, got {value}")
        return None
    return result
