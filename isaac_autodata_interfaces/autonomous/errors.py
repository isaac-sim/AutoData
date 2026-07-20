# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Structured validation errors for autonomous request compilation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_SIMPLE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ValidationIssue:
    """One request or catalog validation failure.

    Args:
        path: Field path components. String components are mapping keys and integer components are
            sequence indices.
        code: Stable machine-readable failure code.
        message: Safe human-readable explanation.
    """

    path: tuple[str | int, ...]
    code: str
    message: str

    @property
    def field_path(self) -> str:
        """Render :attr:`path` as an unambiguous JSONPath-like field path."""

        rendered = "$"
        for component in self.path:
            if isinstance(component, int):
                rendered += f"[{component}]"
            elif _SIMPLE_FIELD_RE.fullmatch(component):
                rendered += f".{component}"
            else:
                rendered += f"[{json.dumps(component, ensure_ascii=False)}]"
        return rendered

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue representation."""

        return {"path": self.field_path, "code": self.code, "message": self.message}

    def __str__(self) -> str:
        return f"{self.field_path} [{self.code}]: {self.message}"


class AutonomousValidationError(ValueError):
    """Raised when semantic input or trusted catalog configuration is invalid.

    All discovered schema issues are available through :attr:`issues`. Fatal YAML construction
    failures, such as duplicate keys, contain one precise issue.
    """

    def __init__(self, issues: list[ValidationIssue] | tuple[ValidationIssue, ...]) -> None:
        assert issues, "AutonomousValidationError requires at least one issue"
        self.issues = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Return all validation issues in a JSON-serializable envelope."""

        return {"issues": [issue.to_dict() for issue in self.issues]}
