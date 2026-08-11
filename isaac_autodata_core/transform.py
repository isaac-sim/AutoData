# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free rigid-transform values used at planner boundaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

Matrix4: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

IDENTITY_MATRIX4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def matrix4(value: Sequence[Sequence[int | float]], field_name: str = "pose") -> Matrix4:
    """Validate and normalize one approximately rigid homogeneous transform.

    Args:
        value: Four rows with four finite numeric values each.
        field_name: Field path used in validation messages.

    Returns:
        Immutable normalized transform.
    """

    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field_name} must have shape [4, 4]")
    rows = tuple(_number_tuple(row, f"{field_name}[{index}]") for index, row in enumerate(value))
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"{field_name} must have shape [4, 4]")
    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(rows[3], expected_last_row)):
        raise ValueError(f"{field_name} must have homogeneous last row [0, 0, 0, 1]")

    rotation = tuple(row[:3] for row in rows[:3])
    for column in range(3):
        norm = sum(rotation[row][column] ** 2 for row in range(3))
        if abs(norm - 1.0) > 2e-3:
            raise ValueError(f"{field_name} rotation columns must have unit norm")
        for other in range(column + 1, 3):
            dot = sum(rotation[row][column] * rotation[row][other] for row in range(3))
            if abs(dot) > 2e-3:
                raise ValueError(f"{field_name} rotation columns must be orthogonal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 2e-3:
        raise ValueError(f"{field_name} rotation determinant must be +1")
    return rows  # type: ignore[return-value]


def matrix4_multiply(first: Matrix4, second: Matrix4) -> Matrix4:
    """Compose two rigid homogeneous transforms."""

    return tuple(
        tuple(sum(first[row][inner] * second[inner][column] for inner in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def matrix4_inverse(value: Matrix4) -> Matrix4:
    """Return the rigid inverse of a homogeneous transform."""

    rotation_transpose = tuple(tuple(value[column][row] for column in range(3)) for row in range(3))
    translation = tuple(value[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_transpose[row][column] * translation[column] for column in range(3)) for row in range(3)
    )
    return tuple(
        tuple(rotation_transpose[row][column] for column in range(3)) + (inverse_translation[row],) for row in range(3)
    ) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def matrix4_error(first: Matrix4, second: Matrix4) -> tuple[float, float]:
    """Return translation [m] and rotation [rad] error between rigid transforms."""

    position_error = math.sqrt(sum((first[row][3] - second[row][3]) ** 2 for row in range(3)))
    relative_trace = sum(first[row][column] * second[row][column] for row in range(3) for column in range(3))
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    return position_error, math.acos(cosine)


def _number_tuple(values: Sequence[int | float], field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a numeric sequence")
    return tuple(_finite_float(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def _finite_float(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


__all__ = [
    "IDENTITY_MATRIX4",
    "Matrix4",
    "matrix4",
    "matrix4_error",
    "matrix4_inverse",
    "matrix4_multiply",
]
