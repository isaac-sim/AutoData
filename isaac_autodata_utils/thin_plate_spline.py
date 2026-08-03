# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Three-dimensional thin-plate-spline fitting and evaluation.

This utility module contains the small numerical surface SoftMimicGen needs for deformable-object
registration. It intentionally avoids the unrelated robotics stack that accompanied the former
Rapprentice dependency.
"""

from __future__ import annotations

import numpy as np

_POINT_DIMENSION = 3


def _points(points: np.ndarray, name: str) -> np.ndarray:
    """Return finite three-dimensional points as double-precision values."""

    points = np.asarray(points, dtype=np.float64)
    assert (
        points.ndim == 2 and points.shape[1] == _POINT_DIMENSION
    ), f"{name} must have shape (N, {_POINT_DIMENSION}), got {points.shape}"
    assert np.isfinite(points).all(), f"{name} contains non-finite values"
    return points


def _corresponding_points(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return corresponding source and target points."""

    source_points = _points(source_points, "source_points")
    target_points = _points(target_points, "target_points")
    assert (
        source_points.shape == target_points.shape
    ), f"source_points and target_points must have matching shapes, got {source_points.shape} and {target_points.shape}"
    assert (
        source_points.shape[0] >= _POINT_DIMENSION + 1
    ), f"TPS requires at least {_POINT_DIMENSION + 1} corresponding points, got {source_points.shape[0]}"
    return source_points, target_points


def _kernel_matrix(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    """Return the three-dimensional TPS kernel ``K(r) = -r``."""

    differences = left_points[:, None, :] - right_points[None, :, :]
    return -np.linalg.norm(differences, axis=-1)


def _transform_parameters(
    linear: np.ndarray,
    translation: np.ndarray,
    weights: np.ndarray,
    source_point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return TPS transform parameters."""

    linear = np.asarray(linear, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    assert linear.shape == (
        _POINT_DIMENSION,
        _POINT_DIMENSION,
    ), f"linear must have shape ({_POINT_DIMENSION}, {_POINT_DIMENSION}), got {linear.shape}"
    assert translation.shape == (
        _POINT_DIMENSION,
    ), f"translation must have shape ({_POINT_DIMENSION},), got {translation.shape}"
    assert weights.shape == (
        source_point_count,
        _POINT_DIMENSION,
    ), f"weights must have shape ({source_point_count}, {_POINT_DIMENSION}), got {weights.shape}"
    return linear, translation, weights


def fit(
    source_points: np.ndarray,
    target_points: np.ndarray,
    bend_coefficient: float,
    rotation_coefficient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a regularized TPS transform between corresponding points.

    Args:
        source_points: Source control points shaped ``(N, 3)``.
        target_points: Target control points shaped ``(N, 3)``.
        bend_coefficient: TPS bending regularization coefficient.
        rotation_coefficient: Affine-rotation regularization coefficient.

    Returns:
        A ``(linear, translation, weights)`` tuple describing the fitted transform.
    """

    source_points, target_points = _corresponding_points(source_points, target_points)
    point_count, dimension = source_points.shape
    kernel = _kernel_matrix(source_points, source_points)
    rotation_ratio = bend_coefficient / rotation_coefficient if rotation_coefficient > 0 else 0.0

    system = np.zeros((point_count + dimension + 1, point_count + dimension + 1))
    system[:point_count, :point_count] = kernel
    diagonal = np.arange(point_count)
    system[diagonal, diagonal] += bend_coefficient
    system[:point_count, point_count : point_count + dimension] = source_points
    system[:point_count, point_count + dimension] = 1.0
    system[point_count : point_count + dimension, :point_count] = source_points.T
    system[point_count + dimension, :point_count] = 1.0
    system[
        point_count : point_count + dimension,
        point_count : point_count + dimension,
    ] = rotation_ratio * np.eye(dimension)

    right_hand_side = np.empty((point_count + dimension + 1, dimension))
    right_hand_side[:point_count] = target_points
    right_hand_side[point_count : point_count + dimension] = rotation_ratio * np.eye(dimension)
    right_hand_side[point_count + dimension] = 0.0

    solution = np.linalg.solve(system, right_hand_side)
    weights = solution[:point_count]
    linear = solution[point_count : point_count + dimension]
    translation = solution[point_count + dimension]
    return linear, translation, weights


def fit_reduced(
    source_points: np.ndarray,
    target_points: np.ndarray,
    bend_coefficient: float,
    rotation_coefficient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a TPS transform in the null space of its affine constraints.

    Args:
        source_points: Source control points shaped ``(N, 3)``.
        target_points: Target control points shaped ``(N, 3)``.
        bend_coefficient: TPS bending regularization coefficient.
        rotation_coefficient: Affine-rotation regularization coefficient.

    Returns:
        A ``(linear, translation, weights)`` tuple describing the fitted transform.
    """

    source_points, target_points = _corresponding_points(source_points, target_points)
    point_count, dimension = source_points.shape
    affine_column_count = dimension + 1
    affine_basis = np.column_stack((source_points, np.ones(point_count)))
    left_singular_vectors, _, _ = np.linalg.svd(affine_basis, full_matrices=True)
    null_space = left_singular_vectors[:, affine_column_count:]

    kernel = _kernel_matrix(source_points, source_points)
    design = np.column_stack((source_points, np.ones(point_count), kernel @ null_space))
    system = design.T @ design
    system[affine_column_count:, affine_column_count:] += bend_coefficient * null_space.T @ kernel @ null_space
    right_hand_side = design.T @ target_points
    system[:dimension, :dimension] += rotation_coefficient * np.eye(dimension)
    right_hand_side[:dimension, :dimension] += rotation_coefficient * np.eye(dimension)

    solution = np.linalg.solve(system, right_hand_side)
    linear = solution[:dimension]
    translation = solution[dimension]
    weights = null_space @ solution[affine_column_count:]
    return linear, translation, weights


def evaluate(
    query_points: np.ndarray,
    linear: np.ndarray,
    translation: np.ndarray,
    weights: np.ndarray,
    source_points: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted TPS transform at query points.

    Args:
        query_points: Points to transform, shaped ``(M, 3)``.
        linear: Affine linear component, shaped ``(3, 3)``.
        translation: Affine translation component, shaped ``(3,)``.
        weights: Non-rigid kernel weights, shaped ``(N, 3)``.
        source_points: Source control points, shaped ``(N, 3)``.

    Returns:
        Transformed points shaped ``(M, 3)``.
    """

    query_points = _points(query_points, "query_points")
    source_points = _points(source_points, "source_points")
    linear, translation, weights = _transform_parameters(
        linear,
        translation,
        weights,
        source_points.shape[0],
    )
    kernel = _kernel_matrix(query_points, source_points)
    return kernel @ weights + query_points @ linear + translation[None, :]


def gradient(
    query_points: np.ndarray,
    linear: np.ndarray,
    translation: np.ndarray,
    weights: np.ndarray,
    source_points: np.ndarray,
) -> np.ndarray:
    """Evaluate the local Jacobian of a fitted TPS transform.

    Args:
        query_points: Points at which to evaluate the Jacobian, shaped ``(M, 3)``.
        linear: Affine linear component, shaped ``(3, 3)``.
        translation: Affine translation component, shaped ``(3,)``.
        weights: Non-rigid kernel weights, shaped ``(N, 3)``.
        source_points: Source control points, shaped ``(N, 3)``.

    Returns:
        Transform Jacobians shaped ``(M, 3, 3)``.
    """

    query_points = _points(query_points, "query_points")
    source_points = _points(source_points, "source_points")
    linear, _, weights = _transform_parameters(linear, translation, weights, source_points.shape[0])
    differences = query_points[:, None, :] - source_points[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    directions = np.divide(
        differences,
        distances[:, :, None],
        out=np.zeros_like(differences),
        where=distances[:, :, None] != 0.0,
    )
    kernel_gradient = np.einsum("mna,ng->mga", directions, weights)
    return linear.T[None, :, :] - kernel_gradient


def cost(
    linear: np.ndarray,
    translation: np.ndarray,
    weights: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    bend_coefficient: float,
) -> float:
    """Return the residual-plus-bending cost of a fitted TPS transform.

    Args:
        linear: Affine linear component, shaped ``(3, 3)``.
        translation: Affine translation component, shaped ``(3,)``.
        weights: Non-rigid kernel weights, shaped ``(N, 3)``.
        source_points: Source control points, shaped ``(N, 3)``.
        target_points: Target control points, shaped ``(N, 3)``.
        bend_coefficient: TPS bending regularization coefficient.

    Returns:
        Scalar residual-plus-bending cost.
    """

    source_points, target_points = _corresponding_points(source_points, target_points)
    linear, translation, weights = _transform_parameters(
        linear,
        translation,
        weights,
        source_points.shape[0],
    )
    kernel = _kernel_matrix(source_points, source_points)
    predicted_points = kernel @ weights + source_points @ linear + translation[None, :]
    residual_cost = np.square(predicted_points - target_points).sum()
    bending_cost = bend_coefficient * np.sum(weights * (kernel @ weights))
    return float(residual_cost + bending_cost)
