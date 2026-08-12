# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for AutoData's three-dimensional TPS implementation."""

import numpy as np

from isaac_autodata_utils import thin_plate_spline as tps

_SOURCE_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
])
_TARGET_POINTS = _SOURCE_POINTS.copy()
_TARGET_POINTS[0] += [0.10, -0.05, 0.02]
_TARGET_POINTS[1] += [0.20, 0.10, -0.03]
_TARGET_POINTS[4] += [-0.10, 0.15, 0.05]
_QUERY_POINTS = np.array([
    [0.25, 0.25, 0.25],
    [0.75, 0.25, 0.25],
])


def test_reduced_fit_matches_previous_tps_reference_outputs():
    linear, translation, weights = tps.fit_reduced(
        source_points=_SOURCE_POINTS,
        target_points=_TARGET_POINTS,
        bend_coefficient=0.1,
        rotation_coefficient=1e-3,
    )

    transformed = tps.evaluate(
        query_points=_QUERY_POINTS,
        linear=linear,
        translation=translation,
        weights=weights,
        source_points=_SOURCE_POINTS,
    )
    expected_transformed = np.array([
        [0.303313771813, 0.251697797185, 0.258325640304],
        [0.821598859240, 0.310808443454, 0.249183096591],
    ])
    np.testing.assert_allclose(transformed, expected_transformed, rtol=1e-10, atol=1e-10)

    jacobians = tps.gradient(
        query_points=_QUERY_POINTS,
        linear=linear,
        translation=translation,
        weights=weights,
        source_points=_SOURCE_POINTS,
    )
    expected_jacobians = np.array([
        [
            [1.032732261752, -0.163893736190, -0.132353662016],
            [0.116302039264, 1.049445488374, 0.002135377114],
            [-0.016366130876, 0.012051710831, 0.996281673744],
        ],
        [
            [1.032732261752, -0.235557062625, -0.167147111810],
            [0.116302039264, 1.050479586533, -0.052135339689],
            [-0.016366130876, 0.047883374048, 1.013678398641],
        ],
    ])
    np.testing.assert_allclose(jacobians, expected_jacobians, rtol=1e-10, atol=1e-10)


def test_regularized_fit_cost_matches_previous_tps_reference_output():
    linear, translation, weights = tps.fit(
        source_points=_SOURCE_POINTS,
        target_points=_TARGET_POINTS,
        bend_coefficient=0.1,
        rotation_coefficient=1e-3,
    )
    registration_cost = tps.cost(
        linear=linear,
        translation=translation,
        weights=weights,
        source_points=_SOURCE_POINTS,
        target_points=_TARGET_POINTS,
        bend_coefficient=0.1,
    )
    np.testing.assert_allclose(registration_cost, 0.0073160059705457805, rtol=1e-10, atol=1e-10)
