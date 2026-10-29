# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline metrics of AutoData datasets and generation runs.

Producers record evidence; :func:`build_report` computes every registered metric from it without a simulator.
Importing this package registers the built-in metrics.
"""

from autodata_metrics import validity, viability
from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import (
    AREAS,
    REGISTERED_CHECKS,
    REGISTERED_METRICS,
    CheckSpec,
    MetricSpec,
    NotApplicable,
    Value,
    check,
    metric,
)
from autodata_metrics.report import build_report, format_report, gate_results
