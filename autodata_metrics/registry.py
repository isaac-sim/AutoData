# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of dataset and run metrics.

A metric is a function of :class:`~autodata_metrics.evidence.Evidence` registered with :func:`metric`. It declares
the evidence it ``requires``; the runner reports it as not applicable, naming the missing key, instead of calling it
without that evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from autodata_metrics.evidence import Evidence

AREAS = ("validity", "diversity", "viability", "diagnostics")


class NotApplicable(Exception):
    """Raised by a metric whose evidence is present but does not let it apply."""


@dataclass(frozen=True)
class Value:
    """What a metric computed.

    Attributes:
        value: A number, or named numbers such as percentiles.
        n: Sample count the value rests on.
        details: Context a reader needs to interpret the value.
    """

    value: float | dict[str, float]
    n: int
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric and how to report it."""

    name: str
    compute: Callable[[Evidence], Value]
    area: str
    signal: str
    scope: Literal["dataset", "run"]
    unit: str
    better: Literal["higher", "lower"] | None
    requires: tuple[str, ...]
    primary: str | None
    version: int


REGISTERED_METRICS: dict[str, MetricSpec] = {}


def metric(
    name: str,
    *,
    area: str,
    signal: str,
    scope: Literal["dataset", "run"],
    unit: str,
    better: Literal["higher", "lower"] | None = None,
    requires: tuple[str, ...] | list[str] = (),
    primary: str | None = None,
    version: int = 1,
) -> Callable[[Callable[[Evidence], Value]], Callable[[Evidence], Value]]:
    """Register the decorated function as the metric ``name``.

    Args:
        name: Unique metric name, used in reports and gates.
        area: One of :data:`AREAS`.
        signal: What the metric measures within its area, e.g. ``"task correctness"``.
        scope: ``"dataset"`` metrics read episodes; ``"run"`` metrics read attempts and the run record.
        unit: Unit of the value.
        better: Which direction is better, if either.
        requires: Evidence keys; see :meth:`~autodata_metrics.evidence.Evidence.has`.
        primary: For a dict value, the entry a gate checks by default.
        version: Raise it whenever the definition changes, so gates pinned to the old one fail.
    """
    assert area in AREAS, f"Metric {name!r} has area {area!r}; expected one of {AREAS}."

    def register(compute: Callable[[Evidence], Value]) -> Callable[[Evidence], Value]:
        assert name not in REGISTERED_METRICS, f"Metric {name!r} is already registered."
        REGISTERED_METRICS[name] = MetricSpec(
            name=name,
            compute=compute,
            area=area,
            signal=signal,
            scope=scope,
            unit=unit,
            better=better,
            requires=tuple(requires),
            primary=primary,
            version=version,
        )
        return compute

    return register
