# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of dataset and run metrics.

A metric is a function of :class:`~autodata_metrics.evidence.Evidence` registered with :func:`metric`. It declares
the evidence it ``requires``; the runner reports it as not applicable, naming the missing key, instead of calling it
without that evidence.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from autodata_metrics.records import is_finite_number

if TYPE_CHECKING:
    from autodata_metrics.evidence import Episode, Evidence

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

    def __post_init__(self) -> None:
        values = self.value.values() if isinstance(self.value, Mapping) else [self.value]
        if not all(is_finite_number(item) for item in values):
            raise ValueError(f"value {self.value!r} is not a finite number or a mapping of finite numbers")
        if not isinstance(self.n, numbers.Integral) or self.n < 0:
            raise ValueError(f"sample count {self.n!r} is not a non-negative integer")
        if not isinstance(self.details, Mapping):
            raise ValueError(f"details {self.details!r} are not a mapping")


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric and how to report it."""

    name: str
    compute: Callable[[Evidence], Value]
    area: str
    signal: str
    scope: Literal["episode", "dataset", "run"]
    unit: str
    sample: str
    better: Literal["higher", "lower"] | None
    requires: tuple[str, ...]
    primary: str | None
    version: int
    baseline: bool = False


@dataclass(frozen=True)
class CheckSpec:
    """A registered episode check."""

    name: str
    check: Callable[[Episode, Evidence], str | None]
    group: str
    requires: tuple[str, ...]


REGISTERED_METRICS: dict[str, MetricSpec] = {}
REGISTERED_CHECKS: dict[str, CheckSpec] = {}
CHECK_GROUPS = ("correctness", "image")


def metric(
    name: str,
    *,
    area: str,
    signal: str,
    scope: Literal["episode", "dataset", "run"],
    unit: str,
    sample: str,
    better: Literal["higher", "lower"] | None = None,
    requires: tuple[str, ...] | list[str] = (),
    primary: str | None = None,
    version: int = 1,
    baseline: bool = False,
) -> Callable[[Callable[[Evidence], Value]], Callable[[Evidence], Value]]:
    """Register the decorated function as the metric ``name``.

    Args:
        name: Unique metric name, used in reports and gates.
        area: One of :data:`AREAS`.
        signal: What the metric measures within its area, e.g. ``"task correctness"``.
        scope: ``"episode"`` metrics summarize the episode checks over every episode; ``"dataset"`` metrics read
            the episodes that passed them; ``"run"`` metrics read attempts and the run record.
        unit: Unit of the value.
        sample: What the value's sample count counts, in the plural, e.g. ``"attempts"`` or ``"episodes"``.
        better: Which direction is better, if either.
        requires: Evidence keys; see :meth:`~autodata_metrics.evidence.Evidence.has`.
        primary: For a dict value, the entry a gate checks by default.
        version: Raise it whenever the definition changes, so gates pinned to the old one fail.
        baseline: Also compute the metric on the source dataset.
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
            sample=sample,
            better=better,
            requires=tuple(requires),
            primary=primary,
            version=version,
            baseline=baseline,
        )
        return compute

    return register


def check(
    name: str, *, group: str = "correctness", requires: tuple[str, ...] | list[str] = ()
) -> Callable[[Callable[[Episode, Evidence], str | None]], Callable[[Episode, Evidence], str | None]]:
    """Register the decorated function as the episode check ``name``.

    A check returns None for an episode that passes, a reason for one that fails, or raises :class:`NotApplicable`
    for an episode it does not cover. Episodes that fail any check are left out of every dataset metric.

    Args:
        name: Unique check name.
        group: One of :data:`CHECK_GROUPS`; the pass-rate metric the check counts toward.
        requires: Evidence keys the check needs; without them it does not run.
    """
    assert group in CHECK_GROUPS, f"Check {name!r} has group {group!r}; expected one of {CHECK_GROUPS}."

    def register(function: Callable[[Episode, Evidence], str | None]) -> Callable[[Episode, Evidence], str | None]:
        assert name not in REGISTERED_CHECKS, f"Check {name!r} is already registered."
        REGISTERED_CHECKS[name] = CheckSpec(name=name, check=function, group=group, requires=tuple(requires))
        return function

    return register
