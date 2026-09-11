# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan providers: the *source* of retargeting examples, decoupled from the replayer.

A :class:`PlanProvider` hands out :class:`Plan` objects (one retargeting example each) and owns the
**stop condition**. The replayer just pulls plans until ``next()`` returns ``None`` and reports each
outcome back via ``observe()``. This separates "which example / how many" (provider) from "execute it
on the target robot" (replayer).

:class:`DatasetReplayProvider` is the current behaviour: replay every source episode once (optionally a
selected subset). Stop when the source is exhausted, or early once a target number of successes or of
total runs is reached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Plan:
    """One retargeting example to replay.

    For :class:`DatasetReplayProvider` this wraps a loaded source episode. ``index`` / ``name`` are for
    logging + output.
    """

    index: int
    name: str
    episode: Any = None  # isaaclab EpisodeData (or an equivalent trajectory the replayer can consume)


@dataclass
class ReplayResult:
    """Outcome of replaying one :class:`Plan`, fed back to the provider via :meth:`PlanProvider.observe`."""

    success: bool
    eef_errors: dict[str, dict[str, Any]] | None = None


class PlanProvider(ABC):
    """Hands out retargeting examples and owns the stop condition.

    Contract: call :meth:`next` to get the next :class:`Plan` or ``None`` when done (target reached or
    source exhausted); after replaying it, report the outcome with :meth:`observe` so success/run targets
    can stop early. Providers are pulled cooperatively (single-env loop, or the async parallel workers),
    so implementations need no locks.
    """

    @abstractmethod
    def next(self) -> Plan | None:
        """Return the next example, or ``None`` when the provider is done."""

    def next_for_env(self, env_id: int) -> Plan | None:
        """Next plan for a specific parallel worker's ``env_id``.

        Defaults to :meth:`next` (env-agnostic -- the copy provider's plans do not depend on which env
        runs them).
        """
        return self.next()

    def observe(self, plan: Plan, result: ReplayResult) -> None:
        """Record one replay outcome (default: no-op). Overridden to drive success/run targets."""

    def in_flight(self) -> int:
        """Plans handed out but not yet observed (used by the parallel drain to know when to exit)."""
        return 0

    @abstractmethod
    def loop_policy(self) -> tuple[bool, int]:
        """``(guarantee_success, num_trials)`` mirroring the stop condition, for the parallel ``env_loop``.

        ``guarantee_success`` True -> the loop terminates on that many *successes*; False -> on that many
        *attempts*. The async workers also exit once the provider is drained, so an unreachable success
        target on a finite source still terminates (via ``env_loop``'s task-completion check).
        """


class DatasetReplayProvider(PlanProvider):
    """Replay each source episode once (optionally a selected subset), i.e. today's full-replay behaviour.

    Args:
        dataset_handler: An opened HDF5 dataset handler exposing ``load_episode(name, device)``.
        episode_names: Ordered source-episode names to replay.
        device: Torch device to load episodes onto.
        target_successes: Stop once this many replays have succeeded (``None`` = no success target).
        target_runs: Stop once this many replays have been attempted (``None`` = no run target).

    With neither target set, it stops when the source is exhausted. The two targets and exhaustion
    compose: whichever is reached first ends the run.
    """

    def __init__(
        self,
        dataset_handler: Any,
        episode_names: list[str],
        device: Any,
        target_successes: int | None = None,
        target_runs: int | None = None,
    ) -> None:
        assert target_successes is None or target_successes > 0, "target_successes must be positive"
        assert target_runs is None or target_runs > 0, "target_runs must be positive"
        self._handler = dataset_handler
        self._names = episode_names
        self._device = device
        self._target_successes = target_successes
        self._target_runs = target_runs
        self._cursor = 0  # next source index to hand out
        self._handed_out = 0
        self._completed = 0
        self._num_success = 0

    def _target_reached(self) -> bool:
        if self._target_successes is not None and self._num_success >= self._target_successes:
            return True
        if self._target_runs is not None and self._completed >= self._target_runs:
            return True
        return False

    def next(self) -> Plan | None:
        if self._target_reached() or self._cursor >= len(self._names):
            return None
        name = self._names[self._cursor]
        index = self._cursor
        self._cursor += 1
        self._handed_out += 1
        return Plan(index=index, name=name, episode=self._handler.load_episode(name, self._device))

    def observe(self, plan: Plan, result: ReplayResult) -> None:
        self._completed += 1
        self._num_success += int(result.success)

    def in_flight(self) -> int:
        return self._handed_out - self._completed

    def loop_policy(self) -> tuple[bool, int]:
        if self._target_successes is not None:
            return True, self._target_successes
        if self._target_runs is not None:
            return False, min(self._target_runs, len(self._names))
        return False, len(self._names)

