# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build object-centric carry segments and subtask boundaries from the descriptor's subtasks.

Each EEF's subtask list (from the retarget descriptor) partitions the source trajectory: a subtask ends
on an event -- a gripper open/close, or a subtask-signal edge -- and a subtask may declare an object it
tracks over its whole span. This module turns those declarations into :class:`CarrySegment`s (where the
replay plans for the *object* instead of the EEF, so the grasped object reproduces its source path
regardless of gripper geometry) plus the boundary steps where the replayer holds a settle. The live
per-step object-centric override lives in the replay loop (it needs the *target* env's grasp transform).
"""

import torch
import warnings
from dataclasses import dataclass
from typing import Any


@dataclass
class CarrySegment:
    """A span in which ``eef`` carries ``obj`` (object-centric planning applies on ``[start, end]``).

    ``grasp_transform`` (target ``eef_T_object``) is re-measured live each replay step (kept here mainly
    for the debug readout).
    """

    eef: str
    obj: str
    start: int
    end: int
    # Object-centric transition smoothing over this segment (see ObjectTracking interpolation params):
    # ramp EEF-path -> object over ``interp_start`` steps at the start, object -> EEF-path over
    # ``interp_after`` steps past the end. Per-segment so different subtasks can ease differently.
    interp_start: int = 0
    interp_after: int = 0
    grasp_transform: torch.Tensor | None = None
    last_target: torch.Tensor | None = None  # last commanded pose in-segment; the start of the end-ramp
    object_start_real: torch.Tensor | None = None  # object's real pose at the carry start (for the ease-in)


def _event_step(mask: torch.Tensor, start: int, rising: bool) -> int | None:
    """First step at or after ``start`` where boolean ``mask`` transitions (rising: F->T, else T->F)."""
    values = mask.flatten().bool()
    for t in range(max(start, 1), values.numel()):
        prev, curr = bool(values[t - 1]), bool(values[t])
        if (rising and curr and not prev) or (not rising and prev and not curr):
            return t
    return None


def _motion_start(frac: torch.Tensor, crossing: int, rising: bool, lower_bound: int) -> int:
    """Back up from a gripper crossing to the last frame *before* the motion began (its leading edge).

    ``frac`` is the per-step closedness (0 open -> 1 closed). From ``crossing`` (the first ``>=`` / ``<``
    close-fraction step), step back through the still-monotonic transition -- rising for a close, falling
    for an open -- and return the frame just before it started, clamped to ``lower_bound``. Speed-agnostic:
    a binary gripper backs up one frame, a slow ramped one backs up to where the ramp left its plateau.
    """
    f = frac.flatten()
    eps = 1e-4
    t = crossing
    while t > lower_bound and ((rising and f[t - 1] < f[t] - eps) or (not rising and f[t - 1] > f[t] + eps)):
        t -= 1
    return t


def _resolve_subtask_end(
    end: Any,
    eef_key: str,
    start: int,
    num_steps: int,
    gripper_closed: dict[str, torch.Tensor],
    signals: dict[str, torch.Tensor],
    name_map: dict[str, str],
    close_fraction: float,
) -> int | None:
    """Trajectory step where subtask-end event ``end`` fires (searching from ``start``), offset applied.

    ``end`` is a :class:`~.config.SubtaskEnd` (duck-typed: ``.method``, ``.signal``, ``.eef``, ``.offset``,
    ``.length``). ``fixed_length`` ends the subtask ``.length`` frames after ``start`` (no source event).
    Gripper events read ``gripper_closed[target_eef]`` (a per-step closedness fraction, keyed by the
    *target* EEF -- the trigger EEF name is mapped through ``name_map``); ``gripper_closing`` /
    ``gripper_opening`` back the crossing up to the motion's leading edge (:func:`_motion_start`). Signal
    events read ``signals[name]``. Returns None (with a warning) if the referenced channel is missing or the
    event never fires, so a partially-specified demo still runs.
    """
    method, offset = end.method, end.offset
    if method == "fixed_length":
        step = start + end.length  # a fixed duration from the subtask's start (no source event)
    elif method in ("gripper_close", "gripper_open", "gripper_closing", "gripper_opening"):
        trigger = end.eef or eef_key
        frac = gripper_closed.get(name_map.get(trigger, trigger))
        if frac is None:
            warnings.warn(
                f"subtask_end {method!r} references EEF {trigger!r} with no source gripper signal; "
                "the subtask boundary is set to the trajectory end. Provide source hand postures.",
                stacklevel=2,
            )
            return None
        f = frac.flatten()
        rising = method in ("gripper_close", "gripper_closing")
        step = _event_step(f >= close_fraction, start, rising=rising)
        if step is not None and method in ("gripper_closing", "gripper_opening"):
            step = _motion_start(f, step, rising, start)  # back up to the leading edge of the motion
    else:  # signal_on / signal_off
        sig = signals.get(end.signal)
        if sig is None:
            warnings.warn(
                f"subtask_end signal {end.signal!r} is not in the source demo's subtask_term_signals "
                f"({sorted(signals)}); the subtask boundary is set to the trajectory end.",
                stacklevel=2,
            )
            return None
        step = _event_step(sig.flatten() > 0.5, start, rising=(method == "signal_on"))
    if step is None:
        warnings.warn(f"subtask_end {method!r} never fired after step {start}; boundary set to trajectory end.", 2)
        return None
    return max(0, min(num_steps - 1, step + offset))


def iter_subtask_spans(
    subtasks: dict[str, list],
    gripper_closed: dict[str, torch.Tensor],
    signals: dict[str, torch.Tensor],
    num_steps: int,
    eef_name_map: dict[str, str] | None = None,
    close_fraction: float = 0.5,
):
    """Yield ``(eef_key, target_eef, index, subtask, start, end, boundary)`` for every subtask.

    Each EEF's subtasks partition ``[0, num_steps)``: subtask ``i`` runs from the previous subtask's end
    (0 for the first) to its own ``subtask_end`` event step; the last subtask (which omits ``subtask_end``)
    runs to the trajectory end. ``boundary`` is the resolved end-event step (equal to ``end`` when the
    event fired) or ``None`` for the last subtask / when the event never fired. ``eef_key`` is the *source*
    EEF name the subtask is listed under; ``target_eef`` is it renamed through ``eef_name_map``.
    """
    name_map = eef_name_map or {}
    for eef_key, entries in subtasks.items():
        target_eef = name_map.get(eef_key, eef_key)
        start = 0
        for i, st in enumerate(entries):
            if i == len(entries) - 1:
                end, boundary = num_steps - 1, None
            else:
                assert (
                    st.subtask_end is not None
                ), f"subtask {i} of EEF {eef_key!r} needs a 'subtask_end' (only the last subtask may omit it)."
                boundary = _resolve_subtask_end(
                    st.subtask_end, eef_key, start, num_steps, gripper_closed, signals, name_map, close_fraction
                )
                end = num_steps - 1 if boundary is None else boundary
            end = max(end, start)
            yield eef_key, target_eef, i, st, start, end, boundary
            start = end


def carry_segments_and_boundaries_from_subtasks(
    subtasks: dict[str, list],
    gripper_closed: dict[str, torch.Tensor],
    signals: dict[str, torch.Tensor],
    num_steps: int,
    eef_name_map: dict[str, str] | None = None,
    default_interp_start: int = 0,
    default_interp_after: int = 0,
    default_settle_steps: int = 0,
    close_fraction: float = 0.5,
) -> tuple[list[CarrySegment], dict[str, dict[int, int]], dict[str, tuple[str, int]]]:
    """Turn the descriptor's per-EEF subtask lists into carry segments + subtask-boundary steps.

    Each EEF's subtasks partition the trajectory: subtask ``i`` runs from the previous subtask's end (0
    for the first) to its own ``subtask_end`` event step; the last subtask (which omits ``subtask_end``)
    runs to the trajectory end. A subtask that declares ``object_tracking`` becomes a
    :class:`CarrySegment` over its whole span (its per-subtask interp overrides, else the defaults). The
    non-final subtask ends are returned as the boundary steps where the replayer holds its settle, each
    mapped to its settle-hold cap (``SubtaskEnd.settle_steps`` if set, else ``default_settle_steps``).

    ``subtasks`` values are :class:`~.config.Subtask` objects (duck-typed). Keyed by *source* EEF name;
    ``eef_name_map`` renames each onto the target EEF the segment commands. ``gripper_closed`` is keyed by
    *target* EEF name (per-step closedness fraction); ``signals`` by signal name.

    Returns:
        ``(carry_segments sorted by start, {target_eef: {boundary_step: settle_cap}}, {name: (target_eef,
        end_step)})``. Each subtask boundary is registered under its **own** EEF (per-EEF settle), so each
        arm settles at its own subtask ends and the executor schedules the arms independently; declare a
        ``synchronization`` group to re-couple them (the third return maps each named subtask to its
        ``(target_eef, end_step)`` for that barrier).
    """
    segments: list[CarrySegment] = []
    boundaries: dict[str, dict[int, int]] = {}
    name_end_step: dict[str, tuple[str, int]] = {}
    for _eef_key, target_eef, _i, st, start, end, boundary in iter_subtask_spans(
        subtasks, gripper_closed, signals, num_steps, eef_name_map, close_fraction
    ):
        eef_boundaries = boundaries.setdefault(target_eef, {})
        if boundary is not None:
            override = st.subtask_end.settle_steps
            cap = default_settle_steps if override is None else override
            eef_boundaries[boundary] = max(eef_boundaries.get(boundary, 0), cap)
        if st.name is not None:
            name_end_step[st.name] = (target_eef, end)
        if st.object_tracking is not None:
            iss, isa = st.object_tracking.interpolation_step_start, st.object_tracking.interpolation_step_after
            segments.append(
                CarrySegment(
                    eef=target_eef,
                    obj=st.object_tracking.object,
                    start=start,
                    end=end,
                    interp_start=default_interp_start if iss is None else iss,
                    interp_after=default_interp_after if isa is None else isa,
                )
            )
    return sorted(segments, key=lambda seg: seg.start), boundaries, name_end_step
