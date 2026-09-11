# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gripper / hand passthrough remapping (binary and interpolation hand policies)."""

import torch
import yaml
from collections.abc import Callable

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter

# Passthrough channel layout: maps each passthrough channel name to its per-step width.
PassthroughLayout = dict[str, int]
# A remapper turns the source passthrough dict ({channel: (num_steps, width)}) into the target's.
PassthroughRemapper = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


def _passthrough_layout(adapter: EmbodimentAdapter) -> PassthroughLayout:
    """Return the ``{channel_name: width}`` passthrough layout an adapter produces/consumes.

    Probes the adapter with a zero action of its own ``action_dim`` so the layout is derived the
    same way for every morphology (single-arm gripper, bimanual hands + extra channels).
    """
    assert hasattr(adapter, "action_dim"), f"{type(adapter).__name__} does not expose action_dim"
    probe = torch.zeros(1, adapter.action_dim)
    channels = adapter.actions_to_passthrough_actions(probe)
    return {name: tensor.shape[-1] for name, tensor in channels.items()}


def _hand_close_fraction(
    source_hand: torch.Tensor,
    src_open: torch.Tensor,
    src_close: torch.Tensor,
    norm: str = "l1",
) -> torch.Tensor:
    """Per-step closedness fraction in ``[0, 1]`` for a source hand trajectory ``(T, dim)``.

    Scores how far the hand sits from its open toward its closed posture as the distance ratio
    ``d_open / (d_open + d_close)`` under the L1 (``norm="l1"``) or L2 (``norm="l2"``) norm: 0 when the
    hand matches ``src_open``, 1 when it matches ``src_close``. This turns a gradual source grasp into a
    smooth percentage (unlike a binary open/closed classification), so the target hand can interpolate
    instead of snapping shut.
    """
    p = 1 if norm == "l1" else 2
    d_open = (source_hand - src_open).norm(p=p, dim=-1)
    d_close = (source_hand - src_close).norm(p=p, dim=-1)
    return (d_open / (d_open + d_close + 1e-6)).clamp(0.0, 1.0)


def _apply_interp_band(alpha: torch.Tensor, band: tuple[float, float] | None) -> torch.Tensor:
    """Linearly remap a closedness fraction so ``[lo, hi]`` spans ``[0, 1]``, clamped outside the band.

    With ``band=(0.1, 0.9)``: ``alpha <= 0.1 -> 0`` (fully open), ``alpha >= 0.9 -> 1`` (fully closed),
    and the middle stretches linearly across the target's full open->close travel. Lets a source grasp
    that never quite reaches its own extremes (dead band near open/close) still drive the target hand all
    the way. ``None`` is the identity.
    """
    if band is None:
        return alpha
    lo, hi = band
    return ((alpha - lo) / (hi - lo)).clamp(0.0, 1.0)


def _apply_joint_mapping(
    source_hand: torch.Tensor,
    src_open: torch.Tensor,
    src_close: torch.Tensor,
    tgt_open: torch.Tensor,
    tgt_close: torch.Tensor,
    joint_mapping: dict[int, list[int]],
) -> torch.Tensor:
    """Drive each target hand joint from a chosen group of source joints, via a linear open->close map.

    For target joint ``X`` with sources ``[Y, Z, ...]`` (``joint_mapping[X]``): take the average source
    position over those joints, score its closedness against the averaged source ``open``/``close``
    (``0`` = open, ``1`` = closed, clamped), and lerp target joint ``X`` between its own ``open`` and
    ``close`` by that fraction. Calibrating through each side's open/close makes it robust to differing
    joint ranges and to the mirrored left/right hands (a raw copy would not). Target joints absent from
    ``joint_mapping`` are left at ``tgt_open``.
    """
    num_steps = source_hand.shape[0]
    out = tgt_open.unsqueeze(0).repeat(num_steps, 1).clone()  # (T, tgt_width); unmapped joints stay open
    for target_index, source_indices in joint_mapping.items():
        src_idx = torch.tensor(source_indices, dtype=torch.long, device=source_hand.device)
        avg = source_hand[:, src_idx].mean(dim=1)  # (T,) averaged source position
        avg_open = src_open[src_idx].mean()
        avg_close = src_close[src_idx].mean()
        alpha = ((avg - avg_open) / (avg_close - avg_open + 1e-6)).clamp(0.0, 1.0)  # (T,) closedness
        out[:, target_index] = tgt_open[target_index] + alpha * (tgt_close[target_index] - tgt_open[target_index])
    return out


def load_hand_postures(embodiment_yaml: str, eef_names: list[str]) -> dict[str, dict[str, list[float]]] | None:
    """Read per-EEF ``{eef: {open, close}}`` hand joint configs from an embodiment YAML, or None.

    Reads the ``hand_open`` / ``hand_close`` sections (each ``{eef: [joint values]}``). Returns None
    if either is absent, so retargeting can require them only for the ``binary``/``interpolation``
    hand policies.

    Bimanual YAMLs (``eefs:`` block) key the postures by their own EEF names (``left``/``right``, shared
    across the pair). A single-arm YAML has exactly one gripper, so its single posture is applied to
    whichever EEF name(s) the retarget requests (``eef_names`` comes from the target adapter) — this
    lets e.g. a Franka source (``eef_name: franka``) map onto a UR10 target (``eef_name: ur10``).
    """
    with open(embodiment_yaml) as f:
        data = yaml.safe_load(f) or {}
    hand_open = data.get("hand_open")
    hand_close = data.get("hand_close")
    if not hand_open or not hand_close:
        return None
    if "eefs" not in data:  # single-arm: one posture, applied to the requested eef name(s)
        (open_vals,) = hand_open.values()
        (close_vals,) = hand_close.values()
        return {eef: {"open": open_vals, "close": close_vals} for eef in eef_names}
    return {eef: {"open": hand_open[eef], "close": hand_close[eef]} for eef in eef_names}


def _fill_non_eef_channel(
    channel_name: str,
    width: int,
    source_passthrough: dict[str, torch.Tensor],
    target_channel_defaults: dict[str, list[float]] | None,
) -> torch.Tensor:
    """Value for a non-EEF target channel: the source's when present, else a configured default.

    Handles a target-only channel (e.g. G1's locomotion ``body`` command absent from a fixed-base
    GR1 source) by holding the ``target_channel_defaults`` value across the demo. Errors if the
    source lacks it and no default is set (zero-filling a locomotion command could drop the robot).
    """
    if channel_name in source_passthrough and source_passthrough[channel_name].shape[-1] == width:
        return source_passthrough[channel_name]
    assert target_channel_defaults and channel_name in target_channel_defaults, (
        f"target passthrough channel {channel_name!r} is not provided by the source embodiment; "
        "set its value under target_channel_defaults in the retarget config."
    )
    reference = next(iter(source_passthrough.values()))
    value = torch.tensor(target_channel_defaults[channel_name], dtype=reference.dtype, device=reference.device)
    assert len(value) == width, f"target_channel_defaults[{channel_name!r}] must have {width} values"
    return value.unsqueeze(0).expand(reference.shape[0], width).contiguous()


def build_passthrough_remapper(
    source_adapter: EmbodimentAdapter,
    target_adapter: EmbodimentAdapter,
    hand_policy: str = "passthrough",
    hand_interp_norm: str = "l1",
    hand_binary_close_threshold: float = 0.5,
    hand_interp_band: tuple[float, float] | None = None,
    joint_mapping: dict[int, list[int]] | None = None,
    source_hand_postures: dict[str, dict[str, list[float]]] | None = None,
    target_hand_postures: dict[str, dict[str, list[float]]] | None = None,
    target_channel_defaults: dict[str, list[float]] | None = None,
    eef_name_map: dict[str, str] | None = None,
) -> PassthroughRemapper:
    """Build a function that maps source passthrough (gripper/hand) actions to the target's layout.

    Policies:

    * ``"passthrough"`` — the two embodiments must share a passthrough layout (matching per-channel
      widths); the source channels are copied verbatim. Channel *names* may differ if bridged by
      ``eef_name_map`` (e.g. a Franka 1-D gripper ``franka`` → a UR10 1-D gripper ``ur10`` both copy
      through under the renamed channel). Differing *widths* (e.g. a Franka parallel gripper → a GR1
      dexterous hand) are rejected here — use ``binary``/``interpolation`` or implement the mapping.
    * ``"binary"`` — per step, score the source hand's closedness fraction (0 = its ``hand_open``,
      1 = its ``hand_close``; same distance ratio as ``interpolation``, under ``hand_interp_norm``) and
      snap the target hand to ``hand_close`` once that fraction reaches ``hand_binary_close_threshold``,
      else ``hand_open``. The threshold sets *how far* the source must close before the target grips
      (e.g. ``0.25`` = switch after 25% closed); ``0.5`` reproduces the nearest-posture classification.
    * ``"interpolation"`` — per step, score how far the source hand is along its ``hand_open`` →
      ``hand_close`` span as a distance ratio (see :func:`_hand_close_fraction`, L1/L2 via
      ``hand_interp_norm``) and linearly interpolate the target hand between its own ``hand_open`` and
      ``hand_close`` by that fraction, so a gradual grasp stays gradual. ``hand_interp_band`` optionally
      remaps that fraction first (see :func:`_apply_interp_band`) so a source that never fully opens/closes
      still drives the target across its whole range.
    * ``"joint_mapping"`` — per **target joint**, average a chosen group of **source joints** and map that
      through open/close (see :func:`_apply_joint_mapping`); the ``joint_mapping`` dict
      ``{target_idx: [source_idx, ...]}`` (indices into each embodiment's per-EEF hand vector) defines the
      wiring. Finer-grained than ``interpolation``'s single whole-hand fraction — use it to route specific
      source fingers to specific target fingers. Target joints not listed stay at ``hand_open``.

    ``"binary"``, ``"interpolation"``, and ``"joint_mapping"`` all need ``hand_open``/``hand_close`` in
    both embodiment YAMLs, and drop source-only channels; a non-EEF channel the target requires but the
    source lacks must be supplied via ``target_channel_defaults``.

    Args:
        source_adapter: Adapter of the embodiment the input dataset was recorded on.
        target_adapter: Adapter of the embodiment to retarget onto.
        hand_policy: ``"passthrough"``, ``"binary"``, or ``"interpolation"`` (see above).
        hand_interp_norm: Distance norm (``"l1"``/``"l2"``) used to score closedness under
            ``"binary"`` and ``"interpolation"``.
        hand_binary_close_threshold: For ``"binary"``, the source closedness fraction in ``[0, 1]`` at
            or above which the target snaps to ``hand_close`` (default ``0.5`` = nearest posture).
        hand_interp_band: For ``"interpolation"``, ``[lo, hi]`` sub-range of the source closedness
            fraction remapped onto the target's full ``[0, 1]`` open->close travel (values outside clamp);
            ``None`` disables it.
        joint_mapping: For ``"joint_mapping"``, ``{target_idx: [source_idx, ...]}`` wiring each target
            hand joint to the average of a group of source hand joints (indices into the per-EEF vector).
        source_hand_postures: ``{eef: {open, close}}`` for the source embodiment (binary/interpolation).
        target_hand_postures: ``{eef: {open, close}}`` for the target embodiment (binary/interpolation).
        eef_name_map: ``{source_eef: target_eef}`` renaming source channels onto the target names for
            ``"passthrough"`` (so a same-width gripper with a different EEF name passes through).

    Returns:
        A remapper ``{channel: (num_steps, width)} -> {channel: (num_steps, width)}``.
    """
    source_layout = _passthrough_layout(source_adapter)
    target_layout = _passthrough_layout(target_adapter)
    target_eef_names = set(target_adapter.get_eef_names())

    if hand_policy in ("binary", "interpolation", "joint_mapping"):
        assert (
            source_hand_postures and target_hand_postures
        ), f"--hand_policy {hand_policy} needs hand_open/hand_close in both the source and target embodiment YAMLs."
        assert hand_interp_norm in ("l1", "l2"), f"hand_interp_norm must be 'l1' or 'l2', got {hand_interp_norm!r}"
        if hand_interp_band is not None:
            lo, hi = hand_interp_band
            assert (
                0.0 <= lo < hi <= 1.0
            ), f"hand_interp_band must be [lo, hi] with 0 <= lo < hi <= 1, got {hand_interp_band}"
            assert hand_policy == "interpolation", "hand_interp_band only applies to --hand_policy interpolation."
        for eef_name in target_eef_names:
            assert (
                len(target_hand_postures[eef_name]["open"]) == target_layout[eef_name]
            ), f"target hand_open[{eef_name!r}] must have {target_layout[eef_name]} values"
        if hand_policy == "joint_mapping":
            assert (
                joint_mapping
            ), "--hand_policy joint_mapping needs a 'joint_mapping' {target_idx: [source_idx, ...]} in the descriptor."
            joint_mapping = {int(target): [int(i) for i in sources] for target, sources in joint_mapping.items()}
            for eef_name in target_eef_names:
                tgt_width, src_width = target_layout[eef_name], source_layout[eef_name]
                bad_t = sorted(x for x in joint_mapping if not 0 <= x < tgt_width)
                assert (
                    not bad_t
                ), f"joint_mapping target indices {bad_t} out of range for eef {eef_name!r} (0..{tgt_width - 1})."
                bad_s = sorted({i for sources in joint_mapping.values() for i in sources if not 0 <= i < src_width})
                assert (
                    not bad_s
                ), f"joint_mapping source indices {bad_s} out of range for eef {eef_name!r} (0..{src_width - 1})."
                assert all(joint_mapping.values()), "each joint_mapping entry needs at least one source joint index."

        def map_hand(source_hand: torch.Tensor, eef_name: str) -> torch.Tensor:
            kwargs = {"dtype": source_hand.dtype, "device": source_hand.device}
            src_open = torch.tensor(source_hand_postures[eef_name]["open"], **kwargs)
            src_close = torch.tensor(source_hand_postures[eef_name]["close"], **kwargs)
            tgt_open = torch.tensor(target_hand_postures[eef_name]["open"], **kwargs)
            tgt_close = torch.tensor(target_hand_postures[eef_name]["close"], **kwargs)
            if hand_policy == "joint_mapping":
                return _apply_joint_mapping(source_hand, src_open, src_close, tgt_open, tgt_close, joint_mapping)
            if hand_policy == "binary":
                # Snap to close once the source is >= threshold of the way closed, else open. The 1e-4
                # tolerance keeps the boundary inclusive despite the epsilon in the distance ratio, so a
                # threshold of 0.25 fires exactly at the "25% closed" point (e.g. Franka 0.03).
                alpha = _hand_close_fraction(source_hand, src_open, src_close, hand_interp_norm)
                is_closed = alpha >= hand_binary_close_threshold - 1e-4
                return torch.where(is_closed.unsqueeze(1), tgt_close.unsqueeze(0), tgt_open.unsqueeze(0))
            # interpolation: lerp the target open->close by the source's closedness fraction, after the
            # optional band remap stretches [lo, hi] of that fraction onto the target's full open->close.
            alpha = _apply_interp_band(
                _hand_close_fraction(source_hand, src_open, src_close, hand_interp_norm), hand_interp_band
            ).unsqueeze(1)
            return tgt_open.unsqueeze(0) + alpha * (tgt_close - tgt_open).unsqueeze(0)

        def remap_hands(source_passthrough: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            remapped: dict[str, torch.Tensor] = {}
            for channel_name, width in target_layout.items():
                if channel_name in target_eef_names:
                    remapped[channel_name] = map_hand(source_passthrough[channel_name], channel_name)
                else:
                    remapped[channel_name] = _fill_non_eef_channel(
                        channel_name, width, source_passthrough, target_channel_defaults
                    )
            return remapped

        return remap_hands

    assert hand_policy == "passthrough", f"unknown hand_policy {hand_policy!r}"
    # Rename the source channels onto the target names (eef_name_map: {source: target}) before comparing
    # layouts, so a same-width gripper with a different EEF name (e.g. franka -> ur10) passes through by
    # copying the value under the renamed channel. Unlisted channels are kept as-is; the rename is
    # idempotent, so pre-renamed input (the generation pool / replay path already rename) is unaffected.
    name_map = eef_name_map or {}
    renamed_source_layout = {name_map.get(channel, channel): width for channel, width in source_layout.items()}
    if renamed_source_layout == target_layout:

        def passthrough(source_passthrough: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {name_map.get(channel, channel): value for channel, value in source_passthrough.items()}

        return passthrough

    raise NotImplementedError(
        "Source and target embodiments have different passthrough (gripper/hand) layouts "
        f"(source={renamed_source_layout} after eef_name_map, target={target_layout}). If only the "
        "channel name differs, add an eef_name_map entry; if the widths differ (different grippers), "
        "pass --hand_policy binary or interpolation to map the hands via each embodiment's "
        "hand_open/hand_close, or add the mapping in build_passthrough_remapper()."
    )
