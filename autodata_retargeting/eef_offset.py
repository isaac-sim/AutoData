# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-EEF SE(3) offset (control link -> canonical grasp frame): build, load, compose."""

import torch
import yaml

import isaaclab.utils.math as math_utils

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter


def build_eef_offsets(
    offset_data: dict[str, dict], eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    """Build per-EEF ``(4, 4)`` SE(3) offsets from ``{eef: {axis_angle, translation}}``.

    Rotate-then-translate convention: the frame is first rotated by ``axis_angle`` (a compact
    axis-angle vector, axis * angle [rad]) then translated by ``translation`` [m] *in that rotated
    (canonical) frame* — so the translation reads as forward/lateral/up at the grasp, consistent
    across embodiments. Missing entries default to identity. Post-multiplied onto the source pose.
    """
    offsets: dict[str, torch.Tensor] = {}
    for eef_name in eef_names:
        entry = offset_data.get(eef_name, {}) or {}
        rotation = torch.eye(3, dtype=torch.float32, device=device)
        axis_angle = torch.tensor(entry.get("axis_angle", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        angle = torch.linalg.norm(axis_angle)
        if float(angle) > 1e-8:
            axis = (axis_angle / angle).unsqueeze(0)
            rotation = math_utils.matrix_from_quat(math_utils.quat_from_angle_axis(angle.unsqueeze(0), axis))[0]
        translation = torch.tensor(entry.get("translation", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        offset = torch.eye(4, dtype=torch.float32, device=device)
        offset[:3, :3] = rotation
        offset[:3, 3] = rotation @ translation  # translation applied in the rotated (canonical) frame
        offsets[eef_name] = offset
    return offsets


def _eef_offset_from_dict(data: dict, eef_names: list[str], device: torch.device) -> dict[str, torch.Tensor] | None:
    """Resolve an embodiment's per-EEF ``offset`` (control link → canonical grasp frame) as ``(4, 4)``.

    Supports both embodiment schemas (see ``calibrate_offset.py``):

    * **Bimanual**: each EEF's ``offset`` (``axis_angle`` + ``translation``) lives under ``eefs.<eef>``.
    * **Single-arm**: a top-level ``eef_offset`` (translation [m]) and ``eef_rotation`` (axis-angle
      [rad]). This is the *same* control-link -> observed-EEF translation
      :class:`~autodata_interfaces.embodiments.single_arm_embodiment_adapter.SingleArmEmbodimentAdapter`
      reads as its own ``eef_offset`` field (``eef_rotation`` has no live-adapter counterpart; it is
      retargeting-only) -- not an unrelated value that happens to share the name. A single-arm
      embodiment has exactly one EEF, so that one offset is applied to whichever EEF name is requested
      (``eef_names`` comes from the target adapter, e.g. ``["ur10"]``).

    Returns None if the embodiment declares no offset (so composition falls back to identity).
    """
    eefs = data.get("eefs") or {}
    if eefs:  # bimanual: per-EEF offset under eefs.<eef>.offset
        offset_data = {name: cfg["offset"] for name, cfg in eefs.items() if isinstance(cfg, dict) and "offset" in cfg}
    elif "eef_offset" in data or "eef_rotation" in data:  # single-arm: top-level eef_offset/eef_rotation
        entry = {
            "translation": data.get("eef_offset", [0.0, 0.0, 0.0]),
            "axis_angle": data.get("eef_rotation", [0.0, 0.0, 0.0]),
        }
        # One-EEF embodiment: apply its single offset to whichever EEF name(s) the retarget requests.
        offset_data = {name: entry for name in eef_names}
    else:
        offset_data = {}
    return build_eef_offsets(offset_data, eef_names, device) if offset_data else None


def load_embodiment_eef_offset(
    embodiment_yaml: str, eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Load an embodiment's per-EEF grasp-frame ``offset`` (see :func:`_eef_offset_from_dict`) as ``(4, 4)``.

    Opens and parses ``embodiment_yaml`` itself; prefer :func:`eef_offset_from_adapter` when an
    :class:`~autodata_interfaces.embodiments.embodiment_adapter.EmbodimentAdapter` built from the same
    file is already in hand, so the file is not parsed twice.
    """
    with open(embodiment_yaml) as f:
        data = yaml.safe_load(f) or {}
    return _eef_offset_from_dict(data, eef_names, device)


def eef_offset_from_adapter(
    adapter: EmbodimentAdapter, eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Read an embodiment's per-EEF grasp-frame offset from an adapter's ``raw_config`` (see
    :attr:`~autodata_interfaces.embodiments.embodiment_adapter.EmbodimentAdapter.raw_config`), reusing
    the config the factory already parsed instead of re-opening the YAML file.
    """
    return _eef_offset_from_dict(adapter.raw_config or {}, eef_names, device)


def compose_retarget_eef_offsets(
    source_embodiment: str, target_embodiment: str, eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Per-EEF ``T_src @ inv(T_tgt)`` from the two embodiments' ``eef_offset`` declarations, or None.

    Expresses the source trajectory in the canonical grasp frame (``@ T_src``) and re-anchors to the
    target's control link (``@ inv(T_tgt)``), so grippers with different geometry/convention align at
    the grasp point. Post-multiplied onto the source pose, exactly like a manual ``eef_offsets``.
    """
    source_offset = load_embodiment_eef_offset(source_embodiment, eef_names, device)
    target_offset = load_embodiment_eef_offset(target_embodiment, eef_names, device)
    if source_offset is None or target_offset is None:
        return None
    return {eef: source_offset[eef] @ torch.linalg.inv(target_offset[eef]) for eef in eef_names}


def compose_retarget_eef_offsets_from_adapters(
    source_adapter: EmbodimentAdapter, target_adapter: EmbodimentAdapter, eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Per-EEF ``T_src @ inv(T_tgt)`` from two already-built adapters' ``raw_config`` (see
    :func:`eef_offset_from_adapter`), reusing the config the factory already parsed for both embodiments
    instead of re-opening their YAML files, otherwise identical to :func:`compose_retarget_eef_offsets`.
    """
    source_offset = eef_offset_from_adapter(source_adapter, eef_names, device)
    target_offset = eef_offset_from_adapter(target_adapter, eef_names, device)
    if source_offset is None or target_offset is None:
        return None
    return {eef: source_offset[eef] @ torch.linalg.inv(target_offset[eef]) for eef in eef_names}
