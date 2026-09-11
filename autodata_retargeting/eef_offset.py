# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-EEF SE(3) offset (control link -> canonical grasp frame): build, load, compose."""

import torch
import yaml

import isaaclab.utils.math as math_utils


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


def load_embodiment_eef_offset(
    embodiment_yaml: str, eef_names: list[str], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Load an embodiment's per-EEF ``offset`` (control link → canonical grasp frame) as ``(4, 4)``.

    Supports both embodiment schemas (see ``calibrate_offset.py``):

    * **Bimanual**: each EEF's ``offset`` (``axis_angle`` + ``translation``) lives under ``eefs.<eef>``.
    * **Single-arm**: a top-level ``eef_offset`` (translation [m]) and ``eef_rotation`` (axis-angle
      [rad]). A single-arm embodiment has exactly one EEF, so that one offset is applied to whichever
      EEF name is requested (``eef_names`` comes from the target adapter, e.g. ``["ur10"]``).

    Returns None if the embodiment declares no offset (so composition falls back to identity).
    """
    with open(embodiment_yaml) as f:
        data = yaml.safe_load(f) or {}
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
