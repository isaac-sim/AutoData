# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode datagen schema consumed by :class:`DataGenerator`.

This is the generator-internal stand-in for the eventual ``Datastream`` abstraction.
"""

from __future__ import annotations

import torch
from copy import deepcopy
from typing import Any


class DatagenInfo:
    """Per-step data extracted from an episode for use during data generation.

    Every field is optional so subtask segments (which carry only a subset) can use the same type.
    The full per-episode record carries all fields populated.

    Attributes:
        eef_pose: ``{eef_name: [T, 4, 4]}`` recorded EEF poses [m, rad].
        object_poses: ``{object_name: [T, 4, 4]}`` recorded object poses [m, rad].
        object_nodal_positions: ``{object_name: [T, N, 3]}`` recorded deformable-object
            nodal positions [m].
        subtask_term_signals: ``{subtask_name: [T]}`` binary completion flag per step.
        subtask_start_signals: ``{subtask_name: [T]}`` binary start flag per step; required by SkillGen.
        target_eef_pose: ``{eef_name: [T, 4, 4]}`` controller target poses [m, rad].
        passthrough_action: ``{channel_name: [T, D]}`` non-pose actions copied verbatim from the
            source demo, keyed by channel (eef grippers/hands plus any non-eef channel such as a
            base/locomotion command).
    """

    def __init__(
        self,
        eef_pose: dict[str, torch.Tensor] | None = None,
        object_poses: dict[str, torch.Tensor] | None = None,
        object_nodal_positions: dict[str, torch.Tensor] | None = None,
        subtask_term_signals: dict[str, Any] | None = None,
        subtask_start_signals: dict[str, Any] | None = None,
        target_eef_pose: dict[str, torch.Tensor] | None = None,
        passthrough_action: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.eef_pose = eef_pose
        self.object_poses = dict(object_poses) if object_poses is not None else None
        self.object_nodal_positions = dict(object_nodal_positions) if object_nodal_positions is not None else None
        self.subtask_term_signals = dict(subtask_term_signals) if subtask_term_signals is not None else None
        self.subtask_start_signals = dict(subtask_start_signals) if subtask_start_signals is not None else None
        self.target_eef_pose = target_eef_pose
        self.passthrough_action = passthrough_action

    def to_dict(self) -> dict[str, Any]:
        """Materialize a dict with only the populated fields."""
        out: dict[str, Any] = {}
        if self.eef_pose is not None:
            out["eef_pose"] = self.eef_pose
        if self.object_poses is not None:
            out["object_poses"] = deepcopy(self.object_poses)
        if self.object_nodal_positions is not None:
            out["object_nodal_positions"] = deepcopy(self.object_nodal_positions)
        if self.subtask_start_signals is not None:
            out["subtask_start_signals"] = deepcopy(self.subtask_start_signals)
        if self.subtask_term_signals is not None:
            out["subtask_term_signals"] = deepcopy(self.subtask_term_signals)
        if self.target_eef_pose is not None:
            out["target_eef_pose"] = self.target_eef_pose
        if self.passthrough_action is not None:
            out["passthrough_action"] = self.passthrough_action
        return out
