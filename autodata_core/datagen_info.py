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

    @classmethod
    def from_episode(cls, ep_grp: dict, eef_name_map: dict[str, str] | None = None) -> DatagenInfo:
        """Decode one episode's ``obs/datagen_info/*`` group into a record.

        The single decode of this group shared by :meth:`DataGenInfoPool._add_episode` and
        ``autodata_retargeting``'s ``source_datagen_poses`` (both previously decoded it by hand, which
        could drift apart). Every field is read with ``.get`` -- absent -> ``None`` on the returned
        record, consistent with every field here being optional -- so a caller that only needs one or
        two fields (e.g. just ``eef_pose``) does not require the dataset to carry the full set
        :class:`DataGenInfoPool` does; it asserts its own required fields.

        Args:
            ep_grp: One episode's raw data mapping (``EpisodeData.data``), with an ``"obs"`` entry.
            eef_name_map: Optional ``{source_eef: target_eef}`` renaming applied to the eef-keyed
                fields (``eef_pose``, ``target_eef_pose``); identity when omitted.

        Returns:
            A populated record (fields absent from the episode are ``None``); no ``passthrough_action``
            (decoding that needs an embodiment adapter, not this episode's ``datagen_info``).
        """
        obs = ep_grp.get("obs", {})
        datagen_info = obs.get("datagen_info") if isinstance(obs, dict) else None
        if not isinstance(datagen_info, dict):
            raise ValueError("Episode lacks 'datagen_info' obs annotations")

        def _rename(poses: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
            if poses is None or not eef_name_map:
                return poses
            return {eef_name_map.get(name, name): pose for name, pose in poses.items()}

        return cls(
            eef_pose=_rename(datagen_info.get("eef_pose")),
            object_poses=datagen_info.get("object_pose"),
            object_nodal_positions=datagen_info.get("object_nodal_position"),
            subtask_term_signals=datagen_info.get("subtask_term_signals"),
            subtask_start_signals=datagen_info.get("subtask_start_signals"),
            target_eef_pose=_rename(datagen_info.get("target_eef_pose")),
        )

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
