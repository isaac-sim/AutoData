# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-demo container for :class:`DataGenerator`.

Holds parsed :class:`DatagenInfo` records plus a per-EEF subtask-boundary index. The HDF5 loader
lives in :meth:`DataGenInfoPool.from_hdf5`; the bare constructor lets callers build pools from
episodes obtained any other way (live recordings, in-memory replays, tests).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaac_autodata_core.datagen_info import DatagenInfo

if TYPE_CHECKING:
    from isaac_autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
    from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor


class DataGenInfoPool:
    """Container of source :class:`DatagenInfo` records keyed by EEF subtask boundary."""

    def __init__(
        self,
        task_descriptor: TaskDescriptor,
        embodiment_adapter: EmbodimentAdapter,
        device,
        uses_start_signals: bool,
        asyncio_lock: asyncio.Lock | None = None,
    ) -> None:
        """
        Args:
            task_descriptor: Source of subtask semantics.
            embodiment_adapter: Projects recorded actions into passthrough actions (per-eef
                grippers plus any non-eef channels) via ``actions_to_passthrough_actions``.
            device: Target torch device for episode tensors.
            uses_start_signals: Whether the algorithm reads subtask start signals
                (SkillGen-only). Selects how subtask boundaries are parsed and validated.
            asyncio_lock: Optional lock guarding concurrent ``add_episode`` calls when a single pool
                feeds multiple async generators.
        """
        self._datagen_infos: list[DatagenInfo] = []
        self._subtask_boundaries: dict[str, list[list[tuple[int, int]]]] = {}

        self.task_descriptor = task_descriptor
        self.embodiment_adapter = embodiment_adapter
        self.device = device
        self.uses_start_signals = uses_start_signals
        self._asyncio_lock = asyncio_lock

        self.subtask_term_signal_names: dict[str, list[str]] = {}
        self.subtask_term_offset_ranges: dict[str, list[tuple[int, int]]] = {}
        self.subtask_start_offset_ranges: dict[str, list[tuple[int, int]]] = {}

        for eef_name in task_descriptor.get_eef_names():
            subtasks = task_descriptor.get_subtasks(eef_name)
            self.subtask_term_signal_names[eef_name] = [st.subtask_term_signal for st in subtasks]
            self.subtask_term_offset_ranges[eef_name] = [st.subtask_term_offset_range for st in subtasks]
            self.subtask_start_offset_ranges[eef_name] = [
                getattr(st.algo_params, "subtask_start_offset_range", (0, 0)) for st in subtasks
            ]

    @classmethod
    def from_hdf5(
        cls,
        file_path: str,
        task_descriptor: TaskDescriptor,
        embodiment_adapter: EmbodimentAdapter,
        device,
        uses_start_signals: bool,
        select_demo_keys: list[str] | None = None,
        asyncio_lock: asyncio.Lock | None = None,
    ) -> DataGenInfoPool:
        """Build a pool by reading every (or selected) episode from an HDF5 dataset."""
        pool = cls(
            task_descriptor=task_descriptor,
            embodiment_adapter=embodiment_adapter,
            device=device,
            uses_start_signals=uses_start_signals,
            asyncio_lock=asyncio_lock,
        )
        handler = HDF5DatasetFileHandler()
        handler.open(file_path)
        episode_names = handler.get_episode_names()
        for episode_name in episode_names:
            if select_demo_keys is not None and episode_name not in select_demo_keys:
                continue
            pool._add_episode(handler.load_episode(episode_name, device))
        return pool

    @property
    def datagen_infos(self) -> list[DatagenInfo]:
        return self._datagen_infos

    @property
    def subtask_boundaries(self) -> dict[str, list[list[tuple[int, int]]]]:
        return self._subtask_boundaries

    @property
    def asyncio_lock(self) -> asyncio.Lock | None:
        return self._asyncio_lock

    @property
    def num_datagen_infos(self) -> int:
        return len(self._datagen_infos)

    async def add_episode(self, episode: EpisodeData) -> None:
        """Append an episode to the pool; locked when ``asyncio_lock`` was provided."""
        if self._asyncio_lock is not None:
            async with self._asyncio_lock:
                self._add_episode(episode)
        else:
            self._add_episode(episode)

    def _add_episode(self, episode: EpisodeData) -> None:
        ep_grp = episode.data

        if "datagen_info" not in ep_grp["obs"]:
            raise ValueError("Episode lacks 'datagen_info' obs annotations")

        eef_pose = ep_grp["obs"]["datagen_info"]["eef_pose"]
        object_poses_dict = ep_grp["obs"]["datagen_info"].get("object_pose")
        object_nodal_positions_dict = ep_grp["obs"]["datagen_info"].get("object_nodal_position")
        target_eef_pose = ep_grp["obs"]["datagen_info"]["target_eef_pose"]
        subtask_term_signals_dict = ep_grp["obs"]["datagen_info"]["subtask_term_signals"]
        subtask_start_signals_dict = ep_grp["obs"]["datagen_info"].get("subtask_start_signals")

        passthrough_actions = self.embodiment_adapter.actions_to_passthrough_actions(ep_grp["actions"])

        ep_datagen_info = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses_dict,
            object_nodal_positions=object_nodal_positions_dict,
            subtask_start_signals=subtask_start_signals_dict,
            subtask_term_signals=subtask_term_signals_dict,
            target_eef_pose=target_eef_pose,
            passthrough_action=passthrough_actions,
        )
        self._datagen_infos.append(ep_datagen_info)

        for eef_name in self.subtask_term_signal_names.keys():
            self._subtask_boundaries.setdefault(eef_name, [])
            self._subtask_boundaries[eef_name].append(self._parse_subtask_boundaries(eef_name, ep_datagen_info, ep_grp))

    def _parse_subtask_boundaries(
        self,
        eef_name: str,
        ep: DatagenInfo,
        ep_grp,
    ) -> list[tuple[int, int]]:
        """Compute ``[(start, end), ...]`` indices for each subtask of ``eef_name`` in this episode."""
        use_skillgen = self.uses_start_signals
        signal_names = self.subtask_term_signal_names[eef_name]
        boundaries: list[tuple[int, int]] = []
        prev_end = 0

        for subtask_index, signal_name in enumerate(signal_names):
            if use_skillgen:
                if ep.subtask_start_signals is None:
                    raise ValueError(
                        f"subtask_start_signals missing for subtask {signal_name!r} but use_skillgen is enabled"
                    )
                start_indicators = ep.subtask_start_signals[signal_name].flatten().int()
                start_index = int((start_indicators[1:] - start_indicators[:-1]).nonzero()[0][0]) + 1
            else:
                start_index = prev_end

            if subtask_index == len(signal_names) - 1:
                end_index = ep_grp["actions"].shape[0]
            else:
                term_indicators = ep.subtask_term_signals[signal_name].flatten().int()
                end_index = int((term_indicators[1:] - term_indicators[:-1]).nonzero()[0][0]) + 2

            if end_index <= start_index:
                raise ValueError(f"subtask {signal_name!r} has non-increasing boundary: {start_index} -> {end_index}")
            boundaries.append((start_index, end_index))
            prev_end = end_index

        self._validate_subtask_offsets(eef_name, boundaries)
        return boundaries

    def _validate_subtask_offsets(self, eef_name: str, boundaries: list[tuple[int, int]]) -> None:
        """Sanity-check that the worst-case randomized boundaries stay non-empty and non-overlapping."""
        start_offsets = self.subtask_start_offset_ranges[eef_name]
        term_offsets = self.subtask_term_offset_ranges[eef_name]

        if self.uses_start_signals:
            for i, (s, e) in enumerate(boundaries):
                assert s + start_offsets[i][1] < e + term_offsets[i][0], f"subtask {i} empty in worst case"
                if i == len(boundaries) - 1:
                    break
                assert (
                    e + term_offsets[i][1] < boundaries[i + 1][0] + start_offsets[i + 1][0]
                ), f"subtasks {i} and {i + 1} overlap in worst case"
        else:
            for i in range(1, len(boundaries)):
                prev_max = term_offsets[i - 1][1]
                assert boundaries[i - 1][1] + prev_max < boundaries[i][1] + term_offsets[i][0], (
                    f"subtask boundary violation: prev_end={boundaries[i - 1][1]} +max_offset={prev_max} "
                    f"vs subtask {i} end={boundaries[i][1]} +min_offset={term_offsets[i][0]}"
                )
