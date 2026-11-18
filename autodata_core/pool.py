# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-demo container for :class:`DataGenerator`.

Holds parsed :class:`DatagenInfo` records plus a per-EEF subtask-boundary index. The HDF5 loader
lives in :meth:`DataGenInfoPool.from_hdf5`; the bare constructor lets callers build pools from
episodes obtained any other way (live recordings, in-memory replays, tests).
"""

from __future__ import annotations

import asyncio
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from autodata_core.datagen_info import DatagenInfo

if TYPE_CHECKING:
    from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
    from autodata_interfaces.tasks.task_descriptor import TaskDescriptor


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
                (SkillGen and ScheduleGen). Selects how subtask boundaries are parsed and validated.
            asyncio_lock: Optional lock guarding concurrent ``add_episode`` calls when a single pool
                feeds multiple async generators.
        """
        self._datagen_infos: list[DatagenInfo] = []
        self._subtask_boundaries: dict[str, list[list[tuple[int, int]]]] = {}
        self._demo_keys: list[str] = []

        self.task_descriptor = task_descriptor
        self.embodiment_adapter = embodiment_adapter
        self.device = device
        self.uses_start_signals = uses_start_signals
        self._asyncio_lock = asyncio_lock

        self.subtask_term_signal_names: dict[str, list[str]] = {}
        self.subtask_start_signal_names: dict[str, list[str]] = {}
        self.subtask_term_offset_ranges: dict[str, list[tuple[int, int]]] = {}
        self.subtask_start_offset_ranges: dict[str, list[tuple[int, int]]] = {}

        for eef_name in task_descriptor.get_eef_names():
            subtasks = task_descriptor.get_subtasks(eef_name)
            self.subtask_term_signal_names[eef_name] = [st.subtask_term_signal for st in subtasks]
            self.subtask_start_signal_names[eef_name] = [
                st.subtask_start_signal or st.subtask_term_signal for st in subtasks
            ]
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
            pool._add_episode(handler.load_episode(episode_name, device), demo_key=episode_name)
        return pool

    @property
    def datagen_infos(self) -> list[DatagenInfo]:
        return self._datagen_infos

    @property
    def subtask_boundaries(self) -> dict[str, list[list[tuple[int, int]]]]:
        return self._subtask_boundaries

    @property
    def demo_keys(self) -> tuple[str, ...]:
        """Source episode keys ordered exactly like :attr:`datagen_infos`."""

        return tuple(self._demo_keys)

    def get_object_event_index(self, demo_index: int, annotation_key: str) -> int:
        """Return the annotated source sample for an object event.

        Args:
            demo_index: Index of the selected source demonstration in this pool.
            annotation_key: Event signal named in the task descriptor.

        Returns:
            First active source-state sample, with no subtask-end offset.

        Raises:
            AssertionError: The selected source has no mark with this name.
            ValueError: The signal is empty, nonbinary, or has no active sample.
        """
        signals = self._datagen_infos[demo_index].object_event_signals
        assert (
            signals is not None and annotation_key in signals
        ), f"source demo {self._demo_keys[demo_index]!r} is missing object event {annotation_key!r}"
        return self._find_first_rising_edge(signals[annotation_key], signal_name=annotation_key)

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

    def _add_episode(self, episode: EpisodeData, *, demo_key: str | None = None) -> None:
        """Parse source annotations and append an episode with its subtask boundaries.

        Args:
            episode: Recorded actions, poses and annotation signals.
            demo_key: Source episode name; generated from the pool index when omitted.
        """
        ep_grp = episode.data

        ep_datagen_info = DatagenInfo.from_episode(ep_grp)
        if ep_datagen_info.subtask_term_signals is None and all(
            len(self.task_descriptor.get_subtasks(eef_name)) == 1 for eef_name in self.task_descriptor.get_eef_names()
        ):
            ep_datagen_info.subtask_term_signals = {}
        missing = [
            name
            for name in ("eef_pose", "target_eef_pose", "subtask_term_signals")
            if getattr(ep_datagen_info, name) is None
        ]
        assert not missing, f"Episode's 'obs/datagen_info' is missing required field(s): {missing}"

        # Legacy single-arm datasets use names such as "robot0". With exactly one
        # recorded and one configured EEF, the correspondence is unambiguous.
        # Normalize both pose mappings in memory without modifying the source episode.
        eef_pose = ep_datagen_info.eef_pose
        target_eef_pose = ep_datagen_info.target_eef_pose
        expected_eefs = set(self.task_descriptor.get_eef_names())
        recorded_eefs = set(eef_pose)
        assert recorded_eefs == set(target_eef_pose), (
            f"Source eef_pose names {sorted(recorded_eefs)} do not match "
            f"target_eef_pose names {sorted(target_eef_pose)}."
        )
        if recorded_eefs != expected_eefs:
            assert len(recorded_eefs) == len(expected_eefs) == 1, (
                f"Source EEF names {sorted(recorded_eefs)} do not match configured names "
                f"{sorted(expected_eefs)}. Automatic name matching requires exactly one "
                "source and one configured EEF; multi-arm datasets require matching names."
            )
            source_eef = next(iter(recorded_eefs))
            configured_eef = next(iter(expected_eefs))
            ep_datagen_info.eef_pose = {configured_eef: eef_pose[source_eef]}
            ep_datagen_info.target_eef_pose = {configured_eef: target_eef_pose[source_eef]}

        ep_datagen_info.passthrough_action = self.embodiment_adapter.actions_to_passthrough_actions(ep_grp["actions"])

        object_event_signals = ep_grp["obs"]["datagen_info"].get("object_event_signals", {})
        required_events = {
            event.annotation_key
            for eef_name in self.task_descriptor.get_eef_names()
            for subtask in self.task_descriptor.get_subtasks(eef_name)
            for segment in subtask.segments
            for event in segment.events
            if event.annotation_key
        }
        assert required_events <= object_event_signals.keys(), (
            f"source demo {demo_key!r} is missing object_event_signals: "
            f"{sorted(required_events - object_event_signals.keys())}"
        )
        for name, signal in object_event_signals.items():
            assert signal.ndim in (1, 2) and signal.shape[0] == signal.numel() == len(
                ep_grp["actions"]
            ), f"object event {name!r} must contain one scalar per source action"
            self._find_first_rising_edge(signal, signal_name=name)
        if demo_key is None:
            index = len(self._datagen_infos)
            while (demo_key := f"demo_{index}") in self._demo_keys:
                index += 1
        assert demo_key not in self._demo_keys, f"duplicate source demo key {demo_key!r}"
        self._datagen_infos.append(ep_datagen_info)
        self._demo_keys.append(demo_key)

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
        term_signal_names = self.subtask_term_signal_names[eef_name]
        start_signal_names = self.subtask_start_signal_names[eef_name]
        boundaries: list[tuple[int, int]] = []
        prev_end = 0

        for subtask_index, term_signal_name in enumerate(term_signal_names):
            if use_skillgen:
                if ep.subtask_start_signals is None:
                    raise ValueError(
                        f"subtask_start_signals missing for subtask {term_signal_name!r} but use_skillgen is enabled"
                    )
                start_signal_name = start_signal_names[subtask_index]
                if not start_signal_name:
                    raise ValueError(f"subtask {eef_name}[{subtask_index}] has no start signal")
                if start_signal_name not in ep.subtask_start_signals:
                    raise ValueError(f"subtask start signal {start_signal_name!r} is missing from the episode")
                start_index = self._find_first_rising_edge(
                    ep.subtask_start_signals[start_signal_name],
                    signal_name=start_signal_name,
                )
            else:
                start_index = prev_end

            if subtask_index == len(term_signal_names) - 1:
                end_index = ep_grp["actions"].shape[0]
            else:
                if not term_signal_name:
                    raise ValueError(f"non-final subtask {eef_name}[{subtask_index}] has no termination signal")
                if ep.subtask_term_signals is None or term_signal_name not in ep.subtask_term_signals:
                    raise ValueError(f"subtask termination signal {term_signal_name!r} is missing from the episode")
                end_index = (
                    self._find_first_rising_edge(
                        ep.subtask_term_signals[term_signal_name],
                        signal_name=term_signal_name,
                    )
                    + 1
                )

            if end_index <= start_index:
                raise ValueError(
                    f"subtask {eef_name}[{subtask_index}] has non-increasing boundary: {start_index} -> {end_index}"
                )
            boundaries.append((start_index, end_index))
            prev_end = end_index

        self._validate_subtask_offsets(eef_name, boundaries)
        return boundaries

    @staticmethod
    def _find_first_rising_edge(signal, *, signal_name: str) -> int:
        """Return the first ``0 -> 1`` index, or zero for an initially true signal.

        Later signal changes do not move a boundary that has already been reached.
        """

        values = torch.as_tensor(signal).flatten()
        if values.numel() == 0:
            raise ValueError(f"annotation signal {signal_name!r} is empty")
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError(f"annotation signal {signal_name!r} must be binary")
        values = values.to(dtype=torch.int8)
        differences = values[1:] - values[:-1]
        rising_edges = (differences == 1).nonzero().flatten() + 1
        if values[0] == 1:
            rising_edges = torch.cat(
                [torch.zeros(1, dtype=rising_edges.dtype, device=rising_edges.device), rising_edges]
            )
        if rising_edges.numel() == 0:
            raise ValueError(f"annotation signal {signal_name!r} has no rising edge")
        return int(rising_edges[0].item())

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
