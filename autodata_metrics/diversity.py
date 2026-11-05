# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diversity metrics: how different the demonstrations in a dataset are from one another.

Each spread averages a distance over pairs of episodes:

- ``start_layout_spread``: how far apart two episodes' starting scenes are [mm].
- ``workspace_spread``: how far one episode's hands pass from where the other's went [mm].

An average over pairs estimates how different two demonstrations from the same producer are, so it does not grow with
the number of episodes. The metrics reference page defines each metric and when two values compare.
"""

from __future__ import annotations

import hashlib
import numpy as np

from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import NotApplicable, Value, metric

MAX_EPISODES = 1000
"""Episodes a spread reads at most."""
MAX_WORKSPACE_PAIRS = 2000
"""Episode pairs the workspace measures at most; the start layout uses every pair."""
WORKSPACE_GRID_M = 0.005
"""Grid in which a hand's path keeps one point per cell, so a path weighs the space it visits, not its length [m]."""
MAX_WORKSPACE_STEP_M = 0.08
"""A hand's move in one recorded step above which the workspace does not apply [m]."""
START_GROUPS = ("rigid_object", "articulation", "deformable_object")
"""``initial_state`` groups whose entities make up a start layout."""


@metric(
    "start_layout_spread",
    area="diversity",
    signal="start layouts",
    scope="dataset",
    unit="mm",
    sample="episodes",
    requires=["actions", "initial_state"],
    baseline=True,
)
def start_layout_spread(ev: Evidence) -> Value:
    """Average distance between two episodes' starting scenes, counting every object.

    For each pair of episodes, each object's horizontal start distance is combined with the others' as a
    root-sum-square: an object that never moves adds nothing, one that moves more adds more. Robots and deformable
    objects count by their root and their nodes' centroid. 0 means every object starts at the same xy in every
    episode.
    """
    episodes = _sample(ev)
    names, layouts = _start_layouts(episodes)
    entities = {name: _pair_mean(1000 * layouts[:, k]) for k, name in enumerate(names)}
    return Value(_pair_mean(1000 * layouts), n=len(episodes), details={"entities_mm": entities})


@metric(
    "workspace_spread",
    area="diversity",
    signal="hand workspace",
    scope="dataset",
    unit="mm",
    sample="episodes",
    requires=["actions", "obs/datagen_info/eef_pose"],
    baseline=True,
)
def workspace_spread(ev: Evidence) -> Value:
    """Average distance from where one episode's hands pass to where the other's went.

    A hand is each recorded end effector, followed by the origin of its frame as the embodiment records it, from the
    episode's first recorded step. Its path keeps one point per :data:`WORKSPACE_GRID_M` cell, so retracing a path adds
    nothing. Each kept point's distance to the nearest kept point of the other episode's path is averaged, both ways,
    and halved; hands are averaged. 0 means the hands pass through the same space.
    """
    episodes = _sample(ev)
    hands = sorted(set().union(*(episode.children("obs/datagen_info/eef_pose") for episode in episodes)))
    if not hands:
        raise NotApplicable("records no end-effector poses")
    hand_paths = [_hand_paths(episode, hands) for episode in episodes]
    for k, hand in enumerate(hands):
        move = max(np.linalg.norm(np.diff(paths[k], axis=0), axis=1).max(initial=0.0) for paths in hand_paths)
        if move > MAX_WORKSPACE_STEP_M:
            raise NotApplicable(
                f"{hand} moves {100 * move:g} cm in one recorded step, over the {100 * MAX_WORKSPACE_STEP_M:g} cm limit"
            )
    kept = [[_grid_points(path) for path in paths] for paths in hand_paths]
    first, second = _workspace_pairs(len(episodes))
    gaps = 1000 * np.array([[_mean_gap(a, b) for a, b in zip(kept[i], kept[j])] for i, j in zip(first, second)])
    hands_mm = {hand: float(gaps[:, k].mean()) for k, hand in enumerate(hands)}
    return Value(float(gaps.mean()), n=len(episodes), details={"pairs": len(first), "hands_mm": hands_mm})


def _digest(episode: Episode) -> bytes:
    actions = np.ascontiguousarray(episode["actions"])
    return hashlib.sha256(f"{actions.dtype.str}{actions.shape}".encode() + actions.tobytes()).digest()


def _sample(ev: Evidence) -> list[Episode]:
    """The episodes a spread reads, at least two: the first :data:`MAX_EPISODES` in a ranking by a digest of each
    episode's actions and how many episodes before it, by name, repeat them. Repeats spread through the ranking, and
    file order matters only between episodes whose actions repeat exactly."""
    copies: dict[bytes, int] = {}
    ranked = []
    for episode in sorted(ev.episodes, key=lambda episode: episode.name):
        digest = _digest(episode)
        copies[digest] = copies.get(digest, -1) + 1
        ranked.append((hashlib.sha256(digest + copies[digest].to_bytes(8, "little")).digest(), episode))
    episodes = [episode for _, episode in sorted(ranked, key=lambda item: item[0])[:MAX_EPISODES]]
    if len(episodes) < 2:
        raise NotApplicable("needs two episodes")
    return episodes


def _pair_mean(rows: np.ndarray) -> float:
    """The mean Euclidean distance between two rows, over every pair of rows."""
    rows = rows.reshape(len(rows), -1)
    return float(np.concatenate([np.linalg.norm(rows[k + 1 :] - rows[k], axis=1) for k in range(len(rows) - 1)]).mean())


def _starts(episode: Episode) -> dict[str, np.ndarray]:
    """Each scene entity's start xy in the episode: rigid objects and articulations by their root, deformable objects
    by their nodes' centroid [m]."""
    found = {}
    for group in START_GROUPS:
        prefix = f"initial_state/{group}"
        for name in episode.children(prefix) if prefix in episode else ():
            if group == "deformable_object":
                found[f"{group}/{name}"] = episode[f"{prefix}/{name}/nodal_position"][0].mean(axis=0)[:2]
            else:
                found[f"{group}/{name}"] = episode[f"{prefix}/{name}/root_pose"][0, :2]
    return found


def _start_layouts(episodes: list[Episode]) -> tuple[list[str], np.ndarray]:
    """The entities with a start pose, and each episode's start xy of each [m]."""
    starts = [_starts(episode) for episode in episodes]
    names = sorted(set().union(*starts))
    if not names:
        raise NotApplicable("no start poses recorded")
    for episode, found in zip(episodes, starts):
        if missing := sorted(set(names) - set(found)):
            raise NotApplicable(f"{missing[0]} has no start pose in episode {episode.name}")
    return names, np.array([[found[name] for name in names] for found in starts], dtype=float)


def _hand_paths(episode: Episode, hands: list[str]) -> list[np.ndarray]:
    """Each hand's position at each recorded step [m]."""
    paths = []
    for hand in hands:
        key = f"obs/datagen_info/eef_pose/{hand}"
        if key not in episode:
            raise NotApplicable(f"episode {episode.name} records no pose of {hand}")
        paths.append(np.asarray(episode[key], dtype=float)[:, :3, 3])
    return paths


def _grid_points(path: np.ndarray) -> np.ndarray:
    """The path sampled every half :data:`WORKSPACE_GRID_M` along its segments, keeping the first sample in each grid
    cell, so every kept point lies on the path."""
    steps = np.diff(path, axis=0)
    counts = np.maximum(1, np.ceil(np.linalg.norm(steps, axis=1) / (WORKSPACE_GRID_M / 2))).astype(int)
    segment = np.repeat(np.arange(len(steps)), counts)
    fraction = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / np.repeat(counts, counts)
    samples = np.concatenate([path[segment] + fraction[:, None] * steps[segment], path[-1:]])
    _, first = np.unique(np.floor(samples / WORKSPACE_GRID_M).astype(np.int64), axis=0, return_index=True)
    return samples[np.sort(first)]


def _mean_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Half the sum of each path's mean distance from its kept points to the other's nearest kept point [m]."""
    distances = np.linalg.norm(a[:, None] - b[None], axis=2)
    return 0.5 * float(distances.min(axis=1).mean() + distances.min(axis=0).mean())


def _workspace_pairs(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of distinct episodes: all, or :data:`MAX_WORKSPACE_PAIRS` drawn with a fixed seed."""
    first, second = np.triu_indices(count, 1)
    if len(first) <= MAX_WORKSPACE_PAIRS:
        return first, second
    chosen = np.sort(np.random.default_rng(0).choice(len(first), MAX_WORKSPACE_PAIRS, replace=False))
    return first[chosen], second[chosen]
