# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diversity metrics: how differently the demonstrations use the workspace, approach what they lift, and start.

The four spread metrics each average, over pairs of episodes or lifts, how much the two differ; the two lift metrics
then average over the lifted objects. That estimates how different two demonstrations drawn from the same generator
are, so it does not grow with their number and is defined from two episodes on; fewer episodes only make it noisier.
The source column holds the same average over the seeds.

Hand and object poses come from ``obs/datagen_info``, which records both in one frame before each step. An object is
lifted when ``states/``, recorded after each step in the world frame, first shows it :data:`LIFT_M` above its start;
lift-off is the last observation before that. The object is carried by the hand in whose frame it moves least while
it rises. The hand's grasp point is where the objects it carries sit in its frame at lift-off, the median over the
dataset, because recorded hand frames differ (a fingertip point, a flange, a wrist); a hand that carries nothing keeps
its frame. An object's approach is the grasp point's path in the object's frame from the last time it was
:data:`APPROACH_M` from the object until lift-off.
"""

from __future__ import annotations

import hashlib
import numpy as np
import weakref
from dataclasses import dataclass

from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import NotApplicable, Value, metric

WORKSPACE_VOXEL_M = 0.05
"""Side of the cubic voxels in which workspace coverage compares the space two episodes' grasp points visit [m]."""
WORKSPACE_POINTS = 64
"""Points each hand's grasp-point path is resampled to, evenly along its length, for path distances."""
MIN_START_EXTENT_M = 0.01
"""Spread along x or y below which an object's start position counts as fixed [m]."""
LIFT_M = 0.01
"""Rise above its start height at which an object counts as lifted; lift-off is the last observation before [m]."""
CARRY_STEPS = 5
"""Steps after lift-off over which the carrying hand is the one in whose frame the object moves least."""
APPROACH_M = 0.15
"""Distance from the object at which an approach starts: the grasp point's path from the last time it was this far
from the object until lift-off [m]."""
APPROACH_POINTS = 16
"""Points an approach path is resampled to, evenly along its length."""
MAX_PAIRS = 2000
"""Pairs the averages draw, with a fixed seed, when there are more."""


@dataclass(frozen=True)
class _Demos:
    voxels: list[set[tuple[int, int, int]]]
    """Per episode, the :data:`WORKSPACE_VOXEL_M` voxels its grasp points visit."""
    paths: np.ndarray | None
    """Per episode, its grasp-point paths resampled and scaled so row distances are root-mean-square per point [m];
    None when episodes record different hands."""
    lifts: dict[str, tuple[np.ndarray, np.ndarray]]
    """Per lifted object, each lift's unit approach direction and its approach path, scaled like ``paths``."""


_DEMOS: weakref.WeakKeyDictionary[Evidence, _Demos] = weakref.WeakKeyDictionary()


@metric(
    "workspace_coverage",
    area="diversity",
    signal="workspace",
    scope="dataset",
    unit="fraction not shared",
    sample="episodes",
    better="higher",
    requires=["obs/datagen_info/eef_pose"],
    baseline=True,
)
def workspace_coverage(ev: Evidence) -> Value:
    """Average share of the space two episodes' grasp points visit that only one of them visits, in 5 cm voxels.

    0 when every episode covers the same space, 1 when no two overlap. The details add the average distance between
    two episodes' grasp-point paths.
    """
    demos = _demos(ev)
    if len(demos.voxels) < 2:
        raise NotApplicable("needs two episodes with recorded hand poses")
    first, second = _pairs(len(demos.voxels))
    not_shared = [
        1 - len(demos.voxels[i] & demos.voxels[j]) / len(demos.voxels[i] | demos.voxels[j])
        for i, j in zip(first, second)
    ]
    details = {"voxel_m": WORKSPACE_VOXEL_M, "pairs": len(first)}
    if demos.paths is not None:
        details["path_distance_mm"] = 1000 * float(
            np.linalg.norm(demos.paths[first] - demos.paths[second], axis=1).mean()
        )
    return Value(float(np.mean(not_shared)), n=len(demos.voxels), details=details)


@metric(
    "approach_spread",
    area="diversity",
    signal="grasp behavior",
    scope="dataset",
    unit="degrees",
    sample="lifts",
    better="higher",
    requires=["obs/datagen_info/eef_pose", "obs/datagen_info/object_pose", "states/rigid_object"],
    baseline=True,
)
def approach_spread(ev: Evidence) -> Value:
    """Average angle between the directions two lifts of an object come from, in its frame, over the lifted objects."""
    objects = {}
    for name, (directions, _) in _lifted(ev).items():
        first, second = _pairs(len(directions))
        angles = np.degrees(np.arccos(np.clip((directions[first] * directions[second]).sum(axis=1), -1.0, 1.0)))
        objects[name] = {"lifts": len(directions), "angle_deg": float(angles.mean())}
    return _over_objects(objects, "angle_deg")


@metric(
    "behavior_diversity",
    area="diversity",
    signal="grasp behavior",
    scope="dataset",
    unit="mm",
    sample="lifts",
    better="higher",
    requires=["obs/datagen_info/eef_pose", "obs/datagen_info/object_pose", "states/rigid_object"],
    baseline=True,
)
def behavior_diversity(ev: Evidence) -> Value:
    """Average distance between two lifts' approach paths to an object, in its frame, over the lifted objects.

    Novelty, in each object's details, is the median over its lifts of the distance from each approach path to the
    nearest seed path.
    """
    seeds = _demos(ev.baseline).lifts if ev.baseline is not None else {}
    objects = {}
    for name, (_, paths) in _lifted(ev).items():
        first, second = _pairs(len(paths))
        distance = 1000 * float(np.linalg.norm(paths[first] - paths[second], axis=1).mean())
        objects[name] = {"lifts": len(paths), "path_distance_mm": distance}
        if name in seeds:
            nearest = _squared_distances(paths, seeds[name][1]).min(axis=1)
            objects[name]["novelty_mm"] = 1000 * float(np.median(np.sqrt(nearest)))
    return _over_objects(objects, "path_distance_mm")


@metric(
    "start_spread",
    area="diversity",
    signal="start configurations",
    scope="dataset",
    unit="mm",
    sample="episodes",
    better="higher",
    requires=["initial_state/rigid_object"],
    baseline=True,
)
def start_spread(ev: Evidence) -> Value:
    """Average distance between two episodes' object layouts in the horizontal plane.

    A layout distance is the root-mean-square, over the objects that start in more than one place, of their xy start
    distances. The details give each object's average distance, with a source the seeds' too, for the objects that
    move here or in the seeds, so an object the seeds move but this dataset pins shows.
    """
    if len(ev.episodes) < 2:
        raise NotApplicable("needs two episodes")
    starts = _start_positions(ev)
    moving = [name for name, points in starts.items() if _moves(points)]
    if not moving:
        raise NotApplicable("no object starts in more than one place")
    first, second = _pairs(len(ev.episodes))
    distances = {name: np.linalg.norm(points[first] - points[second], axis=1) for name, points in starts.items()}
    value = 1000 * float(np.sqrt(np.mean([distances[name] ** 2 for name in moving], axis=0)).mean())
    seeds = _start_positions(ev.source) if ev.source is not None and len(ev.source.episodes) >= 2 else {}
    seed_first, seed_second = _pairs(len(ev.source.episodes)) if seeds else (None, None)
    objects = {}
    for name in starts:
        if name in moving or (name in seeds and _moves(seeds[name])):
            objects[name] = {"spread_mm": 1000 * float(distances[name].mean())}
            if name in seeds:
                seed_distance = np.linalg.norm(seeds[name][seed_first] - seeds[name][seed_second], axis=1)
                objects[name]["seed_spread_mm"] = 1000 * float(seed_distance.mean())
    return Value(value, n=len(ev.episodes), details={"objects": objects})


@metric(
    "exact_duplicates",
    area="diversity",
    signal="repeated episodes",
    scope="dataset",
    unit="fraction",
    sample="episodes",
    better="lower",
    requires=["actions"],
)
def exact_duplicates(ev: Evidence) -> Value:
    """Share of episodes whose actions equal an earlier episode's, bit for bit."""
    first, duplicates = {}, {}
    for episode in ev.episodes:
        actions = np.ascontiguousarray(episode["actions"])
        digest = hashlib.sha256(f"{actions.dtype.str}{actions.shape}".encode() + actions.tobytes()).digest()
        if digest in first:
            duplicates[episode.name] = first[digest]
        else:
            first[digest] = episode.name
    details = {"duplicate_of": dict(list(duplicates.items())[:3])}
    return Value(len(duplicates) / len(ev.episodes), n=len(ev.episodes), details=details)


def _pairs(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of distinct items among ``count``: all of them, or :data:`MAX_PAIRS` drawn with a fixed seed."""
    if count * (count - 1) // 2 <= MAX_PAIRS:
        return np.triu_indices(count, 1)
    rng = np.random.default_rng(0)
    first, second = rng.integers(0, count, MAX_PAIRS), rng.integers(0, count - 1, MAX_PAIRS)
    return first, second + (second >= first)


def _over_objects(objects: dict[str, dict], field: str) -> Value:
    """The average of ``field`` over the objects, each counted once, with every object in the details."""
    value = float(np.mean([entry[field] for entry in objects.values()]))
    return Value(value, n=sum(entry["lifts"] for entry in objects.values()), details={"objects": objects})


def _moves(points: np.ndarray) -> bool:
    return np.ptp(points, axis=0).max() >= MIN_START_EXTENT_M


def _lifted(ev: Evidence) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The objects lifted at least twice from :data:`APPROACH_M` away."""
    lifts = {name: rows for name, rows in _demos(ev).lifts.items() if len(rows[0]) >= 2}
    if not lifts:
        raise NotApplicable(f"no object is lifted twice from {APPROACH_M * 100:g} cm away")
    return lifts


def _start_positions(ev: Evidence) -> dict[str, np.ndarray]:
    """Per rigid object with an initial root pose in every episode, in name order, its xy in each episode [m]."""
    group, starts = "initial_state/rigid_object", {}
    for episode in ev.episodes:
        for name in episode.children(group) if group in episode else ():
            path = f"{group}/{name}/root_pose"
            if path in episode and len(pose := episode[path]):
                starts.setdefault(name, []).append(pose[0, :2])
    return {name: np.array(points) for name, points in sorted(starts.items()) if len(points) == len(ev.episodes)}


def _demos(ev: Evidence) -> _Demos:
    """The grasp-point voxels, paths and lifts of the episodes, computed once per evidence."""
    if ev in _DEMOS:
        return _DEMOS[ev]
    episodes, raw = [], []
    for episode in ev.episodes:
        hands, objects = (_named(episode, f"obs/datagen_info/{group}") for group in ("eef_pose", "object_pose"))
        if not hands:
            continue
        episodes.append(hands)
        for name, poses in objects.items():
            state = f"states/rigid_object/{name}/root_pose"
            if state not in episode:
                continue
            height = episode[state][:, 2]
            raised = np.flatnonzero(height - height[0] > LIFT_M)
            if not len(raised):
                continue
            # States record each step's outcome, so observation raised[0] is the last before the rise.
            step = raised[0]
            later = min(step + CARRY_STEPS, len(poses) - 1)
            shifts = {
                hand: np.linalg.norm(_in_hand(frames, poses, later) - _in_hand(frames, poses, step))
                for hand, frames in hands.items()
            }
            hand = min(shifts, key=shifts.get)
            raw.append((name, hand, hands[hand][: step + 1], poses[: step + 1]))
    offsets = {
        hand: np.median([_in_hand(frames, poses, -1) for _, other, frames, poses in raw if other == hand], axis=0)
        for hand in {hand for _, hand, _, _ in raw}
    }
    voxels, paths = [], []
    for hands in episodes:
        points = {
            hand: poses[:, :3, 3] + poses[:, :3, :3] @ offsets.get(hand, np.zeros(3)) for hand, poses in hands.items()
        }
        cells = np.floor(np.concatenate(list(points.values())) / WORKSPACE_VOXEL_M).astype(np.int64)
        voxels.append(set(map(tuple, cells.tolist())))
        paths.append(np.concatenate([_resample(points[hand], WORKSPACE_POINTS) for hand in sorted(points)]).ravel())
    same_hands = len({tuple(sorted(hands)) for hands in episodes}) == 1
    scale = np.sqrt(WORKSPACE_POINTS * len(episodes[0])) if episodes else 1.0
    lifts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, hand, frames, poses in raw:
        grasp = frames[:, :3, 3] + frames[:, :3, :3] @ offsets[hand]
        approach = np.einsum("sji,sj->si", poses[:, :3, :3], grasp - poses[:, :3, 3])
        outside = np.flatnonzero(np.linalg.norm(approach, axis=1) >= APPROACH_M)
        if len(outside):
            approach = approach[outside[-1] :]
            path = _resample(approach, APPROACH_POINTS).ravel() / np.sqrt(APPROACH_POINTS)
            lifts.setdefault(name, []).append((approach[0] / np.linalg.norm(approach[0]), path))
    demos = _Demos(
        voxels=voxels,
        paths=np.array(paths) / scale if paths and same_hands else None,
        lifts={name: (np.array([d for d, _ in rows]), np.array([p for _, p in rows])) for name, rows in lifts.items()},
    )
    _DEMOS[ev] = demos
    return demos


def _named(episode: Episode, group: str) -> dict[str, np.ndarray]:
    """The arrays under ``group`` by name, such as each hand's poses; empty when the episode lacks the group."""
    return {name: episode[f"{group}/{name}"] for name in episode.children(group)} if group in episode else {}


def _in_hand(frames: np.ndarray, poses: np.ndarray, step: int) -> np.ndarray:
    """Where the object of poses ``poses`` sits in the frame of hand poses ``frames`` at ``step``."""
    return frames[step, :3, :3].T @ (poses[step, :3, 3] - frames[step, :3, 3])


def _resample(path: np.ndarray, points: int) -> np.ndarray:
    """``points`` positions spaced evenly along ``path``, so its speed does not matter."""
    lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    samples = np.linspace(0.0, lengths[-1], points)
    return np.stack([np.interp(samples, lengths, path[:, axis]) for axis in range(3)], axis=1)


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``a`` and of ``b``."""
    return np.maximum((a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None] - 2 * a @ b.T, 0.0)
