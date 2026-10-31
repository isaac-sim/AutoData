# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validity metrics: whether the recorded data is well formed and the demonstrated task holds."""

from __future__ import annotations

import h5py
import math
import numpy as np
import re

from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import NotApplicable, Value, check, metric

ROTATION_TOLERANCE = 1e-3
"""Largest deviation from a proper rotation, or from a unit quaternion, a recorded pose may show."""
BLACK_FRAME_MEAN = 2.0
"""Mean pixel value, out of 255, below which a camera frame counts as black."""
FROZEN_SECONDS = 1.0
"""Recorded time over which a camera that repeats one frame counts as frozen."""
FRAME_CHUNK = 64
"""Steps read at a time, so checks of camera frames and other large arrays stream instead of loading whole episodes."""
PER_EPISODE_PATHS = ("initial_state",)
"""Paths whose arrays are not one entry per step."""
SUBTASK_SIGNAL_PATHS = ("obs/datagen_info/subtask_term_signals/", "obs/datagen_info/subtask_start_signals/")
"""Annotation's per-step subtask signals, stored as a ``(steps,)`` column or a ``(1, steps)`` row."""
OBSERVED_POSE_PATHS = ("obs/datagen_info/eef_pose/", "obs/datagen_info/object_pose/")
"""Recorded poses that are observations; commanded targets are measured by ``action_rotation_error`` instead."""
NEAR_LIMIT = 0.95
"""Share of a datasheet limit from which an episode counts as running at the limit, which a simulator that clamps
there never exceeds."""


def _image_streams(episode: Episode, ev: Evidence) -> dict[str, h5py.Dataset]:
    """The episode's camera streams; not applicable to a dataset that records none."""
    if not ev.camera_paths:
        raise NotApplicable("no camera streams")
    return {path: dataset for path, dataset in episode.datasets() if path in ev.camera_paths}


def _all_finite(dataset: h5py.Dataset) -> bool:
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]))
    return all(np.isfinite(dataset[start : start + FRAME_CHUNK]).all() for start in range(0, len(dataset), FRAME_CHUNK))


@check("finite")
def finite(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with a NaN or infinite value in any floating-point array."""
    bad = [path for path, dataset in episode.datasets() if dataset.dtype.kind == "f" and not _all_finite(dataset)]
    return f"non-finite values in {', '.join(bad)}" if bad else None


@check("aligned_lengths")
def aligned_lengths(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose per-step arrays do not all hold one entry per action; camera streams are checked apart."""
    num_steps = len(episode["actions"]) if "actions" in episode else 0
    if num_steps == 0:
        return "no actions"
    bad = [
        path
        for path, dataset in episode.datasets()
        if not path.startswith(PER_EPISODE_PATHS)
        and path not in ev.camera_paths
        and _steps_held(path, dataset) not in (None, num_steps)
    ]
    return f"{', '.join(bad)} do not hold {num_steps} steps" if bad else None


def _steps_held(path: str, dataset: h5py.Dataset) -> int | None:
    """Steps ``dataset`` holds: a subtask signal counts its entries in either layout; a scalar holds none."""
    if path.startswith(SUBTASK_SIGNAL_PATHS):
        return dataset.size
    return len(dataset) if dataset.ndim else None


def _rotation_errors(poses: np.ndarray) -> np.ndarray:
    """Per 4x4 pose, how far its rotation is from a proper one: the larger of the largest entry of ``|R^T R - I|``
    and ``|det R - 1|``, which also catches reflections."""
    rotations = poses[..., :3, :3]
    orthonormality = np.abs(np.einsum("...ji,...jk->...ik", rotations, rotations) - np.eye(3)).max(axis=(-2, -1))
    return np.maximum(orthonormality, np.abs(np.linalg.det(rotations) - 1.0))


@check("valid_rotations")
def valid_rotations(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with an observed pose rotation that is not a proper rotation, a root quaternion that is not
    unit, or a pose array not shaped ``(steps, 4, 4)`` or ``(steps, 7)``.

    Both tests hold for either quaternion order, so the check needs no convention.
    """
    bad, checked = [], False
    for path, dataset in episode.datasets():
        if path.startswith(OBSERVED_POSE_PATHS):
            shape = (4, 4)
        elif path.endswith("/root_pose"):
            shape = (7,)
        else:
            continue
        checked = True
        if dataset.shape[1:] != shape:
            bad.append(f"{path} of shape {dataset.shape}")
            continue
        poses = dataset[()]
        error = _rotation_errors(poses) if shape == (4, 4) else np.abs(np.linalg.norm(poses[:, 3:7], axis=-1) - 1.0)
        if error.max(initial=0.0) > ROTATION_TOLERANCE:
            bad.append(path)
    if not checked:
        raise NotApplicable("no recorded rotations")
    return f"invalid rotations in {', '.join(bad)}" if bad else None


@check("pose_action_bounds", requires=["env_args.autodata.pose_action"])
def pose_action_bounds(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose pose actions exceed the bound the embodiment clips them to, or lie beyond its actions."""
    actions = episode["actions"]
    outside = [
        f"{eef} {layout['action_slice']}"
        for eef, layout in ev.entry["pose_action"].items()
        if layout["action_slice"][1] > actions.shape[-1]
    ]
    if outside:
        return f"pose action slices beyond the {actions.shape[-1]} action columns: {', '.join(outside)}"
    bad = []
    for eef, layout in ev.entry["pose_action"].items():
        if layout.get("bound") is None:
            continue
        start, end = layout["action_slice"]
        peak = float(np.abs(actions[:, start:end]).max(initial=0.0))
        if peak > layout["bound"] + 1e-6:
            bad.append(f"{eef} reaches {peak:.3g}")
    return f"pose actions beyond their bound: {', '.join(bad)}" if bad else None


@check("one_frame_per_step", group="image")
def one_frame_per_step(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode that lacks a camera stream the dataset records, or whose streams do not hold one frame per action."""
    streams = _image_streams(episode, ev)
    missing = sorted(ev.camera_paths - streams.keys())
    if missing:
        return f"{', '.join(missing)} not recorded"
    num_steps = len(episode["actions"])
    bad = [path for path, dataset in streams.items() if len(dataset) != num_steps]
    return f"{', '.join(bad)} do not hold {num_steps} frames" if bad else None


@check("no_black_frames", group="image")
def no_black_frames(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with a camera frame whose mean pixel value is below :data:`BLACK_FRAME_MEAN`."""
    for path, dataset in _image_streams(episode, ev).items():
        for start in range(0, len(dataset), FRAME_CHUNK):
            means = dataset[start : start + FRAME_CHUNK].mean(axis=(1, 2, 3))
            if (means < BLACK_FRAME_MEAN).any():
                return f"{path} is black at step {start + int(np.argmax(means < BLACK_FRAME_MEAN))}"
    return None


@check("no_frozen_frames", group="image", requires=["env_args.sim_args.dt", "env_args.sim_args.decimation"])
def no_frozen_frames(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose camera repeats one frame for :data:`FROZEN_SECONDS`; needs the recorded control step."""
    streams = _image_streams(episode, ev)
    step = ev.env_args["sim_args"]["dt"] * ev.env_args["sim_args"]["decimation"]
    frozen_frames = max(2, math.ceil(FROZEN_SECONDS / step))
    for path, dataset in streams.items():
        previous, repeats = None, 1
        for start in range(0, len(dataset), FRAME_CHUNK):
            for offset, frame in enumerate(dataset[start : start + FRAME_CHUNK]):
                repeats = repeats + 1 if previous is not None and np.array_equal(frame, previous) else 1
                if repeats >= frozen_frames:
                    return f"{path} repeats one frame for {FROZEN_SECONDS:g} s up to step {start + offset}"
                previous = frame
    return None


def _pass_rate(ev: Evidence, group: str, none_apply: str) -> Value:
    if not ev.all_episodes:
        raise NotApplicable("the dataset has no episodes")
    considered = [name for name, check_group in ev.check_groups.items() if check_group == group]
    names = [name for name in considered if name in ev.failures]
    if not names:
        raise NotApplicable(none_apply)
    failing = {episode for name in names for episode in ev.failures[name]}
    details = {"checks_run": names, "checks_skipped": [name for name in considered if name not in ev.failures]}
    return Value(1 - len(failing) / len(ev.all_episodes), n=len(ev.all_episodes), details=details)


@metric(
    "correctness_pass_rate",
    area="validity",
    signal="data correctness",
    scope="episode",
    unit="fraction",
    sample="episodes",
    better="higher",
)
def correctness_pass_rate(ev: Evidence) -> Value:
    """Share of episodes passing every correctness check: finite, aligned, valid rotations, actions in bounds."""
    return _pass_rate(ev, "correctness", "no correctness check applies to this dataset")


@metric(
    "image_stream_pass_rate",
    area="validity",
    signal="data correctness",
    scope="episode",
    unit="fraction",
    sample="episodes",
    better="higher",
)
def image_stream_pass_rate(ev: Evidence) -> Value:
    """Share of episodes that pass every image check that applies: one frame per step, no black or frozen frames."""
    return _pass_rate(ev, "image", "no camera streams")


@metric(
    "final_success_rate",
    area="validity",
    signal="task correctness",
    scope="dataset",
    unit="fraction",
    sample="episodes",
    better="higher",
    requires=["step_success"],
    baseline=True,
)
def final_success_rate(ev: Evidence) -> Value:
    """Share of valid episodes whose task success holds at their last step."""
    held = [bool(episode["step_success"][-1]) for episode in ev.episodes]
    return Value(sum(held) / len(held), n=len(held))


@metric(
    "action_rotation_error",
    area="validity",
    signal="data correctness",
    scope="dataset",
    unit="max(|R^T R - I|, |det R - 1|)",
    sample="poses",
    better="lower",
    primary="p95",
    requires=["obs/datagen_info/target_eef_pose"],
    baseline=True,
)
def action_rotation_error(ev: Evidence) -> Value:
    """p50 and p95 deviation of the commanded target rotations from a valid rotation, over every step and end effector."""
    errors = []
    for episode in ev.episodes:
        for eef in _pose_channels(ev, episode, "obs/datagen_info/target_eef_pose"):
            targets = episode[f"obs/datagen_info/target_eef_pose/{eef}"]
            errors.append(_rotation_errors(targets[_commanded(targets)]))
    return _percentiles(errors, "no end effector commands a pose")


@metric(
    "tracking_error",
    area="validity",
    signal="motion feasibility",
    scope="dataset",
    unit="mm",
    sample="poses",
    better="lower",
    primary="p95",
    requires=["env_args.autodata.pose_action", "obs/datagen_info/eef_pose", "obs/datagen_info/target_eef_pose"],
    baseline=True,
)
def tracking_error(ev: Evidence) -> Value:
    """p50 and p95 distance from each absolute end-effector target to the end effector one step later."""
    absolute = [eef for eef, layout in ev.entry["pose_action"].items() if layout.get("absolute")]
    distances = []
    for episode in ev.episodes:
        for eef in absolute:
            target, observed = f"obs/datagen_info/target_eef_pose/{eef}", f"obs/datagen_info/eef_pose/{eef}"
            if target not in episode or observed not in episode:
                continue
            targets, reached = episode[target][:-1], episode[observed][1:]
            commanded = _commanded(targets)
            distances.append(np.linalg.norm(targets[commanded, :3, 3] - reached[commanded, :3, 3], axis=1) * 1000)
    return _percentiles(distances, "no pose action is recorded as absolute")


@metric(
    "action_saturation",
    area="validity",
    signal="motion feasibility",
    scope="dataset",
    unit="fraction",
    sample="steps",
    better="lower",
    requires=["env_args.autodata.pose_action"],
    baseline=True,
)
def action_saturation(ev: Evidence) -> Value:
    """Share of steps at which a pose action sits at the bound the embodiment clips it to."""
    layouts = [layout for layout in ev.entry["pose_action"].values() if "bound" in layout]
    if not layouts:
        raise NotApplicable("no pose action is clipped")
    saturated = []
    for episode in ev.episodes:
        actions = np.abs(episode["actions"])
        at_bound = [
            actions[:, slice(*layout["action_slice"])].max(axis=1, initial=0.0) >= layout["bound"] * (1 - 1e-6)
            for layout in layouts
        ]
        saturated.append(np.any(at_bound, axis=0))
    saturated = np.concatenate(saturated)
    return Value(float(saturated.mean()), n=len(saturated))


@metric(
    "motion_smoothness",
    area="validity",
    signal="motion quality",
    scope="dataset",
    unit="m/s³",
    sample="poses",
    better="lower",
    primary="p95",
    requires=["obs/datagen_info/eef_pose", "env_args.sim_args.dt", "env_args.sim_args.decimation"],
    baseline=True,
)
def motion_smoothness(ev: Evidence) -> Value:
    """p50 and p95 end-effector jerk, labelled with the run's largest action-noise scale."""
    step = ev.env_args["sim_args"]["dt"] * ev.env_args["sim_args"]["decimation"]
    jerks = [
        np.linalg.norm(np.diff(episode[f"obs/datagen_info/eef_pose/{eef}"][:, :3, 3], n=3, axis=0), axis=1) / step**3
        for episode in ev.episodes
        for eef in _pose_channels(ev, episode, "obs/datagen_info/eef_pose")
    ]
    noise = {"action_noise": ev.entry.get("action_noise")}
    return _percentiles(jerks, "no end effector that commands a pose records four steps", noise)


@metric(
    "joint_limit_ratio",
    area="validity",
    signal="motion feasibility",
    scope="dataset",
    unit="fraction of episodes",
    sample="episodes",
    better="lower",
    primary="velocity",
    requires=["env_args.autodata.joints.asset", "env_args.sim_args.dt", "env_args.sim_args.decimation"],
    baseline=True,
)
def joint_limit_ratio(ev: Evidence) -> Value:
    """Share of episodes that reach ``NEAR_LIMIT`` of a datasheet velocity or acceleration limit."""
    joints, datasheet = ev.entry["joints"], ev.entry.get("hardware_limits", {})
    limits = {kind: _datasheet(joints["names"], datasheet.get(kind, {})) for kind in ("velocity", "acceleration")}
    limits = {kind: limit for kind, limit in limits.items() if np.isfinite(limit).any()}
    if not limits:
        raise NotApplicable("the embodiment config gives no hardware_limits for these joints")
    path = f"states/articulation/{joints['asset']}/joint_velocity"
    if not all(path in episode for episode in ev.episodes):
        raise NotApplicable(f"not every episode records {path}")
    step = ev.env_args["sim_args"]["dt"] * ev.env_args["sim_args"]["decimation"]
    peaks = {kind: [] for kind in limits}
    for episode in ev.episodes:
        velocity = episode[path]
        if velocity.shape[1:] != (len(joints["names"]),):
            raise ValueError(
                f"{episode.name} records joint velocities of shape {velocity.shape} for {len(joints['names'])} joints"
            )
        motion = {"velocity": velocity, "acceleration": np.diff(velocity, axis=0) / step}
        for kind, limit in limits.items():
            peaks[kind].append((np.abs(motion[kind]) / limit).max(axis=0, initial=0.0))
    value, details = {}, {}
    for kind, per_episode in peaks.items():
        per_episode = np.array(per_episode)
        value[kind] = float((per_episode.max(axis=1) >= NEAR_LIMIT).mean())
        details[f"{kind}_peak_ratio"] = float(per_episode.max())
        per_joint = np.where(np.isfinite(limits[kind]), per_episode.max(axis=0), -1.0)
        details[f"{kind}_worst_joint"] = joints["names"][int(per_joint.argmax())]
    return Value(value, n=len(ev.episodes), details=details)


def _pose_channels(ev: Evidence, episode: Episode, group: str) -> list[str]:
    """The end effectors under ``group`` that command a pose action, or every one when the entry does not say."""
    commanding = ev.entry.get("pose_action")
    return [eef for eef in episode.children(group) if commanding is None or eef in commanding]


def _commanded(targets: np.ndarray) -> np.ndarray:
    """Which steps of ``targets`` hold a pose: all-zero targets mark steps or channels that command none."""
    return np.abs(targets).max(axis=tuple(range(1, targets.ndim))) > 0


def _percentiles(samples: list[np.ndarray], empty: str, details: dict | None = None) -> Value:
    values = np.concatenate(samples) if samples else np.empty(0)
    if not len(values):
        raise NotApplicable(empty)
    p50, p95 = np.percentile(values, [50, 95])
    return Value({"p50": float(p50), "p95": float(p95)}, n=len(values), details=details or {})


def _datasheet(names: list[str], limits: dict[str, float]) -> np.ndarray:
    """Each joint's limit from the first pattern its name fully matches, infinite where none does."""
    return np.array(
        [next((limit for pattern, limit in limits.items() if re.fullmatch(pattern, name)), np.inf) for name in names],
        dtype=float,
    )
