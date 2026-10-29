# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validity metrics: whether the recorded data is well formed and the demonstrated task holds."""

from __future__ import annotations

import h5py
import numpy as np

from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import NotApplicable, Value, check, metric

ROTATION_TOLERANCE = 1e-3
"""Largest deviation from a proper rotation, or from a unit quaternion, a recorded pose may show."""
BLACK_FRAME_MEAN = 2.0
"""Mean pixel value, out of 255, below which a camera frame counts as black."""
FROZEN_FRAMES = 10
"""Identical consecutive camera frames that count as a frozen stream."""
FRAME_CHUNK = 64
"""Steps read at a time, so checks of camera frames and other large arrays stream instead of loading whole episodes."""
PER_EPISODE_PATHS = ("initial_state", "obs/datagen_info/subtask_term_signals", "obs/datagen_info/subtask_start_signals")
"""Paths whose arrays are not one entry per step: the initial state, and annotation's subtask signals."""
OBSERVED_POSE_PATHS = ("obs/datagen_info/eef_pose/", "obs/datagen_info/object_pose/")
"""Recorded poses that are observations; commanded targets are measured by ``action_rotation_error`` instead."""


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
        if dataset.ndim > 0
        and not path.startswith(PER_EPISODE_PATHS)
        and path not in ev.camera_paths
        and len(dataset) != num_steps
    ]
    return f"{', '.join(bad)} do not hold {num_steps} steps" if bad else None


def _rotation_errors(poses: np.ndarray) -> np.ndarray:
    """Per 4x4 pose, how far its rotation is from a proper one: the larger of the largest entry of ``|R^T R - I|``
    and ``|det R - 1|``, which also catches reflections."""
    rotations = poses[..., :3, :3]
    orthonormality = np.abs(np.einsum("...ji,...jk->...ik", rotations, rotations) - np.eye(3)).max(axis=(-2, -1))
    return np.maximum(orthonormality, np.abs(np.linalg.det(rotations) - 1.0))


@check("valid_rotations")
def valid_rotations(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with an observed pose rotation that is not a proper rotation or a root quaternion that is not unit.

    Both tests hold for either quaternion order, so the check needs no convention.
    """
    bad, checked = [], False
    for path, dataset in episode.datasets():
        if path.startswith(OBSERVED_POSE_PATHS) and dataset.shape[-2:] == (4, 4):
            error = _rotation_errors(dataset[()]).max(initial=0.0)
        elif path.endswith("root_pose") and dataset.shape[-1:] == (7,):
            error = np.abs(np.linalg.norm(dataset[()][..., 3:7], axis=-1) - 1.0).max(initial=0.0)
        else:
            continue
        checked = True
        if error > ROTATION_TOLERANCE:
            bad.append(path)
    if not checked:
        raise NotApplicable("no recorded rotations")
    return f"invalid rotations in {', '.join(bad)}" if bad else None


@check("pose_action_bounds", requires=["env_args.autodata.pose_action"])
def pose_action_bounds(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose pose actions exceed the bound the embodiment clips them to."""
    actions = episode["actions"]
    bad = []
    for eef, layout in ev.env_args["autodata"]["pose_action"].items():
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


@check("no_frozen_frames", group="image")
def no_frozen_frames(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose camera repeats one frame :data:`FROZEN_FRAMES` times in a row."""
    for path, dataset in _image_streams(episode, ev).items():
        previous, repeats = None, 1
        for start in range(0, len(dataset), FRAME_CHUNK):
            for offset, frame in enumerate(dataset[start : start + FRAME_CHUNK]):
                repeats = repeats + 1 if previous is not None and np.array_equal(frame, previous) else 1
                if repeats >= FROZEN_FRAMES:
                    return f"{path} repeats one frame {FROZEN_FRAMES} times up to step {start + offset}"
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
    """Share of episodes whose camera streams hold one frame per step, no black frame and no frozen run."""
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
        for eef in episode.children("obs/datagen_info/target_eef_pose"):
            targets = episode[f"obs/datagen_info/target_eef_pose/{eef}"]
            # A channel that commands no pose records all-zero targets, which are not rotations.
            errors.append(_rotation_errors(targets[np.abs(targets).reshape(len(targets), -1).max(axis=1) > 0]))
    errors = np.concatenate(errors) if errors else np.empty(0)
    if not len(errors):
        raise NotApplicable("no end effector commands a pose")
    p50, p95 = np.percentile(errors, [50, 95])
    return Value({"p50": float(p50), "p95": float(p95)}, n=len(errors))
