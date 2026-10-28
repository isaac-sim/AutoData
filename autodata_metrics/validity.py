# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validity metrics: whether the recorded data is well formed and the demonstrated task holds."""

from __future__ import annotations

import h5py
import numpy as np

from autodata_metrics.evidence import Episode, Evidence
from autodata_metrics.registry import NotApplicable, Value, check, metric

ROTATION_TOLERANCE = 1e-3
"""Largest deviation from an orthonormal rotation, or from a unit quaternion, a recorded pose may show."""
BLACK_FRAME_MEAN = 2.0
"""Mean pixel value, out of 255, below which a camera frame counts as black."""
FROZEN_FRAMES = 10
"""Identical consecutive camera frames that count as a frozen stream."""
FRAME_CHUNK = 64
"""Camera frames read at a time, so image checks stream instead of loading whole episodes."""


def _is_image(dataset: h5py.Dataset) -> bool:
    return dataset.dtype == np.uint8 and dataset.ndim == 4


def _image_streams(episode: Episode) -> list[tuple[str, h5py.Dataset]]:
    streams = [(path, dataset) for path, dataset in episode.datasets() if _is_image(dataset)]
    if not streams:
        raise NotApplicable("no camera streams")
    return streams


@check("finite")
def finite(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with a NaN or infinite value in any floating-point array."""
    bad = [
        path for path, dataset in episode.datasets() if dataset.dtype.kind == "f" and not np.isfinite(dataset[()]).all()
    ]
    return f"non-finite values in {', '.join(bad)}" if bad else None


PER_EPISODE_PATHS = ("initial_state", "obs/datagen_info/subtask_term_signals", "obs/datagen_info/subtask_start_signals")
"""Paths whose arrays are not one entry per step: the initial state, and annotation's subtask signals."""


@check("aligned_lengths")
def aligned_lengths(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose per-step arrays do not all hold one entry per action; camera streams are checked apart."""
    if "actions" not in episode:
        return "no actions"
    num_steps = len(episode["actions"])
    bad = [
        path
        for path, dataset in episode.datasets()
        if not path.startswith(PER_EPISODE_PATHS) and not _is_image(dataset) and len(dataset) != num_steps
    ]
    return f"{', '.join(bad)} do not hold {num_steps} steps" if bad else None


OBSERVED_POSE_PATHS = ("obs/datagen_info/eef_pose/", "obs/datagen_info/object_pose/")
"""Recorded poses that are observations; commanded targets are measured by ``action_rotation_error`` instead."""


def _rotation_errors(poses: np.ndarray) -> np.ndarray:
    """Per pose, the largest entry of ``|R^T R - I|``; all-zero poses, a channel's placeholder target, are skipped."""
    rotations = poses[np.abs(poses).reshape(len(poses), -1).max(axis=1) > 0][..., :3, :3]
    return np.abs(np.einsum("...ji,...jk->...ik", rotations, rotations) - np.eye(3)).max(axis=(-2, -1))


@check("valid_rotations")
def valid_rotations(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with an observed pose rotation that is not orthonormal or a root quaternion that is not unit.

    Both tests hold for either quaternion order, so the check needs no convention.
    """
    bad = []
    for path, dataset in episode.datasets():
        if path.startswith(OBSERVED_POSE_PATHS) and dataset.shape[-2:] == (4, 4):
            error = _rotation_errors(dataset[()]).max(initial=0.0)
        elif path.endswith("root_pose") and dataset.shape[-1] == 7:
            error = np.abs(np.linalg.norm(dataset[()][..., 3:7], axis=-1) - 1.0).max(initial=0.0)
        else:
            continue
        if error > ROTATION_TOLERANCE:
            bad.append(path)
    return f"invalid rotations in {', '.join(bad)}" if bad else None


@check("pose_action_bounds", requires=["env_args.autodata.pose_action"])
def pose_action_bounds(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose pose actions exceed the bound the embodiment clips them to."""
    actions = episode["actions"]
    bad = []
    for eef, layout in ev.entry["pose_action"].items():
        if layout["bound"] is None:
            continue
        start, end = layout["action_slice"]
        peak = float(np.abs(actions[:, start:end]).max(initial=0.0))
        if peak > layout["bound"] + 1e-6:
            bad.append(f"{eef} reaches {peak:.3g}")
    return f"pose actions beyond their bound: {', '.join(bad)}" if bad else None


@check("one_frame_per_step", group="image")
def one_frame_per_step(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose camera streams do not hold one frame per action."""
    num_steps = len(episode["actions"])
    bad = [path for path, dataset in _image_streams(episode) if len(dataset) != num_steps]
    return f"{', '.join(bad)} do not hold {num_steps} frames" if bad else None


@check("no_black_frames", group="image")
def no_black_frames(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode with a camera frame whose mean pixel value is below :data:`BLACK_FRAME_MEAN`."""
    for path, dataset in _image_streams(episode):
        for start in range(0, len(dataset), FRAME_CHUNK):
            means = dataset[start : start + FRAME_CHUNK].mean(axis=(1, 2, 3))
            if (means < BLACK_FRAME_MEAN).any():
                return f"{path} is black at step {start + int(np.argmax(means < BLACK_FRAME_MEAN))}"
    return None


@check("no_frozen_frames", group="image")
def no_frozen_frames(episode: Episode, ev: Evidence) -> str | None:
    """Fails an episode whose camera repeats one frame :data:`FROZEN_FRAMES` times in a row."""
    for path, dataset in _image_streams(episode):
        previous, repeats = None, 1
        for start in range(0, len(dataset), FRAME_CHUNK):
            for offset, frame in enumerate(dataset[start : start + FRAME_CHUNK]):
                repeats = repeats + 1 if previous is not None and np.array_equal(frame, previous) else 1
                if repeats >= FROZEN_FRAMES:
                    return f"{path} repeats one frame {FROZEN_FRAMES} times up to step {start + offset}"
                previous = frame
    return None


def _pass_rate(ev: Evidence, group: str) -> Value:
    names = [name for name, check_group in ev.check_groups.items() if check_group == group]
    if not names:
        raise NotApplicable(f"no {group} check applies to this dataset")
    if not ev.all_episodes:
        raise NotApplicable("the dataset has no episodes")
    failing = {episode for name in names for episode in ev.failures[name]}
    details = {f"{name}_failed": len(ev.failures[name]) for name in names}
    return Value(1 - len(failing) / len(ev.all_episodes), n=len(ev.all_episodes), details=details)


@metric(
    "correctness_pass_rate",
    area="validity",
    signal="data correctness",
    scope="episode",
    unit="fraction",
    better="higher",
)
def correctness_pass_rate(ev: Evidence) -> Value:
    """Share of episodes passing every correctness check: finite, aligned, valid rotations, actions in bounds."""
    return _pass_rate(ev, "correctness")


@metric(
    "image_stream_pass_rate",
    area="validity",
    signal="data correctness",
    scope="episode",
    unit="fraction",
    better="higher",
)
def image_stream_pass_rate(ev: Evidence) -> Value:
    """Share of episodes whose camera streams hold one frame per step, no black frame and no frozen run."""
    return _pass_rate(ev, "image")


@metric(
    "final_success_rate",
    area="validity",
    signal="task correctness",
    scope="dataset",
    unit="fraction",
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
    unit="max |R^T R - I|",
    better="lower",
    primary="p95",
    requires=["obs/datagen_info/target_eef_pose"],
    baseline=True,
)
def action_rotation_error(ev: Evidence) -> Value:
    """p50 and p95 deviation of the commanded target rotations from a valid rotation, over every step and end effector."""
    errors = np.concatenate([
        _rotation_errors(episode[f"obs/datagen_info/target_eef_pose/{eef}"])
        for episode in ev.episodes
        for eef in episode.children("obs/datagen_info/target_eef_pose")
    ])
    if not len(errors):
        raise NotApplicable("no end effector commands a pose")
    p50, p95 = np.percentile(errors, [50, 95])
    return Value({"p50": float(p50), "p95": float(p95)}, n=len(errors))
