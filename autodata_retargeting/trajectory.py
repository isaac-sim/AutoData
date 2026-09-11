# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Source-trajectory extraction, subtask-segment detection, and replay-speed resampling."""

import itertools
import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.datasets import EpisodeData

from .util import quat_slerp_batch, se3_inverse

# Component order that maps a recorded ``robot_links_state`` quaternion to Isaac Lab's ``(w, x, y, z)``.
# ``robot_links_state`` comes from ``body_link_pose_w`` and stores its quaternion in an order that this
# permutation reorders to ``(w, x, y, z)`` -- verified to reconstruct the datagen ``eef_pose`` exactly.
# :func:`source_eef_poses_at_link` re-validates it per dataset (matching the observed link), so a future
# format change fails loudly rather than silently producing a wrong reference.
_LINK_STATE_QUAT_ORDER = (1, 2, 3, 0)


def source_datagen_poses(
    episode: EpisodeData,
    eef_names: list[str],
    key: str,
    eef_name_map: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Read a per-step, per-EEF ``(T, 4, 4)`` pose trajectory from a source episode's datagen info.

    ``key`` selects the datagen-info field: ``"target_eef_pose"`` (the commanded controller targets,
    which drive the replay) or ``"eef_pose"`` (the source robot's *achieved* EEF poses, used as the
    consistent-frame reference for the reproduction-error report).

    The source demo keys its poses by the *source* embodiment's EEF names; ``eef_name_map``
    (``{source_eef: target_eef}``) renames them onto the requested target/task ``eef_names`` (identity
    when the pair shares names, e.g. bimanual left/right). Returned dict is keyed by ``eef_names``.
    """
    obs = episode.data.get("obs", {})
    datagen_info = obs.get("datagen_info") if isinstance(obs, dict) else None
    assert datagen_info is not None and key in datagen_info, (
        f"Source episode is missing 'obs/datagen_info/{key}'. Retargeting needs an annotated "
        "dataset — run annotate_demos.py (or use a generate_dataset.py output) first."
    )
    poses = datagen_info[key]
    inverse = {tgt: src for src, tgt in (eef_name_map or {}).items()}
    result: dict[str, torch.Tensor] = {}
    for eef_name in eef_names:
        source_key = inverse.get(eef_name, eef_name)
        assert source_key in poses, f"datagen_info/{key} has no entry for eef '{source_key}'"
        result[eef_name] = poses[source_key]
    return result


def _link_state_pose(links_state: torch.Tensor, body_index: int) -> torch.Tensor:
    """``(T, 4, 4)`` world pose of one body from a recorded ``robot_links_state`` ``(T, num_bodies, 13)``.

    Row layout is ``[pos(3), quat(4), lin_vel(3), ang_vel(3)]``; the quaternion is reordered to
    ``(w, x, y, z)`` via :data:`_LINK_STATE_QUAT_ORDER` before building the rotation.
    """
    row = links_state[:, body_index]
    num_steps = row.shape[0]
    pose = torch.eye(4, dtype=row.dtype, device=row.device).repeat(num_steps, 1, 1)
    pose[:, :3, :3] = math_utils.matrix_from_quat(row[:, 3:7][:, list(_LINK_STATE_QUAT_ORDER)])
    pose[:, :3, 3] = row[:, :3]
    return pose


def _match_body(links_state: torch.Tensor, target: torch.Tensor) -> tuple[int, float]:
    """Body index whose recorded orientation best matches ``target``, plus that mean error [deg].

    Matches on orientation only (translation-free), so it is unaffected by the world-vs-env-origin
    offset between the world-frame link states and the env-relative datagen ``target``. This locates a
    datagen pose (observed ``eef_pose`` or commanded ``target_eef_pose``) among the robot's bodies
    without needing body names, and the match residual doubles as a check of the link-state quaternion
    convention (:data:`_LINK_STATE_QUAT_ORDER`).
    """
    target_quat = math_utils.quat_from_matrix(target[:, :3, :3])
    best_index, best_err = 0, float("inf")
    for body_index in range(links_state.shape[1]):
        quat = links_state[:, body_index, 3:7][:, list(_LINK_STATE_QUAT_ORDER)]
        err = float(torch.rad2deg(math_utils.quat_error_magnitude(target_quat, quat)).mean())
        if err < best_err:
            best_index, best_err = body_index, err
    return best_index, best_err


def source_eef_poses_at_link(
    episode: EpisodeData,
    eef_names: list[str],
    controlled_spec: dict[str, str] | str,
    robot_body_names: list[str] | None = None,
    eef_name_map: dict[str, str] | None = None,
    observed_tol_deg: float = 5.0,
    controlled_tol_deg: float = 20.0,
) -> dict[str, torch.Tensor]:
    """Reconstruct the source *achieved* EEF poses at each EEF's IK-controlled link, from link states.

    The datagen ``eef_pose`` is recorded at whatever link the source env *observed* (e.g. GR1's
    ``hand_roll_link``), which can be a joint short of the link the IK actually *controls*
    (``hand_pitch_link``) -- dropping that joint's rotation from the reference. This rebuilds the
    reference at the controlled link from the recorded per-step link poses::

        achieved_controlled = datagen_eef_pose_observed @ inv(links[observed]) @ links[controlled]

    The datagen pose is the correct-frame anchor; the ``inv(observed) @ controlled`` term is the
    intra-robot observed->controlled transform (the missing joint), and being relative it cancels the
    world/env-origin offset of the link states. Both links are located in ``robot_links_state`` by
    orientation match (:func:`_match_body`), which also validates the link-state format.

    ``controlled_spec`` selects how the controlled link is found:

    * ``"controlled"``/``"auto"`` -- match each EEF's datagen ``target_eef_pose`` (the commanded
      controller target, i.e. the controlled frame by definition). Needs no body names, so it works
      **cross-embodiment** (the source's controlled link is found purely from the source's own datagen).
    * ``{eef_name: link_name}`` -- an explicit override; the named link is located by ``robot_body_names``
      order, so it requires **same-embodiment** retargeting (``robot_body_names`` must be the source
      robot's, taken from the target env).

    Args:
        controlled_spec: ``"controlled"``/``"auto"`` (match ``target_eef_pose``) or a ``{eef: link}`` map.
        robot_body_names: Source robot body names in ``robot_links_state`` order (only for the map form).
        observed_tol_deg: Max orientation error [deg] tolerated when locating the observed link.
        controlled_tol_deg: Max error [deg] when matching the controlled link to ``target_eef_pose``
            (looser than ``observed_tol_deg``: the achieved link lags its command by the tracking error).
    """
    obs = episode.data.get("obs", {})
    links_state = obs.get("robot_links_state") if isinstance(obs, dict) else None
    assert links_state is not None, (
        "eef_reference_link reconstruction needs 'obs/robot_links_state' (per-step link poses) in the "
        "source demo; this env's dataset does not record it. Use reference_pose without eef_reference_link."
    )
    match_controlled = isinstance(controlled_spec, str)
    datagen_observed = source_datagen_poses(episode, eef_names, "eef_pose", eef_name_map)
    datagen_commanded = (
        source_datagen_poses(episode, eef_names, "target_eef_pose", eef_name_map) if match_controlled else {}
    )
    if not match_controlled:
        assert robot_body_names is not None and links_state.shape[1] == len(robot_body_names), (
            "an explicit {eef: link} eef_reference_link needs the source robot's body names in link-state "
            f"order (got {None if robot_body_names is None else len(robot_body_names)} for "
            f"{links_state.shape[1]} bodies); this requires same-embodiment retargeting."
        )

    result: dict[str, torch.Tensor] = {}
    for eef_name in eef_names:
        anchor = datagen_observed[eef_name]
        observed_index, err = _match_body(links_state, anchor)
        assert err <= observed_tol_deg, (
            f"could not locate eef {eef_name!r}'s observed link in the recorded link states (nearest body "
            f"off by {err:.1f} deg > {observed_tol_deg}); the link-state quaternion convention may differ "
            "for this dataset."
        )
        if match_controlled:
            controlled_index, cerr = _match_body(links_state, datagen_commanded[eef_name])
            assert cerr <= controlled_tol_deg, (
                f"could not locate eef {eef_name!r}'s controlled link (matching target_eef_pose): nearest "
                f"body off by {cerr:.1f} deg > {controlled_tol_deg}."
            )
        else:
            controlled_link = controlled_spec[eef_name]
            assert controlled_link in robot_body_names, (
                f"controlled link {controlled_link!r} for eef {eef_name!r} is not a robot body "
                f"(e.g. {robot_body_names[:3]}...)."
            )
            controlled_index = robot_body_names.index(controlled_link)
        relative = se3_inverse(_link_state_pose(links_state, observed_index)) @ _link_state_pose(
            links_state, controlled_index
        )
        result[eef_name] = anchor @ relative
    return result


def source_object_poses(episode: EpisodeData) -> dict[str, torch.Tensor]:
    """Read the source demo's recorded per-object pose trajectories (env-relative ``(T, 4, 4)`` SE(3)).

    Returns ``{object_name: (T, 4, 4)}`` from ``obs/datagen_info/object_pose`` (e.g. ``cube_1/2/3`` for
    cube-stack), or ``{}`` if the annotated source carries no object poses. Used only by the debug
    overlay that draws where each object *was* in the source demo, to see whether the retargeted
    object follows that path or deviates.
    """
    obs = episode.data.get("obs", {})
    datagen_info = obs.get("datagen_info") if isinstance(obs, dict) else None
    object_pose = datagen_info.get("object_pose") if isinstance(datagen_info, dict) else None
    if not isinstance(object_pose, dict):
        return {}
    return {name: traj for name, traj in object_pose.items()}


def source_subtask_signals(episode: EpisodeData) -> dict[str, torch.Tensor]:
    """Read the source demo's per-step subtask-term signals as ``{name: (T, 1)}`` float tensors.

    From ``obs/datagen_info/subtask_term_signals`` (each stored ``(1, T)``); reshaped to ``(T, 1)`` so it
    resamples alongside the other per-step values. Empty when the source carries no signals. Used by the
    signal-driven object tracking (``object_tracking="signals"``) to mark when an object is attached.
    """
    obs = episode.data.get("obs", {})
    datagen_info = obs.get("datagen_info") if isinstance(obs, dict) else None
    signals = datagen_info.get("subtask_term_signals") if isinstance(datagen_info, dict) else None
    if not isinstance(signals, dict):
        return {}
    return {name: value.reshape(-1, 1).to(torch.float32) for name, value in signals.items()}


def _sample_indices(num_src: int, new_len: int, device: torch.device, dtype: torch.dtype):
    """Fractional resampling indices over ``num_src`` waypoints: bracketing index ``i0`` and lerp ``frac``."""
    t = torch.linspace(0.0, num_src - 1, new_len, device=device, dtype=dtype)
    i0 = t.floor().long().clamp(max=num_src - 2)
    return i0, (t - i0.to(dtype)).unsqueeze(-1)


def _resample_pose_trajectory(poses: torch.Tensor, new_len: int) -> torch.Tensor:
    """Resample an SE(3) pose trajectory ``(T, 4, 4)`` to ``new_len`` waypoints along its polyline.

    Position is lerped and rotation slerped between bracketing source waypoints; the first and last
    poses are preserved exactly.
    """
    num_src = poses.shape[0]
    if num_src < 2 or new_len == num_src:
        return poses
    i0, frac = _sample_indices(num_src, new_len, poses.device, poses.dtype)
    p0, p1 = poses[i0], poses[i0 + 1]
    out = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(new_len, 1, 1)
    out[:, :3, 3] = torch.lerp(p0[:, :3, 3], p1[:, :3, 3], frac)
    q0 = math_utils.quat_from_matrix(p0[:, :3, :3])
    q1 = math_utils.quat_from_matrix(p1[:, :3, :3])
    out[:, :3, :3] = math_utils.matrix_from_quat(quat_slerp_batch(q0, q1, frac))
    return out


def _resample_values(values: torch.Tensor, new_len: int) -> torch.Tensor:
    """Nearest-neighbor (step-hold) resample of a per-step value trajectory ``(T, W)`` to ``new_len``.

    These are gripper / passthrough commands, so a discrete open<->close must stay **sharp**. Linear
    interpolation would ramp the transition (a binary ``-1<->+1`` through 0, a parallel target
    ``0.04<->0.0`` through half-open); at ``replay_speed < 1`` that ramps the close over the extra
    waypoints, so the gripper shuts gradually instead of snapping and fails to secure the object while
    the arm moves on. Nearest-neighbor keeps each command at its discrete value. Endpoints are kept.
    """
    num_src = values.shape[0]
    if num_src < 2 or new_len == num_src:
        return values
    t = torch.linspace(0.0, num_src - 1, new_len, device=values.device, dtype=values.dtype)
    nearest = t.round().long().clamp(max=num_src - 1)
    return values[nearest]


def _cap_trajectory_speed(
    poses: torch.Tensor,
    values: dict[str, torch.Tensor],
    max_lin: float | None,
    max_rot_deg: float | None,
    max_expand: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Subdivide a pose trajectory so no step exceeds ``max_lin`` [m] / ``max_rot_deg`` [deg] of motion.

    Each interval ``i -> i+1`` is split into ``k_i = ceil(max(dpos/max_lin, dang/max_rot_deg))`` equal
    sub-steps (position lerp, orientation slerp), so every replayed step stays under the per-waypoint speed
    caps -- a reactive controller then never receives a command faster than it can physically track. Only
    the axes with a non-``None`` cap constrain ``k``. ``values`` (gripper/passthrough) are nearest-neighbor
    resampled onto the new timeline so a discrete open<->close stays sharp. Total length is bounded to
    ``max_expand * T`` (uniform-resample fallback) so a tiny cap cannot explode the trajectory. Returns the
    inputs unchanged when both caps are ``None``, ``T < 2``, or the trajectory is already within the caps.
    """
    num = poses.shape[0]
    if num < 2 or (max_lin is None and max_rot_deg is None):
        return poses, values
    pos = poses[:, :3, 3]
    quat = math_utils.quat_from_matrix(poses[:, :3, :3])
    dpos = torch.linalg.vector_norm(pos[1:] - pos[:-1], dim=1)  # (num-1,) per-step translation [m]
    dots = (quat[1:] * quat[:-1]).sum(dim=1).abs().clamp(max=1.0)
    dang_deg = torch.rad2deg(2.0 * torch.acos(dots))  # (num-1,) per-step rotation [deg]
    k = torch.ones(num - 1, dtype=torch.long, device=poses.device)
    if max_lin is not None:
        k = torch.maximum(k, torch.ceil(dpos / max(max_lin, 1e-9)).long())
    if max_rot_deg is not None:
        k = torch.maximum(k, torch.ceil(dang_deg / max(max_rot_deg, 1e-9)).long())
    total = int(k.sum()) + 1
    if total <= num:
        return poses, values  # already within the caps -- nothing to subdivide
    max_len = int(max_expand * num)
    if total > max_len:  # tiny-cap guard: fall back to a uniform resample at the length ceiling
        return _resample_pose_trajectory(poses, max_len), {c: _resample_values(v, max_len) for c, v in values.items()}
    # Sampling indices: ``k_i`` evenly-spaced fractions per interval, then the final pose exactly.
    i0_list: list[int] = []
    frac_list: list[float] = []
    for i in range(num - 1):
        ki = int(k[i])
        for j in range(ki):
            i0_list.append(i)
            frac_list.append(j / ki)
    i0_list.append(num - 2)
    frac_list.append(1.0)
    i0 = torch.tensor(i0_list, device=poses.device)
    frac = torch.tensor(frac_list, device=poses.device, dtype=poses.dtype).unsqueeze(-1)  # (out, 1)
    out = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(len(i0_list), 1, 1)
    out[:, :3, 3] = torch.lerp(pos[i0], pos[i0 + 1], frac)
    out[:, :3, :3] = math_utils.matrix_from_quat(quat_slerp_batch(quat[i0], quat[i0 + 1], frac))
    nearest = (i0 + frac.squeeze(-1).round().long()).clamp(max=num - 1)  # step-hold gripper onto new timeline
    new_values = {channel: value[nearest] for channel, value in values.items()}
    return out, new_values


def cap_shared_timeline_speed(
    poses: dict[str, torch.Tensor],
    value_dicts: list[dict[str, torch.Tensor]],
    objects: dict[str, torch.Tensor],
    max_lin: float | None,
    max_rot_deg: float | None,
    max_expand: float = 10.0,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Cap the commanded EEF speed on a SHARED multi-EEF timeline (the copy path).

    Unlike :func:`_cap_trajectory_speed` (one trajectory), copy replays every EEF on one shared timeline,
    so the subdivision is computed once as the per-interval **maximum over every EEF's** motion and applied
    to all EEF poses, their passthrough ``value_dicts`` (nearest-neighbor, keeping a discrete open<->close
    sharp), and the tracked ``objects`` (SE(3)) together -- so they stay index-aligned and no step exceeds
    ``max_lin`` [m] / ``max_rot_deg`` [deg]. Total length is bounded to ``max_expand * T``. Returns the
    inputs unchanged when both caps are ``None``, ``T < 2``, or the timeline is already within the caps.
    """
    if (max_lin is None and max_rot_deg is None) or not poses:
        return poses, value_dicts, objects
    any_traj = next(iter(poses.values()))
    num = any_traj.shape[0]
    if num < 2:
        return poses, value_dicts, objects
    # Per-interval subdivision = the max over every EEF (the fastest EEF sets the shared timeline).
    k = torch.ones(num - 1, dtype=torch.long, device=any_traj.device)
    for traj in poses.values():
        pos = traj[:, :3, 3]
        quat = math_utils.quat_from_matrix(traj[:, :3, :3])
        dpos = torch.linalg.vector_norm(pos[1:] - pos[:-1], dim=1)
        dots = (quat[1:] * quat[:-1]).sum(dim=1).abs().clamp(max=1.0)
        dang_deg = torch.rad2deg(2.0 * torch.acos(dots))
        if max_lin is not None:
            k = torch.maximum(k, torch.ceil(dpos / max(max_lin, 1e-9)).long())
        if max_rot_deg is not None:
            k = torch.maximum(k, torch.ceil(dang_deg / max(max_rot_deg, 1e-9)).long())
    total = int(k.sum()) + 1
    if total <= num:
        return poses, value_dicts, objects
    max_len = int(max_expand * num)
    if total > max_len:  # tiny-cap guard: uniform resample at the length ceiling
        return (
            {e: _resample_pose_trajectory(t, max_len) for e, t in poses.items()},
            [{n: _resample_values(v, max_len) for n, v in d.items()} for d in value_dicts],
            {n: _resample_pose_trajectory(t, max_len) for n, t in objects.items()},
        )
    i0_list: list[int] = []
    frac_list: list[float] = []
    for i in range(num - 1):
        ki = int(k[i])
        for j in range(ki):
            i0_list.append(i)
            frac_list.append(j / ki)
    i0_list.append(num - 2)
    frac_list.append(1.0)
    i0 = torch.tensor(i0_list, device=any_traj.device)
    frac = torch.tensor(frac_list, device=any_traj.device, dtype=any_traj.dtype).unsqueeze(-1)  # (out, 1)
    nearest = (i0 + frac.squeeze(-1).round().long()).clamp(max=num - 1)

    def _interp(traj: torch.Tensor) -> torch.Tensor:
        pos, quat = traj[:, :3, 3], math_utils.quat_from_matrix(traj[:, :3, :3])
        out = torch.eye(4, device=traj.device, dtype=traj.dtype).repeat(len(i0_list), 1, 1)
        out[:, :3, 3] = torch.lerp(pos[i0], pos[i0 + 1], frac)
        out[:, :3, :3] = math_utils.matrix_from_quat(quat_slerp_batch(quat[i0], quat[i0 + 1], frac))
        return out

    return (
        {e: _interp(t) for e, t in poses.items()},
        [{n: v[nearest] for n, v in d.items()} for d in value_dicts],
        {n: _interp(t) for n, t in objects.items()},
    )


def resample_trajectory(
    pose_dicts: list[dict[str, torch.Tensor]],
    value_dicts: list[dict[str, torch.Tensor]],
    num_steps: int,
    segment_ends: list[int],
    replay_speed: float,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[int]]:
    """Retime the trajectory by ``1 / replay_speed`` waypoints, resampling each segment independently.

    Splits ``[0, num_steps - 1]`` at the ``segment_ends`` (subtask boundaries) and resamples each chunk
    so its span scales by ``1 / replay_speed`` (e.g. 0.5 -> twice as many waypoints, replayed slower),
    keeping every segment-boundary waypoint exact. ``pose_dicts`` are resampled as SE(3), ``value_dicts``
    (passthrough) linearly. Returns the resampled dicts plus the boundaries mapped to new indices.
    """
    breaks = sorted({0, num_steps - 1} | {b for b in segment_ends if 0 < b < num_steps - 1})
    chunks = list(zip(breaks[:-1], breaks[1:]))
    new_spans = [max(1, round((end - start) / replay_speed)) for start, end in chunks]
    new_bounds = list(itertools.accumulate(new_spans))
    new_segment_ends = new_bounds[:-1]  # internal boundaries only (drop the final trajectory end)

    def resample(traj: torch.Tensor, fn) -> torch.Tensor:
        pieces = []
        for index, (start, end) in enumerate(chunks):
            piece = fn(traj[start : end + 1], new_spans[index] + 1)
            pieces.append(piece if index == 0 else piece[1:])  # boundary waypoint is shared; keep it once
        return torch.cat(pieces, dim=0)

    out_poses = [{eef: resample(traj, _resample_pose_trajectory) for eef, traj in d.items()} for d in pose_dicts]
    out_values = [{name: resample(traj, _resample_values) for name, traj in d.items()} for d in value_dicts]
    return out_poses, out_values, new_segment_ends
