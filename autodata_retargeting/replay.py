# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The retargeting replay engine: reset state, IK warm-start, closed-loop replay, error report."""

import torch
from copy import deepcopy
from typing import Any

import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.datasets import EpisodeData

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter
from autodata_utils.tensor_utils import as_torch

from .config import DefaultObjectTracking
from .gripper_retargeting import PassthroughRemapper, _hand_close_fraction
from .object_tracking import CarrySegment, carry_segments_and_boundaries_from_subtasks, iter_subtask_spans
from .trajectory import (
    cap_shared_timeline_speed,
    resample_trajectory,
    source_datagen_poses,
    source_eef_poses_at_link,
    source_object_poses,
    source_subtask_signals,
)
from .util import (
    pose_tracking_error,
    poses_from_root_pose,
    quat_slerp_batch,
    read_target_base_pose,
    reanchor_to_target_base,
    se3_inverse,
)

# A commanded EEF pose is counted as "reached" (IK effectively tracked it) when the achieved pose
# is within these tolerances. Used only for the IK-performance report, not for success.
_IK_POS_TOL_M = 0.05
_IK_ROT_TOL_DEG = 15.0

# Per-step EEF motion below these tolerances is treated as "settled" (motion has died down), used to
# end a segment settle-hold early. Compared as a delta between consecutive steps, so the pitch/roll
# frame offset that inflates absolute tracking error does not affect it.
_SETTLE_POS_TOL_M = 0.01
_SETTLE_ROT_TOL_DEG = 2.5
# At a subtask boundary (grasp / place) the replayer holds the pose until the robot's joints -- arm AND
# gripper -- stop moving, up to ``config.segment_settle_steps`` extra sim steps (a run knob), then exits
# early, so a slow gripper (e.g. the under-actuated Robotiq 2F-85 linkage) finishes closing/opening before
# the arm advances. Non-boundary waypoints still replay at source speed (one step). "Stopped" is measured
# by per-step joint *position* change, NOT joint_vel: the Robotiq linkage reports a velocity-saturated
# ~1 rad/s joint forever even at rest, whereas the position delta decays cleanly to ~0 once it has closed.
# A fast gripper settles in a few steps and exits well before the cap.
_JOINT_SETTLE_TOL = 0.003  # [rad or m] max per-step joint position change below which the robot is settled

_IK_SOLVE_ITERATIONS = 80
"""Iterations of the differential IK solver used to converge the first-pose joint configuration."""

# A source EEF's gripper is treated as "closed" (grasping) for carry detection once its closedness
# fraction (0 open -> 1 closed, vs the source hand_open/hand_close postures) reaches this.
_CARRY_GRIPPER_CLOSED_FRACTION = 0.5


def validate_eef_agreement(
    source_adapter: EmbodimentAdapter,
    target_adapter: EmbodimentAdapter,
    eef_name_map: dict[str, str] | None = None,
) -> None:
    """Assert both embodiments declare the same EEF names (after ``eef_name_map``).

    The retarget replays the source EEF pose trajectory on the target, so both embodiments must speak of
    the same EEFs. ``eef_name_map`` (``{source_eef: target_eef}``) renames the source EEF names onto the
    target names before the check, so a cross-named single-arm pair (e.g. ``franka`` -> ``ur10``) agrees.
    """
    name_map = eef_name_map or {}
    source_eefs = {name_map.get(eef, eef) for eef in source_adapter.get_eef_names()}
    target_eefs = set(target_adapter.get_eef_names())
    assert (
        source_eefs == target_eefs
    ), f"EEF name mismatch across embodiments: source(mapped)={sorted(source_eefs)}, target={sorted(target_eefs)}" + (
        f" (eef_name_map={name_map})" if name_map else ""
    )


def retargeted_initial_state(
    source_state: dict,
    target_default_state: dict,
    robot_asset_name: str,
    object_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Build a reset state placing the task scene from the source demo on the target robot's env.

    The result uses the *target* env's exact scene schema (from ``target_default_state``) so
    ``reset_to`` never sees an unexpected/missing key, and overrides every non-robot entity's
    matching fields with the source demo's values (object poses, etc.). The robot is kept at the
    target default here; ``init_robot_from_ik`` (in the replay) later overwrites it with the IK
    solution of the first trajectory pose.

    Args:
        source_state: The source episode's recorded ``initial_state`` (scene state dict).
        target_default_state: A freshly-reset target scene state (``scene.get_state``).
        robot_asset_name: Scene key of the robot articulation.
        object_translation: Constant world translation [m] added to every non-robot entity's root
            position, shifting the task objects with the trajectory.

    Returns:
        A scene-state dict compatible with the target env's ``reset_to``.
    """
    merged = deepcopy(target_default_state)
    for category, entities in merged.items():
        source_category = source_state.get(category, {})
        for name, fields in entities.items():
            if name == robot_asset_name:
                continue  # Keep the target robot's default configuration (IK warm-start overrides it).
            source_entity = source_category.get(name)
            if source_entity is None:
                continue
            for field, target_value in fields.items():
                source_value = source_entity.get(field)
                if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
                    fields[field] = source_value

    # Shift the (non-robot) task objects by the same constant translation applied to the trajectory.
    if any(object_translation):
        for category, entities in merged.items():
            if category == "articulation":
                continue
            for name, fields in entities.items():
                root_pose = fields.get("root_pose")
                if name == robot_asset_name or root_pose is None:
                    continue
                translation = torch.tensor(object_translation, dtype=root_pose.dtype, device=root_pose.device)
                shifted = root_pose.clone()
                shifted[..., :3] = shifted[..., :3] + translation
                fields["root_pose"] = shifted
    return merged


def _read_env_object_pose(env: Any, object_name: str, env_id: int = 0) -> torch.Tensor:
    """Read a scene object's current pose as an env-relative ``(4, 4)`` SE(3) transform for ``env_id``.

    The origin is subtracted so it matches the adapter's env-relative EEF frame, letting the two be
    composed into the grasp transform for object-centric planning.
    """
    obj = env.scene[object_name]
    origin = as_torch(env.scene.env_origins)[env_id]
    pos = as_torch(obj.data.root_pos_w)[env_id] - origin
    quat = as_torch(obj.data.root_quat_w)[env_id]
    pose = torch.eye(4, dtype=pos.dtype, device=pos.device)
    pose[:3, :3] = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
    pose[:3, 3] = pos
    return pose


def _blend_pose(pose_a: torch.Tensor, pose_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend two ``(4, 4)`` SE(3) poses: position lerp, orientation slerp. ``alpha=0`` -> a, ``1`` -> b."""
    out = torch.eye(4, dtype=pose_a.dtype, device=pose_a.device)
    out[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
    quat_a = math_utils.quat_from_matrix(pose_a[:3, :3].unsqueeze(0))[:, [0, 1, 2, 3]]
    quat_b = math_utils.quat_from_matrix(pose_b[:3, :3].unsqueeze(0))[:, [0, 1, 2, 3]]
    frac = torch.full((1, 1), float(alpha), dtype=pose_a.dtype, device=pose_a.device)
    blended = quat_slerp_batch(quat_a, quat_b, frac)[:, [0, 1, 2, 3]]
    out[:3, :3] = math_utils.matrix_from_quat(blended)[0]
    return out


def _apply_object_centric_override(
    carry_segments: list[CarrySegment],
    trajectory_step_by_eef: dict[str, int],
    target_eef_pose_dict: dict[str, torch.Tensor],
    source_objects: dict[str, torch.Tensor],
    target_adapter: EmbodimentAdapter,
    env: Any,
    env_id: int = 0,
    controlled_reader=None,
    commanded_poses: dict[str, torch.Tensor] | None = None,
    num_interpolation_steps: int = 0,
    eefs: set[str] | None = None,
) -> None:
    """During a carry segment, replace the carrying EEF's commanded pose with the object-centric one.

    Each step the grasp transform is **re-measured** live in the target env
    (``target_eef_T_object = inv(target_eef) @ target_object``) and the EEF is commanded to
    ``src_object_pose @ inv(grasp_T)`` -- a best-effort, closed-loop move that brings the object from
    wherever it currently sits in the grip to the source object pose this step. Re-measuring every step
    (rather than freezing ``grasp_T`` at the segment start) corrects any object slip relative to the EEF,
    so a non-rigid dexterous grasp still tracks. Mutates ``target_eef_pose_dict``; a no-op outside segments.

    ``grasp_T`` must be measured in the **same frame the command drives** -- the IK-controlled link, via
    ``controlled_reader`` (:func:`read_achieved_eef_poses`), when the reference is reconstructed there.
    Measuring it at the observed link while commanding the controlled link would place the object off by
    the fixed observed->controlled offset (e.g. GR1's ~9 deg wrist_pitch). Defaults to the observed frame.

    Transitions are eased (slerp) rather than snapped: over the first ``interp_start`` steps the command
    ramps from the EEF path to the object-centric pose (step i weights the object by ``(i+1)/interp_start``);
    for ``interp_after`` steps past the segment it ramps from the last tracked pose toward the EEF target
    ``interp_after`` steps ahead (needs ``commanded_poses`` + ``num_interpolation_steps`` to index it),
    landing back on the EEF path.
    """
    for seg in carry_segments:
        if eefs is not None and seg.eef not in eefs:  # skip a holding eef -- do not remeasure its grasp
            continue
        ts = trajectory_step_by_eef[seg.eef]  # this carrying EEF's own trajectory step
        obj_len = source_objects[seg.obj].shape[0]
        in_segment = seg.start <= ts <= seg.end and ts < obj_len
        in_after = 0 < seg.interp_after and seg.end < ts <= seg.end + seg.interp_after
        if not (in_segment or in_after):
            continue
        eef_path_pose = target_eef_pose_dict[seg.eef]  # the source EEF-path pose (this step, pre-override)
        if in_segment:
            eef_now = read_achieved_eef_poses(target_adapter, controlled_reader, env_id)[seg.eef]
            # Live grasp transform (eef_T_object), re-measured every step so slip in the grip self-corrects.
            seg.grasp_transform = se3_inverse(eef_now) @ _read_env_object_pose(env, seg.obj, env_id)
            step_in = ts - seg.start
            if step_in == 0:  # capture where the object really is as the carry begins (the ease-in start)
                seg.object_start_real = _read_env_object_pose(env, seg.obj, env_id)
            if 0 < seg.interp_start and step_in < seg.interp_start:
                # Ease in *object* space: move the object from where it really is (grasp may have nudged it)
                # to the trajectory target ``interp_start`` steps in, then apply the live grasp to get the EEF.
                target_index = min(seg.start + seg.interp_start, seg.end, obj_len - 1)
                object_target = _blend_pose(
                    seg.object_start_real, source_objects[seg.obj][target_index], (step_in + 1) / seg.interp_start
                )
            else:
                object_target = source_objects[seg.obj][ts]
            object_pose = object_target @ se3_inverse(seg.grasp_transform)
            seg.last_target = object_pose
            target_eef_pose_dict[seg.eef] = object_pose
        else:  # end-ramp: ease the last tracked pose -> the EEF target interp_after steps ahead
            start_pose = seg.last_target if seg.last_target is not None else eef_path_pose
            endpoint = eef_path_pose
            if commanded_poses is not None and seg.eef in commanded_poses:
                traj = commanded_poses[seg.eef]
                endpoint = traj[min(seg.end + seg.interp_after + num_interpolation_steps, traj.shape[0] - 1)]
            step_after = ts - seg.end - 1
            target_eef_pose_dict[seg.eef] = _blend_pose(start_pose, endpoint, (step_after + 1) / seg.interp_after)


def _record_signal_frame(env: Any, env_id: int, signal_frame: dict[str, torch.Tensor]) -> None:
    """Append this frame's forwarded subtask signals to the recorded episode (once per ``env.step``).

    ``signal_frame`` maps each signal name to its ``(width,)`` value for the current waypoint (held across
    the waypoint's settle/lead-in frames), recorded under ``obs/datagen_info/subtask_term_signals`` so the
    output dataset's signals stay frame-aligned with the recorded rollout.
    """
    if not signal_frame:
        return
    value = {name: signal.reshape(1, -1) for name, signal in signal_frame.items()}
    env.recorder_manager.add_to_episodes("obs/datagen_info/subtask_term_signals", value, env_ids=[env_id])


def _record_datagen_poses(
    env: Any,
    env_id: int,
    target_adapter: EmbodimentAdapter,
    target_eef_pose_dict: dict[str, torch.Tensor],
    object_names: list[str],
) -> None:
    """Record this frame's ``obs/datagen_info`` poses (once per ``env.step``).

    Writes the observed ``eef_pose`` (``target_adapter.get_eef_poses``), the commanded ``target_eef_pose``,
    and per-object ``object_pose`` -- the same quantities ``annotate_demos``' ``PreStepDatagenInfoRecorder``
    writes -- so a ``copy`` output carries a complete ``datagen_info`` and is a drop-in source for
    ``generate_dataset.py`` (no separate annotation pass). All env-relative ``(1, 4, 4)`` SE(3), per key.
    """
    recorder = env.recorder_manager
    achieved = target_adapter.get_eef_poses(env_ids=[env_id])  # {eef: (1, 4, 4)} observed EEF
    recorder.add_to_episodes("obs/datagen_info/eef_pose", achieved, env_ids=[env_id])
    recorder.add_to_episodes(
        "obs/datagen_info/target_eef_pose",
        {eef: target_eef_pose_dict[eef].reshape(1, 4, 4) for eef in achieved},
        env_ids=[env_id],
    )
    if object_names:
        recorder.add_to_episodes(
            "obs/datagen_info/object_pose",
            {name: _read_env_object_pose(env, name, env_id).reshape(1, 4, 4) for name in object_names},
            env_ids=[env_id],
        )


def _build_commanded_sequences(
    target_adapter: EmbodimentAdapter,
    target_eef_poses: dict[str, torch.Tensor],
    target_passthrough: dict[str, torch.Tensor],
    eef_names: list[str],
    num_interpolation_steps: int,
    env_id: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build the per-step commanded pose + passthrough sequences, with an optional lead-in ramp.

    When ``num_interpolation_steps > 0``, each EEF is linearly interpolated from the target robot's
    *current* pose (read from the freshly-reset env) to the first source-trajectory pose, and the
    passthrough channels hold their first value across the ramp. Both returned dicts share the same
    length ``num_interpolation_steps + T``.
    """
    commanded_poses: dict[str, torch.Tensor] = {}
    if num_interpolation_steps > 0:
        current_poses = target_adapter.get_eef_poses(env_ids=[env_id])
    for eef_name in eef_names:
        trajectory = target_eef_poses[eef_name]
        if num_interpolation_steps > 0:
            # interpolate_poses returns [start, ...N interp..., end]; keep only the N interior poses
            # (the trajectory itself supplies its own first pose).
            lead_in, _ = math_utils.interpolate_poses(
                current_poses[eef_name][0], trajectory[0], num_steps=num_interpolation_steps
            )
            commanded_poses[eef_name] = torch.cat([lead_in[1:-1], trajectory], dim=0)
        else:
            commanded_poses[eef_name] = trajectory

    commanded_passthrough: dict[str, torch.Tensor] = {}
    for name, tensor in target_passthrough.items():
        if num_interpolation_steps > 0:
            held = tensor[0:1].expand(num_interpolation_steps, *tensor.shape[1:])
            commanded_passthrough[name] = torch.cat([held, tensor], dim=0)
        else:
            commanded_passthrough[name] = tensor
    return commanded_poses, commanded_passthrough


def _build_controlled_link_pose_reader(env: Any, target_adapter: EmbodimentAdapter):
    """Return a fn giving each EEF's IK-controlled-link pose ``{eef: (4,4)}``, or None if unavailable.

    Reads the world (env-relative) pose of the link each EEF's commanded pose actually drives (the
    PinkIK ``target_eef_link_names``), mapped to the adapter's EEF names by order. This is the frame
    the retarget commands, so measuring here removes the target's observed-vs-controlled frame offset.
    """
    eef_names = list(target_adapter.get_eef_names())
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return None
    for term_name in getattr(manager, "active_terms", None) or []:
        term = manager.get_term(term_name)
        link_map = getattr(getattr(term, "cfg", None), "target_eef_link_names", None)
        if not link_map or len(link_map) != len(eef_names):
            continue
        robot = env.scene[getattr(target_adapter, "robot_asset_name", "robot")]
        body_names = list(robot.data.body_names)
        idx_by_eef = {eef: body_names.index(link) for eef, link in zip(eef_names, link_map.values())}

        def _torch(x):
            # Body poses come back as warp arrays (need .torch); joint/origin data may already be torch.
            return x.torch if hasattr(x, "torch") else as_torch(x)

        def read_controlled_poses(env_id: int = 0) -> dict[str, torch.Tensor]:
            pos_w = _torch(robot.data.body_pos_w)
            quat_w = _torch(robot.data.body_quat_w)
            origin = _torch(env.scene.env_origins)[env_id]
            poses: dict[str, torch.Tensor] = {}
            for eef, idx in idx_by_eef.items():
                pose = torch.eye(4, dtype=pos_w.dtype, device=pos_w.device)
                pose[:3, :3] = math_utils.matrix_from_quat(quat_w[env_id, idx : idx + 1])[0]
                pose[:3, 3] = pos_w[env_id, idx] - origin
                poses[eef] = pose
            return poses

        return read_controlled_poses
    return None


def read_achieved_eef_poses(
    target_adapter: EmbodimentAdapter, controlled_reader, env_id: int = 0
) -> dict[str, torch.Tensor]:
    """``{eef: (4,4)}`` achieved EEF poses at the controlled link (via ``controlled_reader``) or observed.

    The single frame both the tracking metric and the object-centric grasp transform must use: when the
    reference is reconstructed at the IK-controlled link, ``controlled_reader`` reads there; otherwise
    fall back to the adapter's observed EEF frame. Squeezes the adapter's ``(1,4,4)`` to ``(4,4)``.
    """
    if controlled_reader is not None:
        return controlled_reader(env_id)
    return {eef: pose[0] for eef, pose in target_adapter.get_eef_poses(env_ids=[env_id]).items()}


def resolve_eef_reference_links(
    target_adapter: EmbodimentAdapter, eef_reference_link: dict[str, str] | str | None
) -> dict[str, str] | str | None:
    """Validate/normalize ``eef_reference_link`` (see :func:`.trajectory.source_eef_poses_at_link`).

    ``None`` -> feature off. ``"controlled"``/``"auto"`` -> the universal, cross-embodiment sentinel
    (the controlled link is found per-episode by matching the source's own ``target_eef_pose``, so no
    env introspection is needed). A dict -> an explicit ``{eef: link}`` override (same-embodiment); it
    must cover every EEF. Resolved once, before the replay loop.
    """
    if eef_reference_link is None or eef_reference_link in ("controlled", "auto"):
        return eef_reference_link
    assert isinstance(
        eef_reference_link, dict
    ), f"eef_reference_link must be a {{eef: link}} map or 'controlled'/'auto', got {eef_reference_link!r}."
    eef_names = list(target_adapter.get_eef_names())
    missing = [eef for eef in eef_names if eef not in eef_reference_link]
    assert not missing, f"eef_reference_link is missing entries for EEFs {missing}."
    return {eef: eef_reference_link[eef] for eef in eef_names}


def _find_pink_ik_term(env: Any):
    """Return the env's PinkIK action term (holds the per-env IK controllers), or None."""
    manager = getattr(env, "action_manager", None)
    for term_name in (getattr(manager, "active_terms", None) or []) if manager is not None else []:
        term = manager.get_term(term_name)
        if getattr(term, "_ik_controllers", None) and getattr(term, "_isaaclab_controlled_joint_ids", None) is not None:
            return term
    return None


def _ik_solved_robot_state(
    env: Any,
    target_adapter: EmbodimentAdapter,
    robot_asset_name: str,
    base_state: dict,
    first_pose_dict: dict[str, torch.Tensor],
    first_passthrough_dict: dict[str, torch.Tensor],
    env_id: int = 0,
) -> dict:
    """Solve the IK for the first commanded pose and return a reset state starting the robot there.

    The env's PinkIK is a *differential* solver (it returns joint velocities), so we iterate its
    ``compute`` in configuration space until the controlled joints converge — pure kinematics, with no
    physics, gravity, or settling: the robot never actually moves. The hand/passthrough joints are not
    IK'd, so they are taken straight from the first action. Returns ``base_state`` with the robot's
    ``joint_position`` overwritten by the solved configuration, so ``reset_to`` begins the recorded demo
    exactly at the IK solution of the first trajectory pose (works for any target embodiment).

    Solves for ``env_id`` only: the first-pose target is written into that env's row of the term action
    (others zeroed) and only ``env_id``'s controller/joints are read, so this works per-env in the
    parallel replay. The scratch task targets set on the other envs' controllers are overwritten by the
    next real ``env.step``, so they are harmless.
    """
    term = _find_pink_ik_term(env)
    assert term is not None, "init_robot_from_ik requires a PinkIK action term to solve the first pose."
    first_action = (
        target_adapter.target_eef_pose_to_action(
            target_eef_pose_dict=first_pose_dict, passthrough_action_dict=first_passthrough_dict, env_id=env_id
        )
        .reshape(-1)
        .to(device=env.device)
    )
    # The PinkIK term consumes only its own leading slice of the env action (EEF poses + hand joints);
    # any other action terms (e.g. G1's 4-D body/locomotion channel) own the trailing columns. Write the
    # first-pose slice into this env's row of a full (num_envs, term_dim) action so process_actions sets
    # the right task target for env_id's controller.
    term_dim = term._raw_actions.shape[-1]
    pink_row = first_action[:term_dim]
    full_action = torch.zeros((env.num_envs, term_dim), device=env.device, dtype=pink_row.dtype)
    full_action[env_id] = pink_row
    term.process_actions(full_action)  # set the IK task targets to the first pose (base-relative)

    robot = env.scene[robot_asset_name]
    controller = term._ik_controllers[env_id]
    controlled_ids = list(term._isaaclab_controlled_joint_ids)
    dt = getattr(term, "_sim_dt", None) or env.sim.get_physics_dt()
    joints = as_torch(robot.data.joint_pos)[env_id].clone()  # full joint vector, starting at target default
    for _ in range(_IK_SOLVE_ITERATIONS):
        solved = controller.compute(joints.detach().cpu().numpy(), dt)  # differential step toward the target
        joints[controlled_ids] = solved.to(device=joints.device, dtype=joints.dtype)
    hand_ids = list(getattr(term, "_hand_joint_ids", []) or [])
    if hand_ids:  # hands are passthrough, not IK'd: take them from the term's trailing hand-joint slice
        joints[hand_ids] = pink_row[-len(hand_ids) :].to(device=joints.device, dtype=joints.dtype)

    warm = deepcopy(base_state)
    fields = warm["articulation"][robot_asset_name]
    joint_pos = joints.unsqueeze(0)
    if "joint_position" in fields and tuple(fields["joint_position"].shape) == tuple(joint_pos.shape):
        fields["joint_position"] = joint_pos
    if "joint_velocity" in fields:
        fields["joint_velocity"] = torch.zeros_like(fields["joint_velocity"])
    return warm


def _slice_scene_state(state: dict, env_id: int) -> dict:
    """Return a scene-state dict with every leaf tensor sliced to ``env_id``'s single row ((1, ...)).

    A freshly-read ``scene.get_state`` is shaped ``(num_envs, ...)``; ``reset_to`` (and the source
    episode's ``(1, ...)`` overrides) need one env's row. For a single-env run (``env_id == 0``) this is
    the identity slice.

    Most categories nest ``category -> name -> {field: tensor}`` (3 levels), but surface grippers nest
    ``gripper -> name -> tensor`` (the entity value is a bare tensor, not a field dict), so slicing
    recurses over dicts and slices any tensor leaf regardless of depth.
    """

    def _slice(node):
        if isinstance(node, dict):
            return {key: _slice(value) for key, value in node.items()}
        if isinstance(node, torch.Tensor):
            return node[env_id : env_id + 1]
        return node

    return _slice(state)


def _build_offset_timeline(segs: list, num_steps: int) -> tuple[list, list, list]:
    """Per-step ``(translation, axis_angle, frame)`` for one EEF from its ordered subtask offsets.

    Each subtask *holds* its offset over its core ``[start + interpolation_start, end - interpolation_end]``;
    between cores the offset interpolates **linearly between the two adjacent subtasks' offsets** (not
    through zero), so a 5 cm subtask followed by a 10 cm subtask blends 5 -> 10 across the boundary. A
    leading ramp eases identity -> the first offset over the first ``interpolation_start``. The **last
    subtask HOLDS its offset through the end** (and thus through any success-settle hold) -- it is not
    decayed to identity, since there is no next offset to blend toward. A subtask with no ``offset`` is an
    identity key, so a neighbour eases to/from zero into it. ``segs`` are dicts sorted by ``start``.
    """
    zero = [0.0, 0.0, 0.0]
    keyframes: list[tuple[int, list, list]] = []
    if segs[0]["i_start"] > 0:  # ease in from identity to the first offset
        keyframes.append((segs[0]["start"], zero, zero))
    for s in segs:
        hstart = s["start"] + s["i_start"]
        # The last subtask holds to its end (no ramp-down); interior subtasks reach their own hold end.
        hend = s["end"] if s is segs[-1] else max(hstart, s["end"] - s["i_end"])
        hend = max(hstart, hend)
        keyframes.append((hstart, s["trans"], s["aa"]))
        if hend != hstart:
            keyframes.append((hend, s["trans"], s["aa"]))
    keyframes.sort(key=lambda k: k[0])

    frame_by_step = [segs[-1]["frame"]] * num_steps
    for s in segs:
        for t in range(s["start"], min(s["end"], num_steps)):
            frame_by_step[t] = s["frame"]

    trans_by_step: list = [zero] * num_steps
    aa_by_step: list = [zero] * num_steps
    j = 0
    for t in range(num_steps):
        while j + 1 < len(keyframes) and keyframes[j + 1][0] <= t:
            j += 1
        step0, tr0, aa0 = keyframes[j]
        step1, tr1, aa1 = keyframes[j + 1] if j + 1 < len(keyframes) else keyframes[j]
        alpha = 0.0 if (t <= step0 or step1 <= step0) else (1.0 if t >= step1 else (t - step0) / (step1 - step0))
        trans_by_step[t] = [tr0[k] + alpha * (tr1[k] - tr0[k]) for k in range(3)]
        aa_by_step[t] = [aa0[k] + alpha * (aa1[k] - aa0[k]) for k in range(3)]
    return trans_by_step, aa_by_step, frame_by_step


def _offset_se3(translation, axis_angle, scale: float, ref: torch.Tensor) -> torch.Tensor:
    """Scaled SE(3) offset ``(4, 4)`` from ``translation`` [m] and ``axis_angle`` [rad] (either optional),
    both linearly scaled by ``scale`` (angle scaled, so the rotation eases in). ``ref`` sets dtype/device."""
    mat = torch.eye(4, dtype=ref.dtype, device=ref.device)
    if axis_angle is not None:
        aa = torch.tensor(axis_angle, dtype=ref.dtype, device=ref.device) * scale
        theta = torch.linalg.vector_norm(aa)
        if float(theta) > 1e-8:
            k = aa / theta  # unit axis
            skew = torch.zeros((3, 3), dtype=ref.dtype, device=ref.device)
            skew[0, 1], skew[0, 2], skew[1, 0] = -k[2], k[1], k[2]
            skew[1, 2], skew[2, 0], skew[2, 1] = -k[0], -k[1], k[0]
            eye3 = torch.eye(3, dtype=ref.dtype, device=ref.device)
            mat[:3, :3] = eye3 + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (skew @ skew)
    if translation is not None:
        mat[:3, 3] = torch.tensor(translation, dtype=ref.dtype, device=ref.device) * scale
    return mat


def apply_subtask_offsets(
    target_eef_poses: dict[str, torch.Tensor],
    subtasks: dict,
    gripper_closed: dict[str, torch.Tensor],
    signals: dict[str, torch.Tensor],
    num_steps: int,
    eef_name_map: dict[str, str] | None,
    close_fraction: float,
    source_objects: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply each subtask's ``offset`` to that EEF's commanded trajectory over the subtask's span.

    The offset ``O`` acts in a frame ``F`` as ``P' = F @ O @ inv(F) @ P``, which specializes to
    ``world`` (``F = I`` -> ``O @ P``), ``eef``/``controlled`` (``F = P`` -> ``P @ O``, a nudge in the
    gripper's own frame), and a tracked object name (``F`` = that object's live pose). The frame defaults
    to the subtask's own reference (``object_ref`` else ``frame_ref`` else ``eef``). The offset is a
    continuous per-EEF curve across subtasks (:func:`_build_offset_timeline`): boundaries interpolate
    between adjacent subtask offsets (5 cm -> 10 cm blends 5 -> 10, not 5 -> 0 -> 10), eased by each
    subtask's ``interpolation_start`` / ``interpolation_end``.
    """
    per_eef: dict[str, list] = {}
    for _eef_key, eef, _index, st, start, end, _boundary in iter_subtask_spans(
        subtasks, gripper_closed, signals, num_steps, eef_name_map, close_fraction
    ):
        off = getattr(st, "offset", None)
        per_eef.setdefault(eef, []).append(
            {
                "start": start,
                "end": end,
                "trans": list(off.translation) if (off and off.translation) else [0.0, 0.0, 0.0],
                "aa": list(off.axis_angle) if (off and off.axis_angle) else [0.0, 0.0, 0.0],
                "frame": (off.frame if (off and off.frame) else None) or st.object_ref or st.frame_ref or "eef",
                "i_start": off.interpolation_start if off else 0,
                "i_end": off.interpolation_end if off else 0,
                "has_offset": off is not None,
            }
        )
    for eef, segs in per_eef.items():
        if eef not in target_eef_poses or not any(s["has_offset"] for s in segs):
            continue
        segs.sort(key=lambda s: s["start"])
        poses = target_eef_poses[eef] = target_eef_poses[eef].clone()  # copy-on-write
        num = poses.shape[0]
        trans_by_step, aa_by_step, frame_by_step = _build_offset_timeline(segs, num)
        eye4 = torch.eye(4, dtype=poses.dtype, device=poses.device)
        for step in range(num):
            trans, axis_angle = trans_by_step[step], aa_by_step[step]
            if not any(trans) and not any(axis_angle):  # identity offset -> leave pose untouched
                continue
            o_mat = _offset_se3(trans, axis_angle, 1.0, poses)
            pose = poses[step]
            frame = frame_by_step[step]
            if frame in ("eef", "controlled"):
                frame_pose = pose
            elif source_objects and frame in source_objects:
                frame_pose = source_objects[frame][step]
            else:  # "world" (or an unrecognized frame_ref -> world)
                frame_pose = eye4
            poses[step] = frame_pose @ o_mat @ torch.linalg.inv(frame_pose) @ pose
    return target_eef_poses


def prepare_episode(
    env: Any,
    env_id: int,
    episode: EpisodeData,
    target_adapter: EmbodimentAdapter,
    source_adapter: EmbodimentAdapter,
    remap_passthrough: PassthroughRemapper,
    target_default_state: dict,
    robot_asset_name: str,
    reference_pose: str,
    eef_name_map: dict[str, str] | None,
    replay_speed: float,
    retarget_frame: str,
    scene_translation: tuple[float, float, float],
    eef_offsets: dict[str, torch.Tensor] | None,
    num_interpolation_steps: int,
    init_robot_from_ik: bool,
    subtasks: dict,
    default_object_tracking: DefaultObjectTracking,
    source_hand_postures: dict[str, dict[str, list[float]]] | None,
    need_source_objects: bool,
    reset_sim: bool,
    eef_reference_link: dict[str, str] | str | None = None,
    write_datagen_info: bool = False,
    segment_settle_steps: int = 0,
    max_eef_linear_velocity: float | None = None,
    max_eef_rotation_speed: float | None = None,
    synchronization: list | None = None,
) -> dict:
    """Reset ``env_id`` to the source episode's scene and build its commanded trajectory.

    Shared by the sequential replay and the parallel workers. Everything here is env-side (reset) plus
    pure trajectory math, keyed by ``env_id``. ``reset_sim`` triggers a global ``env.sim.reset()`` (only
    valid single-env; parallel passes False and relies on per-env ``reset_to``). Returns a dict with the
    commanded pose/passthrough sequences, step count, segment boundaries, carry segments, and (when
    requested) the source object trajectories -- ready for the step loop.
    """
    eef_names = list(target_adapter.get_eef_names())

    # When eef_reference_link is set, rebuild the achieved ("eef_pose") reference at each EEF's
    # IK-controlled link (e.g. GR1 hand_pitch) instead of the source's observed link (hand_roll), which
    # can be a joint short of the gripper. Only "eef_pose" is a link-observed quantity; "target_eef_pose"
    # is already the commanded controlled-link pose, so it needs no reconstruction.
    def _achieved_eef_poses() -> dict[str, torch.Tensor]:
        if eef_reference_link is None:
            return source_datagen_poses(episode, eef_names, "eef_pose", eef_name_map)
        body_names = list(env.scene[robot_asset_name].data.body_names)
        return source_eef_poses_at_link(episode, eef_names, eef_reference_link, body_names, eef_name_map)

    if reference_pose == "eef_pose":
        target_eef_poses = _achieved_eef_poses()
    else:
        target_eef_poses = source_datagen_poses(episode, eef_names, reference_pose, eef_name_map)

    source_actions = episode.data["actions"]
    source_passthrough = source_adapter.actions_to_passthrough_actions(source_actions)
    if eef_name_map:
        source_passthrough = {eef_name_map.get(name, name): value for name, value in source_passthrough.items()}
    target_passthrough = remap_passthrough(source_passthrough)

    source_objects = source_object_poses(episode) if need_source_objects else {}
    # Signals are read when a subtask ends on a signal event and/or to forward them into the output.
    needs_signals = write_datagen_info or any(
        st.subtask_end is not None and st.subtask_end.method in ("signal_on", "signal_off")
        for entries in subtasks.values()
        for st in entries
    )
    source_signals = source_subtask_signals(episode) if needs_signals else {}
    # Per-EEF source gripper closedness (0 open -> 1 closed), keyed by target EEF name (source_passthrough
    # was already renamed via eef_name_map); used to detect gripper_open/close subtask boundaries.
    source_gripper_closed: dict[str, torch.Tensor] = {}
    if source_hand_postures:
        for eef in eef_names:
            posture, hand = source_hand_postures.get(eef), source_passthrough.get(eef)
            if posture is None or hand is None:
                continue
            src_open = torch.tensor(posture["open"], dtype=hand.dtype, device=hand.device)
            src_close = torch.tensor(posture["close"], dtype=hand.dtype, device=hand.device)
            source_gripper_closed[eef] = _hand_close_fraction(hand, src_open, src_close).unsqueeze(1)  # (T, 1)

    if replay_speed != 1.0:
        orig_num_steps = target_eef_poses[eef_names[0]].shape[0]
        (target_eef_poses,), (target_passthrough,), _ = resample_trajectory(
            [target_eef_poses],
            [target_passthrough],
            num_steps=orig_num_steps,
            segment_ends=[],
            replay_speed=replay_speed,
        )
        # Per-step value dicts resampled nearest-neighbor (binary/gripper/signals stay sharp).
        extras = [d for d in (source_objects,) if d]
        value_dicts = [d for d in (source_gripper_closed, source_signals) if d]
        if extras or value_dicts:
            resampled, resampled_values, _ = resample_trajectory(
                extras, value_dicts, num_steps=orig_num_steps, segment_ends=[], replay_speed=replay_speed
            )
            resampled_iter, resampled_values_iter = iter(resampled), iter(resampled_values)
            if source_objects:
                source_objects = next(resampled_iter)
            if source_gripper_closed:
                source_gripper_closed = next(resampled_values_iter)
            if source_signals:
                source_signals = next(resampled_values_iter)

    # Reset this env to the source demo's task scene; the target robot starts at its default config.
    env_ids = torch.tensor([env_id], device=env.device)
    if reset_sim:
        env.sim.reset()
    env.recorder_manager.reset(env_ids=env_ids)
    # ``reset_to`` expects a state shaped (len(env_ids), ...) == (1, ...); slice this env's row out of the
    # (num_envs, ...) default so the source object poses (also (1, ...)) shape-match and get applied.
    reset_state = retargeted_initial_state(
        episode.data["initial_state"],
        _slice_scene_state(target_default_state, env_id),
        robot_asset_name,
        object_translation=scene_translation,
    )
    env.reset_to(reset_state, env_ids, is_relative=True)

    if retarget_frame == "robot_base":
        source_root = as_torch(episode.data["states"]["articulation"][robot_asset_name]["root_pose"]).to(env.device)
        source_base = poses_from_root_pose(source_root)
        target_base = read_target_base_pose(env, robot_asset_name, env_id)
        target_eef_poses = {
            eef: reanchor_to_target_base(pose, source_base, target_base) for eef, pose in target_eef_poses.items()
        }

    if any(scene_translation):
        translation = torch.tensor(
            scene_translation, dtype=target_eef_poses[eef_names[0]].dtype, device=target_eef_poses[eef_names[0]].device
        )
        for eef in eef_names:
            target_eef_poses[eef] = target_eef_poses[eef].clone()
            target_eef_poses[eef][:, :3, 3] += translation
        for name in source_objects:
            source_objects[name] = source_objects[name].clone()
            source_objects[name][:, :3, 3] += translation

    if eef_offsets is not None:
        for eef in eef_names:
            target_eef_poses[eef] = target_eef_poses[eef] @ eef_offsets[eef].to(target_eef_poses[eef].dtype)

    # Cap the commanded EEF speed for a reactive controller (e.g. Galbot RmpFlow): subdivide the shared
    # timeline wherever any EEF moves faster than the caps, resampling every EEF pose, the passthrough, the
    # tracked-object paths, gripper closedness, and signals together so they stay index-aligned. Measured on
    # the final (post-offset) commanded poses. A differential-IK robot snaps to each waypoint in one step,
    # so leave the caps ``None`` there. Boundaries below are then computed on this capped timeline.
    if max_eef_linear_velocity is not None or max_eef_rotation_speed is not None:
        target_eef_poses, (target_passthrough, source_gripper_closed, source_signals), source_objects = (
            cap_shared_timeline_speed(
                target_eef_poses,
                [target_passthrough, source_gripper_closed, source_signals],
                source_objects,
                max_eef_linear_velocity,
                max_eef_rotation_speed,
            )
        )

    # Segmentation + object tracking come straight from the descriptor's subtasks, on the final (possibly
    # speed-capped) timeline: each non-final subtask's end marks a per-EEF boundary (motion-aware settle),
    # a subtask with object_tracking a carry segment, and a named subtask's end feeds the cross-EEF
    # ``synchronization`` barriers so a per-EEF settle can't leave the arms desynchronized.
    num_traj_steps = target_eef_poses[eef_names[0]].shape[0]
    # Per-subtask SE(3) offsets: nudge each EEF's commanded pose over its subtask span (in the subtask's
    # frame, eased in/out) before segmentation, so carry/settle are computed on the offset trajectory.
    target_eef_poses = apply_subtask_offsets(
        target_eef_poses,
        subtasks,
        source_gripper_closed,
        source_signals,
        num_traj_steps,
        eef_name_map,
        _CARRY_GRIPPER_CLOSED_FRACTION,
        source_objects,
    )
    carry_segments, segment_ends, name_end_step = carry_segments_and_boundaries_from_subtasks(
        subtasks,
        source_gripper_closed,
        source_signals,
        num_traj_steps,
        eef_name_map=eef_name_map,
        default_interp_start=default_object_tracking.interpolation_step_start,
        default_interp_after=default_object_tracking.interpolation_step_after,
        default_settle_steps=segment_settle_steps,
        close_fraction=_CARRY_GRIPPER_CLOSED_FRACTION,
    )
    sync_of, group_members = _build_sync_barriers(synchronization or [], name_end_step)

    if init_robot_from_ik:
        warm_state = _ik_solved_robot_state(
            env,
            target_adapter,
            robot_asset_name,
            reset_state,
            first_pose_dict={eef: target_eef_poses[eef][0] for eef in eef_names},
            first_passthrough_dict={name: tensor[0] for name, tensor in target_passthrough.items()},
            env_id=env_id,
        )
        env.recorder_manager.reset(env_ids=env_ids)
        env.reset_to(warm_state, env_ids, is_relative=True)
        num_interpolation_steps = 0

    commanded_poses, commanded_passthrough = _build_commanded_sequences(
        target_adapter, target_eef_poses, target_passthrough, eef_names, num_interpolation_steps, env_id
    )
    return {
        "eef_names": eef_names,
        "commanded_poses": commanded_poses,
        "commanded_passthrough": commanded_passthrough,
        "num_interpolation_steps": num_interpolation_steps,
        "num_steps": commanded_poses[eef_names[0]].shape[0],
        "carry_segments": carry_segments,
        "source_objects": source_objects,
        "source_signals": source_signals,  # resampled to trajectory length; forwarded when write_datagen_info
        "segment_ends": segment_ends,  # {eef: {trajectory step: settle cap}} -- per-EEF motion-aware settle
        "sync_of": sync_of,  # {eef: {end step: group id}} -- cross-EEF rendezvous from ``synchronization``
        "group_members": group_members,  # {group id: [(eef, end step)]}
    }


def _monitored_error(
    eef_name: str,
    ts: int,
    carry_segments: list,
    source_objects: dict[str, torch.Tensor],
    target_eef_pose_dict: dict[str, torch.Tensor],
    achieved_poses: dict[str, torch.Tensor],
    env: Any,
    env_id: int,
) -> tuple[float, float, bool]:
    """The tracking error to score for an EEF this step: the tracked **object** vs its source-path pose
    during a carry (so a slipped grasp shows even though the EEF still follows the live grasp), else the
    **EEF** vs its commanded target. Returns ``(pos [m], rot [deg], is_object)``.
    """
    seg = next(
        (
            s
            for s in carry_segments
            if s.eef == eef_name and s.start <= ts <= s.end and ts < source_objects[s.obj].shape[0]
        ),
        None,
    )
    if seg is not None:
        pos_err, rot_err = pose_tracking_error(source_objects[seg.obj][ts], _read_env_object_pose(env, seg.obj, env_id))
        return pos_err, rot_err, True
    pos_err, rot_err = pose_tracking_error(target_eef_pose_dict[eef_name], achieved_poses[eef_name])
    return pos_err, rot_err, False


def _early_stop_hit(
    pos_err: float,
    rot_err: float,
    is_object: bool,
    max_translation_error: float | None,
    max_rotation_error: float | None,
    env_id: int,
    tick: int,
    eef_name: str,
) -> bool:
    """True (and prints the reason) if the monitored error exceeds a threshold (``stop_early_on_failure``)."""
    if (max_translation_error is not None and pos_err > max_translation_error) or (
        max_rotation_error is not None and rot_err > max_rotation_error
    ):
        tag = f"[retarget env{env_id}]" if env_id else "[retarget]"
        print(
            f"{tag} early stop at tick {tick + 1}: {eef_name} {'object' if is_object else 'eef'} miss "
            f"{pos_err * 100:.1f}cm / {rot_err:.1f}deg exceeds threshold -> failure.",
            flush=True,
        )
        return True
    return False


def _monitored_miss(
    eef_names: list[str],
    traj_step: dict[str, int],
    ptr: dict[str, int],
    lengths: dict[str, int],
    carry_segments: list,
    source_objects: dict[str, torch.Tensor],
    target_eef_pose_dict: dict[str, torch.Tensor],
    achieved_poses: dict[str, torch.Tensor],
    env: Any,
    env_id: int,
    max_translation_error: float | None,
    max_rotation_error: float | None,
    tick: int,
) -> bool:
    """True if any active EEF's monitored error (see :func:`_monitored_error`) exceeds the thresholds.

    Prints the reason. Used to early-abort a doomed replay (``stop_early_on_failure``) on the parallel path
    (the single-env path scores + checks together in :func:`_score_step`).
    """
    for eef_name in eef_names:
        ts = traj_step[eef_name]
        if ts < 0 or ptr[eef_name] >= lengths[eef_name]:
            continue
        pos_err, rot_err, is_object = _monitored_error(
            eef_name, ts, carry_segments, source_objects, target_eef_pose_dict, achieved_poses, env, env_id
        )
        if _early_stop_hit(
            pos_err, rot_err, is_object, max_translation_error, max_rotation_error, env_id, tick, eef_name
        ):
            return True
    return False


def _score_step(
    eef_names: list[str],
    traj_step: dict[str, int],
    ptr: dict[str, int],
    lengths: dict[str, int],
    target_eef_pose_dict: dict[str, torch.Tensor],
    achieved_poses: dict[str, torch.Tensor],
    pos_errors: dict[str, list[float]],
    rot_errors: dict[str, list[float]],
    monitor_early: bool,
    carry_segments: list,
    source_objects: dict[str, torch.Tensor],
    env: Any,
    env_id: int,
    max_translation_error: float | None,
    max_rotation_error: float | None,
    tick: int,
) -> bool:
    """Record the per-EEF tracking error and test the early-abort thresholds (real, unfinished steps only).

    The recorded error is the **monitored** quantity (:func:`_monitored_error`): the tracked object vs its
    source path during a carry, else the EEF vs its command -- so the per-episode report reflects how well
    the *object* was tracked, not just how well the arm followed the (object-centric) command. Returns True
    if the replay should early-abort as a failure (``stop_early_on_failure``).
    """
    early_failure = False
    for eef_name in eef_names:
        if traj_step[eef_name] < 0 or ptr[eef_name] >= lengths[eef_name]:
            continue
        pos_err, rot_err, is_object = _monitored_error(
            eef_name,
            traj_step[eef_name],
            carry_segments,
            source_objects,
            target_eef_pose_dict,
            achieved_poses,
            env,
            env_id,
        )
        pos_errors[eef_name].append(pos_err)
        rot_errors[eef_name].append(rot_err)
        if monitor_early and not early_failure:
            early_failure = _early_stop_hit(
                pos_err, rot_err, is_object, max_translation_error, max_rotation_error, env_id, tick, eef_name
            )
    return early_failure


def replay_episode_on_target(
    env: Any,
    episode: EpisodeData | None,
    source_adapter: EmbodimentAdapter | None,
    target_adapter: EmbodimentAdapter,
    remap_passthrough: PassthroughRemapper | None,
    success_term: TerminationTermCfg | None,
    robot_asset_name: str,
    target_default_state: dict | None,
    prep: dict | None = None,
    num_interpolation_steps: int = 0,
    init_robot_from_ik: bool = False,
    replay_speed: float = 1.0,
    success_settle_steps: int = 0,
    segment_settle_steps: int = 0,
    max_eef_linear_velocity: float | None = None,
    max_eef_rotation_speed: float | None = None,
    synchronization: list | None = None,
    stop_early_on_failure: bool = False,
    max_translation_error: float | None = None,
    max_rotation_error: float | None = None,
    settle_pos_tol_m: float = _SETTLE_POS_TOL_M,
    settle_rot_tol_deg: float = _SETTLE_ROT_TOL_DEG,
    settle_joint_tol: float = _JOINT_SETTLE_TOL,
    retarget_frame: str = "world",
    scene_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    eef_offsets: dict[str, torch.Tensor] | None = None,
    eef_name_map: dict[str, str] | None = None,
    reference_pose: str = "eef_pose",
    subtasks: dict | None = None,
    default_object_tracking: DefaultObjectTracking | None = None,
    write_datagen_info: bool = False,
    eef_reference_link: dict[str, str] | str | None = None,
    source_hand_postures: dict[str, dict[str, list[float]]] | None = None,
) -> tuple[bool, dict[str, dict[str, torch.Tensor]], torch.Tensor | None]:
    """Replay one source episode on the target embodiment and record the target rollout.

    Args:
        env: The target env (with a recorder manager) to replay in.
        episode: The source episode to retarget.
        source_adapter: Adapter used to extract the source passthrough (gripper) actions.
        target_adapter: Adapter used to re-encode each step into the target's action space.
        remap_passthrough: Maps source passthrough actions to the target passthrough layout.
        success_term: Optional success termination term evaluated each step.
        robot_asset_name: Scene key of the target robot articulation.
        target_default_state: Snapshot of the target env's home scene state (captured once from a
            clean reset); the target robot starts each replay from its ``robot_asset_name`` entry.
        num_interpolation_steps: Lead-in steps ramping each EEF from the target's start pose to the
            first trajectory pose (0 disables). Ignored when ``init_robot_from_ik`` is set, since the
            robot then already starts at the first pose.
        replay_speed: Retime the extracted trajectory by ``1 / replay_speed`` waypoints per segment
            (1.0 = unchanged; 0.5 = twice as many waypoints, replayed slower so the IK tracks a longer
            source path more closely). Segment-boundary waypoints are preserved exactly.
        init_robot_from_ik: Start the recorded demo at the IK solution of the first trajectory pose
            (solved kinematically, no physics), so there is no startup transient. Works for any target
            embodiment.

    Returns:
        ``(task_succeeded, eef_errors)``. ``eef_errors`` maps each EEF to ``{"pos": tensor, "rot":
        tensor}`` of per-step tracking errors [m], [deg] between the target robot's achieved EEF pose and
        the (ideal) commanded pose it was driven to (measured at the controlled link when
        ``eef_reference_link`` is set; lead-in steps excluded). ``task_succeeded`` is True if the success
        condition held on any step (or no success term).
    """
    subtasks = subtasks or {}
    default_object_tracking = default_object_tracking or DefaultObjectTracking()
    need_source_objects = any(
        st.object_tracking is not None for entries in subtasks.values() for st in entries
    )
    if prep is None:
        prep = prepare_episode(
            env,
            0,
            episode,
            target_adapter,
            source_adapter,
            remap_passthrough,
            target_default_state,
            robot_asset_name,
            reference_pose,
            eef_name_map,
            replay_speed,
            retarget_frame,
            scene_translation,
            eef_offsets,
            num_interpolation_steps,
            init_robot_from_ik,
            subtasks,
            default_object_tracking,
            source_hand_postures=source_hand_postures,
            need_source_objects=need_source_objects,
            reset_sim=True,
            eef_reference_link=eef_reference_link,
            write_datagen_info=write_datagen_info,
            segment_settle_steps=segment_settle_steps,
            max_eef_linear_velocity=max_eef_linear_velocity,
            max_eef_rotation_speed=max_eef_rotation_speed,
            synchronization=synchronization,
        )
    eef_names = prep["eef_names"]
    commanded_poses = prep["commanded_poses"]
    commanded_passthrough = prep["commanded_passthrough"]
    num_interpolation_steps = prep["num_interpolation_steps"]
    num_steps = prep["num_steps"]
    carry_segments = prep["carry_segments"]
    source_objects = prep["source_objects"]
    source_signals = prep["source_signals"]  # {name: (num_traj_steps, 1)} to forward, or {}
    segment_ends = prep["segment_ends"]  # {eef: {trajectory step: settle-hold cap}} for the motion-aware settle
    sync_of = prep.get("sync_of", {})  # {eef: {end step: group id}} -- a cross-EEF rendezvous barrier
    group_members = prep.get("group_members", {})  # {group id: [(eef, end step)]}

    # Where the achieved pose is read for the tracking error. When the reference is reconstructed at the
    # IK-controlled link (eef_reference_link), the command drives that link, so the achieved must be read
    # there too; reading the observed EEF frame would add the fixed observed->controlled offset (e.g.
    # GR1's ~20 deg wrist_pitch: the angle between the hand and the forearm) and inflate the error even
    # when the hand is aligned. Otherwise (no reconstruction) the observed frame is the reference frame.
    controlled_reader = (
        _build_controlled_link_pose_reader(env, target_adapter) if eef_reference_link is not None else None
    )
    if eef_reference_link is not None and controlled_reader is None:
        print("\t  (warning: eef_reference_link set but no PinkIK controlled link found; using observed frame)")

    def read_achieved() -> dict[str, torch.Tensor]:
        return read_achieved_eef_poses(target_adapter, controlled_reader, 0)

    task_succeeded = success_term is None
    pos_errors: dict[str, list[float]] = {eef_name: [] for eef_name in eef_names}
    rot_errors: dict[str, list[float]] = {eef_name: [] for eef_name in eef_names}

    # Single-step per-EEF scheduler (replaces the nested ``_settle_at_pose`` loop): exactly one ``env.step``
    # per tick, each EEF advancing its OWN pointer through its (possibly unequal-length) commanded sequence.
    # A subtask boundary is an inline HOLD -- the EEF repeats its pose and each tick checks its motion-aware
    # settle (arm+gripper below tolerance), advancing once settled or at the cap; cross-EEF sync extends the
    # same hold (M3). Pose / passthrough / object override are (re)computed only for EEFs that just arrived at
    # a new waypoint, so a HOLDING EEF keeps its pose (re-running the override mid-release would remeasure the
    # grasp and corrupt it). Aligned equal-length arms move in lockstep -> reproduces the pre-refactor path.
    lengths = {eef_name: commanded_poses[eef_name].shape[0] for eef_name in eef_names}
    ptr = {eef_name: 0 for eef_name in eef_names}
    hold = {eef_name: 0 for eef_name in eef_names}  # frames held so far at the current segment end
    prev_pose: dict[str, torch.Tensor | None] = {eef_name: None for eef_name in eef_names}
    prev_ptr = {eef_name: -1 for eef_name in eef_names}
    prev_qpos = None
    # Passthrough channels named after an EEF (the de-interleaved hands) advance with that EEF; any extra
    # non-EEF channel (e.g. a mobile base) follows the first EEF's clock.
    channel_eef = {name: name if name in eef_names else eef_names[0] for name in commanded_passthrough}
    target_eef_pose_dict: dict[str, torch.Tensor] = {}
    passthrough_action_dict: dict[str, torch.Tensor] = {}
    record_signals = None
    # When ``write_datagen_info`` (copy), record the full ``obs/datagen_info`` (poses + signals) each step so
    # the output is a drop-in ``generate_dataset.py`` source. The scene's rigid objects (for ``object_pose``).
    object_names = list(env.scene.rigid_objects.keys()) if write_datagen_info else []
    tick = 0
    early_failure = False
    monitor_early = stop_early_on_failure and (max_translation_error is not None or max_rotation_error is not None)
    while any(ptr[eef_name] < lengths[eef_name] for eef_name in eef_names):
        idx = {eef_name: min(ptr[eef_name], lengths[eef_name] - 1) for eef_name in eef_names}
        traj_step = {eef_name: idx[eef_name] - num_interpolation_steps for eef_name in eef_names}
        advanced = {eef_name for eef_name in eef_names if ptr[eef_name] != prev_ptr[eef_name]}

        for eef_name in advanced:
            target_eef_pose_dict[eef_name] = commanded_poses[eef_name][idx[eef_name]]
            prev_ptr[eef_name] = ptr[eef_name]
        for name, tensor in commanded_passthrough.items():
            if channel_eef[name] in advanced or name not in passthrough_action_dict:
                passthrough_action_dict[name] = tensor[idx[channel_eef[name]]]
        # Object-centric override for the EEFs that just advanced (a holding carrier keeps its pose).
        if carry_segments and advanced:
            _apply_object_centric_override(
                carry_segments,
                traj_step,
                target_eef_pose_dict,
                source_objects,
                target_adapter,
                env,
                controlled_reader=controlled_reader,
                commanded_poses=commanded_poses,
                num_interpolation_steps=num_interpolation_steps,
                eefs=advanced,
            )
        ref_ts = max(traj_step.values())  # shared clock for signals only
        signal_frame = {name: sig[min(max(ref_ts, 0), sig.shape[0] - 1)] for name, sig in source_signals.items()}
        record_signals = (lambda sf=signal_frame: _record_signal_frame(env, 0, sf)) if signal_frame else None

        action = target_adapter.target_eef_pose_to_action(
            target_eef_pose_dict=target_eef_pose_dict,
            passthrough_action_dict=passthrough_action_dict,
            env_id=0,
        )
        env.step(action.reshape(1, -1).to(device=env.device))
        if record_signals is not None:
            record_signals()
        if write_datagen_info:
            _record_datagen_poses(env, 0, target_adapter, target_eef_pose_dict, object_names)
        if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
            task_succeeded = True

        # Record per-EEF tracking error (real, unfinished steps only) and test the early-abort thresholds.
        achieved_poses = read_achieved()
        early_failure = _score_step(
            eef_names,
            traj_step,
            ptr,
            lengths,
            target_eef_pose_dict,
            achieved_poses,
            pos_errors,
            rot_errors,
            monitor_early,
            carry_segments,
            source_objects,
            env,
            0,
            max_translation_error,
            max_rotation_error,
            tick,
        )
        if early_failure:
            break

        # Advance each EEF: a mid-segment step advances by one; at a segment end (a per-EEF boundary cap)
        # HOLD until the arm+gripper settle or the cap is hit.
        curr_poses = target_adapter.get_eef_poses(env_ids=[0])
        curr_qpos = as_torch(env.scene[robot_asset_name].data.joint_pos)[0] if robot_asset_name else None
        joint_moved = (
            prev_qpos is not None
            and curr_qpos is not None
            and float(torch.max(torch.abs(curr_qpos - prev_qpos))) > settle_joint_tol
        )
        for eef_name in eef_names:
            if ptr[eef_name] >= lengths[eef_name]:
                continue
            cap = segment_ends.get(eef_name, {}).get(traj_step[eef_name], 0)
            gid = sync_of.get(eef_name, {}).get(traj_step[eef_name])  # a sync barrier at this step, or None
            if cap > 0 or gid is not None:  # a hold point: motion-aware settle and/or a cross-EEF rendezvous
                hold[eef_name] += 1
                settle_ok = True
                if cap > 0:
                    moved = joint_moved
                    if prev_pose[eef_name] is not None:
                        dpos, drot = pose_tracking_error(prev_pose[eef_name], curr_poses[eef_name][0])
                        moved = moved or dpos > settle_pos_tol_m or drot > settle_rot_tol_deg
                    settle_ok = (not moved) or hold[eef_name] >= cap
                # Rendezvous: hold until every group member has reached its own barrier (concluded). ``ptr``
                # is in commanded-index space (lead-in included); the barrier step is a trajectory step.
                sync_ok = gid is None or all(
                    ptr[m_eef] - num_interpolation_steps >= m_step for m_eef, m_step in group_members[gid]
                )
                if settle_ok and sync_ok:
                    ptr[eef_name] += 1
                    hold[eef_name] = 0
            else:
                ptr[eef_name] += 1
            prev_pose[eef_name] = curr_poses[eef_name][0]
        prev_qpos = curr_qpos.clone() if curr_qpos is not None else None
        tick += 1

    # Final success settle: after the last waypoint, hold the final commanded pose (which carries the
    # final gripper *release*) for up to success_settle_steps, re-checking success each step. Unlike the
    # per-waypoint/segment settle this deliberately does NOT stop on arm-settle: a suction cup needs
    # several steps to reach fully-open (state -1) after release, and the cubes may still be settling, so
    # cubes_stacked can only turn True a few steps after the arm has already stopped. Without this a
    # perfectly stacked demo reads as failure because the gripper-open half of the success term is not
    # yet satisfied when success is sampled.
    if (
        not task_succeeded
        and not early_failure
        and success_settle_steps > 0
        and num_steps > 0
        and success_term is not None
    ):
        for _ in range(success_settle_steps):
            action = target_adapter.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                passthrough_action_dict=passthrough_action_dict,
                env_id=0,
            )
            env.step(action.reshape(1, -1).to(device=env.device))
            if record_signals is not None:  # hold the last waypoint's signal across the success settle
                record_signals()
            if write_datagen_info:
                _record_datagen_poses(env, 0, target_adapter, target_eef_pose_dict, object_names)
            if bool(success_term.func(env, **success_term.params)[0]):
                task_succeeded = True
                break

    eef_errors = {
        eef_name: {"pos": torch.tensor(pos_errors[eef_name]), "rot": torch.tensor(rot_errors[eef_name])}
        for eef_name in eef_names
    }
    return task_succeeded, eef_errors


def _build_sync_barriers(
    synchronization: list[list[str]], name_end_step: dict[str, tuple[str, int]]
) -> tuple[dict[str, dict[int, int]], dict[int, list[tuple[str, int]]]]:
    """Turn the descriptor's sync groups into the executor's per-EEF barrier lookups.

    ``name_end_step`` maps each named subtask to ``(eef, end_trajectory_step)`` -- the step where that
    subtask concludes. Returns ``(sync_of, group_members)`` where ``sync_of[eef][end_step]`` is the group
    id an EEF must rendezvous at when it reaches that step, and ``group_members[gid]`` lists every
    ``(eef, end_step)`` in the group. At runtime an EEF holds at ``end_step`` until every member has reached
    its own ``end_step`` (all ``ptr[m_eef] >= m_step``), then all advance -- the hand-off join.
    """
    sync_of: dict[str, dict[int, int]] = {}
    group_members: dict[int, list[tuple[str, int]]] = {}
    for gid, group in enumerate(synchronization):
        for name in group:
            if name not in name_end_step:  # a named subtask that produced no segment this run -- skip
                continue
            eef, end_step = name_end_step[name]
            sync_of.setdefault(eef, {})[end_step] = gid
            group_members.setdefault(gid, []).append((eef, end_step))
    return sync_of, group_members


def format_eef_errors(eef_errors: dict[str, dict[str, torch.Tensor]]) -> str:
    """One line per EEF summarizing position/orientation tracking error and the matched rate."""
    lines: list[str] = []
    for eef_name, errors in eef_errors.items():
        pos = errors["pos"]
        rot = errors["rot"]
        matched = ((pos <= _IK_POS_TOL_M) & (rot <= _IK_ROT_TOL_DEG)).float().mean().item() * 100.0
        lines.append(
            f"\t  {eef_name}: pos err mean {pos.mean() * 100:.1f} / max {pos.max() * 100:.1f} cm | "
            f"rot err mean {rot.mean():.1f} / max {rot.max():.1f} deg | "
            f"matched (<{int(_IK_POS_TOL_M * 100)}cm,<{int(_IK_ROT_TOL_DEG)}deg) {matched:.0f}%"
        )
    return "\n".join(lines)
