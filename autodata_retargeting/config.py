# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task/pair retargeting configuration (the retarget-descriptor YAML schema).

The descriptor is now self-contained: it carries its own task structure (a per-EEF list of
:class:`Subtask`) instead of referencing an external task descriptor. Each subtask declares how it ends
(:class:`SubtaskEnd` -- a gripper or signal event, offsettable) and, optionally, which object the EEF
tracks over it (:class:`SubtaskObjectTracking`), so segmentation and object-centric planning come
straight from this file. Retargeting replays each recorded source trajectory 1:1 onto the target robot.
"""

import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DefaultObjectTracking:
    """Default object-tracking interpolation parameters, overridable per :class:`SubtaskObjectTracking`.

    When a subtask tracks an object, the commanded pose eases between the source EEF path and the
    object-centric pose: over ``interpolation_step_start`` steps at the subtask's start (EEF path ->
    object) and ``interpolation_step_after`` steps past its end (object -> back onto the EEF path).
    0 = switch immediately.
    """

    interpolation_step_start: int = 0
    interpolation_step_after: int = 0

    @classmethod
    def parse(cls, value: "dict | DefaultObjectTracking | None") -> "DefaultObjectTracking":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        assert isinstance(value, dict), f"default_object_tracking must be a section (dict), got {type(value).__name__}."
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        assert not unknown, f"unknown default_object_tracking keys {sorted(unknown)}; allowed: {sorted(allowed)}."
        return cls(**value)


@dataclass
class SubtaskEnd:
    """When a subtask ends: an event detected on the source demo, optionally offset.

    ``method``:
      * ``gripper_close`` / ``gripper_open`` -- the step ``eef``'s gripper crosses closed / open.
      * ``gripper_closing`` / ``gripper_opening`` -- same crossing, but backed up to the frame the gripper
        *begins* to move (the leading edge). For a handoff, end the releaser at ``gripper_opening`` so it
        holds the object until the barrier releases, instead of the fragile ``gripper_open`` + ``offset: -1``.
      * ``signal_on`` / ``signal_off`` -- the rising / falling edge of subtask-term signal ``signal``.
      * ``fixed_length`` -- a fixed ``length`` frames after the subtask's start (no source event).

    ``eef`` selects which gripper the event watches (default: the EEF this subtask is listed under).
    ``length`` (``fixed_length`` only) is the subtask duration in source frames. ``offset`` shifts the
    boundary: negative ends the subtask that many steps *before* the trigger, positive prolongs it that
    many steps *after*. ``settle_steps`` overrides the run-wide ``segment_settle_steps`` cap for the
    motion-aware hold at *this* boundary (None = use the config default; 0 = no hold here).
    """

    method: str
    signal: str | None = None
    eef: str | None = None
    offset: int = 0
    settle_steps: int | None = None
    length: int | None = None

    _SIGNAL_METHODS = ("signal_on", "signal_off")
    _GRIPPER_METHODS = ("gripper_open", "gripper_close", "gripper_opening", "gripper_closing")
    _LENGTH_METHODS = ("fixed_length",)

    def __post_init__(self) -> None:
        allowed = self._SIGNAL_METHODS + self._GRIPPER_METHODS + self._LENGTH_METHODS
        assert self.method in allowed, f"subtask_end method must be one of {allowed}, got {self.method!r}."
        if self.method in self._SIGNAL_METHODS:
            assert self.signal, f"subtask_end method {self.method!r} needs a 'signal' name."
        if self.method in self._LENGTH_METHODS:
            assert (
                self.length is not None and self.length > 0
            ), f"subtask_end method {self.method!r} needs a positive 'length' (frames), got {self.length!r}."

    @classmethod
    def parse(cls, value: "str | dict | SubtaskEnd") -> "SubtaskEnd":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(method=value)
        assert isinstance(value, dict), f"subtask_end must be a string or a section (dict), got {type(value).__name__}."
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        assert not unknown, f"unknown subtask_end keys {sorted(unknown)}; allowed: {sorted(allowed)}."
        return cls(**value)


@dataclass
class SubtaskObjectTracking:
    """Which object an EEF tracks over a subtask (rigid ``eef_T_object``), with optional interp overrides.

    Written as the bare object name (``object_tracking: cube_2``) or a section
    (``{object: cube_2, interpolation_step_start: 10}``). ``interpolation_step_*`` default to
    :class:`DefaultObjectTracking` when unset.
    """

    object: str
    interpolation_step_start: int | None = None
    interpolation_step_after: int | None = None

    @classmethod
    def parse(cls, value: "str | dict | SubtaskObjectTracking | None") -> "SubtaskObjectTracking | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(object=value)
        assert isinstance(
            value, dict
        ), f"object_tracking must be a string or a section (dict), got {type(value).__name__}."
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        assert not unknown, f"unknown object_tracking keys {sorted(unknown)}; allowed: {sorted(allowed)}."
        assert "object" in value, "object_tracking section needs an 'object' field."
        return cls(**value)


@dataclass
class Offset:
    """A per-subtask SE(3) offset applied to the commanded EEF pose over the subtask's span.

    ``frame`` selects the frame the offset acts in; it defaults to the subtask's own reference frame
    (``object_ref`` -> that tracked object's frame, else ``frame_ref``; ``world``/``eef`` are always
    available -- ``eef`` nudges in the gripper's own frame). ``translation`` [m] and ``axis_angle``
    [rad] give the offset transform (either may be omitted -> that part is identity).
    ``interpolation_start`` / ``interpolation_end`` ramp the offset in linearly over the first N steps of
    the span and out over the last N steps, so it can be eased in/out (0 = applied fully across the span).
    """

    frame: str | None = None
    translation: list | None = None
    axis_angle: list | None = None
    interpolation_start: int = 0
    interpolation_end: int = 0

    def __post_init__(self) -> None:
        for name in ("translation", "axis_angle"):
            value = getattr(self, name)
            if value is not None:
                assert (
                    isinstance(value, list | tuple) and len(value) == 3
                ), f"offset.{name} must be a 3-vector [x, y, z], got {value!r}."
                setattr(self, name, [float(v) for v in value])
        self.interpolation_start = int(self.interpolation_start)
        self.interpolation_end = int(self.interpolation_end)
        assert (
            self.interpolation_start >= 0 and self.interpolation_end >= 0
        ), f"offset interpolation_start/end must be >= 0, got {self.interpolation_start}/{self.interpolation_end}."

    @classmethod
    def parse(cls, value: Any) -> "Offset | None":
        if value is None:
            return None
        assert isinstance(value, dict), f"offset must be a section (dict), got {type(value).__name__}."
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        assert not unknown, f"unknown offset keys {sorted(unknown)}; allowed: {sorted(allowed)}."
        return cls(**value)


@dataclass
class Subtask:
    """One subtask in an EEF's sequence.

    ``name`` (optional) is a stable identifier for the subtask; when given it must be unique across all
    subtasks of all EEFs. ``object_ref`` / ``frame_ref`` name the object or frame the subtask is planned
    relative to; they default the reference frame a per-subtask :class:`Offset` acts in. ``subtask_end`` is
    required except on the last subtask of an EEF (which runs to the end of the trajectory).
    ``object_tracking`` (optional) makes the EEF track an object over the whole subtask instead of
    following the source EEF path.
    """

    name: str | None = None
    object_ref: str | None = None
    frame_ref: str | None = None
    description: str = ""
    subtask_end: SubtaskEnd | None = None
    object_tracking: SubtaskObjectTracking | None = None
    # Per-subtask SE(3) pose offset applied to this EEF's commanded trajectory over the subtask's span
    # (see :class:`Offset`). None = no offset.
    offset: "Offset | None" = None

    @classmethod
    def parse(cls, value: dict) -> "Subtask":
        assert isinstance(value, dict), f"each subtask must be a section (dict), got {type(value).__name__}."
        data = dict(value)
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        assert not unknown, f"unknown subtask keys {sorted(unknown)}; allowed: {sorted(allowed)}."
        if data.get("subtask_end") is not None:
            data["subtask_end"] = SubtaskEnd.parse(data["subtask_end"])
        data["object_tracking"] = SubtaskObjectTracking.parse(data.get("object_tracking"))
        data["offset"] = Offset.parse(data.get("offset"))
        return cls(**data)


@dataclass
class RetargetConfig:
    """Self-contained task/pair retargeting parameters, loaded from a retarget-descriptor YAML.

    Bundles what defines a source->target retarget for one task (both embodiments, the target env, the
    per-EEF ``subtasks``) plus every retargeting knob, so a run is one ``--retarget_config`` file instead
    of a long flag list. Runtime/IO (input/output files, keep_failed, diagnostics, device) stay on the CLI.
    """

    source_embodiment: str
    target_embodiment: str
    target_env_name: str
    name: str = "retarget"
    description: str = ""
    hand_policy: str = "passthrough"
    hand_interp_norm: str = "l1"
    hand_binary_close_threshold: float = 0.5
    # For hand_policy "interpolation": remap the source closedness fraction so this ``[lo, hi]`` band
    # spans the target's full open->close travel, clamping outside it. Lets a source grasp with a dead
    # band near its own open/close still drive the target hand fully. None disables (identity).
    hand_interp_band: tuple[float, float] | None = None
    # For hand_policy "joint_mapping": ``{target_joint_idx: [source_joint_idx, ...]}`` wiring each target
    # hand joint to the average of a group of source hand joints (indices into each embodiment's per-EEF
    # hand vector, i.e. gripper_action_indices order), mapped through open/close. Target joints not listed
    # stay at hand_open.
    joint_mapping: dict[int, list[int]] | None = None
    num_interpolation_steps: int = 0
    init_robot_from_ik: bool = False
    replay_speed: float = 1.0
    # Cap the commanded EEF speed by subdividing the trajectory wherever a step moves
    # faster than the limit (extra waypoints are interpolated in). Keeps a reactive controller (e.g. Galbot's
    # RmpFlow) from being handed a command it cannot physically follow -- a differential-IK robot snaps to
    # each waypoint in one step and is unaffected, so leave these ``None`` there. Measured per replayed
    # waypoint on the grasp-frame path: ``max_eef_linear_velocity`` [m/step], ``max_eef_rotation_speed``
    # [deg/step]. ``None`` disables that axis' cap.
    max_eef_linear_velocity: float | None = None
    max_eef_rotation_speed: float | None = None
    # Early-abort a doomed replay to save sim time. When ``stop_early_on_failure`` is set, the replay stops
    # the moment the monitored achieved pose misses its target beyond ``max_translation_error`` [m] or
    # ``max_rotation_error`` [deg] -- the tracked OBJECT vs its source path during a carry, else the EEF vs
    # its commanded target -- and the run is recorded as a failure (success never fired). ``None`` thresholds
    # are not checked. Speeds up generation by not replaying past an unrecoverable miss.
    stop_early_on_failure: bool = False
    max_translation_error: float | None = None
    max_rotation_error: float | None = None
    # At each subtask boundary the replayer holds the pose until the robot's joints (arm AND gripper) stop
    # moving, up to this many extra sim steps -- so a slow gripper (e.g. the Robotiq 2F-85 linkage) finishes
    # closing/opening before the arm moves on. Motion-aware: a fast gripper exits in ~1-2 steps. 0 = no hold.
    # A per-subtask ``SubtaskEnd.settle_steps`` overrides this cap for that boundary.
    segment_settle_steps: int = 30
    # Tolerances for the motion-aware boundary hold -- it exits early once motion falls below all of these.
    # ``settle_joint_tol`` (primary): max per-step joint position change [rad or m]; robust to the Robotiq
    # linkage's saturated velocity reading. ``settle_pos_tol_m`` / ``settle_rot_tol_deg``: max per-step EEF
    # pose change [m] / [deg] (the arm has reached the waypoint).
    settle_joint_tol: float = 0.003
    settle_pos_tol_m: float = 0.01
    settle_rot_tol_deg: float = 2.5
    # Hold the final pose this many steps once the success term fires (re-checking success). From the old
    # task descriptor's generation policy; re-homed here now that the descriptor is self-contained.
    success_settle_steps: int = 0
    # Seed for the reproducible scene resets (was the task descriptor's generation-policy seed).
    seed: int = 1
    retarget_frame: str = "world"
    scene_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    eef_offsets: dict[str, dict] | None = None
    target_channel_defaults: dict[str, list[float]] | None = None
    eef_name_map: dict[str, str] | None = None
    # Which source-demo trajectory drives the replay (and is scored against):
    #   "eef_pose"        -- the source robot's *achieved* (executed, task-successful) path (default).
    #   "target_eef_pose" -- the *commanded* controller targets (the ideal the source may never have
    #                        reached; a more-capable target robot can overshoot toward it).
    reference_pose: str = "eef_pose"
    # Default object-tracking interpolation, overridable per subtask (see DefaultObjectTracking).
    default_object_tracking: "dict | DefaultObjectTracking" = field(default_factory=DefaultObjectTracking)
    # Per-EEF subtask lists (keyed by *source* EEF name; eef_name_map renames onto the target). Each entry
    # is parsed into a Subtask. Segmentation and object tracking come from these.
    subtasks: dict[str, list[Subtask]] = field(default_factory=dict)
    # Bimanual/multi-arm synchronization: a list of barrier groups, each a list of subtask *names* that must
    # conclude together. Every EEF that reaches its named segment holds at that pose until all segments in
    # the group have concluded (a join). Rules: a group's names belong to different EEFs; each subtask is in
    # at most one group; and groups are declared in execution order -- each EEF's participating segments
    # appear in its temporal subtask order (a simple check that also guarantees deadlock-freedom). Empty for
    # single-arm / uncoordinated runs.
    synchronization: list[list[str]] = field(default_factory=list)
    # Write the full ``obs/datagen_info`` into the retargeted dataset so the output is a drop-in source
    # for ``generate_dataset.py`` (no separate ``annotate_demos`` pass): observed ``eef_pose``, commanded
    # ``target_eef_pose`` and per-object ``object_pose`` (read live each step, matching annotate_demos'
    # recorder), plus the forwarded ``subtask_term_signals`` (resampled to the replay trajectory and held
    # across each waypoint's frames -- lead-in, settle, replay_speed retiming -- so boundaries line up with
    # the recorded rollout). Default off.
    write_datagen_info: bool = False
    # Reconstruct the "eef_pose" reference at each EEF's IK-controlled link instead of the source env's
    # *observed* link. Some embodiments observe the EEF a joint short of the link the IK drives (e.g. GR1
    # observes ``hand_roll_link`` but controls ``hand_pitch_link``), so the observed ``eef_pose`` drops
    # that joint's rotation. When set, the reference (and the object-tracking grasp frame) is rebuilt
    # at the controlled link from the recorded per-step link states. Values: ``None`` (off, default),
    # ``"controlled"`` to find the controlled link by matching the source's own ``target_eef_pose``
    # (needs no body names -> works cross-embodiment, e.g. GR1->G1), or a ``{eef_name: link_name}`` map as
    # a same-embodiment override. Only affects ``reference_pose="eef_pose"``.
    eef_reference_link: dict[str, str] | str | None = None

    def __post_init__(self) -> None:
        self.default_object_tracking = DefaultObjectTracking.parse(self.default_object_tracking)
        self.subtasks = {
            eef: [st if isinstance(st, Subtask) else Subtask.parse(st) for st in entries]
            for eef, entries in (self.subtasks or {}).items()
        }
        names = [st.name for entries in self.subtasks.values() for st in entries if st.name is not None]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert not duplicates, f"subtask names must be unique across all subtasks; duplicates: {duplicates}"

        by_name = {
            st.name: (eef, i)
            for eef, entries in self.subtasks.items()
            for i, st in enumerate(entries)
            if st.name is not None
        }
        # Synchronization barriers: names must exist; a group's members belong to different EEFs; each name
        # is in at most one group; and the declaration order is a valid schedule -- for each EEF, the groups
        # it joins (in declaration order) reference its subtasks in temporal order. That last check both
        # keeps authoring intuitive (list groups as they happen) and guarantees the barriers are acyclic.
        group_of: dict[str, int] = {}
        eef_join_order: dict[str, list[tuple[int, int]]] = {}  # eef -> [(group_index, subtask_index)]
        for g_idx, group in enumerate(self.synchronization):
            assert len(group) >= 2, f"synchronization group {g_idx} needs >= 2 subtask names, got {group}."
            group_eefs: set[str] = set()
            for nm in group:
                assert nm in by_name, f"synchronization group {g_idx} references unknown subtask name {nm!r}."
                assert nm not in group_of, (
                    f"subtask {nm!r} is in synchronization groups {group_of[nm]} and {g_idx}; "
                    "each subtask may be in at most one group."
                )
                group_of[nm] = g_idx
                eef, s_i = by_name[nm]
                assert eef not in group_eefs, (
                    f"synchronization group {g_idx} lists two subtasks of EEF {eef!r}; a group's members "
                    "must belong to different EEFs."
                )
                group_eefs.add(eef)
                eef_join_order.setdefault(eef, []).append((g_idx, s_i))
        for eef, joins in eef_join_order.items():
            s_indices = [s_i for _, s_i in sorted(joins)]  # subtask indices in group-declaration order
            assert s_indices == sorted(s_indices) and len(set(s_indices)) == len(s_indices), (
                f"EEF {eef!r} joins synchronization groups out of temporal order; declare groups in the "
                "order their subtasks execute."
            )

    @classmethod
    def from_yaml(cls, path: str) -> "RetargetConfig":
        """Load a retarget descriptor; the embodiment paths resolve relative to the YAML's dir."""
        config_dir = Path(path).resolve().parent
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        def _resolve(ref: str) -> str:
            ref_path = Path(ref)
            return str(ref_path if ref_path.is_absolute() else (config_dir / ref_path).resolve())

        return cls(
            source_embodiment=_resolve(data["source_embodiment"]),
            target_embodiment=_resolve(data["target_embodiment"]),
            target_env_name=data["target_env_name"],
            name=data.get("name", "retarget"),
            description=data.get("description", ""),
            hand_policy=data.get("hand_policy", "passthrough"),
            hand_interp_norm=data.get("hand_interp_norm", "l1"),
            hand_binary_close_threshold=float(data.get("hand_binary_close_threshold", 0.5)),
            hand_interp_band=(tuple(data["hand_interp_band"]) if data.get("hand_interp_band") is not None else None),
            joint_mapping=data.get("joint_mapping"),
            num_interpolation_steps=int(data.get("num_interpolation_steps", 0)),
            init_robot_from_ik=bool(data.get("init_robot_from_ik", False)),
            replay_speed=float(data.get("replay_speed", 1.0)),
            max_eef_linear_velocity=(
                None if data.get("max_eef_linear_velocity") is None else float(data["max_eef_linear_velocity"])
            ),
            max_eef_rotation_speed=(
                None if data.get("max_eef_rotation_speed") is None else float(data["max_eef_rotation_speed"])
            ),
            stop_early_on_failure=bool(data.get("stop_early_on_failure", False)),
            max_translation_error=(
                None if data.get("max_translation_error") is None else float(data["max_translation_error"])
            ),
            max_rotation_error=(None if data.get("max_rotation_error") is None else float(data["max_rotation_error"])),
            segment_settle_steps=int(data.get("segment_settle_steps", 30)),
            settle_joint_tol=float(data.get("settle_joint_tol", 0.003)),
            settle_pos_tol_m=float(data.get("settle_pos_tol_m", 0.01)),
            settle_rot_tol_deg=float(data.get("settle_rot_tol_deg", 2.5)),
            success_settle_steps=int(data.get("success_settle_steps", 0)),
            seed=int(data.get("seed", 1)),
            retarget_frame=data.get("retarget_frame", "world"),
            scene_translation=tuple(data.get("scene_translation", (0.0, 0.0, 0.0))),
            eef_offsets=data.get("eef_offsets"),
            target_channel_defaults=data.get("target_channel_defaults"),
            eef_name_map=data.get("eef_name_map"),
            reference_pose=data.get("reference_pose", "eef_pose"),
            default_object_tracking=data.get("default_object_tracking"),
            subtasks=data.get("subtasks", {}),
            synchronization=[list(group) for group in data.get("synchronization", [])],
            write_datagen_info=bool(data.get("write_datagen_info", False)),
            eef_reference_link=data.get("eef_reference_link"),
        )
