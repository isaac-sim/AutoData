# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed lowering of custream and custream2 commands into AutoData plan IR."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from isaac_autodata_core.autonomous.dense_trace import DensePlanTrace
from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    ConcurrentGroupSegment,
    DetachIntentSegment,
    GripperCommandMode,
    GripperCommandSegment,
    JointTrajectorySegment,
    TaskMotionPlan,
    TaskMotionSegment,
    WaitSegment,
    make_stable_id,
    matrix4,
)
from isaac_autodata_interfaces.autonomous.schedulestream.command_types import (
    MalformedScheduleStreamCommandError,
    ScheduleStreamClosedError,
    ScheduleStreamCommandSymbols,
    ScheduleStreamLimitError,
    ScheduleStreamLoweringContext,
    ScheduleStreamLoweringLimits,
    ScheduleStreamProviderError,
    ScheduleStreamTimingError,
    UnsupportedScheduleStreamCommandError,
)
from isaac_autodata_interfaces.autonomous.schedulestream.symbols import (
    load_schedulestream_command_symbols,
    native_type_name,
)

_Selector = Callable[[str, Any | None], Any]
_SymbolProvider = Callable[[str], ScheduleStreamCommandSymbols]
_ResourceCloser = Callable[[Any], None]


@dataclass
class _Budget:
    limits: ScheduleStreamLoweringLimits
    command_nodes: int = 0
    total_samples: int = 0
    source_duration_s: float = 0.0
    has_concurrency: bool = False

    def charge_node(self, path: tuple[int, ...], depth: int) -> None:
        if depth > self.limits.max_nesting_depth:
            raise ScheduleStreamLimitError(f"command nesting exceeds {self.limits.max_nesting_depth}", path=path)
        self.command_nodes += 1
        if self.command_nodes > self.limits.max_command_nodes:
            raise ScheduleStreamLimitError(f"command stream exceeds {self.limits.max_command_nodes} nodes", path=path)

    def charge_samples(self, count: int, path: tuple[int, ...]) -> None:
        if count < 1:
            raise MalformedScheduleStreamCommandError("command contains no samples", path=path)
        if count > self.limits.max_samples_per_segment:
            raise ScheduleStreamLimitError(
                f"segment has {count} samples; limit is {self.limits.max_samples_per_segment}",
                path=path,
            )
        self.total_samples += count
        if self.total_samples > self.limits.max_total_samples:
            raise ScheduleStreamLimitError(
                f"command stream exceeds {self.limits.max_total_samples} total samples", path=path
            )

    def charge_duration(self, duration_s: float, path: tuple[int, ...]) -> None:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ScheduleStreamTimingError("derived duration must be finite and non-negative", path=path)
        self.source_duration_s += duration_s
        if self.source_duration_s > self.limits.max_duration_s:
            raise ScheduleStreamLimitError(
                f"source command duration exceeds {self.limits.max_duration_s:g} seconds", path=path
            )


@dataclass
class _State:
    context: ScheduleStreamLoweringContext
    symbols: ScheduleStreamCommandSymbols
    budget: _Budget
    segments: list[TaskMotionSegment] = field(default_factory=list)
    traces: list[DensePlanTrace] = field(default_factory=list)
    attachments_by_link: dict[str, str] = field(default_factory=dict)
    gripper_by_eef: dict[str, float] = field(default_factory=dict)

    def branch(self) -> _State:
        return _State(
            context=self.context,
            symbols=self.symbols,
            budget=self.budget,
            segments=self.segments,
            traces=self.traces,
            attachments_by_link=dict(self.attachments_by_link),
            gripper_by_eef=dict(self.gripper_by_eef),
        )


@dataclass(frozen=True)
class _Lowered:
    terminal_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    duration_s: float
    resources: frozenset[str]


@dataclass(frozen=True)
class _JointSamples:
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    dt_s: float
    held_object: str | None


class ScheduleStreamCommandLowerer:
    """Select, lazily load, validate, and lower one ScheduleStream command API.

    Selection delegates to :mod:`isaac_autodata_interfaces.motion_planners.curobo.compat` only
    when the boundary is first used. Native ScheduleStream modules are loaded one step later and
    only for the selected ``custream`` or ``custream2`` application.
    """

    def __init__(
        self,
        requested_motion_backend: str = "auto",
        *,
        capabilities: Any | None = None,
        selector: _Selector | None = None,
        symbols: ScheduleStreamCommandSymbols | None = None,
        symbol_provider: _SymbolProvider | None = None,
        limits: ScheduleStreamLoweringLimits | None = None,
        owned_resource: Any | None = None,
        resource_closer: _ResourceCloser | None = None,
    ) -> None:
        if requested_motion_backend not in ("auto", "curobo_v1", "curobo_v2"):
            raise ValueError("requested_motion_backend must be 'auto', 'curobo_v1', or 'curobo_v2'")
        if symbols is not None and symbol_provider is not None:
            raise ValueError("provide symbols or symbol_provider, not both")
        if owned_resource is not None and resource_closer is None:
            raise ValueError("owned_resource requires an explicit resource_closer")
        self._requested_motion_backend = requested_motion_backend
        self._capabilities = capabilities
        self._selector = selector or _select_backend
        self._symbols = symbols
        self._symbol_provider = symbol_provider or load_schedulestream_command_symbols
        self._limits = limits or ScheduleStreamLoweringLimits()
        self._selection: Any | None = None
        self._owned_resource = owned_resource
        self._resource_closer = resource_closer
        self._closed = False

    @property
    def application(self) -> str:
        """Selected ScheduleStream application name."""

        return self._get_application()

    @property
    def motion_backend(self) -> str:
        """Selected cuRobo backend ID."""

        selection = self._get_selection()
        return _required_selection_text(selection, "motion_backend")

    def dense_trace_from_link_path(
        self,
        command: Any,
        context: ScheduleStreamLoweringContext,
    ) -> DensePlanTrace:
        """Normalize one native link path into a bounded backend-neutral dense trace."""

        self._require_open()
        symbols = self._get_symbols()
        path: tuple[int, ...] = ()
        if type(command) is not symbols.link_path_type:
            raise UnsupportedScheduleStreamCommandError(
                f"expected exact {symbols.link_path_type.__name__}, got {native_type_name(command)}",
                path=path,
            )
        budget = _Budget(self._limits)
        budget.charge_node(path, 0)
        state = _State(
            context=context,
            symbols=symbols,
            budget=budget,
            attachments_by_link=dict(context.initial_attachments_by_link),
        )
        return self._extract_dense_trace(command, path, state)

    def lower(
        self,
        commands: Any,
        context: ScheduleStreamLoweringContext,
    ) -> TaskMotionPlan:
        """Lower a native command stream to a validated :class:`TaskMotionPlan`."""

        self._require_open()
        symbols = self._get_symbols()
        state = _State(
            context=context,
            symbols=symbols,
            budget=_Budget(self._limits),
            attachments_by_link=dict(context.initial_attachments_by_link),
        )
        if isinstance(commands, (list, tuple)):
            lowered = self._lower_sequence(commands, (), (), 0, state, in_composite=False)
        else:
            lowered = self._lower_command(commands, (), (), 0, state, in_composite=False)
        if not lowered.segment_ids:
            raise MalformedScheduleStreamCommandError("command stream produced no executable segments")
        selection = self._get_selection()
        application = self._get_application()
        metadata = dict(context.metadata)
        metadata["schedulestream"] = {
            "application": application,
            "command_nodes": state.budget.command_nodes,
            "dense_trace_count": len(state.traces),
            "has_concurrency": state.budget.has_concurrency,
            "motion_backend": _required_selection_text(selection, "motion_backend"),
            "source_duration_s": state.budget.source_duration_s,
            "total_samples": state.budget.total_samples,
        }
        backend = f"schedulestream_{application}"
        backend_version = context.backend_version or _selection_backend_version(selection)
        return TaskMotionPlan(
            plan_id=make_stable_id(
                "plan",
                context.request_digest,
                context.snapshot_digest,
                backend,
                context.seed,
            ),
            request_digest=context.request_digest,
            snapshot_digest=context.snapshot_digest,
            backend=backend,
            backend_version=backend_version,
            seed=context.seed,
            segments=tuple(state.segments),
            goal=context.goal,
            metadata=metadata,
        )

    def close(self) -> None:
        """Release the explicitly owned native resource exactly once."""

        if self._closed:
            return
        self._closed = True
        resource = self._owned_resource
        closer = self._resource_closer
        self._owned_resource = None
        self._symbols = None
        if resource is not None and closer is not None:
            try:
                closer(resource)
            except Exception as exc:
                message = str(exc).replace("\n", " ")[:500]
                raise ScheduleStreamProviderError(
                    f"failed to close owned ScheduleStream resource ({type(exc).__name__}: {message})"
                ) from exc

    def __enter__(self) -> ScheduleStreamCommandLowerer:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _get_selection(self) -> Any:
        self._require_open()
        if self._selection is None:
            self._selection = self._selector(self._requested_motion_backend, self._capabilities)
            _required_selection_text(self._selection, "motion_backend")
            _required_selection_text(self._selection, "schedulestream_application")
        return self._selection

    def _get_application(self) -> str:
        application = _required_selection_text(self._get_selection(), "schedulestream_application")
        if application not in ("custream", "custream2"):
            raise ScheduleStreamProviderError(
                f"compatibility selector returned unsupported application {application!r}"
            )
        return application

    def _get_symbols(self) -> ScheduleStreamCommandSymbols:
        self._require_open()
        application = self._get_application()
        if self._symbols is None:
            self._symbols = self._symbol_provider(application)
        if not isinstance(self._symbols, ScheduleStreamCommandSymbols):
            raise ScheduleStreamProviderError("symbol provider did not return ScheduleStreamCommandSymbols")
        if self._symbols.application != application:
            raise ScheduleStreamProviderError(
                f"symbol provider returned {self._symbols.application!r} for selected {application!r}"
            )
        return self._symbols

    def _require_open(self) -> None:
        if self._closed:
            raise ScheduleStreamClosedError("ScheduleStream boundary is closed")

    def _lower_sequence(
        self,
        commands: Sequence[Any],
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        depth: int,
        state: _State,
        *,
        in_composite: bool,
    ) -> _Lowered:
        if isinstance(commands, (str, bytes)):
            raise MalformedScheduleStreamCommandError("commands must be a sequence of command objects", path=path)
        if not commands:
            raise MalformedScheduleStreamCommandError("sequential command container must not be empty", path=path)
        if len(commands) > state.budget.limits.max_command_nodes:
            raise ScheduleStreamLimitError("sequential command container is too large", path=path)
        current_dependencies = dependencies
        segment_ids: list[str] = []
        resources: set[str] = set()
        duration_s = 0.0
        index = 0
        while index < len(commands):
            command = commands[index]
            if type(command) is state.symbols.configuration_type:
                run: list[tuple[Any, tuple[int, ...]]] = []
                while index < len(commands) and type(commands[index]) is state.symbols.configuration_type:
                    run.append((commands[index], path + (index,)))
                    index += 1
                lowered = self._lower_configurations(
                    run,
                    current_dependencies,
                    depth,
                    state,
                    in_composite=in_composite,
                )
            else:
                lowered = self._lower_command(
                    command,
                    current_dependencies,
                    path + (index,),
                    depth,
                    state,
                    in_composite=in_composite,
                )
                index += 1
            current_dependencies = lowered.terminal_ids
            segment_ids.extend(lowered.segment_ids)
            resources.update(lowered.resources)
            duration_s += lowered.duration_s
        return _Lowered(current_dependencies, tuple(segment_ids), duration_s, frozenset(resources))

    def _lower_command(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        depth: int,
        state: _State,
        *,
        in_composite: bool,
    ) -> _Lowered:
        state.budget.charge_node(path, depth)
        command_type = type(command)
        symbols = state.symbols
        if command_type is symbols.commands_type:
            children = _native_children(command, path)
            return self._lower_sequence(
                children,
                dependencies,
                path,
                depth + 1,
                state,
                in_composite=in_composite,
            )
        if command_type is symbols.composite_type:
            return self._lower_composite(command, dependencies, path, depth + 1, state)
        if command_type is symbols.configuration_type:
            return self._lower_configurations(
                [(command, path)], dependencies, depth, state, in_composite=in_composite, charged=True
            )
        if command_type is symbols.link_path_type:
            return self._lower_link_path(command, dependencies, path, state)
        if command_type is symbols.trajectory_type:
            return self._lower_trajectory(command, dependencies, path, state)
        if command_type is symbols.open_type:
            return self._lower_gripper(command, dependencies, path, state, GripperCommandMode.OPEN)
        if command_type is symbols.close_type:
            return self._lower_gripper(command, dependencies, path, state, GripperCommandMode.CLOSE)
        if command_type is symbols.attach_type:
            return self._lower_attach(command, dependencies, path, state, in_composite=in_composite)
        if command_type is symbols.detach_type:
            return self._lower_detach(command, dependencies, path, state, in_composite=in_composite)
        supported = sorted({
            symbols.commands_type.__name__,
            symbols.composite_type.__name__,
            symbols.configuration_type.__name__,
            symbols.trajectory_type.__name__,
            symbols.link_path_type.__name__,
            symbols.open_type.__name__,
            symbols.close_type.__name__,
            symbols.attach_type.__name__,
            symbols.detach_type.__name__,
        })
        raise UnsupportedScheduleStreamCommandError(
            f"unsupported exact native type {native_type_name(command)}; supported: {supported}", path=path
        )

    def _lower_configurations(
        self,
        commands: list[tuple[Any, tuple[int, ...]]],
        dependencies: tuple[str, ...],
        depth: int,
        state: _State,
        *,
        in_composite: bool,
        charged: bool = False,
    ) -> _Lowered:
        del in_composite
        samples: list[tuple[_JointSamples, tuple[int, ...]]] = []
        for index, (command, command_path) in enumerate(commands):
            if not charged or index > 0:
                state.budget.charge_node(command_path, depth)
            sample = self._extract_joint_samples(command, command_path, state, configuration=True)
            if len(sample.positions) != 1:
                raise MalformedScheduleStreamCommandError(
                    "Configuration must contain exactly one joint sample", path=command_path
                )
            samples.append((sample, command_path))

        current_dependencies = dependencies
        segment_ids: list[str] = []
        resources: set[str] = set()
        total_duration = 0.0
        index = 0
        while index < len(samples):
            first = samples[index][0]
            end = index + 1
            while end < len(samples):
                candidate = samples[end][0]
                if (
                    candidate.joint_names != first.joint_names
                    or not _same_dt(candidate.dt_s, first.dt_s)
                    or candidate.held_object != first.held_object
                ):
                    break
                end += 1
            group = samples[index:end]
            positions = tuple(item.positions[0] for item, _ in group)
            group_path = group[0][1]
            state.budget.charge_samples(len(group), group_path)
            duration = len(group) * first.dt_s
            state.budget.charge_duration(duration, group_path)
            if len(group) >= 2 and all(position == positions[0] for position in positions[1:]):
                segment_id = make_stable_id(
                    "wait", state.context.request_digest, group_path, len(group), first.joint_names, positions[0]
                )
                segment: TaskMotionSegment = WaitSegment(
                    segment_id=segment_id,
                    depends_on=current_dependencies,
                    duration_s=duration,
                    steps=len(group),
                    metadata={
                        "held_object": first.held_object,
                        "joint_names": list(first.joint_names),
                        "joint_position": list(positions[0]),
                        "source_command": "Configuration",
                        "source_path": list(group_path),
                        "step_dt_s": first.dt_s,
                    },
                )
            else:
                segment_id = make_stable_id(
                    "joint", state.context.request_digest, group_path, first.joint_names, positions
                )
                segment = JointTrajectorySegment(
                    segment_id=segment_id,
                    depends_on=current_dependencies,
                    duration_s=duration,
                    joint_names=first.joint_names,
                    positions=positions,
                    timestamps_s=tuple(sample_index * first.dt_s for sample_index in range(len(group))),
                    attached_object=first.held_object,
                    metadata={
                        "source_command": "Configuration",
                        "source_path": list(group_path),
                        "step_dt_s": first.dt_s,
                    },
                )
            state.segments.append(segment)
            current_dependencies = (segment_id,)
            segment_ids.append(segment_id)
            resources.update(f"joint:{name}" for name in first.joint_names)
            total_duration += duration
            index = end
        return _Lowered(current_dependencies, tuple(segment_ids), total_duration, frozenset(resources))

    def _lower_trajectory(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        state: _State,
    ) -> _Lowered:
        samples = self._extract_joint_samples(command, path, state, configuration=False)
        count = len(samples.positions)
        state.budget.charge_samples(count, path)
        duration = max(0, count - 1) * samples.dt_s
        state.budget.charge_duration(duration, path)
        segment_id = make_stable_id("joint", state.context.request_digest, path, samples.joint_names, samples.positions)
        segment = JointTrajectorySegment(
            segment_id=segment_id,
            depends_on=dependencies,
            duration_s=duration,
            joint_names=samples.joint_names,
            positions=samples.positions,
            timestamps_s=tuple(index * samples.dt_s for index in range(count)),
            attached_object=samples.held_object,
            metadata={
                "source_command": "Trajectory",
                "source_path": list(path),
                "step_dt_s": samples.dt_s,
            },
        )
        state.segments.append(segment)
        resources = {f"joint:{name}" for name in samples.joint_names}
        arm = _optional_command_arm(command, None, path)
        if arm is not None:
            resources.add(f"arm:{arm}")
        return _Lowered((segment_id,), (segment_id,), duration, frozenset(resources))

    def _lower_link_path(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        state: _State,
    ) -> _Lowered:
        trace = self._extract_dense_trace(command, path, state)
        state.traces.append(trace)
        duration = len(trace.poses) * trace.step_dt_s
        state.budget.charge_duration(duration, path)
        link = _optional_text_attribute(command, "link", path)
        eef_name, arm = self._resolve_eef(command, link, path, state.context)
        held_objects = self._held_objects(command, path, state.context)
        segment_id = make_stable_id("cartesian", state.context.request_digest, path, eef_name, trace.poses)
        segment = CartesianTrajectorySegment(
            segment_id=segment_id,
            depends_on=dependencies,
            duration_s=duration,
            eef_name=eef_name,
            frame=trace.frame,
            poses=trace.poses,
            metadata={
                "held_objects": list(held_objects),
                "source_command": state.symbols.link_path_type.__name__,
                "source_link": link,
                "source_path": list(path),
                "step_dt_s": trace.step_dt_s,
            },
        )
        state.segments.append(segment)
        resources = {f"eef:{eef_name}", f"arm:{arm}"}
        return _Lowered((segment_id,), (segment_id,), duration, frozenset(resources))

    def _extract_dense_trace(
        self,
        command: Any,
        path: tuple[int, ...],
        state: _State,
    ) -> DensePlanTrace:
        count = _path_length(command, state.symbols.application, path)
        state.budget.charge_samples(count, path)
        dt_s = _command_dt(command, path, state.context.step_dt_s)
        link = _optional_text_attribute(command, "link", path)
        eef_name, _ = self._resolve_eef(command, link, path, state.context)
        poses = []
        for index in range(count):
            try:
                native_pose = command[index] if state.symbols.application == "custream" else command.pose(index)
                native_matrix = state.symbols.pose_to_matrix(native_pose)
            except Exception as exc:
                raise _malformed_from_exception("failed to read link-path pose", path + (index,), exc) from exc
            matrix_rows = _numeric_matrix(native_matrix, 4, 4, path + (index,))
            try:
                poses.append(matrix4(matrix_rows, f"commands{path}.poses[{index}]"))
            except ValueError as exc:
                raise MalformedScheduleStreamCommandError(str(exc), path=path + (index,)) from exc
        gripper = state.gripper_by_eef.get(eef_name, state.context.initial_gripper_value)
        return DensePlanTrace(
            eef_name=eef_name,
            frame=state.context.frame,
            poses=tuple(poses),
            gripper_values=(gripper,) * count,
            step_dt_s=dt_s,
        )

    def _lower_gripper(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        state: _State,
        mode: GripperCommandMode,
    ) -> _Lowered:
        dt_s = _command_dt(command, path, state.context.step_dt_s)
        steps = _positive_step_count(command, path, default=1)
        state.budget.charge_samples(steps, path)
        duration = steps * dt_s
        state.budget.charge_duration(duration, path)
        eef_name, arm = self._resolve_eef(command, None, path, state.context)
        segment_id = make_stable_id("gripper", state.context.request_digest, path, eef_name, mode.value, steps)
        segment = GripperCommandSegment(
            segment_id=segment_id,
            depends_on=dependencies,
            duration_s=duration,
            eef_name=eef_name,
            command=mode,
            settle_steps=steps,
            metadata={
                "source_arm": arm,
                "source_command": type(command).__name__,
                "source_path": list(path),
                "step_dt_s": dt_s,
            },
        )
        state.segments.append(segment)
        state.gripper_by_eef[eef_name] = 1.0 if mode is GripperCommandMode.OPEN else -1.0
        return _Lowered(
            (segment_id,),
            (segment_id,),
            duration,
            frozenset({f"eef:{eef_name}", f"arm:{arm}"}),
        )

    def _lower_attach(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        state: _State,
        *,
        in_composite: bool,
    ) -> _Lowered:
        if in_composite:
            raise UnsupportedScheduleStreamCommandError(
                "attachment transitions inside Composite are ambiguous and are rejected", path=path
            )
        source_object = _required_text_attribute(command, "obj", path)
        link_field = "parent" if state.symbols.application == "custream" else "link"
        link = _required_text_attribute(command, link_field, path)
        eef_name, arm = self._resolve_eef(command, link, path, state.context)
        object_name = _resolve_object_name(source_object, path, state.context)
        if link in state.attachments_by_link:
            raise MalformedScheduleStreamCommandError(
                f"link {link!r} is already tracked as holding {state.attachments_by_link[link]!r}", path=path
            )
        if object_name in state.attachments_by_link.values():
            raise MalformedScheduleStreamCommandError(
                f"object {object_name!r} is already attached to another link", path=path
            )
        dt_s = _command_dt(command, path, state.context.step_dt_s)
        steps = _positive_step_count(command, path, default=1)
        state.budget.charge_samples(steps, path)
        duration = steps * dt_s
        state.budget.charge_duration(duration, path)
        segment_id = make_stable_id("attach", state.context.request_digest, path, eef_name, object_name)
        segment = AttachIntentSegment(
            segment_id=segment_id,
            depends_on=dependencies,
            duration_s=duration,
            eef_name=eef_name,
            object_name=object_name,
            verifier=state.context.attachment_verifier,
            metadata={
                "source_arm": arm,
                "source_link": link,
                "source_object": source_object,
                "source_path": list(path),
            },
        )
        state.segments.append(segment)
        state.attachments_by_link[link] = object_name
        return _Lowered(
            (segment_id,),
            (segment_id,),
            duration,
            frozenset({f"eef:{eef_name}", f"arm:{arm}"}),
        )

    def _lower_detach(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        state: _State,
        *,
        in_composite: bool,
    ) -> _Lowered:
        if in_composite:
            raise UnsupportedScheduleStreamCommandError(
                "attachment transitions inside Composite are ambiguous and are rejected", path=path
            )
        link_field = "parent" if state.symbols.application == "custream" else "link"
        link = _required_text_attribute(command, link_field, path)
        eef_name, arm = self._resolve_eef(command, link, path, state.context)
        object_name = state.attachments_by_link.get(link)
        if object_name is None:
            raise UnsupportedScheduleStreamCommandError(
                f"Detach({link!r}) does not name its object; bind initial_attachments_by_link or precede it "
                "with a tracked Attach",
                path=path,
            )
        dt_s = _command_dt(command, path, state.context.step_dt_s)
        steps = _positive_step_count(command, path, default=1)
        state.budget.charge_samples(steps, path)
        duration = steps * dt_s
        state.budget.charge_duration(duration, path)
        segment_id = make_stable_id("detach", state.context.request_digest, path, eef_name, object_name)
        segment = DetachIntentSegment(
            segment_id=segment_id,
            depends_on=dependencies,
            duration_s=duration,
            eef_name=eef_name,
            object_name=object_name,
            verifier=state.context.attachment_verifier,
            metadata={"source_arm": arm, "source_link": link, "source_path": list(path)},
        )
        state.segments.append(segment)
        del state.attachments_by_link[link]
        return _Lowered(
            (segment_id,),
            (segment_id,),
            duration,
            frozenset({f"eef:{eef_name}", f"arm:{arm}"}),
        )

    def _lower_composite(
        self,
        command: Any,
        dependencies: tuple[str, ...],
        path: tuple[int, ...],
        depth: int,
        state: _State,
    ) -> _Lowered:
        children = _native_children(command, path)
        if len(children) > state.budget.limits.max_composite_width:
            raise ScheduleStreamLimitError(
                f"Composite width exceeds {state.budget.limits.max_composite_width}", path=path
            )
        if not children:
            dt_s = _command_dt(command, path, state.context.step_dt_s)
            state.budget.charge_samples(1, path)
            state.budget.charge_duration(dt_s, path)
            segment_id = make_stable_id("wait", state.context.request_digest, path, "empty_composite")
            segment = WaitSegment(
                segment_id=segment_id,
                depends_on=dependencies,
                duration_s=dt_s,
                steps=1,
                metadata={
                    "source_command": "Composite",
                    "source_path": list(path),
                    "step_dt_s": dt_s,
                },
            )
            state.segments.append(segment)
            return _Lowered((segment_id,), (segment_id,), dt_s, frozenset())
        if len(children) == 1:
            return self._lower_command(children[0], dependencies, path + (0,), depth, state, in_composite=True)

        state.budget.has_concurrency = True
        base_attachments = dict(state.attachments_by_link)
        base_grippers = dict(state.gripper_by_eef)
        changed_grippers: set[str] = set()
        occupied_resources: set[str] = set()
        terminal_ids: list[str] = []
        member_ids: list[str] = []
        duration = 0.0
        branch_grippers: dict[str, float] = {}
        for index, child in enumerate(children):
            branch = state.branch()
            branch.attachments_by_link = dict(base_attachments)
            branch.gripper_by_eef = dict(base_grippers)
            lowered = self._lower_command(
                child,
                dependencies,
                path + (index,),
                depth,
                branch,
                in_composite=True,
            )
            conflict = occupied_resources & set(lowered.resources)
            if conflict:
                raise UnsupportedScheduleStreamCommandError(
                    f"Composite branches command overlapping resources {sorted(conflict)}", path=path
                )
            occupied_resources.update(lowered.resources)
            terminal_ids.extend(lowered.terminal_ids)
            member_ids.extend(lowered.segment_ids)
            duration = max(duration, lowered.duration_s)
            if branch.attachments_by_link != base_attachments:
                raise UnsupportedScheduleStreamCommandError(
                    "Composite attachment state cannot be merged deterministically", path=path
                )
            for eef_name, value in branch.gripper_by_eef.items():
                if base_grippers.get(eef_name, state.context.initial_gripper_value) == value:
                    continue
                if eef_name in changed_grippers:
                    raise UnsupportedScheduleStreamCommandError(
                        f"Composite changes gripper {eef_name!r} in more than one branch", path=path
                    )
                changed_grippers.add(eef_name)
                branch_grippers[eef_name] = value
        state.gripper_by_eef.update(branch_grippers)
        group_id = make_stable_id("concurrent", state.context.request_digest, path, member_ids)
        group = ConcurrentGroupSegment(
            segment_id=group_id,
            depends_on=tuple(dict.fromkeys(terminal_ids)),
            duration_s=duration,
            member_segment_ids=tuple(member_ids),
            metadata={
                "source_command": "Composite",
                "source_path": list(path),
                "width": len(children),
            },
        )
        state.segments.append(group)
        return _Lowered(
            (group_id,),
            tuple(member_ids) + (group_id,),
            duration,
            frozenset(occupied_resources),
        )

    def _extract_joint_samples(
        self,
        command: Any,
        path: tuple[int, ...],
        state: _State,
        *,
        configuration: bool,
    ) -> _JointSamples:
        try:
            joint_state = command.joint_state
            raw_names = command.joints if hasattr(command, "joints") else joint_state.joint_names
            raw_positions = joint_state.position
        except Exception as exc:
            raise _malformed_from_exception("failed to read joint state", path, exc) from exc
        names = _joint_names(raw_names, path, state.budget.limits.max_joints)
        positions = _numeric_rows(
            raw_positions,
            path,
            max_rows=(1 if configuration else state.budget.limits.max_samples_per_segment),
            expected_columns=len(names),
        )
        dt_s = _command_dt(command, path, state.context.step_dt_s)
        held_objects = self._held_objects(command, path, state.context)
        if len(held_objects) > 1:
            raise UnsupportedScheduleStreamCommandError(
                "TaskMotionPlan v1 can associate at most one attached object with a joint trajectory",
                path=path,
            )
        return _JointSamples(names, positions, dt_s, held_objects[0] if held_objects else None)

    def _held_objects(
        self,
        command: Any,
        path: tuple[int, ...],
        context: ScheduleStreamLoweringContext,
    ) -> tuple[str, ...]:
        raw: Any = None
        for attribute in ("holding", "grasped", "grasps"):
            try:
                raw = getattr(command, attribute)
            except AttributeError:
                continue
            except Exception as exc:
                raise _malformed_from_exception(f"failed to read {attribute}", path, exc) from exc
            else:
                break
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise MalformedScheduleStreamCommandError("held-object collection must be a list or tuple", path=path)
        if len(raw) > 64:
            raise ScheduleStreamLimitError("held-object collection exceeds 64 entries", path=path)
        names = []
        for item in raw:
            source_name = item if isinstance(item, str) else _required_text_attribute(item, "obj", path)
            names.append(_resolve_object_name(source_name, path, context))
        if len(names) != len(set(names)):
            raise MalformedScheduleStreamCommandError("held-object collection contains duplicates", path=path)
        return tuple(names)

    def _resolve_eef(
        self,
        command: Any,
        link: str | None,
        path: tuple[int, ...],
        context: ScheduleStreamLoweringContext,
    ) -> tuple[str, str]:
        if link is not None and link in context.eef_by_link:
            arm = _optional_command_arm(command, link, path) or link
            return context.eef_by_link[link], arm
        arm = _optional_command_arm(command, link, path)
        if arm is None:
            raise UnsupportedScheduleStreamCommandError(
                "command is not associated with a robot arm/end effector", path=path
            )
        if context.eef_by_arm:
            if arm not in context.eef_by_arm:
                raise UnsupportedScheduleStreamCommandError(f"native arm {arm!r} has no eef_by_arm binding", path=path)
            return context.eef_by_arm[arm], arm
        return context.eef_name, arm


def _select_backend(requested_motion_backend: str, capabilities: Any | None) -> Any:
    from isaac_autodata_interfaces.motion_planners.curobo.compat import select_schedulestream_backend

    return select_schedulestream_backend(requested_motion_backend, capabilities)


def _required_selection_text(selection: Any, attribute: str) -> str:
    value = getattr(selection, attribute, None)
    if not isinstance(value, str) or not value:
        raise ScheduleStreamProviderError(f"compatibility selector returned no valid {attribute}")
    return value


def _selection_backend_version(selection: Any) -> str:
    capabilities = getattr(selection, "capabilities", None)
    identity = getattr(capabilities, "schedulestream", None)
    version = getattr(identity, "version", None)
    commit = getattr(identity, "source_commit", None)
    if isinstance(version, str) and version:
        if isinstance(commit, str) and commit:
            return f"{version}+{commit[:12]}"
        return version
    return "unknown"


def _native_children(command: Any, path: tuple[int, ...]) -> tuple[Any, ...]:
    try:
        children = command.commands
    except Exception as exc:
        raise _malformed_from_exception("command container has no readable commands", path, exc) from exc
    if not isinstance(children, (list, tuple)):
        raise MalformedScheduleStreamCommandError("native commands must be stored as a list or tuple", path=path)
    return tuple(children)


def _required_text_attribute(value: Any, attribute: str, path: tuple[int, ...]) -> str:
    try:
        result = getattr(value, attribute)
    except Exception as exc:
        raise _malformed_from_exception(f"missing or unreadable {attribute}", path, exc) from exc
    if not isinstance(result, str) or not result.strip() or len(result) > 512 or "\x00" in result:
        raise MalformedScheduleStreamCommandError(f"{attribute} must be a non-empty bounded string", path=path)
    return result


def _optional_text_attribute(value: Any, attribute: str, path: tuple[int, ...]) -> str | None:
    try:
        result = getattr(value, attribute)
    except AttributeError:
        return None
    except Exception as exc:
        raise _malformed_from_exception(f"unreadable {attribute}", path, exc) from exc
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip() or len(result) > 512 or "\x00" in result:
        raise MalformedScheduleStreamCommandError(f"{attribute} must be null or a non-empty bounded string", path=path)
    return result


def _optional_command_arm(command: Any, link: str | None, path: tuple[int, ...]) -> str | None:
    try:
        arm = getattr(command, "arm")
    except AttributeError:
        arm = None
    except Exception as exc:
        raise _malformed_from_exception("failed to resolve native arm", path, exc) from exc
    if arm is None and link is not None:
        try:
            resolver = command.world.get_link_arm
            arm = resolver(link)
        except Exception as exc:
            raise _malformed_from_exception("failed to map native link to arm", path, exc) from exc
    if arm is None:
        return None
    if not isinstance(arm, str) or not arm.strip() or len(arm) > 512 or "\x00" in arm:
        raise MalformedScheduleStreamCommandError("native arm must be a non-empty bounded string", path=path)
    return arm


def _resolve_object_name(
    source_name: str,
    path: tuple[int, ...],
    context: ScheduleStreamLoweringContext,
) -> str:
    if context.object_name_map:
        if source_name not in context.object_name_map:
            raise UnsupportedScheduleStreamCommandError(
                f"native object {source_name!r} has no object_name_map binding", path=path
            )
        return context.object_name_map[source_name]
    return source_name


def _command_dt(command: Any, path: tuple[int, ...], expected_dt_s: float) -> float:
    try:
        raw_dt = getattr(command, "time_step")
    except AttributeError:
        try:
            raw_dt = command.world.time_step
        except Exception as exc:
            raise _malformed_from_exception("command has no readable time step", path, exc) from exc
    except Exception as exc:
        raise _malformed_from_exception("failed to read command time step", path, exc) from exc
    if isinstance(raw_dt, bool):
        raise ScheduleStreamTimingError("command time step must be numeric", path=path)
    try:
        dt_s = float(raw_dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleStreamTimingError("command time step must be numeric", path=path) from exc
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ScheduleStreamTimingError("command time step must be positive and finite", path=path)
    tolerance = max(1e-9, abs(expected_dt_s) * 1e-6)
    if abs(dt_s - expected_dt_s) > tolerance:
        raise ScheduleStreamTimingError(
            f"native time step {dt_s:g}s does not match executor time step {expected_dt_s:g}s; "
            "resample before lowering",
            path=path,
        )
    return dt_s


def _positive_step_count(command: Any, path: tuple[int, ...], *, default: int) -> int:
    try:
        value = getattr(command, "num_steps")
    except AttributeError:
        value = default
    except Exception as exc:
        raise _malformed_from_exception("failed to read num_steps", path, exc) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MalformedScheduleStreamCommandError("num_steps must be a positive integer", path=path)
    return value


def _path_length(command: Any, application: str, path: tuple[int, ...]) -> int:
    try:
        value = len(command) if application == "custream" else command.length
    except Exception as exc:
        raise _malformed_from_exception("failed to read link-path length", path, exc) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MalformedScheduleStreamCommandError("link-path length must be a positive integer", path=path)
    return value


def _joint_names(raw_names: Any, path: tuple[int, ...], max_joints: int) -> tuple[str, ...]:
    if not isinstance(raw_names, (list, tuple)):
        raise MalformedScheduleStreamCommandError("joint names must be a list or tuple", path=path)
    if not raw_names:
        raise MalformedScheduleStreamCommandError("joint names must not be empty", path=path)
    if len(raw_names) > max_joints:
        raise ScheduleStreamLimitError(f"joint count exceeds {max_joints}", path=path)
    names = []
    for name in raw_names:
        if not isinstance(name, str) or not name.strip() or len(name) > 512 or "\x00" in name:
            raise MalformedScheduleStreamCommandError("joint names must be bounded non-empty strings", path=path)
        names.append(name)
    if len(names) != len(set(names)):
        raise MalformedScheduleStreamCommandError("joint names must be unique", path=path)
    return tuple(names)


def _numeric_rows(
    value: Any,
    path: tuple[int, ...],
    *,
    max_rows: int,
    expected_columns: int,
) -> tuple[tuple[float, ...], ...]:
    shape = _native_shape(value, path)
    if shape is not None:
        if len(shape) != 2:
            raise MalformedScheduleStreamCommandError(f"joint positions must have rank 2, got shape {shape}", path=path)
        if shape[0] < 1 or shape[0] > max_rows or shape[1] != expected_columns:
            raise MalformedScheduleStreamCommandError(
                f"joint positions shape {shape} violates [1..{max_rows}, {expected_columns}]", path=path
            )
    materialized = _to_builtin(value, path)
    if not isinstance(materialized, (list, tuple)):
        raise MalformedScheduleStreamCommandError("joint positions must be a two-dimensional array", path=path)
    if not materialized or len(materialized) > max_rows:
        raise ScheduleStreamLimitError(f"joint sample count must be in [1, {max_rows}]", path=path)
    rows = []
    for row in materialized:
        if not isinstance(row, (list, tuple)) or len(row) != expected_columns:
            raise MalformedScheduleStreamCommandError(
                f"every joint row must contain {expected_columns} values", path=path
            )
        rows.append(tuple(_finite_number(item, path) for item in row))
    return tuple(rows)


def _numeric_matrix(
    value: Any,
    rows: int,
    columns: int,
    path: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    shape = _native_shape(value, path)
    if shape is not None and shape != (rows, columns):
        raise MalformedScheduleStreamCommandError(
            f"pose matrix must have shape [{rows}, {columns}], got {shape}", path=path
        )
    materialized = _to_builtin(value, path)
    if not isinstance(materialized, (list, tuple)) or len(materialized) != rows:
        raise MalformedScheduleStreamCommandError(f"pose matrix must have shape [{rows}, {columns}]", path=path)
    result = []
    for row in materialized:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise MalformedScheduleStreamCommandError(f"pose matrix must have shape [{rows}, {columns}]", path=path)
        result.append(tuple(_finite_number(item, path) for item in row))
    return tuple(result)


def _native_shape(value: Any, path: tuple[int, ...]) -> tuple[int, ...] | None:
    try:
        raw_shape = getattr(value, "shape")
    except AttributeError:
        return None
    except Exception as exc:
        raise _malformed_from_exception("failed to inspect native array shape", path, exc) from exc
    try:
        shape = tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MalformedScheduleStreamCommandError("native array shape is invalid", path=path) from exc
    if any(item < 0 for item in shape):
        raise MalformedScheduleStreamCommandError("native array shape cannot be negative", path=path)
    return shape


def _to_builtin(value: Any, path: tuple[int, ...]) -> Any:
    result = value
    for method_name in ("detach", "cpu", "tolist"):
        method = getattr(result, method_name, None)
        if callable(method):
            try:
                result = method()
            except Exception as exc:
                raise _malformed_from_exception(
                    f"failed to materialize native array via {method_name}", path, exc
                ) from exc
    return result


def _finite_number(value: Any, path: tuple[int, ...]) -> float:
    if isinstance(value, bool):
        raise MalformedScheduleStreamCommandError("numeric arrays must not contain booleans", path=path)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MalformedScheduleStreamCommandError("numeric arrays must contain numbers", path=path) from exc
    if not math.isfinite(result):
        raise MalformedScheduleStreamCommandError("numeric arrays must contain only finite values", path=path)
    return result


def _same_dt(first: float, second: float) -> bool:
    return abs(first - second) <= max(1e-9, abs(first) * 1e-6)


def _malformed_from_exception(
    message: str,
    path: tuple[int, ...],
    exc: Exception,
) -> MalformedScheduleStreamCommandError:
    detail = str(exc).replace("\n", " ")[:300]
    return MalformedScheduleStreamCommandError(f"{message} ({type(exc).__name__}: {detail})", path=path)
