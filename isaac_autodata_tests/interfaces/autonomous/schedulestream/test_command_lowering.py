# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    ConcurrentGroupSegment,
    DetachIntentSegment,
    GoalPredicate,
    GripperCommandMode,
    GripperCommandSegment,
    JointTrajectorySegment,
    WaitSegment,
)
from isaac_autodata_interfaces.autonomous.schedulestream import (
    MalformedScheduleStreamCommandError,
    ScheduleStreamClosedError,
    ScheduleStreamCommandLowerer,
    ScheduleStreamCommandSymbols,
    ScheduleStreamLoweringContext,
    ScheduleStreamTimingError,
    UnsupportedScheduleStreamCommandError,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
TRANSLATED = (
    (1.0, 0.0, 0.0, 0.2),
    (0.0, 1.0, 0.0, -0.1),
    (0.0, 0.0, 1.0, 0.3),
    (0.0, 0.0, 0.0, 1.0),
)


class _World:
    def __init__(self, dt: float = 0.05, arm_by_link: dict[str, str] | None = None) -> None:
        self.time_step = dt
        self.arm_by_link = arm_by_link or {"tool": "arm"}

    def get_link_arm(self, link: str) -> str:
        return self.arm_by_link[link]


class _Command:
    def __init__(self, world: _World) -> None:
        self.world = world

    @property
    def time_step(self) -> float:
        return self.world.time_step


class _Commands(_Command):
    def __init__(self, world: _World, commands: list[object]) -> None:
        super().__init__(world)
        self.commands = commands


class _Composite(_Commands):
    pass


class _JointState:
    def __init__(self, names: tuple[str, ...], positions: list[list[float]]) -> None:
        self.joint_names = names
        self.position = positions


class _Configuration(_Command):
    def __init__(self, world: _World, names: tuple[str, ...], position: list[float]) -> None:
        super().__init__(world)
        self.joint_state = _JointState(names, [position])

    @property
    def joints(self) -> list[str]:
        return list(self.joint_state.joint_names)


class _Trajectory(_Command):
    def __init__(self, world: _World, names: tuple[str, ...], positions: list[list[float]]) -> None:
        super().__init__(world)
        self.joint_state = _JointState(names, positions)

    @property
    def joints(self) -> list[str]:
        return list(self.joint_state.joint_names)


class _V1LinkPath(_Command):
    def __init__(self, world: _World, arm: str, link: str, poses: list[object]) -> None:
        super().__init__(world)
        self.arm = arm
        self.link = link
        self.poses = poses

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index: int) -> object:
        return self.poses[index]


class _V2LinkPath(_Command):
    def __init__(self, world: _World, arm: str, link: str, poses: list[object]) -> None:
        super().__init__(world)
        self.arm = arm
        self.link = link
        self.poses = poses

    @property
    def length(self) -> int:
        return len(self.poses)

    def pose(self, index: int) -> object:
        return self.poses[index]


class _Open(_Command):
    def __init__(self, world: _World, arm: str, num_steps: int = 1) -> None:
        super().__init__(world)
        self.arm = arm
        self.num_steps = num_steps


class _Close(_Open):
    pass


class _Attach(_Command):
    def __init__(
        self,
        world: _World,
        obj: str,
        *,
        arm: str | None,
        parent: str | None = None,
        link: str | None = None,
    ) -> None:
        super().__init__(world)
        self.obj = obj
        self.arm = arm
        self.parent = parent
        self.link = link
        self.num_steps = 1


class _Detach(_Command):
    def __init__(
        self,
        world: _World,
        *,
        arm: str | None,
        parent: str | None = None,
        link: str | None = None,
    ) -> None:
        super().__init__(world)
        self.arm = arm
        self.parent = parent
        self.link = link
        self.num_steps = 1


def _symbols(application: str) -> ScheduleStreamCommandSymbols:
    return ScheduleStreamCommandSymbols(
        application=application,
        commands_type=_Commands,
        composite_type=_Composite,
        configuration_type=_Configuration,
        trajectory_type=_Trajectory,
        link_path_type=_V1LinkPath if application == "custream" else _V2LinkPath,
        open_type=_Open,
        close_type=_Close,
        attach_type=_Attach,
        detach_type=_Detach,
        pose_to_matrix=lambda pose: pose,
    )


def _selector(application: str):
    motion_backend = "curobo_v1" if application == "custream" else "curobo_v2"
    return lambda requested, capabilities: SimpleNamespace(
        motion_backend=motion_backend,
        schedulestream_application=application,
        capabilities=SimpleNamespace(schedulestream=SimpleNamespace(version="test", source_commit="abcdef0123456789")),
    )


def _context(**kwargs) -> ScheduleStreamLoweringContext:
    values = {
        "request_digest": "request",
        "snapshot_digest": "snapshot",
        "seed": 7,
        "goal": (GoalPredicate("on", "cube", "table"),),
        "eef_name": "hand",
        "frame": "world",
        "step_dt_s": 0.05,
    }
    values.update(kwargs)
    return ScheduleStreamLoweringContext(**values)


def _boundary(application: str, **kwargs) -> ScheduleStreamCommandLowerer:
    return ScheduleStreamCommandLowerer(
        selector=_selector(application),
        symbols=_symbols(application),
        **kwargs,
    )


def test_import_is_free_of_schedulestream_and_curobo_modules() -> None:
    forbidden = ("curobo", "schedulestream")
    program = (
        "import json,sys; import isaac_autodata_interfaces.autonomous.schedulestream; "
        f"print(json.dumps([name for name in {forbidden!r} if name in sys.modules]))"
    )
    completed = subprocess.run([sys.executable, "-c", program], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_v1_lowers_every_supported_sequential_command_without_dropping() -> None:
    world = _World()
    commands = _Commands(
        world,
        [
            _Open(world, "arm", num_steps=2),
            _V1LinkPath(world, "arm", "tool", [IDENTITY, TRANSLATED]),
            _Attach(world, "native_cube", arm="arm", parent="tool"),
            _Configuration(world, ("j1", "j2"), [0.1, 0.2]),
            _Configuration(world, ("j1", "j2"), [0.1, 0.2]),
            _Detach(world, arm="arm", parent="tool"),
            _Close(world, "arm", num_steps=3),
            _Trajectory(world, ("j1", "j2"), [[0.1, 0.2], [0.2, 0.3]]),
        ],
    )

    plan = _boundary("custream").lower(
        commands,
        _context(object_name_map={"native_cube": "cube"}),
    )

    assert [type(segment) for segment in plan.segments] == [
        GripperCommandSegment,
        CartesianTrajectorySegment,
        AttachIntentSegment,
        WaitSegment,
        DetachIntentSegment,
        GripperCommandSegment,
        JointTrajectorySegment,
    ]
    assert plan.segments[0].command is GripperCommandMode.OPEN
    assert plan.segments[0].settle_steps == 2
    assert plan.segments[2].object_name == "cube"
    assert plan.segments[3].steps == 2
    assert plan.segments[4].object_name == "cube"
    assert plan.segments[5].command is GripperCommandMode.CLOSE
    assert plan.segments[-1].timestamps_s == (0.0, 0.05)
    assert plan.metadata["schedulestream"]["command_nodes"] == 9


def test_v2_recursively_preserves_disjoint_composite_branches() -> None:
    world = _World(arm_by_link={"left_tool": "left", "right_tool": "right"})
    composite = _Composite(
        world,
        [
            _V2LinkPath(world, "left", "left_tool", [IDENTITY, TRANSLATED]),
            _Commands(
                world,
                [
                    _Open(world, "right"),
                    _V2LinkPath(world, "right", "right_tool", [IDENTITY, TRANSLATED]),
                ],
            ),
        ],
    )

    plan = _boundary("custream2").lower(
        [composite],
        _context(eef_by_arm={"left": "left_hand", "right": "right_hand"}),
    )

    groups = [segment for segment in plan.segments if isinstance(segment, ConcurrentGroupSegment)]
    cartesian = [segment for segment in plan.segments if isinstance(segment, CartesianTrajectorySegment)]
    assert len(groups) == 1
    assert {segment.eef_name for segment in cartesian} == {"left_hand", "right_hand"}
    assert set(groups[0].member_segment_ids) == {segment.segment_id for segment in plan.segments[:-1]}
    assert plan.metadata["schedulestream"]["has_concurrency"] is True


def test_symbol_provider_is_lazy_and_close_is_idempotent() -> None:
    provider_calls = []
    closed = []
    boundary = ScheduleStreamCommandLowerer(
        selector=_selector("custream"),
        symbol_provider=lambda application: provider_calls.append(application) or _symbols(application),
        owned_resource="world",
        resource_closer=closed.append,
    )
    assert provider_calls == []

    boundary.lower([_Open(_World(), "arm")], _context())
    assert provider_calls == ["custream"]
    boundary.close()
    boundary.close()
    assert closed == ["world"]
    with pytest.raises(ScheduleStreamClosedError):
        boundary.lower([_Open(_World(), "arm")], _context())


@pytest.mark.parametrize(
    ("commands", "error_type"),
    [
        ([object()], UnsupportedScheduleStreamCommandError),
        ([_Detach(_World(), arm="arm", parent="tool")], UnsupportedScheduleStreamCommandError),
        ([_Trajectory(_World(), ("j1",), [[float("nan")]])], MalformedScheduleStreamCommandError),
        ([_Open(_World(dt=0.1), "arm")], ScheduleStreamTimingError),
    ],
)
def test_malformed_or_unsupported_commands_fail_closed(commands, error_type) -> None:
    with pytest.raises(error_type):
        _boundary("custream").lower(commands, _context())


def test_exact_type_check_rejects_unknown_trajectory_subclass() -> None:
    class FutureTrajectory(_Trajectory):
        pass

    with pytest.raises(UnsupportedScheduleStreamCommandError):
        _boundary("custream").lower(
            [FutureTrajectory(_World(), ("j1",), [[0.0]])],
            _context(),
        )


def test_composite_rejects_resource_conflicts_and_attachment_transitions() -> None:
    world = _World()
    conflicting = _Composite(
        world,
        [
            _V1LinkPath(world, "arm", "tool", [IDENTITY]),
            _Open(world, "arm"),
        ],
    )
    attachment = _Composite(
        world,
        [_Attach(world, "cube", arm="arm", parent="tool"), _Open(world, "other")],
    )

    with pytest.raises(UnsupportedScheduleStreamCommandError, match="overlapping resources"):
        _boundary("custream").lower([conflicting], _context())
    with pytest.raises(UnsupportedScheduleStreamCommandError, match="inside Composite"):
        _boundary("custream").lower([attachment], _context())
