# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from isaac_autodata_core.autonomous.task_motion import (
    AttachIntentSegment,
    CartesianTrajectorySegment,
    DetachIntentSegment,
    GoalPredicate,
    JointTrajectorySegment,
)
from isaac_autodata_interfaces.autonomous.schedulestream import (
    MalformedScheduleStreamCommandError,
    ScheduleStreamLoweringContext,
    ScheduleStreamProviderError,
    V1IsaacLabCommandPlanner,
    V1IsaacLabPlannerConfig,
    build_v1_isaaclab_world,
    create_v1_isaaclab_command_planner,
)
from isaac_autodata_interfaces.autonomous.schedulestream.custream_v1 import (
    _action_body_offset,
    _filter_v1_ik_joint_limits,
    _scene_state_with_curobo_pose_order,
    _validate_world_graspability,
)
from isaac_autodata_interfaces.autonomous.schedulestream.episode_planner import _world_eef_pose_reader
from isaac_autodata_interfaces.task_planners.schedulestream.goal import ScheduleStreamGoalSymbols

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _ActionManager:
    active_terms = ("arm",)

    def get_term(self, name: str):
        assert name == "arm"
        offset = SimpleNamespace(pos=(0.0, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0))
        return SimpleNamespace(cfg=SimpleNamespace(body_name="tool", body_offset=offset))


class _ActionManager3d:
    active_terms = ("arm",)

    def get_term(self, name: str):
        assert name == "arm"
        offset = SimpleNamespace(pos=(0.1, 0.2, 0.3), rot=(0.0, 0.0, 0.0, 1.0))
        return SimpleNamespace(cfg=SimpleNamespace(body_name="tool", body_offset=offset))


class _NativePlanner:
    def __init__(self) -> None:
        self.semantic_arm = "arm"
        self.semantic_graspable_object = "cube"
        self.base_env = SimpleNamespace(action_manager=_ActionManager())
        self.world = SimpleNamespace(time_step=0.05)
        self.frames = []
        self.goal = "goal"
        self.env = object()

    def current_link_matrix(self, link_name: str):
        assert link_name == "tool"
        return IDENTITY

    def plan_controller(self, env_id: int):
        assert env_id == 0
        return SimpleNamespace(
            joints=("j1", "j2"),
            joint_positions=(
                (0.0, 0.1),
                (0.2, 0.3),
                (0.2, 0.3),
                (0.4, 0.5),
                (0.4, 0.5),
            ),
            link_poses={
                "tool": (
                    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                )
            },
            gripper_actions=(1.0, 1.0, -1.0, -1.0, 1.0),
        )


class _PlannerToolNative(_NativePlanner):
    def __init__(self) -> None:
        super().__init__()
        self.base_env = SimpleNamespace(action_manager=_ActionManager3d())
        self.world = SimpleNamespace(
            get_arm_link=lambda arm: "ee_link",
            time_step=0.05,
        )

    def current_link_matrix(self, link_name: str):
        if link_name == "tool":
            return IDENTITY
        assert link_name == "ee_link"
        return (
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 1.0, 0.0, 0.2),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )


def _observed_pose(env_id: int, eef_name: str):
    assert env_id == 0
    assert eef_name == "hand"
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.1),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotated_observed_pose(env_id: int, eef_name: str):
    assert env_id == 0
    assert eef_name == "hand"
    return (
        (-1.0, 0.0, 0.0, 0.5),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.1),
        (0.0, 0.0, 0.0, 1.0),
    )


def _planner_tool_observed_pose(env_id: int, eef_name: str):
    assert env_id == 0
    assert eef_name == "hand"
    return _PlannerToolNative().current_link_matrix("ee_link")


def _context(seed: int) -> ScheduleStreamLoweringContext:
    return ScheduleStreamLoweringContext(
        request_digest="request",
        snapshot_digest="snapshot",
        seed=seed,
        goal=(GoalPredicate("on", "cube", "table"),),
        eef_name="hand",
        frame="world",
        step_dt_s=0.05,
        eef_by_link={"tool": "hand"},
        backend_version="test",
    )


def test_controller_dense_trace_preserves_native_body_offset_samples_and_attempt_seed() -> None:
    seeds = []

    def set_seed(*, seed: int) -> None:
        seeds.append(seed)

    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=_observed_pose,
        seed_setter=set_seed,
    )

    first = provider.plan_dense_trace(_context(11), link_name="tool")
    second = provider.plan_dense_trace(_context(29), link_name="tool")

    assert seeds == [11, 29]
    assert first is not None and second is not None
    # Match upstream PathController: raw body z=0 plus the exactly configured and live-attested
    # action offset z=0.1, with no inferred calibration transform.
    assert first.poses[0][2][3] == pytest.approx(0.1)
    assert first.poses[1][0][3] == pytest.approx(0.2)
    assert first.poses[1][2][3] == pytest.approx(0.1)
    assert first.joint_names == ("j1", "j2")
    assert first.gripper_values == (1.0, 1.0, -1.0, -1.0, 1.0)
    assert first.gripper_settle_steps == 4
    assert [(event.sample_index, event.operation, event.object_name) for event in first.attachment_events] == [
        (2, "attach", "cube"),
        (4, "detach", "cube"),
    ]


def test_controller_dense_trace_rejects_observed_frame_outside_configured_offset_attestation() -> None:
    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=_rotated_observed_pose,
        seed_setter=lambda *, seed: None,
    )

    with pytest.raises(ScheduleStreamProviderError, match="does not match the configured v1 action offset"):
        provider.plan_dense_trace(_context(7), link_name="tool")


def test_controller_dense_trace_rejects_unconfigured_urdf_to_usd_frame_transform() -> None:
    observed = (
        (-1.0, 0.0, 0.0, -0.13),
        (0.0, 1.0, 0.0, -0.07),
        (0.0, 0.0, -1.0, 0.06),
        (0.0, 0.0, 0.0, 1.0),
    )
    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=lambda env_id, eef_name: observed,
        seed_setter=lambda *, seed: None,
    )

    with pytest.raises(ScheduleStreamProviderError, match="does not match the configured v1 action offset"):
        provider.plan_dense_trace(_context(7), link_name="tool")


def test_controller_dense_trace_attests_configured_offset_on_every_attempt() -> None:
    observations = [_observed_pose(0, "hand"), _observed_pose(0, "hand")]
    second = [list(row) for row in observations[1]]
    second[0][3] = 0.006
    observations[1] = tuple(tuple(row) for row in second)
    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=lambda env_id, eef_name: observations.pop(0),
        seed_setter=lambda *, seed: None,
    )

    assert provider.plan_dense_trace(_context(7), link_name="tool") is not None
    with pytest.raises(ScheduleStreamProviderError, match="does not match the configured v1 action offset"):
        provider.plan_dense_trace(_context(8), link_name="tool")


def test_controller_dense_trace_uses_attested_static_action_to_observed_eef_transform() -> None:
    provider = V1IsaacLabCommandPlanner(
        _PlannerToolNative(),
        eef_pose_reader=_planner_tool_observed_pose,
        seed_setter=lambda *, seed: None,
    )

    trace = provider.plan_dense_trace(_context(13), link_name="tool")

    assert trace is not None
    assert tuple(trace.poses[0][index][3] for index in range(3)) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(trace.poses[1][index][3] for index in range(3)) == pytest.approx((0.3, 0.2, 0.3))
    assert trace.poses[2] == trace.poses[1]
    assert tuple(trace.poses[3][index][3] for index in range(3)) == pytest.approx((0.4, 0.2, 0.3))
    assert trace.poses[4] == trace.poses[3]
    assert provider._last_frame_evidence["planned_link_name"] == "tool"
    attestation = provider._last_frame_evidence["static_frame_attestation"]
    assert attestation["configured_offset_position_error_m"] == pytest.approx(0.0)
    assert attestation["configured_offset_rotation_error_rad"] == pytest.approx(0.0)


def test_controller_dense_trace_records_observed_offset_without_using_it_as_calibration() -> None:
    observed = [list(row) for row in _observed_pose(0, "hand")]
    observed[0][3] = 0.001
    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=lambda env_id, eef_name: observed,
        seed_setter=lambda *, seed: None,
    )

    trace = provider.plan_dense_trace(_context(17), link_name="tool")

    assert trace is not None
    # Lowering remains anchored to the reviewed configured 0.1 m z offset, not the small live
    # residual that is accepted only as attestation tolerance.
    assert tuple(trace.poses[0][index][3] for index in range(3)) == pytest.approx((0.0, 0.0, 0.1))
    attestation = provider._last_frame_evidence["static_frame_attestation"]
    assert tuple(attestation["action_to_observed_eef"][index][3] for index in range(3)) == pytest.approx(
        (0.001, 0.0, 0.1)
    )
    assert tuple(attestation["configured_action_offset"][index][3] for index in range(3)) == pytest.approx(
        (0.0, 0.0, 0.1)
    )
    assert attestation["configured_offset_position_error_m"] == pytest.approx(0.001)


def test_controller_task_plan_contains_only_current_ik_executor_segments() -> None:
    provider = V1IsaacLabCommandPlanner(
        _NativePlanner(),
        eef_pose_reader=_observed_pose,
        seed_setter=lambda *, seed: None,
    )
    destination_placement = {"attested": True, "schema_version": 1}
    grasp_geometry = {"attested": True, "schema_version": 1}
    provider.native_planner.destination_placement_geometry = destination_placement
    provider.native_planner.grasp_geometry = grasp_geometry

    plan = provider.plan_task_motion_plan(_context(3), link_name="tool")

    assert plan is not None
    assert any(isinstance(segment, CartesianTrajectorySegment) for segment in plan.segments)
    assert not any(isinstance(segment, JointTrajectorySegment) for segment in plan.segments)
    cartesian = [segment for segment in plan.segments if isinstance(segment, CartesianTrajectorySegment)]
    assert sum(segment.duration_s for segment in cartesian) == pytest.approx(5 * 0.05)
    gripper = [segment for segment in plan.segments if segment.kind.value == "gripper_command"]
    assert all(segment.settle_steps == 4 for segment in gripper)
    attachments = [segment for segment in plan.segments if isinstance(segment, AttachIntentSegment)]
    detachments = [segment for segment in plan.segments if isinstance(segment, DetachIntentSegment)]
    assert [segment.object_name for segment in attachments] == ["cube"]
    assert [segment.object_name for segment in detachments] == ["cube"]
    assert plan.metadata["schedulestream"]["attachment_events_preserved"] is True
    assert plan.metadata["schedulestream"]["destination_placement"] == destination_placement
    assert plan.metadata["schedulestream"]["frame_evidence"]["destination_placement"] == destination_placement
    assert plan.metadata["schedulestream"]["grasp_geometry"] == grasp_geometry
    assert plan.metadata["schedulestream"]["frame_evidence"]["grasp_geometry"] == grasp_geometry


def test_controller_dense_trace_rejects_place_plan_without_release_transition() -> None:
    native = _NativePlanner()

    def plan_controller(env_id: int):
        controller = _NativePlanner().plan_controller(env_id)
        controller.joint_positions = controller.joint_positions[:2]
        controller.link_poses["tool"] = controller.link_poses["tool"][:2]
        controller.gripper_actions = controller.gripper_actions[:2]
        return controller

    native.plan_controller = plan_controller
    provider = V1IsaacLabCommandPlanner(
        native,
        eef_pose_reader=_observed_pose,
        seed_setter=lambda *, seed: None,
    )

    with pytest.raises(MalformedScheduleStreamCommandError, match="lifecycle does not match"):
        provider.plan_dense_trace(_context(3), link_name="tool")


@pytest.mark.parametrize(
    ("quaternion_xyzw", "expected_rotation"),
    [
        (
            (0.0, 0.0, 0.0, 1.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        (
            (0.0, 0.0, 2**-0.5, 2**-0.5),
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    ],
)
def test_action_body_offset_converts_isaaclab_xyzw_to_curobo_wxyz(
    quaternion_xyzw: tuple[float, float, float, float],
    expected_rotation: tuple[tuple[float, float, float], ...],
) -> None:
    offset = SimpleNamespace(pos=(0.1, 0.2, 0.3), rot=quaternion_xyzw)
    term = SimpleNamespace(cfg=SimpleNamespace(body_name="tool", body_offset=offset))
    manager = SimpleNamespace(active_terms=("arm",), get_term=lambda name: term)

    matrix = _action_body_offset(SimpleNamespace(action_manager=manager), "tool")

    np.testing.assert_allclose(np.asarray(matrix)[:3, :3], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(np.asarray(matrix)[:3, 3], (0.1, 0.2, 0.3), atol=1e-12)


def test_world_reader_applies_nonzero_origin_without_changing_observed_rotation() -> None:
    relative_eef = np.array((
        (0.0, -1.0, 0.0, 0.5),
        (1.0, 0.0, 0.0, -0.25),
        (0.0, 0.0, 1.0, 0.2),
        (0.0, 0.0, 0.0, 1.0),
    ))
    adapter = SimpleNamespace(get_eef_poses=lambda env_ids: {"hand": np.stack((relative_eef,))})
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=np.array(((2.0, 3.0, 4.0),))))
    observed = _world_eef_pose_reader(env, adapter)(0, "hand")

    np.testing.assert_allclose(np.asarray(observed)[:3, :3], relative_eef[:3, :3])
    np.testing.assert_allclose(np.asarray(observed)[:3, 3], (2.5, 2.75, 4.2))
    assert relative_eef[0, 3] == pytest.approx(0.5)


class _FakeWorld:
    instances = []

    def __init__(self, robot_config, objects, **kwargs) -> None:
        self.robot_config = robot_config
        self.objects = objects
        self.kwargs = kwargs
        self.joint_update = None
        self.camera_pose = None
        self.object_poses = {item.name: getattr(item, "pose", _WorldPose()) for item in objects}
        self.__class__.instances.append(self)

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(self.object_poses)

    def get_object_pose(self, name: str):
        return self.object_poses[name]

    def get_object(self, name: str):
        return next(item for item in self.objects if item.name == name)

    def set_object_pose(self, name: str, pose) -> None:
        self.object_poses[name] = pose

    def set_joint_positions(self, names, positions) -> None:
        self.joint_update = (tuple(names), tuple(positions))

    def set_camera_pose(self, pose) -> None:
        self.camera_pose = pose

    @property
    def movable_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.objects if getattr(item, "grasp_config", None) is not None)


class _FakeGraspConfig:
    def __init__(self, *, primitive: str, pitch_interval: str, generator=None) -> None:
        self.primitive = primitive
        self.pitch_interval = pitch_interval
        self.generator = generator


class _FakeSurfaceConfig:
    def __init__(self, *, xy_extend: float, z_offset: float) -> None:
        self.xy_extend = xy_extend
        self.z_offset = z_offset

    @property
    def surface_extend(self):
        return np.array((self.xy_extend, self.xy_extend, 0.0))


class _WorldPose:
    def __init__(self, matrix=None) -> None:
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)

    @classmethod
    def from_pose7(cls, value):
        x, y, z, qw, qx, qy, qz = (float(item) for item in value)
        quaternion_norm = np.linalg.norm((qw, qx, qy, qz))
        qw, qx, qy, qz = (item / quaternion_norm for item in (qw, qx, qy, qz))
        matrix = np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw), x],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw), y],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy), z],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return cls(matrix)

    @classmethod
    def translated(cls, x: float, y: float = 0.0, z: float = 0.0):
        matrix = np.eye(4)
        matrix[:3, 3] = (x, y, z)
        return cls(matrix)

    def inverse(self):
        return _WorldPose(np.linalg.inv(self.matrix))

    def get_numpy_matrix(self):
        return [self.matrix.copy()]


def _converted_object(
    name: str,
    *,
    dimensions: tuple[float, float, float] | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pose=None,
    grasp_config=None,
):
    if dimensions is None:
        dimensions = (0.16, 0.16, 0.05) if name == "destination_bowl" else (0.06, 0.06, 0.06)
    bounding_box = SimpleNamespace(dimensions=np.asarray(dimensions), pose=_WorldPose.translated(*center))
    return SimpleNamespace(
        bounding_box=bounding_box,
        grasp_config=grasp_config,
        name=name,
        pose=_WorldPose() if pose is None else pose,
        surface_config=None,
    )


def _multiply_world_poses(*poses):
    result = np.eye(4)
    for pose in poses:
        result = result @ pose.matrix
    return _WorldPose(result)


def _fake_primitive_grasp_generator(primitive, dimensions, pitch_interval):
    assert primitive == "cuboid"
    assert len(dimensions) == 3
    assert pitch_interval == "top"
    pitch = np.diag((1.0, -1.0, -1.0))
    for yaw in np.linspace(0.0, 2.0 * np.pi, num=4, endpoint=False):
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        yaw_rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
        matrix = np.eye(4)
        matrix[:3, :3] = pitch @ yaw_rotation
        yield _WorldPose(matrix)


def _rigid_object(*, kinematic_enabled: bool | None = None, rigid_props_present: bool = True):
    rigid_props = SimpleNamespace(kinematic_enabled=kinematic_enabled) if rigid_props_present else None
    return SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(rigid_props=rigid_props)))


def _world_scene(*, rigid_objects: dict[str, object], basename: str = "panda_instanceable.usd"):
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=f"omniverse://robots/{basename}")),
        joint_names=("j1", "j2"),
    )
    return SimpleNamespace(
        articulations={"robot": articulation},
        env_origins=((0.0, 0.0, 0.0),),
        rigid_objects=rigid_objects,
        sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
        state={
            "articulation": {
                "robot": {
                    "joint_position": ((0.1, 0.2),),
                    "root_pose": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),),
                }
            }
        },
    )


def _world_module(converted_objects):
    return SimpleNamespace(
        CAMERA_POSE="camera",
        GraspConfig=_FakeGraspConfig,
        SurfaceConfig=_FakeSurfaceConfig,
        primitive_grasp_generator=_fake_primitive_grasp_generator,
        World=_FakeWorld,
        create_objects=lambda scene, env_id: converted_objects,
        load_franka_config=lambda base_poses: {"base_poses": base_poses, "robot": "franka"},
        multiply_poses=_multiply_world_poses,
        to_pose=_WorldPose.from_pose7,
    )


def _reviewed_profile_kwargs() -> dict[str, str]:
    return {
        "destination_asset_name": "bowl_ycb_robolab",
        "destination_object": "destination_bowl",
        "graspable_asset_name": "rubiks_cube_hot3d_robolab",
        "graspable_object": "pick_cube",
    }


def test_scene_state_pose_boundary_converts_xyzw_to_curobo_wxyz_without_mutation() -> None:
    half_sqrt = 2**-0.5
    root_pose = torch.tensor([[0.4, -0.2, 0.1, 0.0, half_sqrt, 0.0, half_sqrt]], dtype=torch.float64)
    state = {
        "articulation": {
            "robot": {
                "joint_position": torch.tensor([[0.1, 0.2]], dtype=torch.float64),
                "root_pose": root_pose,
            }
        }
    }

    converted = _scene_state_with_curobo_pose_order(state, "robot")

    assert converted is not state
    assert converted["articulation"]["robot"] is not state["articulation"]["robot"]
    torch.testing.assert_close(
        converted["articulation"]["robot"]["root_pose"],
        torch.tensor([[0.4, -0.2, 0.1, half_sqrt, 0.0, half_sqrt, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        root_pose,
        torch.tensor([[0.4, -0.2, 0.1, 0.0, half_sqrt, 0.0, half_sqrt]], dtype=torch.float64),
    )


@pytest.mark.parametrize("basename", ["panda_instanceable.usd", "franka_panda_hand_on_stand.usd"])
def test_v1_world_factory_accepts_isaaclab_and_arena_franka_assets(basename: str) -> None:
    _FakeWorld.instances.clear()
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=f"omniverse://robots/{basename}")),
        joint_names=("j1", "j2"),
    )
    scene = SimpleNamespace(
        articulations={"robot": articulation},
        env_origins=((0.0, 0.0, 0.0),),
        rigid_objects={
            "destination_bowl": _rigid_object(kinematic_enabled=False),
            "pick_cube": _rigid_object(rigid_props_present=False),
        },
        sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
        state={
            "articulation": {
                "robot": {
                    "joint_position": ((0.1, 0.2),),
                    "root_pose": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),),
                }
            }
        },
    )
    module = SimpleNamespace(
        CAMERA_POSE="camera",
        GraspConfig=_FakeGraspConfig,
        SurfaceConfig=_FakeSurfaceConfig,
        primitive_grasp_generator=_fake_primitive_grasp_generator,
        World=_FakeWorld,
        create_objects=lambda scene, env_id: [
            _converted_object("pick_cube"),
            _converted_object("destination_bowl"),
        ],
        load_franka_config=lambda base_poses: {"base_poses": base_poses, "robot": "franka"},
        multiply_poses=_multiply_world_poses,
        to_pose=_WorldPose.from_pose7,
    )

    world = build_v1_isaaclab_world(
        module,
        scene,
        V1IsaacLabPlannerConfig(scale_dt=5.0),
        **_reviewed_profile_kwargs(),
    )

    assert world.kwargs["interpolation_dt"] == pytest.approx(0.05)
    assert world.robot_config["base_poses"] is None
    assert world.joint_update == (("j1", "j2"), (0.1, 0.2))
    assert world.camera_pose == "camera"
    assert world.autodata_reference_frame_evidence == {
        "converted_scene_frame": "enclosing_robot_prim",
        "curobo_pose_quaternion_order": "wxyz",
        "isaaclab_pose_quaternion_order": "xyzw",
        "object_count": 2,
        "planner_frame": "articulation_root",
        "robot_prim_to_articulation_root": [list(row) for row in np.eye(4)],
    }


def test_v1_world_factory_rebases_robot_prim_obstacles_into_rotated_articulation_root() -> None:
    half_sqrt = 2**-0.5
    root_relative = _WorldPose.from_pose7((0.4, -0.2, 0.1, half_sqrt, 0.0, 0.0, half_sqrt))
    object_robot_relative = _WorldPose.from_pose7((0.8, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0))
    converted = _converted_object("pick_cube", pose=object_robot_relative)
    destination = _converted_object("destination_bowl")
    scene = _world_scene(
        rigid_objects={
            "destination_bowl": _rigid_object(kinematic_enabled=False),
            "pick_cube": _rigid_object(rigid_props_present=False),
        }
    )
    scene.env_origins = ((2.0, 3.0, 0.5),)
    scene.state["articulation"]["robot"]["root_pose"] = ((2.4, 2.8, 0.6, 0.0, 0.0, half_sqrt, half_sqrt),)

    world = build_v1_isaaclab_world(
        _world_module([converted, destination]),
        scene,
        V1IsaacLabPlannerConfig(),
        **_reviewed_profile_kwargs(),
    )

    expected = np.linalg.inv(root_relative.matrix) @ object_robot_relative.matrix
    np.testing.assert_allclose(world.object_poses["pick_cube"].matrix, expected, atol=1e-12)
    np.testing.assert_allclose(
        world.autodata_reference_frame_evidence["robot_prim_to_articulation_root"],
        root_relative.matrix,
        atol=1e-12,
    )


def test_v1_world_factory_rejects_missing_environment_origin() -> None:
    converted = _converted_object("pick_cube")
    destination = _converted_object("destination_bowl")
    scene = _world_scene(
        rigid_objects={
            "destination_bowl": _rigid_object(kinematic_enabled=False),
            "pick_cube": _rigid_object(rigid_props_present=False),
        }
    )
    del scene.env_origins

    with pytest.raises(ScheduleStreamProviderError, match="scene.env_origins is required"):
        build_v1_isaaclab_world(
            _world_module([converted, destination]),
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )


def test_v1_world_factory_makes_only_task_selected_dynamic_object_graspable() -> None:
    pick_cube = _converted_object("pick_cube", center=(0.01, -0.02, 0.003))
    destination_bowl = _converted_object("destination_bowl", center=(-0.004, 0.005, 0.006))
    kinematic_marker = SimpleNamespace(name="kinematic_marker", grasp_config="stale")
    static_mesh = SimpleNamespace(name="/World/floor", grasp_config=None)
    scene = _world_scene(
        rigid_objects={
            # Arena commonly leaves rigid_props unset for ordinary dynamic RigidObjects.
            "pick_cube": _rigid_object(rigid_props_present=False),
            "destination_bowl": _rigid_object(kinematic_enabled=False),
            "kinematic_marker": _rigid_object(kinematic_enabled=True),
        }
    )

    world = build_v1_isaaclab_world(
        _world_module([pick_cube, destination_bowl, kinematic_marker, static_mesh]),
        scene,
        V1IsaacLabPlannerConfig(),
        **_reviewed_profile_kwargs(),
    )

    assert world.objects == [pick_cube, destination_bowl, kinematic_marker, static_mesh]
    assert isinstance(pick_cube.grasp_config, _FakeGraspConfig)
    assert pick_cube.grasp_config.primitive == "cuboid"
    assert pick_cube.grasp_config.pitch_interval == "top"
    assert isinstance(pick_cube.grasp_config.generator, tuple)
    assert len(pick_cube.grasp_config.generator) == 4
    assert destination_bowl.grasp_config is None
    assert kinematic_marker.grasp_config is None
    assert static_mesh.grasp_config is None
    assert world.movable_names == ("pick_cube",)
    assert destination_bowl.surface_config.xy_extend == pytest.approx(-0.16)
    assert destination_bowl.surface_config.z_offset == pytest.approx(-0.025)
    placement_geometry = world.autodata_destination_placement_geometry
    assert placement_geometry["general_inside_semantics"] is False
    assert placement_geometry["sampled_surface_extent_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert placement_geometry["predicted_aabb_center_offset_m"] == pytest.approx([0.0, 0.0, 0.03])
    assert placement_geometry["subject"]["aabb_center_in_converted_object_origin_m"] == pytest.approx(
        [0.01, -0.02, 0.003]
    )
    assert placement_geometry["destination"]["aabb_center_in_converted_object_origin_m"] == pytest.approx(
        [-0.004, 0.005, 0.006]
    )
    grasp_geometry = world.autodata_grasp_geometry
    assert grasp_geometry["grasp_count"] == 4
    assert grasp_geometry["generator_storage"] == "reusable_finite_tuple"
    assert len(grasp_geometry["primitive_link_from_aabb_center_transforms"]) == 4
    assert len(grasp_geometry["link_from_object_transforms"]) == 4


def test_v1_world_factory_derives_exact_center_profile_from_attested_live_aabbs() -> None:
    source_dimensions = (0.058285847306251526, 0.057770855724811554, 0.05796363018453121)
    destination_dimensions = (0.158, 0.151, 0.054)
    pick_cube = _converted_object("pick_cube", dimensions=source_dimensions)
    destination_bowl = _converted_object("destination_bowl", dimensions=destination_dimensions)
    scene = _world_scene(
        rigid_objects={
            "pick_cube": _rigid_object(rigid_props_present=False),
            "destination_bowl": _rigid_object(kinematic_enabled=False),
        }
    )

    world = build_v1_isaaclab_world(
        _world_module([pick_cube, destination_bowl]),
        scene,
        V1IsaacLabPlannerConfig(),
        **_reviewed_profile_kwargs(),
    )

    expected_z_offset = 0.03 - 0.5 * (destination_dimensions[2] + source_dimensions[2])
    assert destination_bowl.surface_config.xy_extend == pytest.approx(-max(destination_dimensions[:2]))
    assert destination_bowl.surface_config.z_offset == pytest.approx(expected_z_offset)
    placement_geometry = world.autodata_destination_placement_geometry
    assert placement_geometry["sampled_surface_extent_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert placement_geometry["predicted_aabb_center_offset_m"] == pytest.approx([0.0, 0.0, 0.03])
    assert placement_geometry["subject"]["aabb_dimensions_m"] == pytest.approx(source_dimensions)
    assert placement_geometry["subject"]["aabb_dimension_bounds_m"] == [[0.05, 0.065]] * 3
    assert placement_geometry["destination"]["aabb_dimensions_m"] == pytest.approx(destination_dimensions)
    assert placement_geometry["destination"]["aabb_dimension_bounds_m"] == [
        [0.14, 0.17],
        [0.14, 0.17],
        [0.04, 0.07],
    ]
    assert placement_geometry["surface_config"]["derivation"] == {
        "desired_aabb_center_vertical_offset_m": 0.03,
        "inputs": "attested_converted_local_aabb_dimensions_m",
        "xy_extend_formula": "-max(destination_aabb_width_m,destination_aabb_depth_m)",
        "z_offset_formula": "desired_center_z_m-0.5*(destination_aabb_height_m+subject_aabb_height_m)",
    }


def test_v1_analytical_grasps_preserve_world_aabb_center_with_noncommuting_pose() -> None:
    def pose_z(theta: float, translation: tuple[float, float, float]) -> _WorldPose:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        matrix = np.array([
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return _WorldPose(matrix)

    object_pose = pose_z(-0.41, (0.52, -0.13, 0.035))
    bounding_box_pose = pose_z(0.37, (-0.0101597, 0.0289287, -0.002404))
    pick_cube = _converted_object(
        "pick_cube",
        dimensions=(0.0582858473, 0.0577708557, 0.0579636302),
        pose=object_pose,
    )
    pick_cube.bounding_box.pose = bounding_box_pose
    destination_bowl = _converted_object("destination_bowl")
    scene = _world_scene(
        rigid_objects={
            "pick_cube": _rigid_object(rigid_props_present=False),
            "destination_bowl": _rigid_object(kinematic_enabled=False),
        }
    )

    world = build_v1_isaaclab_world(
        _world_module([pick_cube, destination_bowl]),
        scene,
        V1IsaacLabPlannerConfig(),
        **_reviewed_profile_kwargs(),
    )

    primitive_matrices = [
        np.asarray(value) for value in world.autodata_grasp_geometry["primitive_link_from_aabb_center_transforms"]
    ]
    link_from_object_matrices = [pose.matrix for pose in pick_cube.grasp_config.generator]
    world_from_aabb = object_pose.matrix @ bounding_box_pose.matrix
    buggy_center_errors = []
    for primitive, link_from_object in zip(primitive_matrices, link_from_object_matrices):
        world_from_link = object_pose.matrix @ np.linalg.inv(link_from_object)
        expected_world_from_link = world_from_aabb @ np.linalg.inv(primitive)
        np.testing.assert_allclose(world_from_link, expected_world_from_link, atol=1e-12)
        np.testing.assert_allclose(world_from_link[:3, 3], world_from_aabb[:3, 3], atol=1e-12)
        buggy_world_from_link = object_pose.matrix @ np.linalg.inv(primitive) @ bounding_box_pose.matrix
        buggy_center_errors.append(np.linalg.norm(buggy_world_from_link[:3, 3] - world_from_aabb[:3, 3]))
    assert max(buggy_center_errors) > 0.02


@pytest.mark.parametrize("failure", ["count", "matrix", "noncallable"])
def test_v1_world_factory_rejects_malformed_analytical_grasp_sources(failure: str) -> None:
    pick_cube = _converted_object("pick_cube")
    destination_bowl = _converted_object("destination_bowl")
    scene = _world_scene(
        rigid_objects={
            "pick_cube": _rigid_object(rigid_props_present=False),
            "destination_bowl": _rigid_object(kinematic_enabled=False),
        }
    )
    if failure == "count":

        def generator(*args):
            del args
            return iter((_WorldPose(),) * 3)

        message = "exactly four poses"
    elif failure == "matrix":
        malformed = _WorldPose()
        malformed.matrix[0, 0] = np.nan

        def generator(*args):
            del args
            return iter((_WorldPose(), _WorldPose.translated(0.0), malformed, _WorldPose()))

        message = "finite rigid pose"
    else:
        generator = object()
        message = "not callable"

    with pytest.raises(ScheduleStreamProviderError, match=message):
        build_v1_isaaclab_world(
            _world_module([pick_cube, destination_bowl]),
            scene,
            V1IsaacLabPlannerConfig(),
            primitive_grasp_generator=generator,
            **_reviewed_profile_kwargs(),
        )


def test_world_graspability_rejects_missing_or_tampered_grasp_geometry() -> None:
    with pytest.raises(ScheduleStreamProviderError, match="grasp geometry"):
        _validate_world_graspability(SimpleNamespace(movable_names=("cube",)), "cube")

    grasp_geometry = _factory_grasp_geometry("cube")
    grasp_geometry["link_from_object_transforms"][0][0][3] = 1e-6
    world = SimpleNamespace(
        autodata_grasp_geometry=grasp_geometry,
        movable_names=("cube",),
    )
    with pytest.raises(ScheduleStreamProviderError, match="composition formula"):
        _validate_world_graspability(world, "cube")


@pytest.mark.parametrize(
    ("source_dimensions", "destination_dimensions"),
    [
        ((0.049, 0.058, 0.058), (0.16, 0.16, 0.05)),
        ((0.058, 0.058, 0.058), (0.16, 0.171, 0.05)),
    ],
)
def test_v1_world_factory_rejects_geometry_outside_name_pinned_bounds(
    source_dimensions: tuple[float, float, float],
    destination_dimensions: tuple[float, float, float],
) -> None:
    scene = _world_scene(
        rigid_objects={
            "pick_cube": _rigid_object(rigid_props_present=False),
            "destination_bowl": _rigid_object(kinematic_enabled=False),
        }
    )
    converted = [
        _converted_object("pick_cube", dimensions=source_dimensions),
        _converted_object("destination_bowl", dimensions=destination_dimensions),
    ]

    with pytest.raises(ScheduleStreamProviderError, match="outside the reviewed per-axis bounds"):
        build_v1_isaaclab_world(
            _world_module(converted),
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )


@pytest.mark.parametrize(
    ("converted_objects", "message"),
    [
        ([SimpleNamespace(name="not_cube", grasp_config=None)], "did not bind"),
        (
            [
                SimpleNamespace(name="pick_cube", grasp_config=None),
                SimpleNamespace(name="pick_cube", grasp_config=None),
            ],
            "bound ambiguously",
        ),
    ],
)
def test_v1_world_factory_fails_closed_on_missing_or_ambiguous_rigid_binding(
    converted_objects,
    message: str,
) -> None:
    scene = _world_scene(rigid_objects={"pick_cube": _rigid_object(rigid_props_present=False)})

    with pytest.raises(ScheduleStreamProviderError, match=message):
        build_v1_isaaclab_world(
            _world_module(converted_objects),
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )


def test_v1_world_factory_rejects_missing_or_kinematic_task_selected_object() -> None:
    converted = SimpleNamespace(name="destination_bowl", grasp_config=None)
    scene = _world_scene(rigid_objects={"destination_bowl": _rigid_object(kinematic_enabled=False)})

    with pytest.raises(ScheduleStreamProviderError, match="absent"):
        build_v1_isaaclab_world(
            _world_module([converted]),
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )

    scene = _world_scene(rigid_objects={"pick_cube": _rigid_object(kinematic_enabled=True)})
    with pytest.raises(ScheduleStreamProviderError, match="explicitly kinematic"):
        build_v1_isaaclab_world(
            _world_module([SimpleNamespace(name="pick_cube", grasp_config=None)]),
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )


def test_v1_world_factory_rejects_unreviewed_robot_asset() -> None:
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path="omniverse://robots/unknown.usd")),
        joint_names=("j1",),
    )
    scene = SimpleNamespace(
        articulations={"robot": articulation},
        rigid_objects={},
        sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
    )
    module = SimpleNamespace(create_objects=lambda scene, env_id: [], load_franka_config=lambda base_poses: {})

    with pytest.raises(ScheduleStreamProviderError, match="unsupported v1 robot USD"):
        build_v1_isaaclab_world(
            module,
            scene,
            V1IsaacLabPlannerConfig(),
            **_reviewed_profile_kwargs(),
        )


class _Clause:
    def __init__(self, values) -> None:
        self.values = tuple(values)

    def __and__(self, other):
        return _Clause(self.values + other.values)


class _BasePlanner:
    @property
    def scene(self):
        return self.env.scene

    def pose(self, name: str, state=None):
        return getattr(self.env, "root_poses", {}).get(name, _FactoryPose())

    def set_env_state(self, env_id: int, state=None, **kwargs) -> None:
        self.last_env_id = env_id
        for name, root_pose in getattr(self.env, "root_poses", {}).items():
            self.world.set_object_pose(name, root_pose)


class _FactoryPose:
    def __init__(self, matrix=None) -> None:
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)

    @classmethod
    def translated(cls, x: float, y: float = 0.0, z: float = 0.0):
        matrix = np.eye(4)
        matrix[:3, 3] = (x, y, z)
        return cls(matrix)

    def inverse(self):
        return _FactoryPose(np.linalg.inv(self.matrix))

    def get_numpy_matrix(self):
        return [self.matrix.copy()]


def _multiply_factory_poses(*poses):
    result = np.eye(4)
    for pose in poses:
        result = result @ pose.matrix
    return _FactoryPose(result)


class _FactoryWorld:
    arms = ("arm",)
    movable_names = ("cube",)

    def __init__(self, object_poses=None) -> None:
        self.object_poses = object_poses or {"cube": _FactoryPose.translated(0.03, -0.01, 0.002)}
        self.object_pose_updates = []
        grasp_poses = tuple(
            _FactoryPose(pose.matrix) for pose in _fake_primitive_grasp_generator("cuboid", (0.06, 0.06, 0.06), "top")
        )
        self.objects = {
            name: SimpleNamespace(
                grasp_config=(
                    _FakeGraspConfig(primitive="cuboid", pitch_interval="top", generator=grasp_poses)
                    if name == "cube"
                    else None
                )
            )
            for name in self.object_poses
        }

    def set_retract_conf(self) -> None:
        self.retract_set = True

    def initialize(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def link_iterative_inverse_kinematics(self, *args, **kwargs):
        raise AssertionError("test world did not expect live IK")

    def configuration(self):
        return "configuration"

    def get_object_pose(self, name: str):
        return self.object_poses[name]

    def get_object(self, name: str):
        return self.objects[name]

    def set_object_pose(self, name: str, pose) -> None:
        self.object_poses[name] = pose
        self.object_pose_updates.append((name, pose))


def _factory_grasp_geometry(subject: str) -> dict:
    primitive = [pose.matrix.tolist() for pose in _fake_primitive_grasp_generator("cuboid", (0.06,) * 3, "top")]
    link_from_object = [[list(row) for row in transform] for transform in primitive]
    return {
        "asset_name": "rubiks_cube_hot3d_robolab",
        "attested": True,
        "composition_formula": "primitive_link_from_aabb_center*inverse(converted_object_origin_from_aabb_center)",
        "converted_object_origin_from_aabb_center": np.eye(4).tolist(),
        "link_from_object_transforms": link_from_object,
        "generator_storage": "reusable_finite_tuple",
        "grasp_count": 4,
        "link_target_formula": (
            "world_from_object*converted_object_origin_from_aabb_center*inverse(primitive_link_from_aabb_center)"
        ),
        "object_id": subject,
        "pitch_interval": "top",
        "pose_convention": "link_from_object_parent_from_child_homogeneous_4x4",
        "primitive": "cuboid",
        "primitive_link_from_aabb_center_transforms": primitive,
        "profile": "franka_rubiks_cube_offcenter_cuboid_top_v1",
        "schema_version": 1,
        "source": "schedulestream.applications.custream.grasp.primitive_grasp_generator",
    }


def _factory_destination_placement_geometry(subject: str, destination: str) -> dict:
    identity = [list(row) for row in np.eye(4)]
    return {
        "attested": True,
        "destination": {
            "asset_name": "bowl_ycb_robolab",
            "object_id": destination,
            "aabb_center_in_converted_object_origin_m": [0.0, 0.0, 0.0],
            "aabb_dimensions_m": [0.16, 0.16, 0.05],
            "aabb_kind": "converted_mesh_local_axis_aligned_bounding_box",
            "converted_object_origin_from_aabb_center": identity,
        },
        "frame_convention": "parent_from_child_homogeneous_4x4",
        "general_inside_semantics": False,
        "limitations": ["test_surrogate"],
        "placement_model": "destination_local_aabb_top_plane_shifted_downward",
        "predicted_aabb_center_offset_m": [0.0, 0.0, 0.03],
        "profile": "franka_rubiks_cube_to_ycb_bowl_aabb_top_plane_v1",
        "relation": "on",
        "sampled_surface_extent_m": [0.0, 0.0, 0.0],
        "schema_version": 1,
        "subject": {
            "asset_name": "rubiks_cube_hot3d_robolab",
            "object_id": subject,
            "aabb_center_in_converted_object_origin_m": [0.0, 0.0, 0.0],
            "aabb_dimensions_m": [0.06, 0.06, 0.06],
            "aabb_kind": "converted_mesh_local_axis_aligned_bounding_box",
            "converted_object_origin_from_aabb_center": identity,
        },
        "surface_config": {
            "implementation": "schedulestream.applications.custream.object.SurfaceConfig",
            "xy_extend_m": -0.16,
            "z_offset_m": -0.025,
        },
        "vertical_evidence_corridor_m": 0.04,
    }


def test_v1_ik_filter_rejects_violations_and_submilliradian_limit_candidates() -> None:
    positions = torch.tensor([[
        [[0.0, 0.2], [0.1, 0.3]],
        [[1.02, 0.0], [0.0, 0.0]],
        [[0.9995, 0.0], [0.0, 0.0]],
    ]])
    joint_state = SimpleNamespace(position=positions)
    distances = torch.tensor([[0.1, 0.2, 0.3]])

    def get_limit_distances(state):
        lower_difference = -1.0 - state.position
        upper_difference = state.position - 1.0
        return torch.maximum(lower_difference, upper_difference)

    world = SimpleNamespace(
        autodata_ik_joint_limit_evidence={
            "accepted_candidates": 0,
            "calls": 0,
            "candidate_solutions": 0,
            "joint_limit_margin_rad": 1e-3,
            "policy": "all_waypoints_inside_native_position_limits",
            "rejected_candidates": 0,
        },
        get_limit_distances=get_limit_distances,
    )

    returned_state, filtered = _filter_v1_ik_joint_limits(world, joint_state, distances)

    assert returned_state is joint_state
    assert filtered[0, 0].item() == pytest.approx(0.1)
    assert torch.isinf(filtered[0, 1:]).all()
    assert world.autodata_ik_joint_limit_evidence == {
        "accepted_candidates": 1,
        "calls": 1,
        "candidate_solutions": 3,
        "joint_limit_margin_rad": 1e-3,
        "minimum_observed_margin_rad": pytest.approx(-0.02),
        "policy": "all_waypoints_inside_native_position_limits",
        "rejected_candidates": 2,
    }


def test_v1_ik_filter_normalizes_every_nonfinite_distance_sentinel() -> None:
    positions = torch.zeros((1, 4, 1, 1), dtype=torch.float32)
    joint_state = SimpleNamespace(position=positions)
    distances = torch.tensor([[0.1, torch.nan, -torch.inf, torch.inf]], dtype=torch.float32)
    world = SimpleNamespace(
        autodata_ik_joint_limit_evidence={
            "accepted_candidates": 0,
            "calls": 0,
            "candidate_solutions": 0,
            "joint_limit_margin_rad": 1e-3,
            "policy": "all_waypoints_inside_native_position_limits",
            "rejected_candidates": 0,
        },
        get_limit_distances=lambda state: torch.full_like(state.position, -1.0),
    )

    _, filtered = _filter_v1_ik_joint_limits(world, joint_state, distances)

    assert filtered[0, 0].item() == pytest.approx(0.1)
    assert torch.equal(filtered[0, 1:], torch.full((3,), torch.inf))
    assert world.autodata_ik_joint_limit_evidence["candidate_solutions"] == 1
    assert world.autodata_ik_joint_limit_evidence["accepted_candidates"] == 1


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64])
def test_v1_ik_filter_rejects_unsupported_distance_dtype(dtype: torch.dtype) -> None:
    positions = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    joint_state = SimpleNamespace(position=positions)
    distances = torch.zeros((1, 1), dtype=dtype)
    world = SimpleNamespace(get_limit_distances=lambda state: torch.full_like(state.position, -1.0))

    with pytest.raises(ScheduleStreamProviderError, match="distances must use float32 or float64"):
        _filter_v1_ik_joint_limits(world, joint_state, distances)


def test_concrete_v1_factory_compiles_semantic_goal_with_injected_runtime() -> None:
    fake_module = SimpleNamespace(
        CAMERA_POSE="camera",
        Commands=SimpleNamespace(flatten=lambda commands: iter(commands)),
        GraspConfig=_FakeGraspConfig,
        Planner=_BasePlanner,
        World=object,
        animate_commands=lambda *args, **kwargs: [],
        create_controller=lambda *args, **kwargs: None,
        create_objects=lambda *args, **kwargs: [],
        load_franka_config=lambda *args, **kwargs: {},
        multiply_poses=_multiply_factory_poses,
        solve_tamp=lambda *args, **kwargs: None,
        timeout_context=lambda **kwargs: nullcontext(),
    )
    world = _FactoryWorld({
        "cube": _FactoryPose.translated(0.03, -0.01, 0.002),
        "table": _FactoryPose(),
    })
    world.autodata_destination_placement_geometry = _factory_destination_placement_geometry("cube", "table")
    world.autodata_grasp_geometry = _factory_grasp_geometry("cube")
    symbols = ScheduleStreamGoalSymbols(
        attached_equals=lambda subject, target: _Clause((("attached", subject, target),)),
        holding_equals=lambda arm, subject: _Clause((("holding", arm, subject),)),
    )
    env = SimpleNamespace(
        root_poses={"cube": _FactoryPose(), "table": _FactoryPose()},
        scene=SimpleNamespace(
            rigid_objects={
                "cube": _rigid_object(rigid_props_present=False),
                "table": _rigid_object(kinematic_enabled=False),
            }
        ),
    )

    provider = create_v1_isaaclab_command_planner(
        env,
        (GoalPredicate("on", "cube", "table"),),
        graspable_object="cube",
        destination_object="table",
        graspable_asset_name="rubiks_cube_hot3d_robolab",
        destination_asset_name="bowl_ycb_robolab",
        module_loader=lambda: fake_module,
        goal_symbols_loader=lambda application: symbols,
        world_factory=lambda module, scene, config, **kwargs: world,
        eef_pose_reader=_observed_pose,
        seed_setter=lambda *, seed: None,
    )

    assert provider.arm == "arm"
    assert provider.native_planner.goal.values == (("attached", "cube", "table"),)
    assert world.batch_size == 10
    assert provider.native_planner.grasp_configuration["task_selected_graspable_object"] == "cube"
    assert provider.native_planner.grasp_configuration["world_movable_names"] == ["cube"]
    assert provider.native_planner.grasp_configuration["grasp_geometry"]["grasp_count"] == 4
    assert provider.native_planner.grasp_geometry["profile"] == "franka_rubiks_cube_offcenter_cuboid_top_v1"
    np.testing.assert_allclose(
        provider.native_planner.object_pose_offset_evidence["cube"],
        _FactoryPose.translated(0.03, -0.01, 0.002).matrix,
    )
    assert provider.native_planner.destination_placement_geometry["subject"][
        "aabb_center_in_isaac_rigid_root_m"
    ] == pytest.approx([0.03, -0.01, 0.002])

    env.root_poses["cube"] = _FactoryPose.translated(0.2, 0.1, 0.0)
    provider.native_planner.set_env_state(0)

    np.testing.assert_allclose(
        world.object_poses["cube"].matrix,
        _FactoryPose.translated(0.23, 0.09, 0.002).matrix,
    )


def test_root_to_mesh_restore_preserves_rotated_noncommuting_offsets_after_upstream_stomp() -> None:
    def pose_z(theta: float, translation: tuple[float, float, float]) -> _FactoryPose:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        matrix = np.array([
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return _FactoryPose(matrix)

    fake_module = SimpleNamespace(
        CAMERA_POSE="camera",
        Commands=SimpleNamespace(flatten=lambda commands: iter(commands)),
        GraspConfig=_FakeGraspConfig,
        Planner=_BasePlanner,
        World=object,
        animate_commands=lambda *args, **kwargs: [],
        create_controller=lambda *args, **kwargs: None,
        create_objects=lambda *args, **kwargs: [],
        load_franka_config=lambda *args, **kwargs: {},
        multiply_poses=_multiply_factory_poses,
        solve_tamp=lambda *args, **kwargs: None,
        timeout_context=lambda **kwargs: nullcontext(),
    )
    initial_roots = {
        "cube": pose_z(np.pi / 2.0, (0.2, -0.1, 0.0)),
        "bowl": pose_z(-np.pi / 4.0, (-0.3, 0.25, 0.01)),
    }
    offsets = {
        "cube": pose_z(np.pi / 3.0, (0.03, -0.02, 0.005)),
        "bowl": pose_z(-np.pi / 6.0, (-0.04, 0.01, 0.02)),
    }
    world = _FactoryWorld({name: _multiply_factory_poses(initial_roots[name], offsets[name]) for name in initial_roots})
    world.autodata_destination_placement_geometry = _factory_destination_placement_geometry("cube", "bowl")
    world.autodata_grasp_geometry = _factory_grasp_geometry("cube")
    symbols = ScheduleStreamGoalSymbols(
        attached_equals=lambda subject, target: _Clause((("attached", subject, target),)),
        holding_equals=lambda arm, subject: _Clause((("holding", arm, subject),)),
    )
    env = SimpleNamespace(
        root_poses=initial_roots,
        scene=SimpleNamespace(
            rigid_objects={
                "cube": _rigid_object(rigid_props_present=False),
                "bowl": _rigid_object(rigid_props_present=False),
            }
        ),
    )
    provider = create_v1_isaaclab_command_planner(
        env,
        (GoalPredicate("on", "cube", "bowl"),),
        graspable_object="cube",
        destination_object="bowl",
        graspable_asset_name="rubiks_cube_hot3d_robolab",
        destination_asset_name="bowl_ycb_robolab",
        module_loader=lambda: fake_module,
        goal_symbols_loader=lambda application: symbols,
        world_factory=lambda module, scene, config, **kwargs: world,
        eef_pose_reader=_observed_pose,
        seed_setter=lambda *, seed: None,
    )
    new_roots = {
        "cube": pose_z(-np.pi / 5.0, (0.45, 0.15, 0.03)),
        "bowl": pose_z(np.pi / 7.0, (-0.1, -0.2, 0.04)),
    }
    env.root_poses = new_roots

    provider.native_planner.set_env_state(0)

    for name in ("cube", "bowl"):
        expected = _multiply_factory_poses(new_roots[name], offsets[name]).matrix
        np.testing.assert_allclose(world.object_poses[name].matrix, expected, atol=1e-12)
        assert not np.allclose(world.object_poses[name].matrix, new_roots[name].matrix)


def test_concrete_v1_factory_rejects_explicitly_kinematic_goal_subject() -> None:
    fake_module = SimpleNamespace(
        CAMERA_POSE="camera",
        Commands=SimpleNamespace(flatten=lambda commands: iter(commands)),
        GraspConfig=_FakeGraspConfig,
        Planner=_BasePlanner,
        World=object,
        animate_commands=lambda *args, **kwargs: [],
        create_controller=lambda *args, **kwargs: None,
        create_objects=lambda *args, **kwargs: [],
        load_franka_config=lambda *args, **kwargs: {},
        multiply_poses=_multiply_factory_poses,
        solve_tamp=lambda *args, **kwargs: None,
        timeout_context=lambda **kwargs: nullcontext(),
    )
    env = SimpleNamespace(scene=SimpleNamespace(rigid_objects={"cube": _rigid_object(kinematic_enabled=True)}))
    world_factory_calls = []

    with pytest.raises(ScheduleStreamProviderError, match="explicitly kinematic"):
        create_v1_isaaclab_command_planner(
            env,
            (GoalPredicate("on", "cube", "table"),),
            graspable_object="cube",
            destination_object="table",
            graspable_asset_name="rubiks_cube_hot3d_robolab",
            destination_asset_name="bowl_ycb_robolab",
            module_loader=lambda: fake_module,
            world_factory=lambda module, scene, config, **kwargs: world_factory_calls.append(scene),
            eef_pose_reader=_observed_pose,
            seed_setter=lambda *, seed: None,
        )

    assert world_factory_calls == []


def test_concrete_v1_factory_closes_staged_world_when_initialization_fails() -> None:
    fake_module = SimpleNamespace(
        CAMERA_POSE="camera",
        Commands=SimpleNamespace(flatten=lambda commands: iter(commands)),
        GraspConfig=_FakeGraspConfig,
        Planner=_BasePlanner,
        World=object,
        animate_commands=lambda *args, **kwargs: [],
        create_controller=lambda *args, **kwargs: None,
        create_objects=lambda *args, **kwargs: [],
        load_franka_config=lambda *args, **kwargs: {},
        multiply_poses=_multiply_factory_poses,
        solve_tamp=lambda *args, **kwargs: None,
        timeout_context=lambda **kwargs: nullcontext(),
    )

    class FailingWorld(_FactoryWorld):
        def initialize(self, batch_size: int) -> None:
            raise RuntimeError("GPU initialization failed")

    world = FailingWorld()
    closed = []
    env = SimpleNamespace(scene=SimpleNamespace(rigid_objects={"cube": _rigid_object(rigid_props_present=False)}))

    with pytest.raises(ScheduleStreamProviderError, match="failed to construct semantic v1 planner"):
        create_v1_isaaclab_command_planner(
            env,
            (GoalPredicate("on", "cube", "table"),),
            graspable_object="cube",
            destination_object="table",
            graspable_asset_name="rubiks_cube_hot3d_robolab",
            destination_asset_name="bowl_ycb_robolab",
            module_loader=lambda: fake_module,
            world_factory=lambda module, scene, config, **kwargs: world,
            eef_pose_reader=_observed_pose,
            seed_setter=lambda *, seed: None,
            world_closer=closed.append,
        )

    assert closed == [world]
