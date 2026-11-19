# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision-world frames, obstacle state and plan sampling without simulator startup."""

import math
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("curobo.geom.sdf.world")
from curobo.geom.types import Cuboid, WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner  # noqa: E402


def _planner(world):
    """Create a CPU-only planner shell with a mutable native world and enable flags."""
    planner = object.__new__(CuroboPlanner)
    planner.tensor_args = TensorDeviceType(device=torch.device("cpu"))
    planner.env_id = 0
    planner.logger = Mock()
    planner.datastream = SimpleNamespace(device="cpu")
    planner.config = SimpleNamespace(static_objects=[], include_configured_world=True)
    planner._base_pose_in_env = torch.eye(4)
    planner._env_to_base_transform = torch.eye(4)
    planner._base_frame_is_identity = True
    planner._static_world_base_pose = torch.eye(4)
    planner._static_world_config = world
    planner._collision_world_owner = planner
    planner._cached_object_mappings = None
    planner._object_obstacle_transforms = {}
    planner.primitive_types = ["cuboid"]
    names = [obstacle.name for obstacle in world.cuboid or []]
    flags = torch.ones(1, len(names), dtype=torch.uint8)

    def enable(name, enable=True):
        flags[0, names.index(name)] = enable

    def update(name, pose, **kwargs):
        next(obstacle for obstacle in world if obstacle.name == name).pose = planner._object_pose_to_curobo_list(
            pose.get_matrix().reshape(4, 4)
        )

    checker = SimpleNamespace(
        world_model=world,
        _env_obbs_names=[names],
        _cube_tensor_list=[None, None, flags],
        enable_obstacle=enable,
        update_obstacle_pose=update,
    )
    planner.motion_gen = SimpleNamespace(world_coll_checker=checker)
    return planner, flags


def test_nested_exclusions_restore_the_original_enable_flags():
    world = WorldConfig(
        cuboid=[Cuboid(name=name, pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]) for name in ("held", "wall")]
    )
    planner, flags = _planner(world)
    flags[0, 0] = 0
    with planner._world_obstacles_disabled(("held", "wall")):
        with planner._world_obstacles_disabled(("wall",)):
            assert not flags.any()
        assert not flags.any()
    assert flags.tolist() == [[0, 1]]


@pytest.mark.parametrize("root_name", [None, "base"])
def test_configured_embodiment_frame_is_rejected_before_world_conversion(root_name):
    planner, _ = _planner(WorldConfig())
    planner.root_pose_source_frame_name = root_name
    planner.datastream.embodiment_adapter = SimpleNamespace(
        get_reference_frame_pose=lambda **kwargs: torch.eye(4).unsqueeze(0)
    )
    with pytest.raises(AssertionError, match="configured embodiment reference frames"):
        planner._refresh_base_frame_transform()


def test_self_collision_offsets_remove_native_world_padding(tmp_path):
    from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGenerator, CudaRobotGeneratorConfig

    urdf = tmp_path / "two_links.urdf"
    urdf.write_text(
        '<robot name="test"><link name="base"/><link name="tip"/>'
        '<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    generator = CudaRobotGenerator(
        CudaRobotGeneratorConfig(
            base_link="base",
            ee_link="tip",
            urdf_path=str(urdf),
            tensor_args=TensorDeviceType(device=torch.device("cpu")),
            collision_link_names=["base", "tip"],
            collision_spheres={name: [{"center": [0, 0, 0], "radius": 0.1}] for name in ("base", "tip")},
            collision_sphere_buffer=0.05,
            self_collision_buffer={},
            self_collision_ignore={"base": [], "tip": []},
        )
    )
    planner, _ = _planner(WorldConfig())
    planner._kinematics = SimpleNamespace(self_collision_config=generator.self_collision_config)
    planner._self_collision_pair_cache = None
    planner._self_collision_ignore = {"base": (), "tip": ()}
    planner.collision_sphere_link_names = lambda count: ("base", "tip")
    spheres = torch.tensor([
        [[0, 0, 0, 0.15], [0.25, 0, 0, 0.15]],
        [[0, 0, 0, 0.15], [0.15, 0, 0, 0.15]],
    ])
    assert planner._collision_spheres_self_collisions(spheres, 0).tolist() == [False, True]


def test_shared_context_keeps_the_owner_world_and_disabled_obstacles():
    world = WorldConfig(cuboid=[Cuboid(name="held", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])])
    owner, flags = _planner(world)
    flags.zero_()
    shared, _ = _planner(WorldConfig())
    owner.planning_root_frame_name = shared.planning_root_frame_name = "base"
    shared._collision_world_owner = owner
    shared._initialize_static_world(activate=False)
    assert shared._static_world_config is world
    assert not flags.any()


@pytest.mark.parametrize("rotate", [False, True])
def test_root_motion_reexpresses_the_world_without_reenabling_obstacles(rotate):
    world = WorldConfig(cuboid=[Cuboid(name="wall", pose=[3, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])])
    planner, flags = _planner(world)
    flags.zero_()
    planner._base_pose_in_env[0, 3] = 1
    if rotate:
        planner._base_pose_in_env[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    planner._env_to_base_transform = torch.linalg.inv(planner._base_pose_in_env)
    planner._static_world_refresh()
    expected = [0, -2, 0] if rotate else [2, 0, 0]
    assert world.cuboid[0].pose[:3] == pytest.approx(expected, abs=1e-6)
    assert not flags.any()
    planner._static_world_refresh()
    assert world.cuboid[0].pose[:3] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("child_count", [1, 2])
@pytest.mark.parametrize("rotate", [False, True])
def test_object_pose_sync_preserves_child_offsets_and_compound_geometry(child_count, rotate):
    root_path = "/World/envs/env_0/Object"
    stage = Usd.Stage.CreateInMemory()
    env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.AddTranslateOp().Set((3, 0, 0))
    if rotate:
        root.AddRotateZOp().Set(90)
    world = WorldConfig(
        cuboid=[
            Cuboid(
                name=f"{root_path}/part{index}",
                pose=(
                    [3, index / 4, 0, math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
                    if rotate
                    else [3 + index / 4, 0, 0, 1, 0, 0, 0]
                ),
                dims=[0.1] * 3,
            )
            for index in range(1, child_count + 1)
        ]
    )
    planner, _ = _planner(world)
    live_pose = torch.eye(4)
    live_pose[0, 3] = 10
    planner.datastream = SimpleNamespace(
        device="cpu",
        get_usd_stage=lambda: stage,
        get_env_prim_path=lambda index: str(env.GetPath()),
        get_object_prim_path=lambda name, index: root_path,
        get_object_poses=lambda **kwargs: {"object": live_pose.unsqueeze(0)},
    )
    planner._sync_object_poses_with_isaaclab()
    assert [obstacle.pose[0] for obstacle in world] == pytest.approx(
        [10 + index / 4 for index in range(1, child_count + 1)]
    )
    if child_count == 1:
        assert torch.allclose(planner.get_object_pose("object").position, torch.tensor([[10.0, 0, 0]]))
    else:
        assert planner._get_object_mappings() == {}


@pytest.mark.parametrize("retreat, approach", [(1.0, 0.0), (0.0, 1.0)])
def test_world_offsets_are_rotated_into_the_planning_frame(retreat, approach):
    planner, _ = _planner(WorldConfig())
    planner.config = SimpleNamespace(approach_direction=(0, 0, -1), approach_retreat_frame="world")
    planner._env_to_base_transform[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    goal = planner._make_pose(position=[0, 0, 0])
    planner.get_ee_pose = lambda state: goal
    planner._plan_to_contact_pose = Mock(return_value=False)
    assert not planner._plan_to_contact(None, goal, retreat, approach)
    target = planner._plan_to_contact_pose.call_args.kwargs["goal_pose"]
    assert torch.allclose(target.position, torch.tensor([[0.0, 1.0, 0.0]]))


def test_gripper_lock_reapplies_live_joint_limits():
    planner, _ = _planner(WorldConfig())
    planner.config = SimpleNamespace(gripper_closed_positions={"finger": 0.1})
    planner.robot_cfg = {}
    planner.motion_gen.update_locked_joints = Mock()
    planner.motion_gen.kinematics = SimpleNamespace(get_joint_limits=lambda: "limits")
    planner._joint_limits_sync = Mock()
    planner._set_gripper_state(True)
    planner._joint_limits_sync.assert_called_once_with("limits")


@pytest.mark.parametrize("interpolate", [False, True])
def test_legacy_and_controller_rate_plan_sampling_are_explicit(interpolate):
    planner, _ = _planner(WorldConfig())
    planner.config = SimpleNamespace(hand_link_names=[], planning_group_name="arm", interpolate_plan=interpolate)
    planner.datastream.get_robot_joint_names = lambda: ["joint"]
    optimized = SimpleNamespace(position=torch.zeros(2, 1), joint_names=["joint"])
    interpolated = SimpleNamespace(position=torch.zeros(5, 1), joint_names=["joint"])
    for value in (optimized, interpolated):
        value.get_ordered_joint_state = lambda names, value=value: value
    planner.motion_gen.plan_single = Mock(
        return_value=SimpleNamespace(
            success=torch.tensor(True),
            optimized_plan=optimized,
            get_interpolated_plan=lambda: interpolated,
        )
    )
    planner.motion_gen.rollout_fn = SimpleNamespace(
        primitive_collision_constraint=Mock(),
        robot_self_collision_constraint=Mock(),
    )
    planner._count_active_spheres = lambda: {"total": 0, "robot_links": 0, "attached_objects": 0}
    planner._planning_joint_state_expand = lambda state: state
    planner.plan_config = None
    assert planner._plan_to_contact_pose(None, None, contact=False)
    assert planner.current_plan is (interpolated if interpolate else optimized)
