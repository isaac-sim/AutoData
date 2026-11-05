# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the AutoData-owned Arena Franka rope environment."""

import gymnasium as gym
import sys
import torch
from functools import partial
from types import SimpleNamespace

import pytest
from isaaclab.managers import TerminationTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import task_success_from_progress
from isaaclab_arena.utils.physics_backend import PhysicsBackend

import autodata_examples.envs as env_registration
import autodata_examples.envs.isaac_lab_arena as arena_registration
from autodata_examples.envs.isaac_lab_arena import registration
from autodata_examples.envs.isaac_lab_arena.franka_rope import mdp
from autodata_examples.envs.isaac_lab_arena.franka_rope.embodiment import (
    FrankaRopeIKEmbodiment,
    FrankaRopeObservationsCfg,
)
from autodata_examples.envs.isaac_lab_arena.franka_rope.environment import (
    FRANKA_ROPE_ARENA_ENV_ID,
    FrankaRopeArenaEnvironment,
    FrankaRopeArenaEnvironmentCfg,
    _configure_franka_rope_env,
)
from autodata_examples.envs.isaac_lab_arena.franka_rope.rope_asset import FrankaRopeAsset, rope_newton_spawn_cfg
from autodata_examples.envs.isaac_lab_arena.franka_rope.task import FrankaRopeTask


def test_arena_environment_factory_has_typed_config():
    assert FrankaRopeArenaEnvironment.name == FRANKA_ROPE_ARENA_ENV_ID
    assert FrankaRopeArenaEnvironment._legacy_argparse_cfg_type is FrankaRopeArenaEnvironmentCfg


def test_arena_environment_factory_is_registered():
    registry = EnvironmentRegistry()
    assert registry.get_component_by_name(FRANKA_ROPE_ARENA_ENV_ID) is FrankaRopeArenaEnvironment
    assert registry.get_environment_cfg_type(FrankaRopeArenaEnvironment) is FrankaRopeArenaEnvironmentCfg


@pytest.mark.parametrize("presets", [None, "physx", "newton"])
def test_arena_registration_uses_run_configuration_and_preserves_make_kwargs(monkeypatch, presets):
    captured = {}
    variation_recorder = object()

    class FakeArenaEnvBuilder:
        def __init__(self, arena_environment, cfg):
            captured["arena_environment"] = arena_environment
            captured["cfg"] = cfg

        def build_registered(self):
            return FRANKA_ROPE_ARENA_ENV_ID, object(), {"variation_recorder": variation_recorder}

    monkeypatch.delitem(gym.registry, FRANKA_ROPE_ARENA_ENV_ID, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.environments.arena_env_builder",
        SimpleNamespace(ArenaEnvBuilder=FakeArenaEnvBuilder),
    )
    monkeypatch.setattr(
        FrankaRopeArenaEnvironment,
        "build",
        lambda _self, cfg: SimpleNamespace(
            name=FRANKA_ROPE_ARENA_ENV_ID,
            embodiment=SimpleNamespace(enable_cameras=cfg.enable_cameras),
        ),
    )

    make_kwargs = env_registration.register_environment_for_run(
        env_name=FRANKA_ROPE_ARENA_ENV_ID,
        enable_cameras=True,
        num_envs=8,
        device="cuda:1",
        seed=23,
        presets=presets,
    )

    builder_cfg = captured["cfg"]
    assert captured["arena_environment"].name == FRANKA_ROPE_ARENA_ENV_ID
    assert captured["arena_environment"].embodiment.enable_cameras is True
    assert builder_cfg.num_envs == 8
    assert builder_cfg.env_spacing == 2.5
    assert builder_cfg.seed == 23
    assert builder_cfg.solve_relations is False
    assert builder_cfg.device == "cuda:1"
    assert builder_cfg.presets == (None if presets is None else PhysicsBackend(presets))
    assert make_kwargs == {FRANKA_ROPE_ARENA_ENV_ID: {"variation_recorder": variation_recorder}}


def test_regular_isaac_lab_run_does_not_build_arena(monkeypatch):
    arena_registration_called = False

    def register_arena_environment(**_kwargs):
        nonlocal arena_registration_called
        arena_registration_called = True
        return {}

    monkeypatch.setattr(arena_registration, "build_and_register_arena_environment", register_arena_environment)

    make_kwargs = env_registration.register_environment_for_run(
        env_name="Isaac-Regular-Lab-Env-v0",
        enable_cameras=False,
        num_envs=4,
        device="cuda:0",
        seed=17,
    )

    assert make_kwargs == {}
    assert arena_registration_called is False


def test_external_callback_routes_arena_task(monkeypatch):
    callback_called = False

    def register_arena_environment_from_cli():
        nonlocal callback_called
        callback_called = True
        return ["hydra.option=value"]

    monkeypatch.setattr(arena_registration, "register_environment_from_cli", register_arena_environment_from_cli)
    monkeypatch.setattr(sys, "argv", ["replay_demos.py", "--task", FRANKA_ROPE_ARENA_ENV_ID])

    remaining_args = env_registration.register_environments()

    assert callback_called is True
    assert remaining_args == ["hydra.option=value"]


def test_external_callback_skips_arena_for_regular_lab_task(monkeypatch):
    callback_called = False

    def register_arena_environment_from_cli():
        nonlocal callback_called
        callback_called = True
        return []

    monkeypatch.setattr(arena_registration, "register_environment_from_cli", register_arena_environment_from_cli)
    monkeypatch.setattr(sys, "argv", ["replay_demos.py", "--task", "Isaac-Regular-Lab-Env-v0"])

    assert env_registration.register_environments() == []
    assert callback_called is False


def test_external_callback_binds_make_kwargs_without_gym_deepcopy(monkeypatch):
    test_env_name = FRANKA_ROPE_ARENA_ENV_ID
    variation_recorder = object()
    env_spec = gym.envs.registration.EnvSpec(id=test_env_name, entry_point=lambda **kwargs: kwargs)
    monkeypatch.setitem(gym.registry, test_env_name, env_spec)
    registration._bind_gym_make_kwargs(  # noqa: SLF001
        test_env_name,
        {"variation_recorder": variation_recorder},
    )

    assert isinstance(env_spec.entry_point, partial)
    assert env_spec.entry_point.keywords["variation_recorder"] is variation_recorder
    assert "variation_recorder" not in env_spec.kwargs

    make_kwargs = env_registration.register_environment_for_run(
        env_name=test_env_name,
        enable_cameras=False,
        num_envs=1,
        device="cpu",
        seed=1,
        presets="physx",
    )

    assert make_kwargs == {}
    assert env_spec.entry_point(**make_kwargs.get(test_env_name, {}))["variation_recorder"] is variation_recorder


def test_rope_asset_uses_deformable_scene_key():
    rope = FrankaRopeAsset()
    name, object_cfg = rope.get_object_cfg()
    assert name == "object"
    assert object_cfg.prim_path == "{ENV_REGEX_NS}/Object"
    assert object_cfg.spawn.usd_path.endswith("/isaac_lab_arena/franka_rope/assets/Rope.usd")


def test_newton_rope_preserves_simulation_topology():
    from pxr import Usd

    physx_stage = Usd.Stage.Open(FrankaRopeAsset().object_cfg.spawn.usd_path)
    newton_stage = Usd.Stage.Open(rope_newton_spawn_cfg().usd_path)
    for attribute in ("points", "tetVertexIndices", "omniphysics:restShapePoints", "omniphysics:restTetVtxIndices"):
        original = physx_stage.GetPrimAtPath("/World/simulation_mesh").GetAttribute(attribute).Get()
        adapted = newton_stage.GetPrimAtPath("/World/simulation_mesh").GetAttribute(attribute).Get()
        assert original is not None
        assert original == adapted
    assert len(newton_stage.GetPrimAtPath("/World/simulation_mesh").GetAttribute("points").Get()) == 549
    assert physx_stage.GetPrimAtPath("/World/collision_mesh").IsActive()
    assert not newton_stage.GetPrimAtPath("/World/collision_mesh").IsActive()
    assert physx_stage.GetPrimAtPath("/World/Xform").IsActive()
    assert newton_stage.GetPrimAtPath("/World/Xform").IsActive()
    for attribute in ("points", "faceVertexCounts", "faceVertexIndices", "primvars:st"):
        original = physx_stage.GetPrimAtPath("/World/Xform").GetAttribute(attribute).Get()
        adapted = newton_stage.GetPrimAtPath("/World/Xform").GetAttribute(attribute).Get()
        assert adapted == original
    assert sum(prim.GetTypeName() == "TetMesh" for prim in newton_stage.Traverse()) == 1


def test_rope_mdp_terms_are_owned_by_arena_package():
    arena_module_prefix = "autodata_examples.envs.isaac_lab_arena.franka_rope.mdp"
    assert mdp.object_nodal_pos.__module__.startswith(arena_module_prefix)
    assert mdp.reset_rope_nodal_state.__module__.startswith(arena_module_prefix)
    assert mdp.rope_ends_close_tracked.__module__.startswith(arena_module_prefix)


def test_rope_task_exposes_annotation_and_success_terms():
    task = FrankaRopeTask(episode_length_s=15.0)
    assert task.get_observation_cfg().subtask_terms.grasp is not None
    terminations = task.get_termination_cfg()
    assert isinstance(terminations, TaskTerminationCfg)
    assert terminations.timeout_s == 15.0
    assert len(terminations.success) == 1
    assert terminations.success[0].predicate_sequence == [mdp.rope_ends_close_tracked]
    assert terminations.failures["object_dropping"].func is mdp.rope_below_minimum
    assert task.get_events_cfg().reset_object_position is not None


def test_rope_environment_configures_lab_timing():
    env_cfg = SimpleNamespace(
        sim=SimulationCfg(),
        scene=SimpleNamespace(replicate_physics=True),
        terminations=SimpleNamespace(success=None),
    )

    assert _configure_franka_rope_env(env_cfg) is env_cfg
    assert env_cfg.decimation * env_cfg.sim.dt == 0.05
    assert env_cfg.sim.render_interval == 2
    assert env_cfg.num_rerenders_on_reset == 1
    assert env_cfg.scene.replicate_physics is False
    assert set(env_cfg.teleop_devices.devices) == {"keyboard", "spacemouse"}
    assert env_cfg.terminations.success.func is mdp.rope_success_current


def test_rope_embodiment_preserves_arena_observation_names():
    embodiment = FrankaRopeIKEmbodiment(enable_cameras=False)
    assert embodiment.name == "franka_rope_ik"
    assert embodiment.observation_config.policy.eef_pos is not None
    assert embodiment.observation_config.policy.eef_quat is not None
    assert embodiment.observation_config.policy.object_nodal_pos is not None
    assert embodiment.event_config.randomize_franka_joint_state.params["position_range"] == (-0.02, 0.02)
    assert embodiment.reward_config is None
    for camera_name in embodiment.camera_config.camera_names():
        camera = getattr(embodiment.camera_config, camera_name)
        assert camera.renderer_cfg.renderer_type == "isaac_rtx"
        assert camera.renderer_cfg.global_settings.antialiasing_mode == "DLSS"


def test_gripper_observation_reads_finger_joints():
    joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
    positions = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.04, 0.03]])
    robot = SimpleNamespace(
        joint_names=joint_names,
        num_joints=len(joint_names),
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=positions)),
        find_joints=lambda names, preserve_order=False: resolve_matching_names(
            names, joint_names, preserve_order=preserve_order
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})
    term = FrankaRopeObservationsCfg().policy.gripper_pos
    term.params["robot_cfg"].resolve(env.scene)

    observed = term.func(env, **term.params)

    torch.testing.assert_close(observed, torch.tensor([[0.04, -0.03]]))


def test_newton_embodiment_compensates_gravity_without_changing_physx():
    physx = FrankaRopeIKEmbodiment()
    newton = FrankaRopeIKEmbodiment()
    physx.configure_physics_backend(PhysicsBackend.PHYSX)
    newton.configure_physics_backend(PhysicsBackend.NEWTON)
    assert physx.scene_config.robot.spawn.rigid_props.disable_gravity is True
    assert physx.scene_config.robot.spawn.collision_props is None
    assert newton.scene_config.robot.spawn.rigid_props.gravcomp == 1.0
    assert newton.scene_config.robot.spawn.make_uninstanceable is True
    assert newton.scene_config.robot.spawn.collision_props.sdf_max_resolution == 64
    assert physx.scene_config.robot.actuators["panda_shoulder"].stiffness == 400.0
    assert newton.scene_config.robot.actuators["panda_shoulder"].stiffness == 800.0
    assert physx.action_config.gripper_action.close_command_expr == {"panda_finger_.*": 0.0}
    assert newton.action_config.gripper_action.close_command_expr == {"panda_finger_.*": 0.0}
    for name in newton.camera_config.camera_names():
        assert getattr(newton.camera_config, name).renderer_cfg.renderer_type == "newton_warp"
        assert getattr(physx.camera_config, name).renderer_cfg.renderer_type == "isaac_rtx"


def test_newton_rope_reset_preserves_tetrahedra(monkeypatch):
    """Reset poses must not invert or excessively stretch the volume mesh."""
    import itertools
    import numpy as np
    import torch

    from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
    from isaaclab.utils import math as math_utils
    from pxr import Gf, Usd, UsdGeom

    from autodata_examples.envs.isaac_lab_arena.franka_rope.mdp import events

    stage = Usd.Stage.Open(FrankaRopeAsset().object_cfg.spawn.usd_path)
    mesh = UsdGeom.TetMesh(stage.GetPrimAtPath("/World/simulation_mesh"))
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    nodes = torch.tensor(
        np.asarray([transform.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get()]),
        dtype=torch.float32,
    )
    # Match the task's initial orientation: the rope's long axis is world Y.
    nodes = math_utils.quat_apply(torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]).expand(len(nodes), -1), nodes)
    corners = torch.tensor(list(itertools.product((0.0, 0.1), (0.0, 0.1), (-torch.pi / 6, torch.pi / 6))))
    count = len(corners)
    state = torch.zeros(count, len(nodes), 6)
    state[..., :3] = nodes
    written = {}
    rope = SimpleNamespace(device="cpu", data=SimpleNamespace(default_nodal_state_w=SimpleNamespace(torch=state)))
    rope.transform_nodal_pos = lambda *args: BaseDeformableObject.transform_nodal_pos(rope, *args)
    rope.write_nodal_state_to_sim_index = lambda values, env_ids: written.update(state=values, env_ids=env_ids)
    translations = torch.zeros(count, 3)
    translations[:, :2] = corners[:, :2]
    rotations = math_utils.quat_from_euler_xyz(torch.zeros(count), torch.zeros(count), corners[:, 2])
    samples = iter(
        ((torch.zeros(count, 3), torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(count, 1)), (translations, rotations))
    )
    monkeypatch.setattr(events, "_sample_pose_delta", lambda *_: next(samples))
    env_ids = torch.arange(count)
    events.reset_rope_nodal_state(SimpleNamespace(scene={"object": rope}), env_ids, {}, {}, transition_length=0.4)
    indices = torch.tensor(np.asarray(mesh.GetTetVertexIndicesAttr().Get()), dtype=torch.long)
    rest = nodes[indices[:, 1:]] - nodes[indices[:, :1]]
    result = written["state"][..., :3]
    deformed = result[:, indices[:, 1:]] - result[:, indices[:, :1]]
    gradient = torch.linalg.solve(rest.transpose(-1, -2), deformed.transpose(-1, -2), left=False)
    assert torch.linalg.det(gradient).min() > 0.25
    assert torch.linalg.svdvals(gradient).max() < 2.0
    torch.testing.assert_close(written["state"][..., 3:], torch.zeros_like(state[..., 3:]))
    torch.testing.assert_close(written["env_ids"], env_ids)
    torch.testing.assert_close(state[..., :3], nodes.expand(count, -1, -1))


def test_rope_reset_smoothing_is_newton_only():
    from isaaclab_newton.physics import NewtonCfg

    for physics in (None, NewtonCfg()):
        task = FrankaRopeTask()
        cfg = SimpleNamespace(
            sim=SimulationCfg() if physics is None else SimulationCfg(physics=physics),
            scene=SimpleNamespace(
                object=FrankaRopeAsset().object_cfg,
                table=SimpleNamespace(spawn=SimpleNamespace()),
            ),
            events=task.get_events_cfg(),
            terminations=SimpleNamespace(success=None),
        )
        _configure_franka_rope_env(cfg)
        params = cfg.events.reset_object_position.params
        if physics is None:
            assert "transition_length" not in params
        else:
            assert params["transition_length"] == 0.4
    assert "transition_length" not in FrankaRopeTask().get_events_cfg().reset_object_position.params


def test_newton_rope_contact_buffer_fits_full_surface():
    from collections import Counter
    from itertools import combinations

    from pxr import Usd, UsdGeom

    from autodata_examples.envs.isaac_lab_arena.franka_rope.newton_cfg import rope_newton_physics_cfg

    stage = Usd.Stage.Open(rope_newton_spawn_cfg().usd_path)
    mesh = UsdGeom.TetMesh(stage.GetPrimAtPath("/World/simulation_mesh"))
    face_counts = Counter(
        tuple(sorted(face)) for tet in mesh.GetTetVertexIndicesAttr().Get() for face in combinations(tet, 3)
    )
    surface_faces = [face for face, count in face_counts.items() if count == 1]
    surface_edges = {edge for face in surface_faces for edge in combinations(face, 2)}
    contact_count = len(mesh.GetPointsAttr().Get()) + len(surface_faces) + len(surface_edges)
    soft_solver = next(
        entry.solver_cfg for entry in rope_newton_physics_cfg().solver_cfg.entries if entry.name == "soft"
    )
    assert soft_solver.rigid_body_particle_contact_buffer_size >= contact_count


def test_recording_success_rechecks_rope_condition_without_losing_progress():
    nodes = torch.zeros(1, 6, 3)
    rope = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=SimpleNamespace(torch=nodes)))
    env = SimpleNamespace(scene={"object": rope}, device="cpu", num_envs=1, _rope_end_indices={0: (0, 5)})
    tracker = ProgressTracker(FrankaRopeTask().get_termination_cfg().success, 1, "cpu", env=env)
    env.progress_tracker = tracker
    env.update_task_progress = lambda: tracker.step(env)
    env_cfg = SimpleNamespace(
        sim=SimulationCfg(),
        scene=SimpleNamespace(replicate_physics=True),
        terminations=SimpleNamespace(success=TerminationTermCfg(func=task_success_from_progress)),
    )
    _configure_franka_rope_env(env_cfg)
    success_term = env_cfg.terminations.success
    current_success, completed_progress = [], []

    for end_x in (0.06, 1.0, 1.0):
        nodes[0, :, 0] = torch.tensor([0.0, 0.001, 0.002, end_x, end_x + 0.001, end_x + 0.002])
        current_success.append(bool(success_term.func(env, **success_term.params)[0]))
        completed_progress.append(bool(tracker.is_complete()[0]))

    assert current_success == [True, False, False]
    assert completed_progress == [True, True, True]
