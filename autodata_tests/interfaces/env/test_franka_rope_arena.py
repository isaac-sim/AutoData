# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the AutoData-owned Arena Franka rope environment."""

import gymnasium as gym
import sys
import torch
from functools import partial
from types import SimpleNamespace

import pytest
from isaaclab.sim import SimulationCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
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
from autodata_examples.envs.isaac_lab_arena.franka_rope.rope_asset import FrankaRopeAsset
from autodata_examples.envs.isaac_lab_arena.franka_rope.task import FrankaRopeTask


def test_arena_environment_factory_has_typed_config():
    assert FrankaRopeArenaEnvironment.name == FRANKA_ROPE_ARENA_ENV_ID
    assert FrankaRopeArenaEnvironment._legacy_argparse_cfg_type is FrankaRopeArenaEnvironmentCfg


def test_arena_environment_factory_is_registered():
    registry = EnvironmentRegistry()
    assert registry.get_component_by_name(FRANKA_ROPE_ARENA_ENV_ID) is FrankaRopeArenaEnvironment
    assert registry.get_environment_cfg_type(FrankaRopeArenaEnvironment) is FrankaRopeArenaEnvironmentCfg


@pytest.mark.parametrize("presets", [None, "physx"])
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
    env_cfg = SimpleNamespace(sim=SimulationCfg(), scene=SimpleNamespace(replicate_physics=True))

    assert _configure_franka_rope_env(env_cfg) is env_cfg
    assert env_cfg.decimation * env_cfg.sim.dt == 0.05
    assert env_cfg.sim.render_interval == 2
    assert env_cfg.num_rerenders_on_reset == 1
    assert env_cfg.scene.replicate_physics is False
    assert set(env_cfg.teleop_devices.devices) == {"keyboard", "spacemouse"}


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
