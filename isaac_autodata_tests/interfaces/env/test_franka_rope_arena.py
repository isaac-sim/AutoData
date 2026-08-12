# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the AutoData-owned Arena Franka rope environment."""

import gymnasium as gym
import sys
from functools import partial
from types import SimpleNamespace

from isaaclab_arena.assets.registries import EnvironmentRegistry

import isaac_autodata_examples.envs as env_registration
import isaac_autodata_examples.envs.isaac_lab_arena as arena_registration
from isaac_autodata_examples.envs.isaac_lab_arena import registration
from isaac_autodata_examples.envs.isaac_lab_arena.franka_rope import mdp
from isaac_autodata_examples.envs.isaac_lab_arena.franka_rope.embodiment import FrankaRopeIKEmbodiment
from isaac_autodata_examples.envs.isaac_lab_arena.franka_rope.environment import (
    FRANKA_ROPE_ARENA_ENV_ID,
    FrankaRopeArenaEnvironment,
    FrankaRopeArenaEnvironmentCfg,
)
from isaac_autodata_examples.envs.isaac_lab_arena.franka_rope.rope_asset import FrankaRopeAsset
from isaac_autodata_examples.envs.isaac_lab_arena.franka_rope.task import FrankaRopeTask


def test_arena_environment_factory_has_typed_config():
    assert FrankaRopeArenaEnvironment.name == FRANKA_ROPE_ARENA_ENV_ID
    assert FrankaRopeArenaEnvironment._legacy_argparse_cfg_type is FrankaRopeArenaEnvironmentCfg


def test_arena_environment_factory_is_registered():
    registry = EnvironmentRegistry()
    assert registry.get_component_by_name(FRANKA_ROPE_ARENA_ENV_ID) is FrankaRopeArenaEnvironment
    assert registry.get_environment_cfg_type(FrankaRopeArenaEnvironment) is FrankaRopeArenaEnvironmentCfg


def test_arena_registration_uses_run_configuration_and_preserves_make_kwargs(monkeypatch):
    from isaaclab_arena.environments import arena_env_builder

    captured = {}
    variation_recorder = object()

    class FakeArenaEnvBuilder:
        def __init__(self, arena_environment, cfg):
            captured["arena_environment"] = arena_environment
            captured["cfg"] = cfg

        def build_registered(self):
            return FRANKA_ROPE_ARENA_ENV_ID, object(), {"variation_recorder": variation_recorder}

    monkeypatch.delitem(gym.registry, FRANKA_ROPE_ARENA_ENV_ID, raising=False)
    monkeypatch.setattr(arena_env_builder, "ArenaEnvBuilder", FakeArenaEnvBuilder)
    monkeypatch.setattr(
        FrankaRopeArenaEnvironment,
        "build",
        lambda _self, cfg: SimpleNamespace(
            name=FRANKA_ROPE_ARENA_ENV_ID,
            embodiment=SimpleNamespace(enable_cameras=cfg.enable_cameras),
        ),
    )

    make_kwargs = registration.build_and_register_arena_environment(
        enable_cameras=True,
        num_envs=8,
        device="cuda:1",
        seed=23,
    )

    builder_cfg = captured["cfg"]
    assert captured["arena_environment"].name == FRANKA_ROPE_ARENA_ENV_ID
    assert captured["arena_environment"].embodiment.enable_cameras is True
    assert builder_cfg.num_envs == 8
    assert builder_cfg.env_spacing == 2.5
    assert builder_cfg.seed == 23
    assert builder_cfg.solve_relations is False
    assert builder_cfg.device == "cuda:1"
    assert make_kwargs == {FRANKA_ROPE_ARENA_ENV_ID: {"variation_recorder": variation_recorder}}


def test_regular_isaac_lab_run_does_not_build_arena(monkeypatch):
    arena_registration_called = False

    def register_arena_environment(**_kwargs):
        nonlocal arena_registration_called
        arena_registration_called = True
        return {}

    monkeypatch.setattr(
        arena_registration, "build_and_register_arena_environment", register_arena_environment
    )

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


def test_external_callback_binds_make_kwargs_without_gym_deepcopy():
    test_env_name = "AutoData-Test-Arena-Callback-v0"
    variation_recorder = object()
    gym.register(id=test_env_name, entry_point=lambda **_kwargs: None)
    try:
        registration._bind_gym_make_kwargs(  # noqa: SLF001
            test_env_name,
            {"variation_recorder": variation_recorder},
        )

        env_spec = gym.spec(test_env_name)
        assert isinstance(env_spec.entry_point, partial)
        assert env_spec.entry_point.keywords["variation_recorder"] is variation_recorder
        assert "variation_recorder" not in env_spec.kwargs
    finally:
        del gym.registry[test_env_name]


def test_rope_asset_uses_deformable_scene_key():
    rope = FrankaRopeAsset()
    name, object_cfg = rope.get_object_cfg()
    assert name == "object"
    assert object_cfg.prim_path == "{ENV_REGEX_NS}/Object"
    assert object_cfg.spawn.usd_path.endswith("/isaac_lab_arena/franka_rope/assets/Rope.usd")


def test_rope_mdp_terms_are_owned_by_arena_package():
    arena_module_prefix = "isaac_autodata_examples.envs.isaac_lab_arena.franka_rope.mdp"
    assert mdp.object_nodal_pos.__module__.startswith(arena_module_prefix)
    assert mdp.reset_rope_nodal_state.__module__.startswith(arena_module_prefix)
    assert mdp.rope_ends_close_tracked.__module__.startswith(arena_module_prefix)


def test_rope_task_exposes_annotation_and_success_terms():
    task = FrankaRopeTask()
    assert task.get_observation_cfg().subtask_terms.grasp is not None
    assert task.get_termination_cfg().success is not None
    assert task.get_events_cfg().reset_object_position is not None


def test_rope_embodiment_preserves_arena_observation_names():
    embodiment = FrankaRopeIKEmbodiment(enable_cameras=False)
    assert embodiment.name == "franka_rope_ik"
    assert embodiment.observation_config.policy.eef_pos is not None
    assert embodiment.observation_config.policy.eef_quat is not None
    assert embodiment.observation_config.policy.object_nodal_pos is not None
    assert embodiment.event_config.randomize_franka_joint_state.params["position_range"] == (-0.02, 0.02)
    assert embodiment.reward_config is None
