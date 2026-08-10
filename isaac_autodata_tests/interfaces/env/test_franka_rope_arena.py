# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the AutoData-owned Arena Franka rope environment."""

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
