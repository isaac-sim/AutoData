# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :func:`autodata_interfaces.env.isaaclab_env_interface.apply_env_profile`.

Applies the shipped bin-stack profile to the parsed neutral env config. Imports ``isaaclab``
config modules (no simulation app); skipped where they are unavailable.
"""

import copy
import os

import pytest

pytest.importorskip("isaaclab")

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from autodata_interfaces.env import EnvironmentProfile, apply_env_profile  # noqa: E402
from autodata_tests.utils.constants import TestPaths  # noqa: E402

BASE_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-v0"


def _parse_base_cfg():
    return parse_env_cfg(BASE_ENV, device="cpu", num_envs=1)


def _load_bin_profile() -> EnvironmentProfile:
    return EnvironmentProfile.from_yaml(os.path.join(TestPaths.env_profiles_dir, "franka_bin_stack.yaml"))


def test_apply_bin_stack_profile_overlays_scene_and_events():
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()

    apply_env_profile(env_cfg, profile, BASE_ENV)

    # Added bin asset.
    bin_cfg = env_cfg.scene.blue_sorting_bin
    assert isinstance(bin_cfg, RigidObjectCfg)
    assert bin_cfg.prim_path == "{ENV_REGEX_NS}/BlueSortingBin"
    assert bin_cfg.spawn.usd_path.endswith("Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd")
    assert "{ISAACLAB_NUCLEUS_DIR}" not in bin_cfg.spawn.usd_path
    assert bin_cfg.spawn.scale == (1.1, 1.6, 3.3)
    assert bin_cfg.init_state.pos == (0.4, 0.0, 0.0203)
    assert bin_cfg.init_state.rot == (0.0, 0.0, 0.0, 1.0)  # identity in (x, y, z, w)

    # Patched cube physics.
    for cube in ("cube_1", "cube_2", "cube_3"):
        assert getattr(env_cfg.scene, cube).spawn.rigid_props.solver_position_iteration_count == 40

    # Removed and added reset events.
    assert env_cfg.events.randomize_cube_positions is None
    for term_name in ("reset_blue_bin_pose", "reset_cube_1_pose", "reset_cube_pose"):
        term = getattr(env_cfg.events, term_name)
        assert term.func is franka_stack_events.randomize_object_pose
        assert term.mode == "reset"
        assert all(isinstance(cfg, SceneEntityCfg) for cfg in term.params["asset_cfgs"])

    assert [cfg.name for cfg in env_cfg.events.reset_blue_bin_pose.params["asset_cfgs"]] == ["blue_sorting_bin"]
    assert [cfg.name for cfg in env_cfg.events.reset_cube_pose.params["asset_cfgs"]] == ["cube_2", "cube_3"]
    assert env_cfg.events.reset_cube_pose.params["pose_range"]["x"] == [0.65, 0.70]
    assert env_cfg.events.reset_cube_pose.params["min_separation"] == 0.1

    # Untouched base config: success term and unrelated events survive the overlay.
    assert env_cfg.terminations.success is not None
    assert env_cfg.events.randomize_franka_joint_state is not None


def test_apply_rejects_base_env_mismatch():
    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()
    with pytest.raises(AssertionError, match="written against base env"):
        apply_env_profile(env_cfg, profile, "Isaac-Some-Other-Task-v0")


def test_apply_rejects_removing_unknown_event_term():
    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()
    profile.events.remove = ["no_such_term"]
    with pytest.raises(AssertionError, match="Cannot remove event term"):
        apply_env_profile(env_cfg, profile, BASE_ENV)


def test_apply_rejects_adding_colliding_scene_asset():
    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()
    profile.scene.rigid_objects.add["cube_1"] = copy.deepcopy(profile.scene.rigid_objects.add["blue_sorting_bin"])
    with pytest.raises(AssertionError, match="already defines it"):
        apply_env_profile(env_cfg, profile, BASE_ENV)


def test_apply_rejects_event_params_referencing_unknown_asset():
    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()
    profile.events.add["reset_cube_pose"].params["asset_cfgs"] = ["cube_2", "no_such_asset"]
    with pytest.raises(AssertionError, match="no_such_asset"):
        apply_env_profile(env_cfg, profile, BASE_ENV)


def test_apply_rejects_overriding_unknown_scene_asset():
    from autodata_interfaces.env.env_profile import RigidObjectOverrideSpec

    env_cfg = _parse_base_cfg()
    profile = _load_bin_profile()
    profile.scene.rigid_objects.override["no_such_asset"] = RigidObjectOverrideSpec(
        rigid_props={"solver_position_iteration_count": 40}
    )
    with pytest.raises(AssertionError, match="Cannot override scene asset"):
        apply_env_profile(env_cfg, profile, BASE_ENV)


def test_apply_override_merges_existing_term_params():
    env_cfg = _parse_base_cfg()
    original_range = env_cfg.events.randomize_cube_positions.params["pose_range"]
    profile = EnvironmentProfile.from_dict({
        "name": "wide",
        "base_env": BASE_ENV,
        "events": {
            "override": {"randomize_cube_positions": {"params": {"pose_range": {**original_range, "y": [-0.23, 0.23]}}}}
        },
    })

    apply_env_profile(env_cfg, profile, BASE_ENV)

    term = env_cfg.events.randomize_cube_positions
    assert term.params["pose_range"]["y"] == [-0.23, 0.23]
    assert term.params["min_separation"] == 0.1  # untouched sibling param
