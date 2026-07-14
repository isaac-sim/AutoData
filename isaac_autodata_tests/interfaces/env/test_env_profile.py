# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :mod:`isaac_autodata_interfaces.env.env_profile` schema.

Schema loading and validation only; apply-time behavior against a parsed env config is covered
in :mod:`test_env_profile_apply`.
"""

import copy
import os

import pytest

from isaac_autodata_interfaces.env.env_profile import EnvironmentProfile, validate_profile_dict
from isaac_autodata_tests.utils.constants import TestPaths


def _minimal_valid() -> dict:
    return {"name": "p", "base_env": "Isaac-Stack-Cube-Franka-IK-Rel-v0"}


def _full_valid() -> dict:
    return {
        "name": "p",
        "description": "d",
        "base_env": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "planner": "franka_stack_cube_bin",
        "scene": {
            "rigid_objects": {
                "add": {
                    "bin": {
                        "prim_path": "{ENV_REGEX_NS}/Bin",
                        "usd_path": "{ISAACLAB_NUCLEUS_DIR}/bin.usd",
                        "position": [0.4, 0.0, 0.02],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.1, 1.6, 3.3],
                        "rigid_props": {"solver_position_iteration_count": 40},
                    }
                },
                "override": {"cube_1": {"rigid_props": {"solver_position_iteration_count": 40}}},
            },
        },
        "events": {
            "remove": ["randomize_cube_positions"],
            "override": {"init_franka_arm_pose": {"params": {"default_pose": [0.0]}}},
            "add": {
                "reset_bin_pose": {
                    "func": "some.module:some_function",
                    "mode": "reset",
                    "params": {"pose_range": {"x": [0.4, 0.4]}, "asset_cfgs": ["bin"]},
                }
            },
        },
    }


# ---------------------------------------------------------------------------------------------------
# validate_profile_dict
# ---------------------------------------------------------------------------------------------------
def test_validate_minimal_valid_passes():
    validate_profile_dict(_minimal_valid())  # must not raise


def test_validate_full_valid_passes():
    validate_profile_dict(_full_valid())  # must not raise


def test_validate_missing_required_keys_fails():
    for key in ("name", "base_env"):
        data = _minimal_valid()
        del data[key]
        with pytest.raises(AssertionError, match="Missing required top-level keys"):
            validate_profile_dict(data)


def test_validate_unknown_top_level_key_fails():
    data = _minimal_valid()
    data["scene_objects"] = {}
    with pytest.raises(AssertionError, match="Unknown top-level keys"):
        validate_profile_dict(data)


def test_validate_unknown_scene_key_fails():
    data = _minimal_valid()
    data["scene"] = {"assets": {}}
    with pytest.raises(AssertionError, match="'scene' has unknown keys"):
        validate_profile_dict(data)


def test_validate_unknown_rigid_objects_key_fails():
    data = _minimal_valid()
    data["scene"] = {"rigid_objects": {"remove": []}}
    with pytest.raises(AssertionError, match="'scene.rigid_objects' has unknown keys"):
        validate_profile_dict(data)


def test_validate_added_rigid_object_missing_required_keys_fails():
    data = _minimal_valid()
    data["scene"] = {"rigid_objects": {"add": {"bin": {"prim_path": "{ENV_REGEX_NS}/Bin"}}}}
    with pytest.raises(AssertionError, match="missing required keys"):
        validate_profile_dict(data)


def test_validate_added_rigid_object_unknown_key_fails():
    data = _full_valid()
    data["scene"]["rigid_objects"]["add"]["bin"]["pose"] = [0, 0, 0]
    with pytest.raises(AssertionError, match="has unknown keys"):
        validate_profile_dict(data)


def test_validate_rigid_object_override_unknown_key_fails():
    data = _full_valid()
    data["scene"]["rigid_objects"]["override"]["cube_1"]["prim_path"] = "{ENV_REGEX_NS}/Cube_1"
    with pytest.raises(AssertionError, match="has unknown keys"):
        validate_profile_dict(data)


def test_validate_rigid_object_override_requires_rigid_props():
    data = _full_valid()
    data["scene"]["rigid_objects"]["override"]["cube_1"] = {}
    with pytest.raises(AssertionError, match="non-empty 'rigid_props'"):
        validate_profile_dict(data)


def test_validate_unknown_events_key_fails():
    data = _minimal_valid()
    data["events"] = {"delete": []}
    with pytest.raises(AssertionError, match="'events' has unknown keys"):
        validate_profile_dict(data)


def test_validate_added_event_requires_func():
    data = _full_valid()
    del data["events"]["add"]["reset_bin_pose"]["func"]
    with pytest.raises(AssertionError, match="missing required key 'func'"):
        validate_profile_dict(data)


def test_validate_added_event_func_must_be_module_colon_function():
    data = _full_valid()
    data["events"]["add"]["reset_bin_pose"]["func"] = "some.module.some_function"
    with pytest.raises(AssertionError, match="'<module>:<function>'"):
        validate_profile_dict(data)


def test_validate_added_event_bad_mode_fails():
    data = _full_valid()
    data["events"]["add"]["reset_bin_pose"]["mode"] = "always"
    with pytest.raises(AssertionError, match="mode must be one of"):
        validate_profile_dict(data)


def test_validate_event_override_requires_params():
    data = _full_valid()
    data["events"]["override"]["init_franka_arm_pose"] = {}
    with pytest.raises(AssertionError, match="non-empty 'params'"):
        validate_profile_dict(data)


# ---------------------------------------------------------------------------------------------------
# EnvironmentProfile.from_dict / from_yaml
# ---------------------------------------------------------------------------------------------------
def test_from_dict_builds_specs_and_coerces_vectors():
    profile = EnvironmentProfile.from_dict(_full_valid())

    assert profile.name == "p"
    assert profile.base_env == "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    assert profile.planner == "franka_stack_cube_bin"

    bin_spec = profile.scene.rigid_objects.add["bin"]
    assert bin_spec.position == (0.4, 0.0, 0.02)
    assert bin_spec.rotation == (0.0, 0.0, 0.0, 1.0)
    assert bin_spec.scale == (1.1, 1.6, 3.3)
    assert bin_spec.rigid_props == {"solver_position_iteration_count": 40}
    assert profile.scene.rigid_objects.override["cube_1"].rigid_props == {"solver_position_iteration_count": 40}

    assert profile.events.remove == ["randomize_cube_positions"]
    assert profile.events.add["reset_bin_pose"].mode == "reset"
    # Event params pass through untouched at load time; asset-name conversion happens at apply.
    assert profile.events.add["reset_bin_pose"].params["asset_cfgs"] == ["bin"]


def test_from_dict_defaults_for_minimal_profile():
    profile = EnvironmentProfile.from_dict(_minimal_valid())
    assert profile.description == ""
    assert profile.planner is None
    assert profile.scene.rigid_objects.add == {}
    assert profile.scene.rigid_objects.override == {}
    assert profile.events.remove == []


def test_from_dict_does_not_mutate_input():
    data = _full_valid()
    snapshot = copy.deepcopy(data)
    EnvironmentProfile.from_dict(data)
    assert data == snapshot


def test_referenced_asset_names_collects_event_asset_refs():
    profile = EnvironmentProfile.from_dict(_full_valid())
    assert profile.referenced_asset_names() == {"bin"}


def test_shipped_bin_stack_profile_parses():
    path = os.path.join(TestPaths.env_profiles_dir, "franka_bin_stack.yaml")
    profile = EnvironmentProfile.from_yaml(path)
    assert profile.name == "franka_bin_stack"
    assert profile.base_env == "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    assert profile.planner == "franka_stack_cube_bin"
    assert "blue_sorting_bin" in profile.scene.rigid_objects.add
    assert set(profile.scene.rigid_objects.override) == {"cube_1", "cube_2", "cube_3"}
    assert profile.events.remove == ["randomize_cube_positions"]
    assert set(profile.events.add) == {"reset_blue_bin_pose", "reset_cube_1_pose", "reset_cube_pose"}
