# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.embodiments.factory`."""

import os

import pytest

from isaac_autodata_interfaces.embodiments import (
    EMBODIMENT_TYPE_REGISTRY,
    AbsolutePoseWholeBodyBimanualAdapter,
    DeltaPoseIKSingleArmAdapter,
    embodiment_adapter_from_dict,
    embodiment_adapter_from_yaml,
)
from isaac_autodata_tests.interfaces.mocks import MockEnv
from isaac_autodata_tests.utils.constants import TestPaths

_SINGLE_ARM_DICT = {
    "type": "delta_pose_ik_single_arm",
    "name": "franka",
    "eef_name": "franka",
    "pose_obs_keys": {"pos": "eef_pos", "quat": "eef_quat"},
    "action_layout": {"gripper_dim": 1},
}


def test_registry_contents():
    assert EMBODIMENT_TYPE_REGISTRY == {
        "delta_pose_ik_single_arm": DeltaPoseIKSingleArmAdapter,
        "absolute_pose_whole_body_bimanual": AbsolutePoseWholeBodyBimanualAdapter,
    }


def test_from_dict_dispatches_single_arm():
    embodiment_adapter = embodiment_adapter_from_dict(_SINGLE_ARM_DICT)
    assert isinstance(embodiment_adapter, DeltaPoseIKSingleArmAdapter)
    assert embodiment_adapter.name == "franka"
    assert embodiment_adapter.env is None  # not bound when no env passed


def test_from_dict_strips_type_key_before_delegating():
    # The 'type' discriminator must not be forwarded to the concrete from_dict (which would
    # reject it as an unexpected kwarg).
    embodiment_adapter = embodiment_adapter_from_dict(dict(_SINGLE_ARM_DICT))
    assert embodiment_adapter.eef_name == "franka"


def test_from_dict_binds_env_when_provided():
    env = MockEnv()
    embodiment_adapter = embodiment_adapter_from_dict(_SINGLE_ARM_DICT, env=env)
    assert embodiment_adapter.env is env


def test_from_dict_missing_type():
    data = {k: v for k, v in _SINGLE_ARM_DICT.items() if k != "type"}
    with pytest.raises(AssertionError, match="Missing required top-level key 'type'"):
        embodiment_adapter_from_dict(data)


def test_from_dict_non_str_type():
    data = dict(_SINGLE_ARM_DICT, type=5)
    with pytest.raises(AssertionError, match="'type' must be a string"):
        embodiment_adapter_from_dict(data)


def test_from_dict_unknown_type():
    data = dict(_SINGLE_ARM_DICT, type="nope")
    with pytest.raises(AssertionError, match="Unknown embodiment type"):
        embodiment_adapter_from_dict(data)


def test_from_dict_non_dict():
    with pytest.raises(AssertionError, match="Expected top-level dict"):
        embodiment_adapter_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_from_yaml_round_trip(tmp_path):
    import yaml

    path = tmp_path / "embodiment.yaml"
    path.write_text(yaml.safe_dump(_SINGLE_ARM_DICT))
    embodiment_adapter = embodiment_adapter_from_yaml(str(path))
    assert isinstance(embodiment_adapter, DeltaPoseIKSingleArmAdapter)
    assert embodiment_adapter.name == "franka"


@pytest.mark.parametrize(
    "yaml_name, expected_cls",
    [
        ("franka_ik_rel.yaml", DeltaPoseIKSingleArmAdapter),
        ("franka_ik_rel_skillgen.yaml", DeltaPoseIKSingleArmAdapter),
        ("g1_ik_abs.yaml", AbsolutePoseWholeBodyBimanualAdapter),
        ("gr1_ik_abs.yaml", AbsolutePoseWholeBodyBimanualAdapter),
    ],
)
def test_shipped_example_embodiment_yamls_build(yaml_name, expected_cls):
    """Every shipped embodiment descriptor must build into the expected adapter type."""
    path = os.path.join(TestPaths.embodiments_dir, yaml_name)
    embodiment_adapter = embodiment_adapter_from_yaml(path)
    assert isinstance(embodiment_adapter, expected_cls)
    assert embodiment_adapter.name
    assert len(embodiment_adapter.get_eef_names()) >= 1
    # action_dim must be computable (regression guard for the bimanual empty-channels case).
    assert embodiment_adapter.action_dim > 0
