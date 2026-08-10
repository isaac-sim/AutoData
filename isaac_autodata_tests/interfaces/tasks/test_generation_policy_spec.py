# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.tasks.generation_policy_spec`."""

from isaac_autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy


def test_generation_policy_defaults():
    p = GenerationPolicy()
    assert p.name == "demo"
    assert p.seed == 1
    assert p.num_trials == 10
    assert p.guarantee_success is True
    assert p.keep_failed is False
    assert p.source_dataset_path is None
    assert p.generation_path is None
    assert p.task_name is None
    assert p.use_skillgen is False
    assert p.use_navigation_controller is False
    assert p.select_src_per_subtask is False
    assert p.select_src_per_arm is False
    assert p.transform_first_robot_pose is False
    assert p.interpolate_from_last_target_pose is True


def test_generation_policy_from_kwargs():
    p = GenerationPolicy(name="run", seed=42, num_trials=100, use_skillgen=True)
    assert p.name == "run"
    assert p.seed == 42
    assert p.num_trials == 100
    assert p.use_skillgen is True
    # Unspecified fields keep their defaults.
    assert p.guarantee_success is True
    assert p.interpolate_from_last_target_pose is True


def test_generation_policy_from_dict_unpacking():
    # This mirrors how TaskDescriptor.from_dict builds the policy: GenerationPolicy(**block).
    data = {"name": "x", "num_trials": 5, "keep_failed": True, "select_src_per_subtask": True}
    p = GenerationPolicy(**data)
    assert p.name == "x"
    assert p.num_trials == 5
    assert p.keep_failed is True
    assert p.select_src_per_subtask is True


def test_generation_policy_instances_are_independent():
    a = GenerationPolicy()
    b = GenerationPolicy()
    a.num_trials = 999
    assert b.num_trials == 10
