# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision geometry and generated model contracts."""

from autodata_interfaces.motion_planners.curobo import collision_spheres
from autodata_interfaces.motion_planners.curobo.robot_config import curobo_robot_config_create


def test_cache_reuses_geometry_and_invalidates_geometry_changes(tmp_path, monkeypatch):
    calls = []

    def fit(meshes, config, link_name):
        calls.append(link_name)
        return (collision_spheres.CollisionSphere((0.0, 0.0, 0.0), 0.1),)

    monkeypatch.setattr(collision_spheres, "_fit_link", fit)
    model = tmp_path / "model.urdf"
    cache = tmp_path / "cache"

    def load(name, size="1 1 1", origin="0 0 0", config=None):
        model.write_text(
            f'<robot name="{name}"><link name="body"><visual><origin xyz="{origin}"/>'
            f'<geometry><box size="{size}"/></geometry></visual></link></robot>'
        )
        return collision_spheres.collision_spheres_fit_urdf(model, cache_dir=cache, config=config)

    first = load("first_export")
    assert load("another_export").cache_key == first.cache_key
    assert calls == ["body"]
    assert load("another_export", size="1 1 2").cache_key != first.cache_key
    assert load("another_export", origin="0 0 1").cache_key != first.cache_key
    assert (
        load("another_export", config=collision_spheres.CollisionSphereFitConfig(inflation_m=0.02)).cache_key
        != first.cache_key
    )
    assert len(calls) == 4


def test_locked_auxiliary_joints_preserve_collision_geometry(tmp_path):
    """Planning fixes the commanded hand without removing it from the collision model."""
    model = tmp_path / "model.urdf"
    model.write_text(
        '<robot name="arm"><link name="base"/><link name="tool"><visual>'
        '<geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>'
        '<link name="finger"><visual><geometry><box size="0.02 0.02 0.1"/></geometry></visual></link>'
        '<joint name="shoulder" type="revolute"><parent link="base"/><child link="tool"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
        '<joint name="finger" type="revolute"><parent link="tool"/><child link="finger"/>'
        '<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    spheres = {name: (collision_spheres.CollisionSphere((0.0, 0.0, 0.0), 0.01),) for name in ("tool", "finger")}
    options = dict(auxiliary_joint_names=("finger",), attachment_sphere_count=0)
    query = curobo_robot_config_create(model, "tool", {"shoulder": 0.1, "finger": -0.2}, spheres, **options)
    planning = curobo_robot_config_create(
        model, "tool", {"shoulder": 0.1, "finger": -0.2}, spheres, lock_auxiliary_joints=True, **options
    )
    query, planning = query["robot_cfg"]["kinematics"], planning["robot_cfg"]["kinematics"]
    assert query["cspace"]["joint_names"] == ["shoulder", "finger"]
    assert planning["cspace"]["joint_names"] == ["shoulder"]
    assert planning["lock_joints"] == {"finger": -0.2}
    assert planning["collision_link_names"] == query["collision_link_names"]
    assert planning["collision_spheres"] == query["collision_spheres"]
