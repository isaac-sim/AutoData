# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate the shared rope-folding task with each supported physics backend."""

import os

import pytest
import warp as wp

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess


@wp.kernel
def _read_rope_visual_points(
    points: wp.fabricarrayarray(dtype=wp.vec3f),
    matrices: wp.fabricarray(dtype=wp.mat44d),
    offsets: wp.fabricarray(dtype=wp.uint32),
    result: wp.array2d(dtype=wp.vec3f),
    sim_count: int,
):
    i, node = wp.tid()
    env_index = int(offsets[i]) // sim_count
    result[env_index, node] = wp.transform_point(wp.transpose(wp.mat44f(matrices[i])), points[i][node])


def _assert_kit_rope_visual(env):
    """Compare Kit's actual GPU mesh points with the simulated rope in each world."""
    import torch

    import usdrt
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom

    env.sim.render()
    nodes = env.scene["object"].data.nodal_pos_w.torch
    stage = get_current_stage()
    for env_index in range(env.num_envs):
        root = f"/World/envs/env_{env_index}/Object"
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(f"{root}/Xform"))
        assert visual.GetPrim().IsActive()
        assert len(visual.GetPointsAttr().Get()) == 1056
        assert len(visual.GetFaceVertexCountsAttr().Get()) == 1026
    # CPU attribute reads are not valid once Fabric owns the point data on the GPU.
    selection = get_current_stage(fabric=True).SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.UInt, "autodata:rope:particleOffset", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
        ],
        device=env.device,
    )
    assert selection.GetCount() == env.num_envs, f"Only {selection.GetCount()}/{env.num_envs} rope skins found"
    visual_points = wp.empty((env.num_envs, 1056), dtype=wp.vec3f, device=env.device)
    wp.launch(
        _read_rope_visual_points,
        dim=visual_points.shape,
        inputs=[
            wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
            wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
            wp.fabricarray(data=selection, attrib="autodata:rope:particleOffset"),
            visual_points,
            nodes.shape[1],
        ],
        device=env.device,
    )
    remap = env.sim.physics_manager._rope_vis_remap
    indices = wp.to_torch(remap.tet_vertex_indices).long()
    weights = wp.to_torch(remap.bary_weights)
    expected = (nodes[:, indices] * weights[None, :, :, None]).sum(2)
    torch.testing.assert_close(wp.to_torch(visual_points), expected, atol=1.0e-5, rtol=0)


@pytest.mark.with_subprocess
@pytest.mark.parametrize("presets", ["physx", "newton"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_rope_environment(presets, num_envs, tmp_path):
    result_file = tmp_path / "passed"
    run_subprocess(
        [
            TestPaths.python_path,
            "-m",
            __name__,
            "--presets",
            presets,
            "--num_envs",
            str(num_envs),
            "--viz",
            "kit",
            "--result_file",
            str(result_file),
            "--enable_cameras",
        ],
        env={**os.environ, "HEADLESS": "1"},
    )
    assert result_file.exists(), "Simulation exited before completing the checks"


@pytest.mark.with_subprocess
def test_rope_gripper_table_contacts_newton(tmp_path):
    """Require physical table support under downward commands with open and closed fingers."""
    result_file = tmp_path / "table_contacts_passed"
    run_subprocess(
        [
            TestPaths.python_path,
            "-m",
            __name__,
            "--presets",
            "newton",
            "--num_envs",
            "1",
            "--viz",
            "kit",
            "--enable_cameras",
            "--table_contacts",
            "--result_file",
            str(result_file),
        ],
        env={**os.environ, "HEADLESS": "1"},
    )
    assert result_file.exists(), "Simulation exited before completing the table-contact checks"


def _assert_newton_gripper_table_contacts(env):
    import itertools
    import math
    import torch

    from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse
    from newton import ShapeFlags

    assert env.num_envs == 1
    model = env.sim.physics_manager.get_model()
    shape_names = list(map(str, model.shape_label))
    finger_ids = [i for i, name in enumerate(shape_names) if "finger/collisions" in name]
    table_ids = [i for i, name in enumerate(shape_names) if "/table/Collisions/Cube" in name]
    assert len(finger_ids) == 2 and len(table_ids) == 1
    for entry in ("rigid", "soft"):
        view = env.sim.physics_manager._solver.view(entry)
        table_flags = [
            int(view.shape_flags.numpy()[i])
            for i, name in enumerate(view.shape_label)
            if "/table/Collisions/Cube" in str(name)
        ]
        assert table_flags and all(flags & int(ShapeFlags.COLLIDE_SHAPES) for flags in table_flags)

    shape_poses = wp.to_torch(model.shape_transform)
    shape_scales = wp.to_torch(model.shape_scale)
    shape_bodies = wp.to_torch(model.shape_body).long()
    signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), device=env.device)

    def box_corners(shape_ids):
        poses = shape_poses[shape_ids]
        points = quat_apply(
            poses[:, None, 3:].expand(-1, 8, -1).reshape(-1, 4),
            (shape_scales[shape_ids, None] * signs).reshape(-1, 3),
        ).reshape(-1, 8, 3)
        points += poses[:, None, :3]
        bodies = wp.to_torch(env.sim.physics_manager.get_state_0().body_q)
        for index, body in enumerate(shape_bodies[shape_ids].tolist()):
            if body >= 0:
                points[index] = quat_apply(bodies[body, 3:].expand(8, -1), points[index]) + bodies[body, :3]
        return points

    table_top = box_corners(table_ids)[..., 2].max().item()
    robot = env.scene["robot"]
    action = torch.zeros((1, 7), device=env.device)
    orientation = torch.tensor([[math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]], device=env.device)
    with torch.inference_mode():
        action[:, -1] = 1.0
        env.step(action)  # Refresh the frame sensor after reset.
        for gripper in (1.0, -1.0):
            for height, count in ((0.13, 60), (-0.04, 75)):
                target = torch.tensor([[0.45, -0.35, height]], device=env.device)
                max_penetration = 0.0
                for _ in range(count):
                    frame = env.scene["ee_frame"].data
                    pos, quat = frame.target_pos_w.torch[:, 0], frame.target_quat_w.torch[:, 0]
                    _, rotation = compute_pose_error(pos, quat, target, orientation, rot_error_type="axis_angle")
                    root_quat = robot.data.root_quat_w.torch
                    action[:, :3] = (quat_apply_inverse(root_quat, target - pos) / 0.5).clamp(-0.04, 0.04)
                    action[:, 3:6] = (quat_apply_inverse(root_quat, rotation) / 0.5).clamp(-0.2, 0.2)
                    action[:, -1] = gripper
                    env.step(action)
                    max_penetration = max(max_penetration, table_top - box_corners(finger_ids)[..., 2].min().item())
                    assert torch.isfinite(env.scene["object"].data.nodal_pos_w.torch).all()
                if height < table_top:
                    pos = env.scene["ee_frame"].data.target_pos_w.torch[0, 0]
                    assert torch.linalg.vector_norm(pos[:2] - target[0, :2]) < 0.025, "Missed the test contact area"
                    assert pos[2] < 0.04, "Gripper never reached the tabletop"
                    assert max_penetration < 0.001, f"Fingers penetrated the table by {max_penetration:.6f} m"
                    print(f"ROPE table contact: gripper={gripper}, max_penetration={max_penetration:.6f} m", flush=True)


def main() -> None:
    """Check spawning, stepping, contacts, observations and indexed resets."""
    import argparse
    import tempfile

    wp.config.enable_backward = False
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", choices=("physx", "newton"), required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--table_contacts", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--enable_cameras", action="store_true")
    args = parser.parse_args()
    enable_cameras = args.enable_cameras
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    from autodata_examples.envs.isaac_lab_arena.franka_rope.environment import FRANKA_ROPE_ARENA_ENV_ID
    from autodata_examples.envs.isaac_lab_arena.registration import register_environment_from_cli

    # Use the same backend-selection callback as the public Lab teleop/replay scripts.
    register_environment_from_cli()
    cfg = gym.spec(FRANKA_ROPE_ARENA_ENV_ID).kwargs["env_cfg_entry_point"]
    cfg.seed = 42
    cfg.recorders.dataset_export_dir_path = tempfile.mkdtemp(prefix="rope_env_check_")
    if args.table_contacts:
        assert args.presets == "newton"
        cfg.recorders = {}
        cfg.terminations = {}
    env = gym.make(FRANKA_ROPE_ARENA_ENV_ID, cfg=cfg).unwrapped
    exit_code = 0
    try:
        if args.table_contacts:
            env.reset()
            _assert_newton_gripper_table_contacts(env)
            if args.result_file:
                from pathlib import Path

                Path(args.result_file).touch()
            return

        def camera_observations():
            # Headless Kit renders on demand in the pinned Lab version. Request
            # an actual frame before reading the RTX camera annotators.
            if args.presets == "physx":
                for visualizer in env.sim.visualizers:
                    if visualizer.pumps_app_update():
                        visualizer.render_rgb_array()
                for sensor in env.scene.sensors.values():
                    if hasattr(sensor.cfg, "renderer_cfg"):
                        sensor.update(0.0, force_recompute=True)
            return env.observation_manager.compute()["camera_obs"]

        env.reset()
        if args.presets == "newton":
            _assert_kit_rope_visual(env)
        rope = env.scene["object"]
        robot = env.scene["robot"]
        assert rope.data.nodal_pos_w.torch.shape == (args.num_envs, 549, 3)
        expected_module = "isaaclab_physx" if args.presets == "physx" else "isaaclab_contrib.deformable"
        assert type(rope).__module__.startswith(expected_module)
        print(f"ROPE backend={args.presets} implementation={type(rope)}", flush=True)
        initial_nodes = rope.data.nodal_pos_w.torch.clone()
        actions = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
        actions[:, -1] = 1.0
        with torch.inference_mode():
            for step in range(100):
                obs, _, terminated, _, _ = env.step(actions)
                nodes = rope.data.nodal_pos_w.torch
                assert torch.isfinite(nodes).all(), f"Non-finite rope nodes at step {step}"
                assert not terminated.any(), (
                    f"Unexpected task termination at step {step}: "
                    f"{env.termination_manager.get_active_iterable_terms(int(terminated.nonzero()[0]))}"
                )
                assert torch.isfinite(obs["policy"]["object_nodal_pos"]).all()
                if step % 20 == 0:
                    local = nodes - env.scene.env_origins[:, None, :]
                    print(f"ROPE step={step} min={local.amin((0, 1))} max={local.amax((0, 1))}", flush=True)
                    if args.presets == "newton":
                        _assert_kit_rope_visual(env)
        assert (rope.data.nodal_pos_w.torch - initial_nodes).abs().max() > 1.0e-4

        # Gravity must settle the rope on the tabletop, without penetrating or exploding.
        local = rope.data.nodal_pos_w.torch - env.scene.env_origins[:, None, :]
        assert local[..., 2].min() > -0.01
        assert local[..., 2].max() < 0.15
        assert (local.amax(1) - local.amin(1)).amax() < 1.5

        if enable_cameras:
            images = camera_observations()
            assert len(images) == 2
            for image in images.values():
                assert image.shape[:3] == (args.num_envs, 128, 128)
                assert image.float().std() > 1.0
            if args.snapshot:
                from PIL import Image

                snapshot = next(value for name, value in images.items() if "agentview" in name)
                Image.fromarray(snapshot[0, ..., :3].cpu().numpy()).save(args.snapshot)

        # Relative IK and gripper actions must drive the Newton/PhysX articulation.
        before_joints = robot.data.joint_pos.torch.clone()
        before_images = {name: value.clone() for name, value in images.items()} if enable_cameras else {}
        actions[:, 2] = 0.005
        actions[:, -1] = -1.0
        for _ in range(12):
            obs, *_ = env.step(actions)
        after_joints = robot.data.joint_pos.torch
        assert torch.isfinite(after_joints).all()
        assert (after_joints[:, :7] - before_joints[:, :7]).abs().max() > 0.005
        assert after_joints[:, -2:].mean() < before_joints[:, -2:].mean() - 0.005
        if enable_cameras:
            images = camera_observations()
            assert any(
                (images[name].float() - before.float()).abs().mean() > 0.01 for name, before in before_images.items()
            ), "Camera images did not change when the robot moved"

        # Reset one environment without changing the other environments' nodes.
        before_reset = rope.data.nodal_pos_w.torch.clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        after_reset = rope.data.nodal_pos_w.torch
        assert (after_reset[0] - before_reset[0]).abs().max() > 0.001
        torch.testing.assert_close(after_reset[1:], before_reset[1:], atol=1.0e-5, rtol=0)
        if args.presets == "newton":
            _assert_kit_rope_visual(env)

        # Restoring a saved state must preserve nodal coordinates, including env offsets.
        state = env.scene.get_state(is_relative=True)
        saved_nodes = rope.data.nodal_pos_w.torch.clone()
        env.reset()
        env.reset_to(state, env_ids=None, is_relative=True)
        torch.testing.assert_close(rope.data.nodal_pos_w.torch, saved_nodes, atol=1.0e-5, rtol=0)
        if args.presets == "newton":
            _assert_kit_rope_visual(env)
        actions.zero_()
        actions[:, -1] = 1.0
        for _ in range(5):
            env.step(actions)
        assert torch.isfinite(rope.data.nodal_pos_w.torch).all()
        print("ROPE smoke passed", flush=True)
        if args.result_file:
            from pathlib import Path

            Path(args.result_file).touch()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        env.close()
        # Kit terminates the process directly; preserve failures across shutdown.
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
