# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate the rope-folding task with the PhysX physics backend."""

import os

import pytest
import warp as wp

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
@pytest.mark.parametrize("presets", ["physx"])
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


def main() -> None:
    """Check spawning, stepping, contacts, observations and indexed resets."""
    import argparse
    import tempfile

    wp.config.enable_backward = False
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", choices=("physx",), required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)
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
    env = gym.make(FRANKA_ROPE_ARENA_ENV_ID, cfg=cfg).unwrapped
    exit_code = 0
    try:

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
        rope = env.scene["object"]
        robot = env.scene["robot"]
        assert rope.data.nodal_pos_w.torch.shape == (args.num_envs, 549, 3)
        expected_module = "isaaclab_physx"
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

        # Relative IK and gripper actions must drive the PhysX articulation.
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

        # Restoring a saved state must preserve nodal coordinates, including env offsets.
        state = env.scene.get_state(is_relative=True)
        saved_nodes = rope.data.nodal_pos_w.torch.clone()
        env.reset()
        env.reset_to(state, env_ids=None, is_relative=True)
        torch.testing.assert_close(rope.data.nodal_pos_w.torch, saved_nodes, atol=1.0e-5, rtol=0)
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
