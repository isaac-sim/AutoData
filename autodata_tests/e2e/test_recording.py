# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the bundled Lab recorder with simulated SpaceMouse input."""

import h5py
import json
import os
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess


def configure_recording_smoke() -> None:
    """Replace physical HID I/O and export three real simulation steps for the test."""
    import torch

    import __main__ as recorder
    from isaaclab.devices import Se3SpaceMouse
    from isaaclab.envs import ManagerBasedRLEnv

    # This assertion observes the actual script's parser after Python startup,
    # before its task lookup. No task or environment resolution is mocked.
    assert recorder.args_cli.task == "IsaacContrib-Stack-Cube-Franka-IK-Rel"
    Se3SpaceMouse._find_device = lambda self: None
    Se3SpaceMouse._run_device = lambda self: None
    original_step = ManagerBasedRLEnv.step
    steps = 0

    def step_and_finish(env, actions):
        nonlocal steps
        result = original_step(env, actions)
        steps += 1
        if steps == 3:
            # Synthetic input does not solve the task. Explicitly export this
            # short test episode to verify recording and metadata, then stop.
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            raise KeyboardInterrupt
        return result

    ManagerBasedRLEnv.step = step_and_finish


@pytest.mark.with_subprocess
def test_lab_record_demos_task_startup(tmp_path):
    """The direct recorder uses the registered task and records real actions and states."""
    output_file = tmp_path / "recording.hdf5"
    script = Path(TestPaths.repo_root) / "submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py"
    run_subprocess(
        [
            TestPaths.python_path,
            str(script),
            "--viz",
            "kit",
            "--task",
            "IsaacContrib-Stack-Cube-Franka-IK-Rel",
            "--teleop_device",
            "spacemouse",
            "--dataset_file",
            str(output_file),
            "--num_demos",
            "1",
            "--external_callback",
            "autodata_tests.e2e.test_recording.configure_recording_smoke",
        ],
        env={**os.environ, "HEADLESS": "1"},
    )
    with h5py.File(output_file, "r") as dataset:
        assert json.loads(dataset["data"].attrs["env_args"])["env_name"] == "IsaacContrib-Stack-Cube-Franka-IK-Rel"
        assert len(dataset["data"]) == 1
        episode = next(iter(dataset["data"].values()))
        assert episode["actions"].shape == (3, 7)
        assert "states" in episode
