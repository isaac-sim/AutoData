# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record, annotate, and generate Newton demonstrations through the public tools."""

import h5py
import json
import numpy as np
import os
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_valid_dataset


def configure_newton_recording() -> list[str]:
    """Feed reproducible SpaceMouse commands while keeping real physics and success checks."""
    import torch

    import __main__ as recorder
    from isaaclab.devices import Se3SpaceMouse
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    from autodata_examples.envs import register_environments

    assert recorder.args_cli.teleop_device == "spacemouse"
    assert not recorder.args_cli.reset_sim_buffer_each_episode
    remaining_args = register_environments()
    reader = HDF5DatasetFileHandler()
    reader.open(str(Path(TestPaths.test_data_dir) / "annotated_dataset_franka_rope_softmimicgen_newton.hdf5"))
    # Only the initial state and robot actions serve as test input. No annotations,
    # subsequent states, or success labels are copied into the new recording.
    episode = reader.load_episode("demo_0", "cpu")
    reader.close()
    actions = episode.data["actions"]
    original_reset = ManagerBasedRLEnv.reset

    def reset_from_source(env, *args, **kwargs):
        original_reset(env, *args, **kwargs)
        assert type(env.scene["object"]).__module__.startswith("isaaclab_contrib.deformable")
        return env.reset_to(episode.get_initial_state(), env_ids=None, is_relative=True)

    def reset_device(device):
        device._test_action_index = 0

    def advance(device):
        index = device._test_action_index
        device._test_action_index += 1
        assert index < len(actions) + 30, "Scripted recording did not reach the real task success condition"
        if index < len(actions):
            return actions[index].clone()
        hold = torch.zeros_like(actions[-1])
        hold[-1] = actions[-1, -1]
        return hold

    ManagerBasedRLEnv.reset = reset_from_source
    Se3SpaceMouse._find_device = lambda self: None
    Se3SpaceMouse._run_device = lambda self: None
    Se3SpaceMouse.reset = reset_device
    Se3SpaceMouse.advance = advance
    return remaining_args


@pytest.mark.with_subprocess
def test_newton_rope_record_annotate_generate(tmp_path):
    """Require newly recorded raw data to become usable SoftMimicGen input."""
    raw_file = tmp_path / "recorded.hdf5"
    annotated_file = tmp_path / "annotated.hdf5"
    generated_file = tmp_path / "generated.hdf5"
    lab_record = (
        Path(TestPaths.repo_root) / "submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py"
    )
    process_env = {**os.environ, "HEADLESS": "1"}
    env_name = "Isaac-Rope-Franka-Arena-IK-Rel-v0"
    camera_args = ["--viz", "kit", "--enable_cameras"]
    task_args = [
        "--task_descriptor",
        str(Path(TestPaths.tasks_dir) / "franka_rope_softmimicgen.yaml"),
        "--embodiment",
        str(Path(TestPaths.embodiments_dir) / "franka_ik_rel.yaml"),
    ]

    run_subprocess(
        [
            TestPaths.python_path,
            str(lab_record),
            "--external_callback",
            "autodata_tests.e2e.test_newton_rope_workflow.configure_newton_recording",
            "--task",
            env_name,
            "--presets",
            "newton",
            "--teleop_device",
            "spacemouse",
            "--no-reset_sim_buffer_each_episode",
            "--dataset_file",
            str(raw_file),
            "--num_demos",
            "2",
            "--step_hz",
            "20",
            *camera_args,
        ],
        env=process_env,
    )
    assert_valid_dataset(str(raw_file), min_num_demos=2)
    with h5py.File(raw_file, "r") as raw:
        assert len(raw["data"]) == 2
        for episode in raw["data"].values():
            assert episode.attrs["success"]
            assert "datagen_info" not in episode["obs"]

    run_subprocess(
        [
            TestPaths.python_path,
            str(Path(TestPaths.scripts_dir) / "annotate_demos.py"),
            "--auto",
            "--presets",
            "newton",
            "--device",
            "cuda",
            "--input_file",
            str(raw_file),
            "--output_file",
            str(annotated_file),
            *task_args,
            *camera_args,
        ],
        env=process_env,
    )
    assert_valid_dataset(str(annotated_file), min_num_demos=2)
    with h5py.File(raw_file, "r") as raw, h5py.File(annotated_file, "r") as annotated:
        assert len(annotated["data"]) == 2
        assert json.loads(annotated["data"].attrs["env_args"])["env_name"] == env_name
        for name, episode in annotated["data"].items():
            np.testing.assert_array_equal(episode["actions"][:], raw[f"data/{name}/actions"][:])
            info = episode["obs/datagen_info"]
            signal = info["subtask_term_signals/grasp"][:]
            assert len(signal) == len(episode["actions"])
            assert not signal[0] and signal[-1]
            assert np.all(np.diff(signal.astype(int)) >= 0)
            nodes = info["object_nodal_position/object"][:]
            assert nodes.shape == (len(signal), 549, 3)
            assert np.isfinite(nodes).all()
            assert set(info["eef_pose"]) == {"franka"}
            for camera in ("agentview_image_rgb", "robot0_eye_in_hand_image_rgb"):
                assert episode[f"obs/{camera}"].shape[:3] == (len(signal), 128, 128)

    run_subprocess(
        [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            env_name,
            "--presets",
            "newton",
            "--alg",
            "softmimicgen",
            "--input_file",
            str(annotated_file),
            "--output_file",
            str(generated_file),
            "--generation_num_trials",
            "1",
            "--num_envs",
            "3",
            "--device",
            "cuda",
            *task_args,
            *camera_args,
        ],
        env=process_env,
    )
    assert_valid_dataset(str(generated_file), min_num_demos=1)
    with h5py.File(generated_file, "r") as generated:
        for episode in generated["data"].values():
            assert episode.attrs["success"]
            nodes = episode["initial_state/deformable_object/object/nodal_position"][:]
            assert np.isfinite(nodes).all() and np.abs(nodes).max() < 2.0
    assert not (tmp_path / "generated_failed.hdf5").exists()
