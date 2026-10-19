# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay source demonstrations and annotate them with the current Isaac Lab tasks."""

import h5py
import json
import subprocess
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_valid_dataset


@pytest.mark.with_subprocess
def test_automatic_annotation_from_dataset(tmp_path):
    """Use the stored task name and export usable termination signals during replay."""
    input_file = Path(TestPaths.test_data_dir) / "annotated_dataset_franka_stack_mimicgen.hdf5"
    output_file = tmp_path / "annotated.hdf5"
    run_subprocess([
        TestPaths.python_path,
        str(Path(TestPaths.scripts_dir) / "annotate_demos.py"),
        "--auto",
        "--viz",
        "none",
        "--device",
        "cpu",
        "--task_descriptor",
        str(Path(TestPaths.tasks_dir) / "franka_cube_stack.yaml"),
        "--embodiment",
        str(Path(TestPaths.embodiments_dir) / "franka_ik_rel.yaml"),
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_file),
    ])
    assert_valid_dataset(str(output_file), min_num_demos=1)
    with h5py.File(input_file, "r") as source, h5py.File(output_file, "r") as annotated:
        assert len(annotated["data"]) == len(source["data"])
        assert json.loads(annotated["data"].attrs["env_args"])["env_name"] == "IsaacContrib-Stack-Cube-Franka-IK-Rel"
        for episode in annotated["data"].values():
            signals = episode["obs/datagen_info/subtask_term_signals"]
            assert set(signals) == {"grasp_1", "stack_1", "grasp_2"}
            for signal in signals.values():
                assert len(signal) == len(episode["actions"])
                assert signal[:].any()


@pytest.mark.with_subprocess
def test_manual_annotation_requires_window(tmp_path, capfd):
    """Kit shutdown must preserve the error and exit status for an invalid launch."""
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_subprocess([
            TestPaths.python_path,
            str(Path(TestPaths.scripts_dir) / "annotate_demos.py"),
            "--viz",
            "none",
            "--device",
            "cpu",
            "--task_descriptor",
            str(Path(TestPaths.tasks_dir) / "franka_cube_stack.yaml"),
            "--embodiment",
            str(Path(TestPaths.embodiments_dir) / "franka_ik_rel.yaml"),
            "--input_file",
            str(Path(TestPaths.test_data_dir) / "annotated_dataset_franka_stack_mimicgen.hdf5"),
            "--output_file",
            str(tmp_path / "annotated.hdf5"),
        ])
    assert error.value.returncode == 1
    out, err = capfd.readouterr()
    assert "Re-run with a window (--viz kit), or pass --auto." in out + err
