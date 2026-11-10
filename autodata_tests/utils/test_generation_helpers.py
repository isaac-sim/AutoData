# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation and replay must use the selected environment, not an ambient Python."""

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths


@pytest.mark.parametrize("helper", ["run_generation.sh", "replay_franka_generation.sh"])
@pytest.mark.parametrize("selection", ["project", "virtual", "conda", "incomplete"])
def test_helpers_select_environment_and_forward_arguments(tmp_path, helper, selection):
    if selection == "project" and os.access("/isaac-sim/python.sh", os.X_OK):
        pytest.skip("The legacy container interpreter takes precedence over a mounted project environment.")

    root = tmp_path / "checkout with spaces"
    root.mkdir()
    helpers = root / "helpers"
    shutil.copytree(Path(TestPaths.repo_root) / "skills/autodata-generate-datasets/scripts", helpers)
    environments = {
        "project": root / ".venv",
        "virtual": tmp_path / "active environment",
        "conda": tmp_path / "conda",
        "ambient": tmp_path / "ambient",
    }
    for name, path in environments.items():
        venv.EnvBuilder(with_pip=False).create(path)
        if name != "virtual" or selection != "incomplete":
            site_packages = path / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            (site_packages / "isaaclab_arena.py").write_text("")

    arena = root / "submodules/IsaacLab-Arena"
    arena.mkdir(parents=True)
    for repository in (root, arena):
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(arena),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
    )
    (root / ".gitmodules").write_text('[submodule "arena"]\n\tpath = submodules/IsaacLab-Arena\n\turl = ./arena\n')
    subprocess.run(
        ["git", "-C", str(root), "add", ".gitmodules", "submodules/IsaacLab-Arena"], check=True, capture_output=True
    )
    subprocess.run(["git", "-C", str(root), "submodule", "init"], check=True, capture_output=True)

    log = tmp_path / "commands.jsonl"
    for script in (
        "scripts/generate_dataset.py",
        "scripts/validate_dataset.py",
        "submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py",
    ):
        path = root / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import json, os, sys\n"
            "with open(os.environ['TEST_LOG'], 'a') as log:\n"
            "    log.write(json.dumps([sys.prefix, sys.argv, os.environ.get('HDF5_USE_FILE_LOCKING')]) + '\\n')\n"
        )

    dataset = "dataset with spaces.hdf5"
    (root / dataset).touch()
    environment = os.environ.copy()
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH", "HDF5_USE_FILE_LOCKING"):
        environment.pop(name, None)
    environment.update(PATH=f"{environments['ambient'] / 'bin'}:{os.defpath}", TEST_LOG=str(log))
    if selection != "project":
        environment["CONDA_PREFIX"] = str(environments["conda"])
    if selection in ("virtual", "incomplete"):
        environment["VIRTUAL_ENV"] = str(environments["virtual"])

    if helper == "run_generation.sh":
        arguments = [
            "--alg",
            "mimicgen",
            "--task",
            "task with spaces.yaml",
            "--embodiment",
            "embodiment.yaml",
            "--input",
            dataset,
            "--output",
            "generated dataset.hdf5",
        ]
        expected_calls = [
            (
                [
                    "scripts/generate_dataset.py",
                    "--alg",
                    "mimicgen",
                    "--input_file",
                    dataset,
                    "--task_descriptor",
                    "task with spaces.yaml",
                    "--embodiment",
                    "embodiment.yaml",
                    "--output_file",
                    "generated dataset.hdf5",
                ],
                None,
            ),
            (["scripts/validate_dataset.py", "generated dataset.hdf5"], "FALSE"),
        ]
    else:
        arguments = [dataset]
        expected_calls = [(
            [
                "submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py",
                "--viz",
                "kit",
                "--task",
                "IsaacContrib-Stack-Cube-Franka-IK-Rel",
                "--num_envs",
                "1",
                "--dataset_file",
                dataset,
            ],
            None,
        )]

    result = subprocess.run(
        ["bash", str(helpers / helper), *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if selection == "incomplete":
        assert result.returncode != 0
        assert "Isaac Lab-Arena was not found" in result.stderr
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert calls == [[str(environments[selection]), arguments, locking] for arguments, locking in expected_calls]
