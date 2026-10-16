# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import h5py
import numpy as np
import os
import tempfile

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_valid_dataset

HEADLESS = True
GENERATION_NUM_TRIALS = 1


def _run_franka_rope_softmimicgen(num_envs: int, device: str, presets: str = "physx") -> None:
    """Run SoftMimicGen data generation for the Franka rope task."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "generated.hdf5")
        dataset_suffix = "_newton" if presets == "newton" else ""

        args = [
            TestPaths.python_path,
            TestPaths.generate_dataset_script,
            "--env_name",
            "Isaac-Rope-Franka-Arena-IK-Rel-v0",
            "--presets",
            presets,
            "--alg",
            "softmimicgen",
            "--task_descriptor",
            os.path.join(TestPaths.tasks_dir, "franka_rope_softmimicgen.yaml"),
            "--embodiment",
            os.path.join(TestPaths.embodiments_dir, "franka_ik_rel.yaml"),
            "--input_file",
            os.path.join(TestPaths.test_data_dir, f"annotated_dataset_franka_rope_softmimicgen{dataset_suffix}.hdf5"),
            "--output_file",
            output_file,
            "--generation_num_trials",
            str(GENERATION_NUM_TRIALS),
            "--num_envs",
            str(num_envs),
            "--device",
            device,
            "--viz",
            "none" if HEADLESS else "kit",
        ]
        run_subprocess(args)

        assert_valid_dataset(output_file, min_num_demos=GENERATION_NUM_TRIALS)
        with h5py.File(output_file, "r") as dataset:
            for episode in dataset["data"].values():
                assert episode.attrs["success"]
                initial_nodes = episode["initial_state/deformable_object/object/nodal_position"][:]
                assert initial_nodes.shape == (1, 549, 3)
                assert np.isfinite(initial_nodes).all()
                assert np.abs(initial_nodes).max() < 2.0
                nodes = episode["states/deformable_object/object/nodal_position"][:]
                assert nodes.shape == (len(episode["actions"]), 549, 3)
                assert np.isfinite(nodes).all()
                # Parallel environment origins must not leak into recorded positions.
                assert np.abs(nodes).max() < 2.0


@pytest.mark.with_subprocess
@pytest.mark.parametrize("presets", ["physx", "newton"])
def test_franka_rope_softmimicgen_data_generation_single_env_cuda(presets):
    """SoftMimicGen generation for the Franka rope task on a single env on GPU."""
    _run_franka_rope_softmimicgen(num_envs=1, device="cuda", presets=presets)


@pytest.mark.with_subprocess
@pytest.mark.parametrize("presets", ["physx", "newton"])
def test_franka_rope_softmimicgen_data_generation_multi_env_cuda(presets):
    """SoftMimicGen generation for the Franka rope task on multiple parallel envs on GPU."""
    _run_franka_rope_softmimicgen(num_envs=10, device="cuda", presets=presets)


if __name__ == "__main__":
    for presets in ("physx", "newton"):
        test_franka_rope_softmimicgen_data_generation_single_env_cuda(presets)
        test_franka_rope_softmimicgen_data_generation_multi_env_cuda(presets)
