# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate one successful demonstration for each supported Arena workflow."""

import yaml
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths
from autodata_tests.utils.subprocess import run_subprocess
from autodata_tests.utils.utils import assert_valid_dataset, read_generation_result


@pytest.mark.with_subprocess
@pytest.mark.parametrize(
    "env_name, task_descriptor, embodiment, dataset_file, env_profile",
    [
        pytest.param(
            "gr1_open_microwave",
            "gr1_open_microwave.yaml",
            "gr1_ik_abs.yaml",
            "arena_gr1_manipulation_dataset_annotated.hdf5",
            None,
            id="gr1_open_microwave",
        ),
        pytest.param(
            "put_item_in_fridge_and_close_door",
            "gr1_put_and_close_door.yaml",
            "gr1_ik_abs.yaml",
            "ranch_bottle_into_fridge_annotated.hdf5",
            None,
            id="gr1_put_and_close_door",
        ),
        pytest.param(
            "galileo_g1_locomanip_pick_and_place",
            "g1_locomanip.yaml",
            "g1_wbc_pink.yaml",
            "arena_g1_loco_manipulation_dataset_annotated.hdf5",
            "g1_locomanip.yaml",
            id="g1_locomanip",
        ),
    ],
)
def test_arena_data_generation(
    env_name: str,
    task_descriptor: str,
    embodiment: str,
    dataset_file: str,
    env_profile: str | None,
    tmp_path: Path,
) -> None:
    """Run one CPU environment until one successful episode is generated and exported."""
    with (Path(TestPaths.tasks_dir) / task_descriptor).open(encoding="utf-8") as descriptor_file:
        descriptor = yaml.safe_load(descriptor_file)
    # These examples normally count attempts; the test requests one actual success.
    descriptor["generation_policy"]["guarantee_success"] = True
    test_descriptor = tmp_path / "task_descriptor.yaml"
    test_descriptor.write_text(yaml.safe_dump(descriptor), encoding="utf-8")

    output_file = tmp_path / "generated.hdf5"
    result_file = tmp_path / "generation_result.json"
    args = [
        TestPaths.python_path,
        TestPaths.generate_dataset_script,
        "--external_callback",
        "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback",
        "--env_name",
        env_name,
        "--alg",
        "dexmimicgen",
        "--task_descriptor",
        str(test_descriptor),
        "--embodiment",
        str(Path(TestPaths.embodiments_dir) / embodiment),
        "--input_file",
        str(Path(TestPaths.test_data_dir) / "arena_data_generation" / dataset_file),
        "--output_file",
        str(output_file),
        "--result_file",
        str(result_file),
        "--generation_num_trials",
        "1",
        "--num_envs",
        "1",
        "--device",
        "cpu",
        "--viz",
        "none",
    ]
    if env_profile is not None:
        args.extend(["--env_profile", str(Path(TestPaths.env_profiles_dir) / env_profile)])

    run_subprocess(args)

    num_success, _ = read_generation_result(str(result_file))
    assert num_success == 1, f"Expected one successful episode for {env_name}, got {num_success}."
    assert_valid_dataset(str(output_file), min_num_demos=1)
