# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from autodata_utils.dataset_metadata import exported_dataset_files


def test_failed_file_left_by_an_earlier_run_is_not_exported_again(tmp_path: Path) -> None:
    for name in ("generated.hdf5", "generated_failed.hdf5"):
        (tmp_path / name).touch()

    assert exported_dataset_files(str(tmp_path), "generated", kept_failed=False) == [str(tmp_path / "generated.hdf5")]
    assert len(exported_dataset_files(str(tmp_path), "generated", kept_failed=True)) == 2
