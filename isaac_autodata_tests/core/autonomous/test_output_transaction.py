# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import h5py
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import isaac_autodata_core.autonomous.output_transaction as output_transaction_module
from isaac_autodata_core.autonomous.output_transaction import (
    DatasetCommitUncertainError,
    OutputTransaction,
    RequestDirectoryAnchor,
)


def _write_dataset(transaction: OutputTransaction, *, episodes: int = 1, failed: bool = False) -> bytes:
    targets = transaction.recording_targets
    suffix = "_failed" if failed else ""
    path = Path(targets.dataset_export_dir_path) / f"{targets.dataset_filename}{suffix}.hdf5"
    with h5py.File(path, "w") as dataset:
        dataset.attrs["format_version"] = 1
        data = dataset.create_group("data")
        for index in range(episodes):
            episode = data.create_group(f"demo_{index}")
            episode.attrs["num_samples"] = 1
            episode.attrs["success"] = not failed
            initial_robot = episode.create_group("initial_state/articulation/robot")
            initial_robot.create_dataset("joint_position", data=[[float(index)]])
            episode.create_group("obs").create_dataset("joint_pos", data=[[float(index)]])
            episode.create_group("states").create_dataset("joint_pos", data=[[float(index)]])
            episode.create_dataset("actions", data=[[float(index)]])
            episode.create_dataset("processed_actions", data=[[float(index)]])
    return path.read_bytes()


def _write_zero_sample_failed_dataset(transaction: OutputTransaction) -> bytes:
    targets = transaction.recording_targets
    path = Path(targets.dataset_export_dir_path) / f"{targets.dataset_filename}_failed.hdf5"
    with h5py.File(path, "w") as dataset:
        dataset.attrs["format_version"] = 1
        episode = dataset.create_group("data/demo_0")
        episode.attrs["num_samples"] = 0
        episode.attrs["success"] = False
        episode.create_group("initial_state/articulation/robot").create_dataset(
            "joint_position",
            data=[[0.0]],
        )
    return path.read_bytes()


def _reserve(tmp_path: Path, *, run_log: bool = True, keep_failed: bool = False) -> OutputTransaction:
    return OutputTransaction.reserve(
        request_directory=tmp_path,
        dataset_path=tmp_path / "nested" / "episodes.hdf5",
        run_log_path=tmp_path / "audit" / "run.jsonl" if run_log else None,
        keep_failed=keep_failed,
    )


def test_transaction_publishes_validated_inode_and_terminal_commit(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path)
    targets = transaction.recording_targets
    staging_path = Path(targets.dataset_export_dir_path)
    assert stat.S_IMODE(staging_path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "audit" / "run.jsonl").stat().st_mode) == 0o600
    original = _write_dataset(transaction, episodes=2)
    assert not transaction.dataset_path.exists()
    writer = transaction.open_run_log_writer()
    assert writer is not None
    writer.append({"record_type": "started"})

    artifacts = transaction.publish(
        run_log_writer=writer,
        commit_record={"record_type": "run_committed"},
        expected_successful_episodes=2,
    )
    writer.close()
    transaction.close()

    assert transaction.dataset_path.read_bytes() == original
    assert len(artifacts) == 1
    assert artifacts[0].sha256 == hashlib.sha256(original).hexdigest()
    assert artifacts[0].episode_count == 2
    records = [json.loads(line) for line in (tmp_path / "audit" / "run.jsonl").read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["started", "run_committed"]
    assert records[-1]["artifacts"] == [artifacts[0].to_dict()]
    assert not staging_path.exists()


def test_publication_never_replaces_file_created_after_reservation(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path)
    _write_dataset(transaction)
    transaction.dataset_path.write_bytes(b"human-owned")

    with pytest.raises(FileExistsError):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert transaction.dataset_path.read_bytes() == b"human-owned"


def test_successful_link_is_rolled_back_if_postlink_identity_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    original_stat = output_transaction_module.os.stat
    fail_once = True

    def fail_final_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal fail_once
        if (
            fail_once
            and path == transaction.dataset_path.name
            and kwargs.get("dir_fd") == transaction._dataset_parent_fd
        ):
            fail_once = False
            raise OSError("post-link identity probe failed")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(output_transaction_module.os, "stat", fail_final_stat)
    with pytest.raises(OSError, match="identity probe"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_publication_links_held_inode_if_staging_name_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    original = _write_dataset(transaction)
    original_link = output_transaction_module._link_fd_to_name

    def swap_then_link(source_fd: int, destination_dir_fd: int, destination_name: str) -> None:
        os.rename(
            "episodes.hdf5",
            "validated-original.hdf5",
            src_dir_fd=transaction._staging_fd,
            dst_dir_fd=transaction._staging_fd,
        )
        malicious = os.open(
            "episodes.hdf5",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=transaction._staging_fd,
        )
        os.write(malicious, b"not the validated inode")
        os.close(malicious)
        original_link(source_fd, destination_dir_fd, destination_name)

    monkeypatch.setattr(output_transaction_module, "_link_fd_to_name", swap_then_link)
    transaction.publish(
        run_log_writer=None,
        commit_record=None,
        expected_successful_episodes=1,
    )
    transaction.close()

    assert transaction.dataset_path.read_bytes() == original


def test_staged_symlink_is_rejected_without_reading_target(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    outside = tmp_path / "outside.hdf5"
    outside.write_bytes(b"outside")
    os.symlink(outside, "episodes.hdf5", dir_fd=transaction._staging_fd)

    with pytest.raises(OSError):
        transaction.publish(run_log_writer=None, commit_record=None)
    transaction.close()

    assert outside.read_bytes() == b"outside"
    assert not transaction.dataset_path.exists()


def test_symlinked_output_parent_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        OutputTransaction.reserve(
            request_directory=tmp_path,
            dataset_path=tmp_path / "linked" / "episodes.hdf5",
            run_log_path=None,
            keep_failed=False,
        )

    assert list(real.iterdir()) == []


def test_dataset_parent_rename_after_reservation_prevents_misreported_publication(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path)
    _write_dataset(transaction)
    original_parent = tmp_path / "nested-original"
    transaction.dataset_path.parent.rename(original_parent)
    transaction.dataset_path.parent.mkdir(mode=0o700)
    writer = transaction.open_run_log_writer()

    with pytest.raises(RuntimeError, match="dataset parent"):
        transaction.publish(
            run_log_writer=writer,
            commit_record={"record_type": "run_committed"},
            expected_successful_episodes=1,
        )
    assert writer is not None
    writer.close()
    transaction.close()

    assert not transaction.dataset_path.exists()
    assert list(transaction.dataset_path.parent.iterdir()) == []


def test_run_log_conflict_removes_private_staging(tmp_path: Path) -> None:
    run_log = tmp_path / "audit" / "run.jsonl"
    run_log.parent.mkdir()
    run_log.write_text("owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _reserve(tmp_path)

    assert run_log.read_text(encoding="utf-8") == "owned\n"
    dataset_parent = tmp_path / "nested"
    assert list(dataset_parent.iterdir()) == []


def test_reservation_failure_does_not_unlink_replacement_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = output_transaction_module._open_exclusive_run_log

    def replace_after_open(name: str, parent_fd: int) -> int:
        descriptor = original_open(name, parent_fd)
        os.unlink(name, dir_fd=parent_fd)
        replacement = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        os.write(replacement, b"replacement\n")
        os.close(replacement)
        return descriptor

    monkeypatch.setattr(output_transaction_module, "_open_exclusive_run_log", replace_after_open)
    with pytest.raises(RuntimeError, match="held inode"):
        _reserve(tmp_path)

    assert (tmp_path / "audit" / "run.jsonl").read_text() == "replacement\n"


def test_episode_count_mismatch_prevents_publication_and_commit(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path)
    _write_dataset(transaction, episodes=1)
    writer = transaction.open_run_log_writer()
    assert writer is not None
    writer.append({"record_type": "started"})

    with pytest.raises(ValueError, match="expected 2"):
        transaction.publish(
            run_log_writer=writer,
            commit_record={"record_type": "run_committed"},
            expected_successful_episodes=2,
        )
    writer.close()
    transaction.close()

    assert not transaction.dataset_path.exists()
    records = [json.loads(line) for line in (tmp_path / "audit" / "run.jsonl").read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["started"]


def test_structurally_incomplete_hdf5_is_not_attested(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    targets = transaction.recording_targets
    staged = Path(targets.dataset_export_dir_path) / f"{targets.dataset_filename}.hdf5"
    with h5py.File(staged, "w") as dataset:
        dataset.attrs["format_version"] = 1
        dataset.create_group("data").create_group("demo_0")

    with pytest.raises(ValueError, match="initial_state"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_numeric_success_metadata_is_not_accepted_as_boolean(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    targets = transaction.recording_targets
    staged = Path(targets.dataset_export_dir_path) / f"{targets.dataset_filename}.hdf5"
    with h5py.File(staged, "r+") as dataset:
        dataset["data/demo_0"].attrs["success"] = 1

    with pytest.raises(ValueError, match="success metadata"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_successful_episode_requires_initial_state_datasets(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    staged = Path(transaction.recording_targets.dataset_export_dir_path) / "episodes.hdf5"
    with h5py.File(staged, "r+") as dataset:
        del dataset["data/demo_0/initial_state/articulation"]

    with pytest.raises(ValueError, match="initial_state.*no datasets"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_successful_episode_rejects_empty_initial_state_dataset(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    staged = Path(transaction.recording_targets.dataset_export_dir_path) / "episodes.hdf5"
    with h5py.File(staged, "r+") as dataset:
        robot = dataset["data/demo_0/initial_state/articulation/robot"]
        del robot["joint_position"]
        robot.create_dataset("joint_position", shape=(0, 1), dtype="f4")

    with pytest.raises(ValueError, match="initial_state.*must not be empty"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=("nan", "inf"))
@pytest.mark.parametrize(
    "dataset_path",
    [
        "processed_actions",
        "obs/joint_pos",
        "states/joint_pos",
        "initial_state/articulation/robot/joint_position",
    ],
)
def test_nonfinite_episode_data_is_not_published(tmp_path: Path, dataset_path: str, value: float) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    staged = Path(transaction.recording_targets.dataset_export_dir_path) / "episodes.hdf5"
    with h5py.File(staged, "r+") as dataset:
        dataset[f"data/demo_0/{dataset_path}"][0, 0] = value

    with pytest.raises(ValueError, match="contains non-finite values"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


@pytest.mark.parametrize(
    "dataset_path",
    [
        "processed_actions",
        "obs/joint_pos",
        "states/joint_pos",
        "initial_state/articulation/robot/joint_position",
    ],
)
def test_nonnumeric_episode_data_is_not_published(tmp_path: Path, dataset_path: str) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    _write_dataset(transaction)
    staged = Path(transaction.recording_targets.dataset_export_dir_path) / "episodes.hdf5"
    with h5py.File(staged, "r+") as dataset:
        episode = dataset["data/demo_0"]
        del episode[dataset_path]
        episode.create_dataset(dataset_path, data=[[b"not numeric"]])

    with pytest.raises(ValueError, match="must be numeric"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            expected_successful_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_nested_product10_like_episode_is_published(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, run_log=False)
    targets = transaction.recording_targets
    staged = Path(targets.dataset_export_dir_path) / f"{targets.dataset_filename}.hdf5"
    with h5py.File(staged, "w") as dataset:
        dataset.attrs["format_version"] = 1
        episode = dataset.create_group("data/demo_0")
        episode.attrs["num_samples"] = 2
        episode.attrs["success"] = True
        initial_state = episode.create_group("initial_state")
        initial_state.create_dataset("articulation/robot/joint_position", data=[[0.0, 0.1]])
        initial_state.create_dataset("rigid_object/pick_cube/root_pose", data=[[0.0] * 7])
        obs = episode.create_group("obs")
        obs.create_dataset("policy/joint_pos", data=[[0.0, 0.1], [0.1, 0.2]])
        states = episode.create_group("states")
        states.create_dataset("articulation/robot/joint_position", data=[[0.0, 0.1], [0.1, 0.2]])
        states.create_dataset("rigid_object/pick_cube/root_pose", data=[[0.0] * 7, [0.1] * 7])
        episode.create_dataset("actions", data=[[0.0], [0.1]])
        episode.create_dataset("processed_actions", data=[[0.0, 1.0], [0.1, 1.0]])

    artifacts = transaction.publish(
        run_log_writer=None,
        commit_record=None,
        expected_successful_episodes=1,
    )
    transaction.close()

    assert len(artifacts) == 1
    assert artifacts[0].episode_count == 1
    assert transaction.dataset_path.exists()


def test_post_append_ledger_replacement_is_ambiguous_and_does_not_roll_back_dataset(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path)
    _write_dataset(transaction)
    writer = transaction.open_run_log_writer()
    assert writer is not None
    writer.append({"record_type": "started"})
    original_append = writer.append

    def replace_after_append(record: dict[str, object]) -> None:
        original_append(record)
        assert transaction._run_log_parent_fd is not None
        assert transaction.run_log_path is not None
        os.unlink(transaction.run_log_path.name, dir_fd=transaction._run_log_parent_fd)
        replacement = os.open(
            transaction.run_log_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
            dir_fd=transaction._run_log_parent_fd,
        )
        os.write(replacement, b'{"attacker":true}\n')
        os.close(replacement)

    writer.append = replace_after_append
    with pytest.raises(DatasetCommitUncertainError, match="do not retry"):
        transaction.publish(
            run_log_writer=writer,
            commit_record={"record_type": "run_committed"},
            expected_successful_episodes=1,
        )
    writer.close()
    transaction.close()

    assert transaction.dataset_path.exists()
    assert json.loads((tmp_path / "audit" / "run.jsonl").read_text()) == {"attacker": True}


def test_failed_dataset_is_validated_and_published_as_a_separate_artifact(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, keep_failed=True)
    successful = _write_dataset(transaction, episodes=2)
    failed = _write_dataset(transaction, episodes=3, failed=True)

    artifacts = transaction.publish(
        run_log_writer=None,
        commit_record=None,
        require_failed_dataset=True,
        expected_successful_episodes=2,
        expected_failed_episodes=3,
    )
    transaction.close()

    assert [artifact.kind for artifact in artifacts] == ["successful", "failed"]
    assert transaction.dataset_path.read_bytes() == successful
    assert transaction.failed_dataset_path is not None
    assert transaction.failed_dataset_path.read_bytes() == failed


def test_zero_sample_failed_episode_without_task_stream_is_published(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, keep_failed=True)
    _write_dataset(transaction)
    failed = _write_zero_sample_failed_dataset(transaction)

    artifacts = transaction.publish(
        run_log_writer=None,
        commit_record=None,
        require_failed_dataset=True,
        expected_successful_episodes=1,
        expected_failed_episodes=1,
    )
    transaction.close()

    assert [artifact.kind for artifact in artifacts] == ["successful", "failed"]
    assert transaction.failed_dataset_path is not None
    assert transaction.failed_dataset_path.read_bytes() == failed


def test_nonzero_failed_episode_still_requires_complete_task_stream(tmp_path: Path) -> None:
    transaction = _reserve(tmp_path, keep_failed=True)
    _write_dataset(transaction)
    _write_zero_sample_failed_dataset(transaction)
    staged = Path(transaction.recording_targets.dataset_export_dir_path) / "episodes_failed.hdf5"
    with h5py.File(staged, "r+") as dataset:
        dataset["data/demo_0"].attrs["num_samples"] = 1

    with pytest.raises(ValueError, match="has no 'obs' group"):
        transaction.publish(
            run_log_writer=None,
            commit_record=None,
            require_failed_dataset=True,
            expected_successful_episodes=1,
            expected_failed_episodes=1,
        )
    transaction.close()

    assert not transaction.dataset_path.exists()


def test_request_anchor_detects_parent_directory_replacement(tmp_path: Path) -> None:
    request_directory = tmp_path / "request-root"
    request_directory.mkdir()
    request_path = request_directory / "request.yaml"
    request_path.write_text("schema_version: 1\n", encoding="utf-8")
    anchor = RequestDirectoryAnchor.open(request_path)
    original_directory = tmp_path / "original-root"
    request_directory.rename(original_directory)
    request_directory.mkdir()
    (request_directory / "request.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="renamed or replaced"):
        anchor.verify_current()
    anchor.close()


def test_request_anchor_detects_request_file_replacement(tmp_path: Path) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: 1\n", encoding="utf-8")
    anchor = RequestDirectoryAnchor.open(request_path)
    request_path.rename(tmp_path / "old-request.yaml")
    request_path.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity changed"):
        anchor.verify_current()
    anchor.close()
