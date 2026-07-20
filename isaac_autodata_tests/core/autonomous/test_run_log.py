# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import pytest

import isaac_autodata_core.autonomous.run_log as run_log_module
from isaac_autodata_core.autonomous.run_log import (
    MAX_RUN_RECORD_COLLECTION_ITEMS,
    MAX_RUN_RECORD_JSON_NODES,
    RunLogWriter,
    RunLogWriteUncertainError,
    normalize_run_record,
    safe_exception_record,
)


class _MisreportedMapping(Mapping):
    def __getitem__(self, key):
        return key

    def __iter__(self):
        yield "first"
        yield "unexpected"

    def __len__(self):
        return 1


def test_run_log_writer_appends_complete_json_lines(tmp_path):
    path = tmp_path / "attempts.jsonl"
    writer = RunLogWriter(path)

    writer.append({"attempt_id": "attempt-1", "status": "failed"})
    writer.append({"attempt_id": "attempt-2", "status": "succeeded"})
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["attempt_id"] for line in lines] == ["attempt-1", "attempt-2"]


def test_run_log_writer_requires_jsonl_extension(tmp_path):
    with pytest.raises(ValueError, match="jsonl"):
        RunLogWriter(tmp_path / "attempts.json")


def test_run_log_writer_rejects_oversized_record(tmp_path):
    writer = RunLogWriter(tmp_path / "attempts.jsonl", max_record_bytes=1024)
    with pytest.raises(ValueError, match="maximum"):
        writer.append({"payload": "x" * 2000})
    writer.close()


def test_run_log_writer_rejects_non_json_objects(tmp_path):
    writer = RunLogWriter(tmp_path / "attempts.jsonl")
    with pytest.raises(ValueError, match="unsupported"):
        writer.append({"payload": object()})
    writer.close()


def test_run_log_value_normalization_matches_writer_types_and_byte_bound():
    normalized = normalize_run_record(
        {"nested": {"values": (1, 2, 3)}},
        max_serialized_bytes=128,
    )

    assert normalized == {"nested": {"values": [1, 2, 3]}}
    with pytest.raises(ValueError, match="maximum is 32"):
        normalize_run_record({"payload": "x" * 64}, max_serialized_bytes=32)


def test_run_log_value_normalization_rejects_excessive_depth_and_collection_items():
    nested = {}
    for _ in range(14):
        nested = {"nested": nested}

    with pytest.raises(ValueError, match="nesting depth"):
        normalize_run_record(nested)
    with pytest.raises(ValueError, match="maximum item count"):
        normalize_run_record([None] * (MAX_RUN_RECORD_COLLECTION_ITEMS + 1))

    shared_values = [None] * 1_000
    repeated_mapping = {
        str(index): shared_values for index in range(MAX_RUN_RECORD_JSON_NODES // len(shared_values) + 1)
    }
    with pytest.raises(ValueError, match="maximum JSON node count"):
        normalize_run_record(repeated_mapping)
    with pytest.raises(ValueError, match="misreports"):
        normalize_run_record(_MisreportedMapping())


def test_partial_append_permanently_poisons_ledger_without_later_writes(tmp_path, monkeypatch):
    path = tmp_path / "attempts.jsonl"
    writer = RunLogWriter(path)
    original_write = run_log_module.os.write
    first_write = True

    def partial_then_fail(descriptor, data):
        nonlocal first_write
        if first_write:
            first_write = False
            original_write(descriptor, data[:7])
            raise OSError("simulated partial append")
        return original_write(descriptor, data)

    monkeypatch.setattr(run_log_module.os, "write", partial_then_fail)
    with pytest.raises(RunLogWriteUncertainError, match="durability is unknown"):
        writer.append({"record_type": "first"})
    poisoned_size = path.stat().st_size

    with pytest.raises(RunLogWriteUncertainError, match="poisoned"):
        writer.append({"record_type": "must_not_be_written"})
    writer.close()

    assert poisoned_size == 7
    assert path.stat().st_size == poisoned_size


def test_run_log_writer_uses_and_closes_held_append_descriptor(tmp_path):
    path = tmp_path / "attempts.jsonl"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW,
        0o600,
    )
    writer = RunLogWriter(path, fd=descriptor)

    writer.append({"record_type": "held_fd"})
    writer.close()

    assert json.loads(path.read_text())["record_type"] == "held_fd"
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_run_log_writer_rejects_descriptor_without_append(tmp_path):
    path = tmp_path / "attempts.jsonl"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with pytest.raises(ValueError, match="O_APPEND"):
            RunLogWriter(path, fd=descriptor)
    finally:
        os.close(descriptor)


def test_run_log_writer_does_not_follow_final_symlink(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text("owned\n", encoding="utf-8")
    (tmp_path / "attempts.jsonl").symlink_to(target)

    with pytest.raises(OSError):
        RunLogWriter(tmp_path / "attempts.jsonl")

    assert target.read_text(encoding="utf-8") == "owned\n"


def test_safe_exception_record_is_bounded_and_optional_traceback():
    try:
        raise RuntimeError("bad\x00message" + "x" * 3000)
    except RuntimeError as exc:
        short = safe_exception_record(exc)
        debug = safe_exception_record(exc, include_traceback=True)

    assert short["exception_type"].endswith("RuntimeError")
    assert "\x00" not in short["message"]
    assert len(short["message"]) == 2048
    assert "traceback" not in short
    assert "RuntimeError" in debug["traceback"]
