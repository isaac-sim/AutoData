# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-safe reservation and publication of autonomous run outputs.

The transaction is intentionally Linux/POSIX specific. Autonomous Isaac execution already runs in
the Linux runtime image, where directory descriptors and ``/proc/self/fd`` let the recorder write
through an identity that cannot be redirected by a later path or symlink replacement.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePath
from typing import Any

from isaac_autodata_core.autonomous.run_log import RunLogWriter

DEFAULT_MAX_DATASET_BYTES = 1 << 40
DEFAULT_MAX_REQUEST_BYTES = 4 << 20
_HASH_CHUNK_BYTES = 8 << 20
_HDF5_VALIDATION_CHUNK_BYTES = 8 << 20
_MAX_STAGING_NAME_ATTEMPTS = 32


@dataclass(frozen=True)
class _ValidatedDataset:
    staged_name: str
    final_name: str
    descriptor: int
    device: int
    inode: int
    mtime_ns: int
    artifact: DatasetArtifact


class RequestDirectoryAnchor:
    """Held identity of the request file and its containing output root."""

    def __init__(self, path: Path, directory_fd: int, request_fd: int) -> None:
        self.path = path
        self.directory = path.parent
        self._directory_fd = directory_fd
        self._request_fd = request_fd
        self._directory_identity = _device_inode(os.fstat(directory_fd))
        self._request_identity = _device_inode(os.fstat(request_fd))
        self._closed = False

    @classmethod
    def open(
        cls,
        request_path: str | Path,
        *,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> RequestDirectoryAnchor:
        """Open a regular request and its parent without following symlink components."""

        if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes < 1:
            raise ValueError("max_request_bytes must be a positive integer")
        path = _absolute_lexical_path(request_path)
        directory_fd = _open_absolute_directory(path.parent)
        request_fd: int | None = None
        try:
            request_fd = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
            request_stat = os.fstat(request_fd)
            if not stat.S_ISREG(request_stat.st_mode) or request_stat.st_nlink != 1:
                raise ValueError("task request must be a regular file with exactly one link")
            if request_stat.st_size <= 0 or request_stat.st_size > max_request_bytes:
                raise ValueError(
                    f"task request size must be in [1, {max_request_bytes}] bytes, got {request_stat.st_size}"
                )
            return cls(path, directory_fd, request_fd)
        except Exception:
            if request_fd is not None:
                os.close(request_fd)
            os.close(directory_fd)
            raise

    def verify_current(self) -> None:
        """Require the request pathname and parent to retain their held identities."""

        if self._closed:
            raise RuntimeError("request directory anchor is closed")
        if _device_inode(os.fstat(self._directory_fd)) != self._directory_identity:
            raise RuntimeError("attested request directory identity changed")
        visible_directory_fd = _open_absolute_directory(self.directory)
        try:
            if _device_inode(os.fstat(visible_directory_fd)) != self._directory_identity:
                raise RuntimeError("attested request directory path was renamed or replaced")
        finally:
            os.close(visible_directory_fd)
        request_stat = os.fstat(self._request_fd)
        path_stat = os.stat(self.path.name, dir_fd=self._directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise RuntimeError("attested request path is no longer a single-link regular file")
        if _device_inode(request_stat) != self._request_identity or _device_inode(path_stat) != self._request_identity:
            raise RuntimeError("attested request path identity changed during compilation")

    def duplicate_directory_fd(self) -> int:
        """Return a caller-owned duplicate of the verified request-directory descriptor."""

        self.verify_current()
        return os.dup(self._directory_fd)

    def close(self) -> None:
        """Close the held request and parent descriptors."""

        if self._closed:
            return
        self._closed = True
        os.close(self._request_fd)
        os.close(self._directory_fd)


class DatasetCommitUncertainError(RuntimeError):
    """Dataset links are durable but terminal run-log durability is unknown."""


@dataclass(frozen=True)
class RecordingTargets:
    """Recorder destination backed by a held private staging-directory descriptor.

    Attributes:
        dataset_export_dir_path: Stable process-local path to the held staging directory.
        dataset_filename: Dataset filename stem expected by Isaac Lab's recorder.
    """

    dataset_export_dir_path: str
    dataset_filename: str


@dataclass(frozen=True)
class DatasetArtifact:
    """Validated identity of one published HDF5 artifact."""

    path: str
    sha256: str
    size_bytes: int
    episode_count: int
    kind: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-compatible artifact record."""

        return {
            "episode_count": self.episode_count,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class OutputTransaction:
    """Own output directory identities from reservation through durable publication.

    Use :meth:`reserve` after child request attestation and before the Isaac recorder is built.
    The caller must close the Arena environment before calling :meth:`publish`, because recorder
    shutdown is the operation that flushes and closes the staged HDF5 file.
    """

    def __init__(
        self,
        *,
        request_directory: Path,
        dataset_path: Path,
        failed_dataset_path: Path | None,
        run_log_path: Path | None,
        request_dir_fd: int,
        dataset_parent_fd: int,
        run_log_parent_fd: int | None,
        run_log_fd: int | None,
        staging_name: str,
        staging_fd: int,
        max_dataset_bytes: int,
    ) -> None:
        self.request_directory = request_directory
        self.dataset_path = dataset_path
        self.failed_dataset_path = failed_dataset_path
        self.run_log_path = run_log_path
        self._request_dir_fd = request_dir_fd
        self._dataset_parent_fd = dataset_parent_fd
        self._run_log_parent_fd = run_log_parent_fd
        self._run_log_fd = run_log_fd
        self._staging_name = staging_name
        self._staging_fd = staging_fd
        self._max_dataset_bytes = max_dataset_bytes
        self._published = False
        self._closed = False

    @classmethod
    def reserve(
        cls,
        *,
        request_directory: str | Path | None = None,
        request_anchor: RequestDirectoryAnchor | None = None,
        dataset_path: str | Path,
        run_log_path: str | Path | None,
        keep_failed: bool,
        max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
    ) -> OutputTransaction:
        """Reserve fresh outputs and a private dataset staging directory.

        Every path component is traversed relative to a held directory descriptor with
        ``O_DIRECTORY | O_NOFOLLOW``. Missing output-parent components are created mode ``0700``.
        The run_log file is created exclusively and retained open with ``O_APPEND``.

        Args:
            request_directory: Canonical directory containing the attested request.
            dataset_path: Canonical final ``.hdf5`` path below ``request_directory``.
            run_log_path: Optional canonical final ``.jsonl`` path below the request directory.
            keep_failed: Reserve publication support for the recorder's failed-episode dataset.
            max_dataset_bytes: Maximum accepted size of each staged dataset.
        """

        if isinstance(max_dataset_bytes, bool) or not isinstance(max_dataset_bytes, int) or max_dataset_bytes < 1:
            raise ValueError("max_dataset_bytes must be a positive integer")
        if request_anchor is None and request_directory is None:
            raise ValueError("request_directory or request_anchor is required")
        if request_anchor is not None:
            request_anchor.verify_current()
            request_dir = request_anchor.directory
            if request_directory is not None and _absolute_lexical_path(request_directory) != request_dir:
                raise ValueError("request_directory does not match the held request anchor")
        else:
            assert request_directory is not None
            request_dir = _absolute_lexical_path(request_directory)
        dataset = _absolute_lexical_path(dataset_path)
        run_log = None if run_log_path is None else _absolute_lexical_path(run_log_path)
        if dataset.suffix.lower() != ".hdf5":
            raise ValueError("dataset path must end in .hdf5")
        if run_log is not None and run_log.suffix.lower() != ".jsonl":
            raise ValueError("run_log path must end in .jsonl")
        dataset_relative = _relative_output_path(dataset, request_dir, "dataset")
        run_log_relative = None if run_log is None else _relative_output_path(run_log, request_dir, "run_log")

        request_fd = (
            request_anchor.duplicate_directory_fd()
            if request_anchor is not None
            else _open_absolute_directory(request_dir)
        )
        dataset_parent_fd: int | None = None
        run_log_parent_fd: int | None = None
        run_log_fd: int | None = None
        staging_name: str | None = None
        staging_fd: int | None = None
        try:
            dataset_parent_fd = _open_relative_directory(
                request_fd,
                dataset_relative.parent.parts,
                create=True,
            )
            _require_missing_entry(dataset_relative.name, dataset_parent_fd, "dataset")
            failed_dataset = dataset.with_name(f"{dataset.stem}_failed{dataset.suffix}") if keep_failed else None
            if failed_dataset is not None:
                _require_missing_entry(failed_dataset.name, dataset_parent_fd, "failed dataset")

            staging_name, staging_fd = _create_staging_directory(dataset_parent_fd)
            _probe_output_filesystem(staging_fd, dataset_parent_fd)

            if run_log_relative is not None:
                run_log_parent_fd = _open_relative_directory(
                    request_fd,
                    run_log_relative.parent.parts,
                    create=True,
                )
                run_log_fd = _open_exclusive_run_log(run_log_relative.name, run_log_parent_fd)
                _fsync_directory(run_log_parent_fd)
                _require_visible_fd_identity(run_log_relative.name, run_log_parent_fd, run_log_fd)

            return cls(
                request_directory=request_dir,
                dataset_path=dataset,
                failed_dataset_path=failed_dataset,
                run_log_path=run_log,
                request_dir_fd=request_fd,
                dataset_parent_fd=dataset_parent_fd,
                run_log_parent_fd=run_log_parent_fd,
                run_log_fd=run_log_fd,
                staging_name=staging_name,
                staging_fd=staging_fd,
                max_dataset_bytes=max_dataset_bytes,
            )
        except Exception:
            if run_log_fd is not None:
                created_identity = _device_inode(os.fstat(run_log_fd))
                os.close(run_log_fd)
                assert run_log_relative is not None
                assert run_log_parent_fd is not None
                try:
                    visible = os.stat(
                        run_log_relative.name,
                        dir_fd=run_log_parent_fd,
                        follow_symlinks=False,
                    )
                    if stat.S_ISREG(visible.st_mode) and _device_inode(visible) == created_identity:
                        os.unlink(run_log_relative.name, dir_fd=run_log_parent_fd)
                        _fsync_directory(run_log_parent_fd)
                except FileNotFoundError:
                    pass
            if staging_fd is not None and staging_name is not None and dataset_parent_fd is not None:
                with suppress(Exception):
                    _cleanup_private_staging(staging_fd, staging_name, dataset_parent_fd)
            _close_distinct_fds_best_effort(run_log_parent_fd, dataset_parent_fd, request_fd)
            raise

    @property
    def recording_targets(self) -> RecordingTargets:
        """Return the recorder override bound to the held staging directory."""

        self._require_open()
        return RecordingTargets(
            dataset_export_dir_path=f"/proc/self/fd/{self._staging_fd}",
            dataset_filename=self.dataset_path.stem,
        )

    def open_run_log_writer(
        self,
        writer_factory: Any = RunLogWriter,
    ) -> RunLogWriter | Any | None:
        """Create a writer on a duplicate of the exclusively reserved run_log descriptor.

        The returned writer owns its duplicate. The transaction retains the original descriptor so
        the reserved inode stays held even if a caller closes the writer early.
        """

        self._require_open()
        if self._run_log_fd is None or self.run_log_path is None:
            return None
        duplicate = os.dup(self._run_log_fd)
        try:
            return writer_factory(self.run_log_path, fd=duplicate, close_fd=True)
        except Exception:
            os.close(duplicate)
            raise

    def publish(
        self,
        *,
        run_log_writer: RunLogWriter | Any | None,
        commit_record: Mapping[str, Any] | None,
        require_failed_dataset: bool = False,
        expected_successful_episodes: int | None = None,
        expected_failed_episodes: int | None = None,
    ) -> tuple[DatasetArtifact, ...]:
        """Validate, atomically link, and durably commit staged artifacts.

        Final links are created with no replacement. If publication of a later artifact fails, links
        created by this call are removed before the error is returned. The run_log commit record
        is appended only after all final links and their parent directory have been fsynced.
        """

        self._require_open()
        if self._published:
            raise RuntimeError("output transaction has already been published")
        terminal_record: dict[str, Any] | None = None
        if commit_record is not None:
            if run_log_writer is None:
                raise ValueError("a run_log commit record requires a run_log writer")
            terminal_record = dict(commit_record)
            if "artifacts" in terminal_record:
                raise ValueError("commit_record must not define the transaction-owned artifacts field")
        specifications = [(self.dataset_path.name, self.dataset_path, "successful")]
        if self.failed_dataset_path is not None:
            failed_name = self.failed_dataset_path.name
            if _entry_exists(failed_name, self._staging_fd):
                specifications.append((failed_name, self.failed_dataset_path, "failed"))
            elif require_failed_dataset:
                raise FileNotFoundError("recorder did not produce the expected failed-episode dataset")

        expected_counts = {
            "successful": expected_successful_episodes,
            "failed": expected_failed_episodes,
        }
        validated: list[_ValidatedDataset] = []
        linked_items: list[_ValidatedDataset] = []
        try:
            self._verify_output_parent_identities()
            self._verify_run_log_identity()
            for staged_name, final_path, kind in specifications:
                validated.append(
                    self._validate_staged_dataset(
                        staged_name,
                        final_path,
                        kind,
                        expected_episode_count=expected_counts[kind],
                    )
                )
            artifacts = tuple(item.artifact for item in validated)
            if terminal_record is not None:
                terminal_record["artifacts"] = [artifact.to_dict() for artifact in artifacts]
            for item in validated:
                _link_fd_to_name(item.descriptor, self._dataset_parent_fd, item.final_name)
                linked_items.append(item)
                linked = os.stat(item.final_name, dir_fd=self._dataset_parent_fd, follow_symlinks=False)
                if not stat.S_ISREG(linked.st_mode) or (linked.st_dev, linked.st_ino) != (item.device, item.inode):
                    raise RuntimeError("published dataset identity did not match the validated inode")
            _fsync_directory(self._dataset_parent_fd)
            for item in validated:
                self._unlink_staged_dataset_inode(item)
            _fsync_directory(self._staging_fd)
            for item in validated:
                self._verify_published_dataset(item)
            self._verify_output_parent_identities()
            self._verify_run_log_identity()
        except Exception:
            for item in reversed(linked_items):
                try:
                    visible = os.stat(item.final_name, dir_fd=self._dataset_parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if _device_inode(visible) == (item.device, item.inode):
                    os.unlink(item.final_name, dir_fd=self._dataset_parent_fd)
            if linked_items:
                _fsync_directory(self._dataset_parent_fd)
            for item in validated:
                with suppress(OSError):
                    os.close(item.descriptor)
            validated.clear()
            raise

        self._published = True
        try:
            if terminal_record is not None:
                try:
                    assert run_log_writer is not None
                    run_log_writer.append(terminal_record)
                    self._verify_run_log_identity()
                    assert self._run_log_parent_fd is not None
                    _fsync_directory(self._run_log_parent_fd)
                except Exception as exc:
                    raise DatasetCommitUncertainError(
                        "dataset links are durable, but terminal run_log append durability is unknown; "
                        "do not retry or delete outputs until run-log recovery is complete"
                    ) from exc
            return artifacts
        finally:
            for item in validated:
                with suppress(OSError):
                    os.close(item.descriptor)

    def close(self) -> None:
        """Close held descriptors and remove the exact private staging directory."""

        if self._closed:
            return
        self._closed = True
        issues: list[BaseException] = []
        try:
            _cleanup_private_staging(self._staging_fd, self._staging_name, self._dataset_parent_fd)
        except BaseException as exc:
            issues.append(exc)
        if self._run_log_fd is not None:
            try:
                os.close(self._run_log_fd)
            except OSError as exc:
                issues.append(exc)
        for descriptor in _distinct_fds(
            self._run_log_parent_fd,
            self._dataset_parent_fd,
            self._request_dir_fd,
        ):
            try:
                os.close(descriptor)
            except OSError as exc:
                issues.append(exc)
        if issues:
            raise OSError("; ".join(str(issue) for issue in issues))

    def _validate_staged_dataset(
        self,
        staged_name: str,
        final_path: Path,
        kind: str,
        *,
        expected_episode_count: int | None,
    ) -> _ValidatedDataset:
        descriptor = os.open(
            staged_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._staging_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"staged {kind} dataset is not a regular file")
            if before.st_nlink != 1:
                raise ValueError(f"staged {kind} dataset must have exactly one link before publication")
            if before.st_size <= 0:
                raise ValueError(f"staged {kind} dataset is empty")
            if before.st_size > self._max_dataset_bytes:
                raise ValueError(
                    f"staged {kind} dataset is {before.st_size} bytes; maximum is {self._max_dataset_bytes}"
                )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            digest = _sha256_fd(descriptor, expected_size=before.st_size)
            episode_count = _hdf5_episode_count(descriptor, kind=kind)
            if expected_episode_count is not None and episode_count != expected_episode_count:
                raise ValueError(
                    f"staged {kind} dataset has {episode_count} episodes; expected {expected_episode_count}"
                )
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise RuntimeError(f"staged {kind} dataset changed during validation")
            return _ValidatedDataset(
                staged_name=staged_name,
                final_name=final_path.name,
                descriptor=descriptor,
                device=before.st_dev,
                inode=before.st_ino,
                mtime_ns=before.st_mtime_ns,
                artifact=DatasetArtifact(
                    path=str(final_path),
                    sha256=digest,
                    size_bytes=before.st_size,
                    episode_count=episode_count,
                    kind=kind,
                ),
            )
        except Exception:
            os.close(descriptor)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("output transaction is closed")

    def _verify_run_log_identity(self) -> None:
        if self._run_log_fd is None:
            return
        assert self._run_log_parent_fd is not None
        assert self.run_log_path is not None
        held = os.fstat(self._run_log_fd)
        visible = os.stat(
            self.run_log_path.name,
            dir_fd=self._run_log_parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(held.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or held.st_nlink != 1
            or visible.st_nlink != 1
            or _device_inode(held) != _device_inode(visible)
        ):
            raise RuntimeError("reserved run_log path no longer names the held run-log inode")
        os.fsync(self._run_log_fd)

    def _verify_output_parent_identities(self) -> None:
        _require_visible_directory_identity(
            self.request_directory,
            self._request_dir_fd,
            "request root",
        )
        _require_visible_directory_identity(
            self.dataset_path.parent,
            self._dataset_parent_fd,
            "dataset parent",
        )
        if self.run_log_path is not None:
            assert self._run_log_parent_fd is not None
            _require_visible_directory_identity(
                self.run_log_path.parent,
                self._run_log_parent_fd,
                "run_log parent",
            )

    def _verify_published_dataset(self, item: _ValidatedDataset) -> None:
        held = os.fstat(item.descriptor)
        visible = os.stat(item.final_name, dir_fd=self._dataset_parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(visible.st_mode)
            or held.st_nlink != 1
            or visible.st_nlink != 1
            or _device_inode(held) != (item.device, item.inode)
            or _device_inode(visible) != (item.device, item.inode)
            or held.st_size != item.artifact.size_bytes
            or visible.st_size != item.artifact.size_bytes
            or held.st_mtime_ns != item.mtime_ns
            or visible.st_mtime_ns != item.mtime_ns
        ):
            raise RuntimeError("published dataset path no longer names the validated inode")
        if _sha256_fd(item.descriptor, expected_size=item.artifact.size_bytes) != item.artifact.sha256:
            raise RuntimeError("published dataset content changed after validation")

    def _unlink_staged_dataset_inode(self, item: _ValidatedDataset) -> None:
        matches: list[str] = []
        names = os.listdir(self._staging_fd)
        if len(names) > 32:
            raise RuntimeError("private staging directory contains too many entries during publication")
        for name in names:
            visible = os.stat(name, dir_fd=self._staging_fd, follow_symlinks=False)
            if _device_inode(visible) == (item.device, item.inode):
                matches.append(name)
        if len(matches) > 1:
            raise RuntimeError("validated dataset acquired unexpected additional staging links")
        if matches:
            os.unlink(matches[0], dir_fd=self._staging_fd)


def _absolute_lexical_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(candidate))


def _device_inode(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _relative_output_path(path: Path, request_directory: Path, label: str) -> PurePath:
    try:
        relative = path.relative_to(request_directory)
    except ValueError as exc:
        raise ValueError(f"{label} path must be below the request directory") from exc
    if relative == PurePath(".") or not relative.name:
        raise ValueError(f"{label} path must name a file")
    if any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{label} path contains an unsafe component")
    return relative


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("directory anchor must be absolute")
    descriptor = os.open("/", _directory_open_flags())
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_relative_directory(anchor_fd: int, components: tuple[str, ...], *, create: bool) -> int:
    descriptor = os.dup(anchor_fd)
    try:
        for component in components:
            if component in ("", ".", "..") or "/" in component:
                raise ValueError("output directory contains an unsafe component")
            try:
                next_descriptor = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                next_descriptor = os.open(component, _directory_open_flags(), dir_fd=descriptor)
                _fsync_directory(descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_missing_entry(name: str, parent_fd: int, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"refusing to overwrite existing {label} output: {name}")


def _require_visible_fd_identity(name: str, parent_fd: int, descriptor: int) -> None:
    held = os.fstat(descriptor)
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or held.st_nlink != 1
        or visible.st_nlink != 1
        or _device_inode(held) != _device_inode(visible)
    ):
        raise RuntimeError("exclusively created output path no longer names its held inode")


def _require_visible_directory_identity(path: Path, held_fd: int, label: str) -> None:
    visible_fd = _open_absolute_directory(path)
    try:
        if _device_inode(os.fstat(visible_fd)) != _device_inode(os.fstat(held_fd)):
            raise RuntimeError(f"visible {label} path no longer names its held directory inode")
    finally:
        os.close(visible_fd)


def _entry_exists(name: str, parent_fd: int) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_staging_directory(parent_fd: int) -> tuple[str, int]:
    for _ in range(_MAX_STAGING_NAME_ATTEMPTS):
        name = f".autodata-{secrets.token_hex(16)}.staging"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        os.fchmod(descriptor, 0o700)
        _fsync_directory(parent_fd)
        return name, descriptor
    raise FileExistsError("could not allocate a private dataset staging directory")


def _open_exclusive_run_log(name: str, parent_fd: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _sha256_fd(descriptor: int, *, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        chunk = os.pread(descriptor, min(_HASH_CHUNK_BYTES, expected_size - offset), offset)
        if not chunk:
            raise OSError("staged dataset ended while it was being hashed")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _hdf5_episode_count(descriptor: int, *, kind: str) -> int:
    signature = os.pread(descriptor, 8, 0)
    if signature != b"\x89HDF\r\n\x1a\n":
        raise ValueError(f"staged {kind} dataset does not have an HDF5 signature")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to validate staged autonomous datasets") from exc

    duplicate = os.dup(descriptor)
    with os.fdopen(duplicate, "rb", closefd=True) as stream:
        with h5py.File(stream, "r") as dataset:
            data = dataset.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError(f"staged {kind} dataset has no HDF5 'data' group")
            if kind == "filesystem probe":
                return len(data)
            format_version = dataset.attrs.get("format_version")
            if isinstance(format_version, bool) or not isinstance(format_version, Integral) or format_version != 1:
                raise ValueError(f"staged {kind} dataset has unsupported or missing format_version")
            episode_names = list(data)
            episode_indices: list[int] = []
            expected_success = kind == "successful"
            for episode_name in episode_names:
                match = re.fullmatch(r"demo_(\d+)", episode_name)
                if match is None:
                    raise ValueError(f"staged {kind} dataset has invalid episode name {episode_name!r}")
                episode_indices.append(int(match.group(1)))
                episode = data[episode_name]
                if not isinstance(episode, h5py.Group):
                    raise ValueError(f"staged {kind} dataset episode {episode_name!r} is not a group")
                _validate_hdf5_episode(episode, episode_name, expected_success=expected_success, h5py=h5py)
            if sorted(episode_indices) != list(range(len(episode_indices))):
                raise ValueError(f"staged {kind} dataset episode indices must be contiguous from zero")
            return len(episode_names)


def _validate_hdf5_episode(episode: Any, episode_name: str, *, expected_success: bool, h5py: Any) -> None:
    import numpy as np

    initial_state = episode.get("initial_state")
    if not isinstance(initial_state, h5py.Group):
        raise ValueError(f"staged dataset episode {episode_name!r} has no 'initial_state' group")
    num_samples = episode.attrs.get("num_samples")
    minimum_samples = 1 if expected_success else 0
    if isinstance(num_samples, bool) or not isinstance(num_samples, Integral) or num_samples < minimum_samples:
        raise ValueError(f"staged dataset episode {episode_name!r} has invalid num_samples")
    success = episode.attrs.get("success")
    if not isinstance(success, (bool, np.bool_)) or bool(success) is not expected_success:
        raise ValueError(f"staged dataset episode {episode_name!r} has inconsistent success metadata")

    # Isaac Lab can retain a failed attempt before its first task step. Such an episode has a
    # reset-time initial state and num_samples=0, but no task-stream groups or action datasets.
    # Preserve that diagnostic without weakening successful or partially executed episodes.
    require_task_stream = expected_success or num_samples > 0
    task_groups: dict[str, Any] = {}
    for name in ("obs", "states"):
        value = episode.get(name)
        if value is None and not require_task_stream:
            continue
        if not isinstance(value, h5py.Group):
            raise ValueError(f"staged dataset episode {episode_name!r} has no {name!r} group")
        task_groups[name] = value
    task_datasets: dict[str, Any] = {}
    for name in ("actions", "processed_actions"):
        value = episode.get(name)
        if value is None and not require_task_stream:
            continue
        if not isinstance(value, h5py.Dataset) or len(value.shape) != 2 or value.shape[0] != num_samples:
            raise ValueError(f"staged dataset episode {episode_name!r} {name!r} must be rank-2 with num_samples rows")
        if value.shape[1] < 1:
            raise ValueError(f"staged dataset episode {episode_name!r} {name!r} must have a nonempty feature axis")
        task_datasets[name] = value
    for name, value in task_datasets.items():
        _require_finite_hdf5_dataset(value, episode_name, name)
    _validate_hdf5_dataset_group(
        initial_state,
        episode_name,
        "initial_state",
        h5py=h5py,
        require_datasets=expected_success,
        require_nonempty_datasets=expected_success,
    )
    for group_name, group in task_groups.items():
        _validate_hdf5_dataset_group(
            group,
            episode_name,
            group_name,
            h5py=h5py,
            expected_rows=num_samples,
            require_datasets=require_task_stream,
        )


def _validate_hdf5_dataset_group(
    group: Any,
    episode_name: str,
    group_name: str,
    *,
    h5py: Any,
    expected_rows: int | None = None,
    require_datasets: bool,
    require_nonempty_datasets: bool = False,
) -> None:
    datasets: list[tuple[str, Any]] = []

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            datasets.append((name, value))

    group.visititems(collect)
    if require_datasets and not datasets:
        raise ValueError(f"staged dataset episode {episode_name!r} {group_name!r} group has no datasets")
    for dataset_name, value in datasets:
        qualified_name = f"{group_name}/{dataset_name}"
        if expected_rows is not None and (len(value.shape) < 1 or value.shape[0] != expected_rows):
            raise ValueError(
                f"staged dataset episode {episode_name!r} {qualified_name!r} must have num_samples leading rows"
            )
        _require_finite_hdf5_dataset(
            value,
            episode_name,
            qualified_name,
            require_nonempty=require_nonempty_datasets,
        )


def _require_finite_hdf5_dataset(
    dataset: Any,
    episode_name: str,
    dataset_name: str,
    *,
    require_nonempty: bool = False,
) -> None:
    import numpy as np

    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} must be numeric")
    if require_nonempty and dataset.size < 1:
        raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} must not be empty")
    for selection in _bounded_hdf5_selections(dataset.shape, dataset.dtype.itemsize):
        values = dataset[()] if not selection else dataset[selection]
        if not np.isfinite(values).all():
            raise ValueError(f"staged dataset episode {episode_name!r} {dataset_name!r} contains non-finite values")


def _bounded_hdf5_selections(shape: tuple[int, ...], itemsize: int) -> Iterator[tuple[slice, ...]]:
    """Yield hyperslabs whose materialized numeric payload is at most the validation byte bound."""

    if not shape:
        yield ()
        return
    if any(size == 0 for size in shape):
        return
    max_items = max(1, _HDF5_VALIDATION_CHUNK_BYTES // max(1, itemsize))
    block_shape = [1] * len(shape)
    remaining_items = max_items
    for axis in range(len(shape) - 1, -1, -1):
        block_shape[axis] = min(shape[axis], remaining_items)
        remaining_items = max(1, remaining_items // block_shape[axis])
    starts = (range(0, size, block_size) for size, block_size in zip(shape, block_shape))
    for offsets in itertools.product(*starts):
        yield tuple(
            slice(offset, min(offset + block_size, size))
            for offset, block_size, size in zip(offsets, block_shape, shape)
        )


def _link_fd_to_name(source_fd: int, destination_dir_fd: int, destination_name: str) -> None:
    """Hard-link the exact held inode through its process-local descriptor identity."""

    os.link(
        f"/proc/self/fd/{source_fd}",
        destination_name,
        dst_dir_fd=destination_dir_fd,
        follow_symlinks=True,
    )


def _fsync_directory(descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISDIR(descriptor_stat.st_mode):
        raise ValueError("directory durability operation received a non-directory descriptor")
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise RuntimeError(
            "output filesystem does not support the required durable directory fsync "
            f"(errno={exc.errno}: {exc.strerror})"
        ) from exc


def _probe_output_filesystem(staging_fd: int, parent_fd: int) -> None:
    """Exercise the exact HDF5, descriptor-link, and durability primitives before generation."""

    token = secrets.token_hex(12)
    staged_name = f".autodata-probe-{token}.hdf5"
    published_name = f".autodata-probe-{token}.published"
    descriptor: int | None = None
    staged_exists = False
    published_exists = False
    try:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for the autonomous output filesystem probe") from exc
        staged_path = f"/proc/self/fd/{staging_fd}/{staged_name}"
        with h5py.File(staged_path, "x") as dataset:
            dataset.create_group("data")
        staged_exists = True
        descriptor = os.open(staged_name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=staging_fd)
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise RuntimeError("filesystem probe did not create a private single-link regular HDF5 file")
        os.fsync(descriptor)
        if _hdf5_episode_count(descriptor, kind="filesystem probe") != 0:
            raise RuntimeError("filesystem probe HDF5 episode group was not empty")
        _link_fd_to_name(descriptor, parent_fd, published_name)
        published_exists = True
        published_stat = os.stat(published_name, dir_fd=parent_fd, follow_symlinks=False)
        if _device_inode(published_stat) != _device_inode(source_stat):
            raise RuntimeError("filesystem probe hard link did not preserve the held inode")
        _fsync_directory(parent_fd)
        os.unlink(published_name, dir_fd=parent_fd)
        published_exists = False
        os.unlink(staged_name, dir_fd=staging_fd)
        staged_exists = False
        _fsync_directory(staging_fd)
        _fsync_directory(parent_fd)
    except Exception as exc:
        raise RuntimeError(
            "output filesystem does not support the required descriptor-safe HDF5 transaction: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if published_exists:
            with suppress(OSError):
                os.unlink(published_name, dir_fd=parent_fd)
        if staged_exists:
            with suppress(OSError):
                os.unlink(staged_name, dir_fd=staging_fd)


def _cleanup_private_staging(staging_fd: int, staging_name: str, parent_fd: int) -> None:
    issue: BaseException | None = None
    try:
        names = os.listdir(staging_fd)
        if len(names) > 32:
            raise OSError("private staging directory contains too many unexpected entries")
        for name in names:
            if name in ("", ".", "..") or "/" in name:
                raise OSError("private staging directory contains an unsafe entry")
            entry = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            if stat.S_ISDIR(entry.st_mode):
                os.rmdir(name, dir_fd=staging_fd)
            else:
                os.unlink(name, dir_fd=staging_fd)
    except BaseException as exc:
        issue = exc
    finally:
        os.close(staging_fd)
    try:
        os.rmdir(staging_name, dir_fd=parent_fd)
        _fsync_directory(parent_fd)
    except BaseException as exc:
        if issue is None:
            issue = exc
    if issue is not None:
        raise issue


def _distinct_fds(*descriptors: int | None) -> tuple[int, ...]:
    return tuple(dict.fromkeys(descriptor for descriptor in descriptors if descriptor is not None))


def _close_distinct_fds_best_effort(*descriptors: int | None) -> None:
    for descriptor in _distinct_fds(*descriptors):
        with suppress(OSError):
            os.close(descriptor)
