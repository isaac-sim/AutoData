# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import-free cuRobo detection and ScheduleStream backend selection.

ScheduleStream has two distinct manipulation applications: ``custream`` targets the cuRobo v1
module layout, while ``custream2`` targets cuRobo v2. This module detects concrete API markers on
disk without importing either stack, then makes backend selection explicit and reproducible.
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class CuroboApiGeneration(StrEnum):
    """Detected public API family."""

    UNAVAILABLE = "unavailable"
    V1 = "v1"
    V2 = "v2"
    UNKNOWN = "unknown"


CUROBO_V1_MARKERS = (
    "curobo.wrap.reacher.motion_gen",
    "curobo.types.math",
    "curobo.types.state",
    "curobo.util.usd_helper",
)
CUROBO_V2_MARKERS = (
    "curobo.motion_planner",
    "curobo.scene",
    "curobo.types",
    "curobo._src.util.usd_scene_parser",
)
SCHEDULESTREAM_V1_MARKERS = (
    "schedulestream.applications.custream.example",
    "schedulestream.applications.custream.world",
)
SCHEDULESTREAM_V2_MARKERS = (
    "schedulestream.applications.custream2.policy",
    "schedulestream.applications.custream2.world",
)


@dataclass(frozen=True)
class DistributionIdentity:
    """Installed distribution version and optional direct-source commit."""

    name: str
    version: str
    source_url: str | None = None
    source_commit: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "source_commit": self.source_commit,
            "source_url": self.source_url,
            "version": self.version,
        }


@dataclass(frozen=True)
class CuroboRuntimeCapabilities:
    """Concrete motion/planning capabilities present in one Python runtime."""

    api_generation: CuroboApiGeneration
    curobo: DistributionIdentity | None
    schedulestream: DistributionIdentity | None
    has_schedulestream_v1: bool
    has_schedulestream_v2: bool
    missing_v1_markers: tuple[str, ...]
    missing_v2_markers: tuple[str, ...]
    missing_schedulestream_v1_markers: tuple[str, ...]
    missing_schedulestream_v2_markers: tuple[str, ...]

    @property
    def motion_backend_id(self) -> str | None:
        if self.api_generation is CuroboApiGeneration.V1:
            return "curobo_v1"
        if self.api_generation is CuroboApiGeneration.V2:
            return "curobo_v2"
        return None

    @property
    def schedulestream_application(self) -> str | None:
        if self.api_generation is CuroboApiGeneration.V1 and self.has_schedulestream_v1:
            return "custream"
        if self.api_generation is CuroboApiGeneration.V2 and self.has_schedulestream_v2:
            return "custream2"
        return None

    @property
    def supports_schedulestream(self) -> bool:
        return self.schedulestream_application is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_generation": self.api_generation.value,
            "curobo": None if self.curobo is None else self.curobo.to_dict(),
            "has_schedulestream_v1": self.has_schedulestream_v1,
            "has_schedulestream_v2": self.has_schedulestream_v2,
            "missing_markers": {
                "curobo_v1": list(self.missing_v1_markers),
                "curobo_v2": list(self.missing_v2_markers),
                "schedulestream_v1": list(self.missing_schedulestream_v1_markers),
                "schedulestream_v2": list(self.missing_schedulestream_v2_markers),
            },
            "motion_backend_id": self.motion_backend_id,
            "schedulestream": None if self.schedulestream is None else self.schedulestream.to_dict(),
            "schedulestream_application": self.schedulestream_application,
            "supports_schedulestream": self.supports_schedulestream,
        }


@dataclass(frozen=True)
class MotionBackendSelection:
    """Resolved cuRobo/ScheduleStream implementation for one run."""

    motion_backend: str
    schedulestream_application: str
    capabilities: CuroboRuntimeCapabilities


class BackendCompatibilityError(RuntimeError):
    """Raised when requested planner and installed API capabilities do not agree."""


def detect_curobo_runtime(
    *,
    module_available: Callable[[str], bool] | None = None,
    distribution_reader: Callable[[tuple[str, ...]], DistributionIdentity | None] | None = None,
) -> CuroboRuntimeCapabilities:
    """Inspect cuRobo/ScheduleStream API markers without importing their packages.

    Args:
        module_available: Optional test hook returning whether a module path exists.
        distribution_reader: Optional test hook resolving installed distribution identity.
    """

    module_available = module_available or _module_available_without_import
    distribution_reader = distribution_reader or _read_distribution_identity

    curobo_root = module_available("curobo")
    missing_v1 = tuple(marker for marker in CUROBO_V1_MARKERS if not module_available(marker))
    missing_v2 = tuple(marker for marker in CUROBO_V2_MARKERS if not module_available(marker))
    has_v1 = curobo_root and not missing_v1
    has_v2 = curobo_root and not missing_v2
    if not curobo_root:
        api_generation = CuroboApiGeneration.UNAVAILABLE
    elif has_v2:
        # A v2 install may retain compatibility modules with v1 names. Prefer its defining v2 API.
        api_generation = CuroboApiGeneration.V2
    elif has_v1:
        api_generation = CuroboApiGeneration.V1
    else:
        api_generation = CuroboApiGeneration.UNKNOWN

    schedulestream_root = module_available("schedulestream")
    missing_schedule_v1 = tuple(marker for marker in SCHEDULESTREAM_V1_MARKERS if not module_available(marker))
    missing_schedule_v2 = tuple(marker for marker in SCHEDULESTREAM_V2_MARKERS if not module_available(marker))
    return CuroboRuntimeCapabilities(
        api_generation=api_generation,
        curobo=distribution_reader(("nvidia-curobo", "curobo")) if curobo_root else None,
        schedulestream=(distribution_reader(("schedulestream",)) if schedulestream_root else None),
        has_schedulestream_v1=schedulestream_root and not missing_schedule_v1,
        has_schedulestream_v2=schedulestream_root and not missing_schedule_v2,
        missing_v1_markers=missing_v1,
        missing_v2_markers=missing_v2,
        missing_schedulestream_v1_markers=missing_schedule_v1,
        missing_schedulestream_v2_markers=missing_schedule_v2,
    )


def select_schedulestream_backend(
    requested_motion_backend: str,
    capabilities: CuroboRuntimeCapabilities | None = None,
) -> MotionBackendSelection:
    """Select the ScheduleStream application compatible with the installed cuRobo API.

    Args:
        requested_motion_backend: ``auto``, ``curobo_v1``, or ``curobo_v2``.
        capabilities: Precomputed capabilities; detected from the current runtime when omitted.
    """

    if requested_motion_backend not in ("auto", "curobo_v1", "curobo_v2"):
        raise BackendCompatibilityError(
            f"motion_backend must be one of ['auto', 'curobo_v1', 'curobo_v2'], got {requested_motion_backend!r}"
        )
    capabilities = capabilities or detect_curobo_runtime()
    detected = capabilities.motion_backend_id
    if detected is None:
        raise BackendCompatibilityError(
            "No supported cuRobo API was detected. "
            f"Detected state: {capabilities.api_generation.value}; "
            f"missing v1 markers: {list(capabilities.missing_v1_markers)}; "
            f"missing v2 markers: {list(capabilities.missing_v2_markers)}."
        )
    if requested_motion_backend != "auto" and requested_motion_backend != detected:
        raise BackendCompatibilityError(
            f"Requested {requested_motion_backend}, but this runtime provides {detected}. "
            "Use motion_backend: auto or select a matching runtime profile."
        )
    application = capabilities.schedulestream_application
    if application is None:
        expected_markers = (
            capabilities.missing_schedulestream_v1_markers
            if detected == "curobo_v1"
            else capabilities.missing_schedulestream_v2_markers
        )
        expected_application = "custream" if detected == "curobo_v1" else "custream2"
        raise BackendCompatibilityError(
            f"cuRobo backend {detected} is available, but ScheduleStream's {expected_application} "
            f"application is not complete; missing markers: {list(expected_markers)}. "
            "Install the reviewed pinned ScheduleStream runtime in its development image; "
            "the AutoData host environment will not be modified automatically."
        )
    return MotionBackendSelection(
        motion_backend=detected,
        schedulestream_application=application,
        capabilities=capabilities,
    )


def _module_available_without_import(module_name: str) -> bool:
    """Return whether ``module_name`` exists on disk without importing a parent package."""

    parts = module_name.split(".")
    try:
        root_spec = importlib.util.find_spec(parts[0])
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    if root_spec is None:
        return False
    if len(parts) == 1:
        return True
    search_locations = root_spec.submodule_search_locations
    if search_locations is None:
        return False
    candidates = [Path(location).joinpath(*parts[1:]) for location in search_locations]
    suffixes = tuple(importlib.machinery.all_suffixes())
    return any(
        candidate.is_dir() or any(Path(f"{candidate}{suffix}").is_file() for suffix in suffixes)
        for candidate in candidates
    )


def _read_distribution_identity(names: tuple[str, ...]) -> DistributionIdentity | None:
    for name in names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        source_url = None
        source_commit = None
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError:
                direct_url = {}
            if isinstance(direct_url, dict):
                raw_url = direct_url.get("url")
                if isinstance(raw_url, str):
                    source_url = _sanitize_source_url(raw_url)
                vcs_info = direct_url.get("vcs_info")
                if isinstance(vcs_info, dict):
                    raw_commit = vcs_info.get("commit_id")
                    if isinstance(raw_commit, str) and re.fullmatch(r"[0-9a-fA-F]{7,128}", raw_commit):
                        source_commit = raw_commit
        return DistributionIdentity(
            name=distribution.metadata.get("Name", name),
            version=distribution.version,
            source_url=source_url,
            source_commit=source_commit,
        )
    return None


def _sanitize_source_url(raw_url: str) -> str | None:
    """Remove credentials and request data before an install URL enters runtime identity records."""

    if not raw_url or len(raw_url) > 8192 or any(character in raw_url for character in "\r\n\x00"):
        return None
    try:
        parsed = urlsplit(raw_url)
        if parsed.scheme.lower() == "file":
            return "file:///<local-source>"
        hostname = parsed.hostname
        if not parsed.scheme or hostname is None:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        sanitized = urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))
    except ValueError:
        return None
    return sanitized[:2048]
