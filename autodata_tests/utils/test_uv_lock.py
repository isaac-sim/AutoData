# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoData's uv lock must keep the stack that Arena's lock validated."""

import tomllib
from collections import defaultdict
from pathlib import Path

from autodata_tests.utils.constants import TestPaths

ROOT = Path(TestPaths.repo_root)
ARENA = ROOT / "submodules/IsaacLab-Arena"
RESEED = "cp submodules/IsaacLab-Arena/uv.lock uv.lock && uv lock"


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _resolution(lock: Path) -> dict[str, list]:
    # A lock can hold several records of one package, one per resolution fork. Dependency markers are left out:
    # uv writes equivalent markers differently depending on the root project.
    records = defaultdict(list)
    for package in _load(lock)["package"]:
        dependencies = sorted(
            (dependency["name"], dependency.get("version", ""), dependency.get("extra", []))
            for dependency in package.get("dependencies", [])
        )
        records[package["name"]].append((package.get("version", ""), dependencies))
    return {name: sorted(entries) for name, entries in records.items()}


def test_shared_packages_keep_arena_resolution():
    autodata = _resolution(ROOT / "uv.lock")
    arena = _resolution(ARENA / "uv.lock")
    drift = {
        name: (arena[name], autodata[name]) for name in arena.keys() & autodata.keys() if arena[name] != autodata[name]
    }
    assert not drift, f"Run `{RESEED}` with the uv version pinned in CI. Arena vs AutoData: {drift}"


def test_resolver_settings_mirror_arena():
    autodata = _load(ROOT / "pyproject.toml")["tool"]["uv"]
    arena = _load(ARENA / "pyproject.toml")["tool"]["uv"]
    # uv deprecates Arena's prerelease mode; if-necessary resolves this stack identically.
    if arena["prerelease"] == "if-necessary-or-explicit":
        arena["prerelease"] = "if-necessary"
    for key in ("environments", "prerelease", "index-strategy", "override-dependencies", "constraint-dependencies"):
        assert autodata[key] == arena[key], f"[tool.uv] {key} differs from Arena's"
