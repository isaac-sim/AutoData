# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI must mount a writable home cache for Arena's composed robot assets."""

import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths


@pytest.fixture
def ci_launcher(tmp_path):
    """Exercise the real shell entry points while replacing only Docker and X11."""
    root = tmp_path / "checkout with spaces"
    for relative in ("scripts/ci/run_tests.sh", "docker/run_docker.sh"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(TestPaths.repo_root) / relative, target)
        target.chmod(0o755)

    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['images', '-q']:\n"
        "    print('test-image')\n"
        "elif args[:2] == ['container', 'inspect']:\n"
        "    print('false')\n"
        "elif args[:1] == ['run']:\n"
        "    with open(os.environ['TEST_DOCKER_LOG'], 'w') as output:\n"
        "        json.dump(args, output)\n"
        "elif args[:2] != ['ps', '-a']:\n"
        "    raise SystemExit(f'Unexpected Docker command: {args}')\n"
    )
    docker.chmod(0o755)
    xhost = commands / "xhost"
    xhost.write_text("#!/bin/sh\nexit 0\n")
    xhost.chmod(0o755)
    runner_temp = tmp_path / "runner temp"
    runner_temp.mkdir()
    log = tmp_path / "docker.json"
    environment = os.environ.copy()
    environment.pop("AUTODATA_DOCKER_CACHE_DIR", None)
    environment.update(
        PATH=f"{commands}:{environment['PATH']}",
        RUNNER_TEMP=str(runner_temp),
        TEST_DOCKER_LOG=str(log),
        FORCE_REBUILD="false",
        AUTODATA_RESULTS_DIR="",
    )

    def run(**variables):
        result = subprocess.run(
            ["bash", str(root / "scripts/ci/run_tests.sh")],
            env=environment | variables,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        args = json.loads(log.read_text())
        target = f"/home/{pwd.getpwuid(os.getuid()).pw_name}/.cache"
        sources = [args[i + 1].split(":")[0] for i, arg in enumerate(args[:-1]) if arg == "-v"]
        mounts = [source for source in sources if f"{source}:{target}" in args]
        assert len(mounts) == 1
        source = Path(mounts[0])
        assert source.is_dir()
        assert source.stat().st_uid == os.getuid()
        (source / "arena-cache-write-check").write_text("writable")
        assert "/isaac-sim/python.sh" in args and "pytest" in args
        return source

    return run, runner_temp


def test_ci_cache_is_fresh_and_owned_by_runner(ci_launcher):
    run, runner_temp = ci_launcher
    first, second = run(), run()
    assert first.parent == second.parent == runner_temp
    assert first != second


def test_ci_cache_respects_explicit_mount(ci_launcher, tmp_path):
    run, _ = ci_launcher
    selected = tmp_path / "selected cache"
    assert run(AUTODATA_DOCKER_CACHE_DIR=str(selected)) == selected
