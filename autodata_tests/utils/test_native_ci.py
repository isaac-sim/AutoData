# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CI cache retention and required-check behavior."""

import os
import signal
import subprocess
import sys
import time
import yaml
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _step(workflow, job, name):
    steps = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"][job]["steps"]
    return next(step["run"] for step in steps if step.get("name") == name)


def _wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"Timed out waiting for {path}"
        time.sleep(0.01)


@pytest.fixture
def native_setup(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    uv = commands / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "cache = Path(os.environ['UV_CACHE_DIR'])\n"
        "cache.mkdir(parents=True, exist_ok=True)\n"
        "(cache / 'payload').touch()\n"
        "Path(os.environ['TEST_STARTED']).touch()\n"
        "while not Path(os.environ['TEST_RELEASE']).exists():\n"
        "    time.sleep(0.01)\n"
        "assert (cache / 'payload').exists(), 'Active cache was evicted'\n"
        "raise SystemExit(int(os.environ.get('TEST_EXIT', '0')))\n"
    )
    uv.chmod(0o755)
    root = tmp_path / "managed cache"
    processes = []

    def start(letter, *, hold=False, exit_code=0):
        key = f"uv0.12.21-python3.12-cuda12.8.1-8.9-{letter * 64}"
        started = tmp_path / f"started-{len(processes)}"
        release = tmp_path / f"release-{len(processes)}"
        if not hold:
            release.touch()
        environment = os.environ.copy()
        environment.pop("UV_CACHE_DIR", None)
        process = subprocess.Popen(
            ["bash", "-euo", "pipefail", "-c", _step("native.yml", "test", "Install the locked environment")],
            cwd=ROOT,
            env=environment
            | {
                "PATH": f"{commands}:{os.environ['PATH']}",
                "NATIVE_CACHE_ROOT": str(root),
                "NATIVE_CACHE_KEY": key,
                "TEST_STARTED": str(started),
                "TEST_RELEASE": str(release),
                "TEST_EXIT": str(exit_code),
            },
            start_new_session=True,
        )
        processes.append(process)
        if not hold:
            assert process.wait(timeout=10) == exit_code
        return process, root / "keys" / key, started, release

    yield root, start
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def test_cache_retains_most_recent_successful_use(native_setup):
    root, start = native_setup
    _, a, _, _ = start("a")
    start("b")
    start("a")
    _, c, _, _ = start("c")
    assert set((root / "keys").iterdir()) == {a, c}
    assert (a / "payload").exists() and (c / "payload").exists()


@pytest.mark.parametrize(("interruption", "key"), [("failure", "c"), ("cancel", "a")])
def test_interrupted_sync_keeps_fallback_and_partial_cache(native_setup, interruption, key):
    root, start = native_setup
    start("a")
    _, b, _, _ = start("b")
    process, partial, started, _ = start(key, hold=interruption == "cancel", exit_code=7)
    if interruption == "cancel":
        _wait_for(started)
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) != 0
    assert set((root / "keys").iterdir()) == {b, partial}
    assert (b / ".last-success").is_file()
    assert not (partial / ".last-success").exists()
    _, d, _, _ = start("d")
    assert set((root / "keys").iterdir()) == {b, d}


def test_concurrent_sync_protects_current_and_only_prunes_managed_keys(native_setup, tmp_path):
    root, start = native_setup
    _, a, _, _ = start("a")
    _, b, _, _ = start("b")
    incomplete = root / "keys" / a.name.replace("a" * 64, "d" * 64)
    incomplete.mkdir()
    legacy = tmp_path / "autodata-uv-cache"
    legacy.mkdir()
    (legacy / "payload").touch()
    foreign = root / "keys" / "unmanaged"
    foreign.mkdir()
    link = root / "keys" / a.name.replace("a" * 64, "e" * 64)
    link.symlink_to(legacy, target_is_directory=True)
    first, _, started, release = start("a", hold=True)
    _wait_for(started)
    assert set((root / "keys").iterdir()) == {a, b, foreign, link}
    second, c, second_started, second_release = start("c", hold=True)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            second.wait(timeout=0.3)
        assert not second_started.exists()
        assert (a / "payload").exists()
    finally:
        release.touch()
    assert first.wait(timeout=10) == 0
    _wait_for(second_started)
    assert set((root / "keys").iterdir()) == {a, c, foreign, link}
    second_release.touch()
    assert second.wait(timeout=10) == 0
    assert (legacy / "payload").exists()


@pytest.mark.parametrize("trusted", [True, False])
@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_native_gate_reports_forks_only_for_required_fork_skips(trusted, required, result):
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step("ci.yml", "native_check", "Check native installation result")],
        env=os.environ
        | {
            "CHANGES_RESULT": "success",
            "NATIVE_REQUIRED": str(required).lower(),
            "NATIVE_RESULT": result,
            "TRUSTED": str(trusted).lower(),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (completed.returncode == 0) == (result == ("success" if required else "skipped"))
    assert ("maintainer-controlled" in completed.stdout) == (required and not trusted and result == "skipped")


@pytest.mark.parametrize("changes_result", ["failure", "cancelled", "skipped"])
def test_native_gate_rejects_unavailable_change_detection(changes_result):
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step("ci.yml", "native_check", "Check native installation result")],
        env=os.environ
        | {"CHANGES_RESULT": changes_result, "NATIVE_REQUIRED": "false", "NATIVE_RESULT": "skipped", "TRUSTED": "true"},
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode != 0
