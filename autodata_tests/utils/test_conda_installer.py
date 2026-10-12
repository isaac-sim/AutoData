# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installer control flow without changing a real conda environment."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths


@pytest.fixture
def installer(tmp_path):
    """Provide fake conda/uv executables and a checkout whose path contains spaces."""
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copyfile(Path(TestPaths.repo_root) / "conda_installer.sh", root / "conda_installer.sh")
    arena = root / "submodules/IsaacLab-Arena"
    lab = arena / "submodules/IsaacLab"
    lab.mkdir(parents=True)
    for path in (arena / "pyproject.toml", arena / "uv.lock", lab / "pyproject.toml"):
        path.write_text("")

    commands = tmp_path / "bin"
    commands.mkdir()
    conda_base = tmp_path / "conda"
    (conda_base / "bin").mkdir(parents=True)
    (conda_base / "bin/python").symlink_to(sys.executable)
    target = conda_base / "envs/chosen"
    (target / "bin").mkdir(parents=True)
    (target / "bin/python").symlink_to(sys.executable)
    hooks = conda_base / "etc/profile.d"
    hooks.mkdir(parents=True)
    (hooks / "conda.sh").write_text(
        "conda() {\n"
        '  command conda "$@" || return $?\n'
        '  if [[ "$1" == activate ]]; then export CONDA_PREFIX="$TEST_TARGET"; fi\n'
        '  export PATH="${TEST_ACTIVATE_PATH:-$PATH}"\n'
        "}\n"
    )

    def executable(name, source):
        path = commands / name
        path.write_text(source)
        path.chmod(0o755)

    executable(
        "conda",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(['conda', *sys.argv[1:]]) + '\\n')\n"
        "if sys.argv[1:] == ['info', '--base']:\n"
        "    print(os.environ['TEST_CONDA_BASE'])\n"
        "elif sys.argv[1:] == ['info', '--json']:\n"
        "    envs = [os.environ['TEST_TARGET']] if os.environ.get('TEST_EXISTING') else []\n"
        "    dirs = [os.path.dirname(os.environ['TEST_TARGET'])] if os.environ.get('TEST_OCCUPIED') else []\n"
        "    print(json.dumps({'envs': envs, 'envs_dirs': dirs}))\n",
    )
    executable(
        "uv",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('uv ' + os.environ.get('TEST_UV_VERSION', '0.12.21'))\n"
        "else:\n"
        "    with open(os.environ['TEST_LOG'], 'a') as log:\n"
        "        log.write(json.dumps(['uv', *sys.argv[1:]]) + '\\n')\n"
        "    if os.environ.get('TEST_INSTALL_FAIL') and sys.argv[1:3] == ['pip', 'install']:\n"
        "        sys.exit(17)\n",
    )
    executable("uname", '#!/bin/bash\nif [[ "$1" == -s ]]; then echo Linux; else echo x86_64; fi\n')
    for name in ("git", "cmake", "c++"):
        executable(name, "#!/bin/bash\nexit 0\n")

    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX"):
        env.pop(name, None)
    env.update(
        PATH=f"{commands}:{env['PATH']}",
        TEST_LOG=str(tmp_path / "commands.jsonl"),
        TEST_CONDA_BASE=str(conda_base),
        TEST_TARGET=str(target),
    )

    def run(*arguments, **variables):
        result = subprocess.run(
            ["bash", str(root / "conda_installer.sh"), *arguments],
            env=env | variables,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        log = Path(env["TEST_LOG"])
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls, target

    return run


@pytest.mark.parametrize("arguments", [("-c", "-i", "-n", "chosen"), ("-i", "-n", "chosen", "-c")])
def test_combined_install_targets_named_environment_in_either_order(installer, arguments):
    result, calls, target = installer(*arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    create = next(i for i, call in enumerate(calls) if call[:2] == ["conda", "create"])
    activate = calls.index(["conda", "activate", "chosen"])
    assert create < activate
    installs = [call for call in calls if call[:3] == ["uv", "pip", "install"]]
    assert len(installs) == 3
    assert all(call[call.index("--python") + 1] == str(target / "bin/python") for call in installs)
    assert "--check" in installs[-1]
    assert "--editable" in installs[1]
    assert installs[1][-1].endswith("checkout with spaces[dev]")
    constraints = installs[1][installs[1].index("--constraint") + 1]
    assert constraints.endswith(".cache/conda-installer/runtime-constraints.txt")
    assert any("--no-emit-local" in call and constraints in call for call in calls)
    export = next(call for call in calls if call[:2] == ["uv", "export"])
    assert "--frozen" in export
    assert "pylock.toml" in export


def test_install_existing_environment_does_not_create_one(installer):
    result, calls, _ = installer("-i", "-n", "chosen", TEST_EXISTING="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ["conda", "activate", "chosen"] in calls
    assert not any(call[:2] == ["conda", "create"] for call in calls)


def test_uv_remains_available_after_conda_changes_path(installer):
    result, calls, _ = installer("-i", TEST_ACTIVATE_PATH="/usr/bin:/bin")
    assert result.returncode == 0, result.stdout + result.stderr
    assert any("--check" in call for call in calls)


@pytest.mark.parametrize("variables", [{"TEST_EXISTING": "1"}, {"TEST_OCCUPIED": "1"}])
def test_create_refuses_to_replace_existing_environment_or_directory(installer, variables):
    result, calls, _ = installer("-c", "-i", "-n", "chosen", **variables)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert not any(call[:2] in (["conda", "create"], ["conda", "activate"], ["uv", "pip"]) for call in calls)


@pytest.mark.parametrize("arguments", [("-c", "-h"), ("-i", "-c", "-h")])
def test_help_has_no_side_effects(installer, arguments):
    result, calls, _ = installer(*arguments)
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert not calls


@pytest.mark.parametrize("arguments", [("-c", "--unknown"), ("-c", "unexpected"), ("-n",), ("-c", "-n", "base")])
def test_invalid_arguments_fail_before_mutation(installer, arguments):
    result, calls, _ = installer(*arguments)
    assert result.returncode != 0
    assert not calls


def test_old_uv_fails_before_creating_environment(installer):
    result, calls, _ = installer("-c", "-i", TEST_UV_VERSION="0.6.0")
    assert result.returncode != 0
    assert "uv >= 0.12.21" in result.stderr
    assert not any(call[:2] == ["conda", "create"] for call in calls)


def test_failed_stack_install_stops_before_autodata(installer):
    result, calls, _ = installer("-i", TEST_INSTALL_FAIL="1")
    assert result.returncode == 17
    assert not any("--editable" in call for call in calls)
    assert "Installation complete" not in result.stdout
