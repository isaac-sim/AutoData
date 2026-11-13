# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python startup must tolerate a stale editable AutoData installation."""

import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import distributions
from pathlib import Path

import pytest

from autodata_tests.utils.constants import TestPaths


@pytest.mark.skipif(not Path("/isaac-sim").is_dir(), reason="requires the Isaac Sim base image")
def test_docker_uses_bundled_sim_with_isolated_dependencies(tmp_path):
    import torch

    import isaacsim

    assert Path(isaacsim.__file__).is_relative_to("/isaac-sim")
    assert Path(torch.__file__).is_relative_to(sys.prefix)
    packages = {distribution.metadata["Name"] for distribution in distributions(path=[sysconfig.get_path("purelib")])}
    assert not any(name == "isaacsim" or name.startswith("isaacsim-") for name in packages)
    installed = subprocess.run(
        ["uv", "pip", "show", "autodata"], capture_output=True, text=True, check=True, timeout=30
    )
    assert f"Location: {sysconfig.get_path('purelib')}" in installed.stdout
    for directory in ("cache", "data", "logs"):
        with tempfile.TemporaryFile(dir=Path("/isaac-sim/kit") / directory):
            pass

    other = tmp_path / "other"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(other)], check=True, timeout=120)
    subprocess.run(
        [str(other / "bin/python"), "-c", "from importlib.util import find_spec; assert find_spec('torch') is None"],
        check=True,
        timeout=30,
    )


def test_venv_creation_with_sitecustomize_but_without_autodata(tmp_path):
    """Reproduce a startup hook exposed without the renamed AutoData packages."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)

    def run_python(python, *arguments):
        result = subprocess.run(
            [str(python), *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Error in sitecustomize" not in result.stderr
        return result

    parent = tmp_path / "stale_install"
    run_python(sys.executable, "-m", "venv", "--without-pip", str(parent))
    python = parent / "bin/python"
    purelib = run_python(python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))").stdout.strip()
    source = Path(TestPaths.repo_root) / "sitecustomize.py"
    (Path(purelib) / "sitecustomize.py").write_text(source.read_text())

    run_python(
        python,
        "-c",
        "import sitecustomize; from importlib.util import find_spec; assert find_spec('autodata_utils') is None",
    )
    docs_env = tmp_path / "venv_docs"
    run_python(python, "-m", "venv", str(docs_env))
    run_python(docs_env / "bin/python", "-m", "pip", "--version")
