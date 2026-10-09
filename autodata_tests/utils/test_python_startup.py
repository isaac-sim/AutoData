# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python startup must tolerate a stale editable AutoData installation."""

import os
import subprocess
import sys
from pathlib import Path

from autodata_tests.utils.constants import TestPaths


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
