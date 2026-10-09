#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ENV_NAME="autodata"
PYTHON_VERSION="3.12"
UV_MIN_VERSION="0.12.21"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n ENV_NAME] [-c] [-i] [-h]

  -n    Conda environment name (default: ${ENV_NAME})
  -c    Create conda env "${ENV_NAME}" with Python ${PYTHON_VERSION}
  -i    Install Arena's locked Sim/Lab/PyTorch stack and AutoData with test tools
        into the selected env (env must already exist, or combine with -c)
  -h    Show this help

Requires Linux x86_64, conda, uv >= ${UV_MIN_VERSION}, Git, CMake, and a C++ compiler.
Flags can be combined in any order, e.g. "$(basename "$0") -c -i -n autodata".
The installer never replaces an existing environment when -c is specified.
cuRobo is optional and installed separately; see the installation guide.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command '$1' not found on PATH" >&2
        exit 1
    fi
}

create_env() {
    local conda_base
    conda_base="$(conda info --base)"
    # Refuse to let conda create --yes overwrite an existing environment.
    conda info --json | "${conda_base}/bin/python" -I -c '
import json, pathlib, sys
name = sys.argv[1]
info = json.load(sys.stdin)
registered = any(pathlib.Path(path).name == name for path in info["envs"])
targets = [pathlib.Path(directory) / name for directory in info["envs_dirs"]]
if registered or any(path.exists() or path.is_symlink() for path in targets):
    sys.exit(f"error: environment or directory {name!r} already exists; use -i to update an env, or choose a new name.")
' "${ENV_NAME}"
    echo ">>> Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
    echo ">>> Env created. Activate with: conda activate ${ENV_NAME}"
}

activate_env() {
    require_cmd conda
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"

    if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "error: Python was not found in the '${ENV_NAME}' conda env." >&2
        exit 1
    fi

    # Do not import another environment's packages during installation.
    unset PYTHONPATH PYTHONHOME
    "${CONDA_PREFIX}/bin/python" -I -c '
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit("error: the selected conda environment must use Python 3.12.")
'
}

check_install_prerequisites() {
    local arena_root="${REPO_ROOT}/submodules/IsaacLab-Arena"
    require_cmd uv
    # Activation may remove the environment that provided uv from PATH.
    UV_EXE="$(command -v uv)"
    require_cmd git
    require_cmd cmake
    require_cmd c++
    local file
    for file in "${arena_root}/pyproject.toml" "${arena_root}/uv.lock" \
        "${arena_root}/submodules/IsaacLab/pyproject.toml"; do
        if [[ ! -f "${file}" ]]; then
            echo "error: ${file} not found." >&2
            echo "       Run 'git submodule update --init --recursive' first." >&2
            exit 1
        fi
    done
    local conda_base
    conda_base="$(conda info --base)"
    "${conda_base}/bin/python" -I -c '
import re, sys
match = re.match(r"uv (\d+)\.(\d+)\.(\d+)", sys.argv[1])
if not match or tuple(map(int, match.groups())) < tuple(map(int, sys.argv[2].split("."))):
    sys.exit(f"error: uv >= {sys.argv[2]} is required; found {sys.argv[1]!r}.")
' "$("${UV_EXE}" --version)" "${UV_MIN_VERSION}"
}

install_all() {
    activate_env
    local conda_python="${CONDA_PREFIX}/bin/python"
    local arena_root="${REPO_ROOT}/submodules/IsaacLab-Arena"
    local export_dir="${REPO_ROOT}/.cache/conda-installer"
    local lock_export="${export_dir}/pylock.toml"
    local constraints="${export_dir}/runtime-constraints.txt"
    mkdir -p "${export_dir}"

    # PEP 751 preserves Arena's wheel URLs, hashes, overrides, and editable Lab
    # sources. Export without re-resolving or writing inside either submodule.
    echo ">>> Exporting Arena's locked Sim/Lab/PyTorch environment"
    "${UV_EXE}" export --project "${arena_root}" --frozen --no-dev --format pylock.toml \
        --output-file "${lock_export}" --quiet
    # Keep later AutoData/optional installs from changing any locked version.
    # Editable requirements are not valid constraints; their dependencies remain.
    "${UV_EXE}" export --project "${arena_root}" --frozen --no-dev --format requirements.txt \
        --no-emit-local --no-hashes --output-file "${constraints}" --quiet

    # pip install retains unrelated packages (including conda's pip); pip sync
    # would remove them. Explicit --python prevents targeting a stray .venv.
    echo ">>> Installing the locked stack into '${ENV_NAME}'"
    "${UV_EXE}" pip install --no-config --python "${conda_python}" --requirements "${lock_export}"

    echo ">>> Installing AutoData and test tools"
    "${UV_EXE}" pip install --no-config --python "${conda_python}" \
        --constraint "${constraints}" --editable "${REPO_ROOT}[dev]"

    echo ">>> Checking that the installed stack still matches Arena's lock"
    "${UV_EXE}" pip install --no-config --python "${conda_python}" --check --requirements "${lock_export}"
    echo ">>> Installation complete. Activate with: conda activate ${ENV_NAME}"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

create=false
install=false
while getopts ":cin:h" opt; do
    case "${opt}" in
        c) create=true ;;
        i) install=true ;;
        n) ENV_NAME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "error: unknown option -${OPTARG}" >&2; usage; exit 1 ;;
        :) echo "error: option -${OPTARG} requires an argument" >&2; usage; exit 1 ;;
    esac
done
shift "$((OPTIND - 1))"
if [[ $# -ne 0 || ( "${create}" == false && "${install}" == false ) ]]; then
    usage
    exit 1
fi
if [[ ! "${ENV_NAME}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ || "${ENV_NAME}" == base ]]; then
    echo "error: specify an environment name (not a path or 'base')." >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "error: Arena's locked environment supports Linux x86_64 only." >&2
    exit 1
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "error: deactivate the active Python virtual environment before running this installer." >&2
    exit 1
fi
# Conda itself may use a different Python version from the active shell.
unset PYTHONPATH PYTHONHOME
require_cmd conda
if "${install}"; then
    check_install_prerequisites
fi
if "${create}"; then
    create_env
fi
if "${install}"; then
    install_all
fi
