#!/usr/bin/env bash
# Copyright (c) 2025, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ENV_NAME="isaac_autodata"
PYTHON_VERSION="3.12"
ISAAC_SIM_VERSION="6.0.1.0"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-c] [-i] [-h]

  -c    Create conda env "${ENV_NAME}" with Python ${PYTHON_VERSION}
  -i    Install Isaac Sim, PyTorch, Isaac Lab, Arena, and Isaac Auto Data
        into the "${ENV_NAME}" env (env must already exist; run with -c first)
  -h    Show this help

Flags can be combined, e.g. "$(basename "$0") -c -i" creates the env and installs everything.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command '$1' not found on PATH" >&2
        exit 1
    fi
}

create_env() {
    require_cmd conda
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

    # Ensure every uv invocation targets this conda environment explicitly.
    export UV_PYTHON="${CONDA_PREFIX}/bin/python"
}

install_all() {
    local isaaclab_sh="${REPO_ROOT}/submodules/IsaacLab-Arena/submodules/IsaacLab/isaaclab.sh"
    local arena_root="${REPO_ROOT}/submodules/IsaacLab-Arena"
    local torch_cuda_version
    local torch_index_url

    if [[ ! -x "${isaaclab_sh}" ]]; then
        echo "error: ${isaaclab_sh} not found." >&2
        echo "       Did you forget 'git submodule update --init --recursive'?" >&2
        exit 1
    fi
    if [[ ! -f "${arena_root}/setup.py" ]]; then
        echo "error: ${arena_root}/setup.py not found." >&2
        echo "       Did you forget 'git submodule update --init --recursive'?" >&2
        exit 1
    fi

    case "$(uname -m)" in
        x86_64|amd64)
            torch_cuda_version="12.8"
            torch_index_url="https://download.pytorch.org/whl/cu128"
            ;;
        aarch64|arm64)
            torch_cuda_version="13.0"
            torch_index_url="https://download.pytorch.org/whl/cu130"
            ;;
        *)
            echo "error: unsupported architecture '$(uname -m)' for the Isaac Lab PyTorch install." >&2
            exit 1
            ;;
    esac

    activate_env
    require_cmd uv

    echo ">>> Upgrading pip"
    uv pip install --upgrade pip

    echo ">>> Installing Isaac Sim ${ISAAC_SIM_VERSION}"
    uv pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match \
        --prerelease=allow

    echo ">>> Installing PyTorch (CUDA ${torch_cuda_version})"
    uv pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${torch_index_url}"

    echo ">>> Installing Isaac Lab"
    "${isaaclab_sh}" -i all

    echo ">>> Installing Isaac Lab - Arena"
    uv pip install --editable "${arena_root}"

    echo ">>> Installing Isaac Auto Data"
    uv pip install --editable "${REPO_ROOT}"

    echo ">>> Installation complete."
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

while getopts ":cih" opt; do
    case "${opt}" in
        c) create_env ;;
        i) install_all ;;
        h) usage; exit 0 ;;
        \?) echo "error: unknown option -${OPTARG}" >&2; usage; exit 1 ;;
    esac
done
