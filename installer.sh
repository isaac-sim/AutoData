#!/usr/bin/env bash
# Copyright (c) 2025, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ENV_NAME="isaac_autodata"
PYTHON_VERSION="3.12"

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
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    echo ">>> Env created. Activate with: conda activate ${ENV_NAME}"
}

activate_env() {
    require_cmd conda
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
}

install_all() {
    activate_env
    require_cmd uv

    local isaaclab_sh="${REPO_ROOT}/submodules/IsaacLab-Arena/submodules/IsaacLab/isaaclab.sh"
    if [[ ! -f "${isaaclab_sh}" ]]; then
        echo "error: ${isaaclab_sh} not found." >&2
        echo "       Did you forget 'git submodule update --init --recursive'?" >&2
        exit 1
    fi

    echo ">>> Upgrading pip"
    uv pip install --upgrade pip

    echo ">>> Installing Isaac Sim 6.0.0"
    uv pip install "isaacsim[all,extscache]==6.0.0" \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match \
        --prerelease=allow

    echo ">>> Installing PyTorch (CUDA 12.8)"
    uv pip install -U torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

    echo ">>> Installing Isaac Lab"
    "${isaaclab_sh}" -i

    echo ">>> Installing Isaac Lab - Arena"
    pip install -e "${REPO_ROOT}/submodules/IsaacLab-Arena"

    echo ">>> Installing Isaac Auto Data"
    pip install -e "${REPO_ROOT}"

    echo ">>> Pinning daqp to 0.8.5 for Pink IK"
    uv pip install "daqp==0.8.5"

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
