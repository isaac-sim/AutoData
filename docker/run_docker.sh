#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

DOCKER_IMAGE_NAME='isaac_autodata'
DOCKER_VERSION_TAG='sim-6.0.1'
# Override with BASE_IMAGE when testing against a different compatible Isaac Sim release.
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"

INSTALL_CUROBO=false
CUROBO_VERSION_TAG='sim-6.0.1-curobo'

# Resolve TORCH_CUDA_ARCH_LIST for the cuRobo build. Honour an explicit override if set, else
# auto-detect the host GPU's compute capability via nvidia-smi (e.g. "12.0" -> "12.0+PTX").
detect_cuda_arch() {
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        echo "${TORCH_CUDA_ARCH_LIST}"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi not found on host; cannot auto-detect GPU arch." >&2
        echo "       Set TORCH_CUDA_ARCH_LIST (e.g. export TORCH_CUDA_ARCH_LIST=12.0+PTX) and retry." >&2
        return 1
    fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
    if [ -z "${cc}" ]; then
        echo "error: could not read compute capability from nvidia-smi." >&2
        echo "       Set TORCH_CUDA_ARCH_LIST (e.g. export TORCH_CUDA_ARCH_LIST=12.0+PTX) and retry." >&2
        return 1
    fi
    echo "${cc}+PTX"
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)

# Path the repo is mounted to inside the container (kept in sync with the Dockerfile's WORKDIR).
WORKDIR="/workspaces/isaac_autodata"
ISAACLAB_PATH="${WORKDIR}/submodules/IsaacLab-Arena/submodules/IsaacLab"

# Optional host datasets directory mounted at /datasets (in addition to the repo's own datasets/).
DATASETS_HOST_MOUNT_DIRECTORY="$HOME/datasets"

FORCE_REBUILD=false
NO_CACHE=""

while getopts ":d:crRvh" OPTION; do
    case $OPTION in
        d) DATASETS_HOST_MOUNT_DIRECTORY=$OPTARG ;;
        c) INSTALL_CUROBO=true ;;
        r) FORCE_REBUILD=true ;;
        R) FORCE_REBUILD=true; NO_CACHE="--no-cache" ;;
        v) set -x ;;
        h)
            script_name=$(basename "$0")
            echo "Build and run the Isaac Auto Data dev container."
            echo ""
            echo "Usage: $script_name [options] [command...]"
            echo ""
            echo "Options:"
            echo "  -d <dir>  Host datasets directory to mount at /datasets (default \"$DATASETS_HOST_MOUNT_DIRECTORY\")."
            echo "  -c        Install cuRobo, auto-detects the GPU arch (override with the TORCH_CUDA_ARCH_LIST env var)."
            echo "  -r        Force rebuilding the image."
            echo "  -R        Force rebuilding the image without cache."
            echo "  -v        Verbose (set -x)."
            echo "  -h        Show this help."
            echo ""
            echo "Any trailing arguments are run as a command inside the container, then it exits."
            exit 0
            ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Set separate tag for cuRobo image.
if [ "$INSTALL_CUROBO" = "true" ]; then
    DOCKER_VERSION_TAG="${CUROBO_VERSION_TAG}"
fi

CONTAINER_NAME="${DOCKER_IMAGE_NAME}-${DOCKER_VERSION_TAG}"

echo "Using Docker image: ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} (base: ${BASE_IMAGE})"

# Build the image if it doesn't exist yet, or if a rebuild was requested.
if [ "$(docker images -q "${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}" 2>/dev/null)" ] && [ "$FORCE_REBUILD" = false ]; then
    echo "Image ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} already exists. Use -r to force a rebuild."
else
    BUILD_ARGS=("--build-arg" "WORKDIR=${WORKDIR}" "--build-arg" "BASE_IMAGE=${BASE_IMAGE}")
    if [ "$INSTALL_CUROBO" = "true" ]; then
        # Detect the host GPU arch so cuRobo's kernels are compiled for it.
        ARCH_LIST=$(detect_cuda_arch) || exit 1
        echo "cuRobo enabled — building for TORCH_CUDA_ARCH_LIST=${ARCH_LIST}"
        BUILD_ARGS+=("--build-arg" "INSTALL_CUROBO=true" "--build-arg" "TORCH_CUDA_ARCH_LIST=${ARCH_LIST}")
    fi
    docker build --pull \
        $NO_CACHE \
        --progress=plain \
        "${BUILD_ARGS[@]}" \
        -t "${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}" \
        --file "${SCRIPT_DIR}/Dockerfile.isaac_autodata" \
        "${REPO_ROOT}"
fi

# Remove a previously-exited container of the same name so we can recreate it.
if [ "$(docker ps -a --quiet --filter status=exited --filter "name=^${CONTAINER_NAME}$")" ]; then
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

# If it's already running, just attach a shell as the host user.
if [ "$(docker container inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" = "true" ]; then
    echo "Container already running. Attaching."
    docker exec -it "${CONTAINER_NAME}" su "$(id -un)"
    exit 0
fi

add_volume_if_it_exists() {
    [ -d "$1" ] && echo "-v $1:$2"
}

# Forward the host SSH agent if available (so git over SSH works in-container).
SSH_DOCKER_ARGS=()
if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "$SSH_AUTH_SOCK" ]; then
    SSH_DOCKER_ARGS+=("-v" "$SSH_AUTH_SOCK:/ssh-agent" "--env" "SSH_AUTH_SOCK=/ssh-agent")
fi

DOCKER_RUN_ARGS=(
    "--name" "${CONTAINER_NAME}"
    "--privileged"
    "--ulimit" "memlock=-1"
    "--ulimit" "stack=-1"
    "--ipc=host"
    "--net=host"
    "--runtime=nvidia"
    "--gpus=all"
    # Live-mount the repo: host edits are reflected in the container's editable installs.
    "-v" "${REPO_ROOT}:${WORKDIR}"
    $(add_volume_if_it_exists "$DATASETS_HOST_MOUNT_DIRECTORY" /datasets)
    # Share the host Kit/pip cache to speed up shader warmup and reinstalls across runs.
    "-v" "$HOME/.cache:/home/$(id -un)/.cache"
    # X11 passthrough so "--viz kit" can open a window.
    "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    "${SSH_DOCKER_ARGS[@]}"
    "--env" "DISPLAY=${DISPLAY:-}"
    "--env" "ACCEPT_EULA=Y"
    "--env" "PRIVACY_CONSENT=Y"
    "--env" "ISAACLAB_PATH=${ISAACLAB_PATH}"
    # Make AutoData's sitecustomize compatibility hook visible to direct upstream script entrypoints.
    "--env" "PYTHONPATH=${WORKDIR}"
    # Used by the entrypoint to recreate the host user inside the container.
    "--env" "DOCKER_RUN_USER_ID=$(id -u)"
    "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
    "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
    "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
)

# Allow local X11 clients from the container (for the Kit viewer).
if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || true
fi

# Only allocate a TTY when stdin is a terminal. CI has no TTY, and passing --tty
# there fails with "cannot attach stdin to a TTY-enabled container".
TTY_ARGS=()
if [ -t 0 ]; then
    TTY_ARGS+=("--tty")
fi

docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm "${TTY_ARGS[@]}" "${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}" "${@}"
