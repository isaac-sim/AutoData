#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DEFAULT_IMAGE="isaac_autodata:schedulestream-v1"
readonly ISAAC_CACHE_VERSION="isaac-sim-5-1"
readonly CONTAINER_REPO="/workspaces/isaac_autodata"
readonly CONTAINER_RUN_ROOT="/autonomous-run"
readonly CONTAINER_TASK="${CONTAINER_RUN_ROOT}/task.yaml"
readonly CONTAINER_HOME="/autodata-home"
readonly CONTAINER_XAUTHORITY="/autodata-xauthority"
readonly MAX_HOST_PATH_LENGTH=4096
readonly MAX_TASK_BYTES=$((4 * 1024 * 1024))

usage() {
    cat <<'EOF'
Run one autonomous AutoData task in a reviewed ScheduleStream runtime image.

Usage:
  docker/run_autonomous_task.sh [--gui | --headless] [--run-dir PATH] [--image IMAGE] TASK.yaml

Options:
  --gui           Launch the Kit window. Internally this selects Isaac Lab's Kit visualizer.
  --headless      Run without a window (default).
  --run-dir PATH  Writable host run directory. By default, create a unique directory
                  below datasets/autonomous_runs.
  --image IMAGE   Local reviewed image (default: isaac_autodata:schedulestream-v1).
  -h, --help      Show this help.

The caller must explicitly export ACCEPT_EULA=Y. The launcher never accepts the EULA on the
caller's behalf. The repository is mounted read-only; only the printed run directory and isolated,
version-namespaced Docker cache volumes are writable.
EOF
}

fail() {
    echo "error: $*" >&2
    exit 2
}

require_safe_mount_path() {
    local label="$1"
    local value="$2"
    [[ -n "${value}" ]] || fail "${label} is empty"
    ((${#value} <= MAX_HOST_PATH_LENGTH)) || fail "${label} exceeds ${MAX_HOST_PATH_LENGTH} characters"
    [[ "${value}" != *","* ]] || fail "${label} cannot contain a comma"
    [[ ! "${value}" =~ [[:cntrl:]] ]] || fail "${label} cannot contain control characters"
}

gui_auth_dir=""
gui_auth_file=""
cleanup() {
    if [[ -n "${gui_auth_file}" ]]; then
        rm -f -- "${gui_auth_file}"
    fi
    if [[ -n "${gui_auth_dir}" ]]; then
        rmdir -- "${gui_auth_dir}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
entrypoint_path="${repo_root}/docker/autonomous_entrypoint.sh"

gui=false
display_mode=""
image="${DEFAULT_IMAGE}"
requested_run_dir=""
task_argument=""

while (($# > 0)); do
    case "$1" in
        --gui)
            [[ -z "${display_mode}" || "${display_mode}" == "gui" ]] || fail "--gui conflicts with --headless"
            gui=true
            display_mode="gui"
            shift
            ;;
        --headless)
            [[ -z "${display_mode}" || "${display_mode}" == "headless" ]] || fail "--headless conflicts with --gui"
            gui=false
            display_mode="headless"
            shift
            ;;
        --run-dir)
            (($# >= 2)) || fail "--run-dir requires a path"
            requested_run_dir="$2"
            shift 2
            ;;
        --run-dir=*)
            requested_run_dir="${1#*=}"
            shift
            ;;
        --image)
            (($# >= 2)) || fail "--image requires an image reference"
            image="$2"
            shift 2
            ;;
        --image=*)
            image="${1#*=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 1)) || fail "exactly one task YAML is required after --"
            [[ -z "${task_argument}" ]] || fail "task YAML was provided more than once"
            task_argument="$1"
            shift
            ;;
        -*)
            fail "unknown option: $1"
            ;;
        *)
            [[ -z "${task_argument}" ]] || fail "exactly one task YAML is required"
            task_argument="$1"
            shift
            ;;
    esac
done

[[ -n "${task_argument}" ]] || fail "task YAML is required"
[[ "${ACCEPT_EULA:-}" == "Y" ]] || fail "set ACCEPT_EULA=Y after reviewing the NVIDIA EULA"

((${#image} <= 255)) || fail "image reference exceeds 255 characters"
[[ "${image}" =~ ^[A-Za-z0-9][A-Za-z0-9._/@:-]*$ ]] || fail "image reference contains unsupported characters"
require_safe_mount_path "repository path" "${repo_root}"
require_safe_mount_path "task path" "${task_argument}"
task_path=$(realpath --canonicalize-existing -- "${task_argument}") || fail "task YAML does not exist"
require_safe_mount_path "resolved task path" "${task_path}"
[[ -f "${task_path}" && -r "${task_path}" ]] || fail "task YAML must be a readable regular file"
case "${task_path,,}" in
    *.yaml | *.yml) ;;
    *) fail "task input must use a .yaml or .yml suffix" ;;
esac
task_bytes=$(stat --format='%s' -- "${task_path}")
((task_bytes <= MAX_TASK_BYTES)) || fail "task YAML exceeds ${MAX_TASK_BYTES} bytes"

[[ -f "${entrypoint_path}" && -x "${entrypoint_path}" ]] || fail "container entrypoint is not executable"
command -v docker >/dev/null || fail "docker is not available"
command -v install >/dev/null || fail "install is not available"

image_id=$(docker image inspect --format '{{.Id}}' "${image}" 2>/dev/null) || fail "Docker image is unavailable: ${image}"
[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "Docker returned an invalid image ID for ${image}"
image_contract=$(
    docker image inspect \
        --format '{{index .Config.Labels "org.opencontainers.image.schedulestream.application"}}|{{index .Config.Labels "org.opencontainers.image.curobo.api-generation"}}|{{index .Config.Labels "org.opencontainers.image.schedulestream.commit"}}|{{index .Config.Labels "org.opencontainers.image.base.digest"}}' \
        "${image_id}" 2>/dev/null
) || fail "could not inspect the autonomous runtime image"
IFS='|' read -r schedulestream_application curobo_api schedulestream_commit base_image_digest extra_contract <<<"${image_contract}"
[[ -z "${extra_contract}" ]] || fail "autonomous runtime image labels are malformed"
case "${curobo_api}:${schedulestream_application}" in
    v1:custream | v2:custream2) ;;
    *) fail "image is not a supported ScheduleStream/cuRobo runtime" ;;
esac
[[ "${schedulestream_commit}" =~ ^[0-9a-f]{40}$ ]] || fail "image has no valid ScheduleStream source identity"
[[ "${base_image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "image has no valid base-image identity"

host_uid=$(id -u)
host_gid=$(id -g)
[[ "${host_uid}" =~ ^[0-9]+$ && "${host_gid}" =~ ^[0-9]+$ ]] || fail "host UID/GID are not numeric"
((host_uid > 0 && host_gid > 0)) || fail "refusing to run the autonomous container as host root"

if ${gui}; then
    [[ "${DISPLAY:-}" =~ ^:([0-9]+)(\.[0-9]+)?$ ]] || fail "--gui requires a local DISPLAY such as :0 or :1"
    display_number=$((10#${BASH_REMATCH[1]}))
    x11_socket="/tmp/.X11-unix/X${display_number}"
    [[ -S "${x11_socket}" ]] || fail "X11 socket is unavailable: ${x11_socket}"
    command -v xauth >/dev/null || fail "--gui requires the host xauth command"

    host_xauthority="${XAUTHORITY:-${HOME:-}/.Xauthority}"
    host_xauthority=$(realpath --canonicalize-existing -- "${host_xauthority}") \
        || fail "Xauthority file does not exist"
    require_safe_mount_path "Xauthority path" "${host_xauthority}"
    [[ -f "${host_xauthority}" && -r "${host_xauthority}" ]] || fail "Xauthority must be a readable regular file"

    runtime_temp_root="${XDG_RUNTIME_DIR:-/tmp}"
    [[ -d "${runtime_temp_root}" && -w "${runtime_temp_root}" ]] || fail "no writable runtime directory for Xauthority"
    gui_auth_dir=$(mktemp -d -- "${runtime_temp_root}/isaac-autodata-xauth.XXXXXXXX")
    gui_auth_file="${gui_auth_dir}/Xauthority"
    install -m 0600 /dev/null "${gui_auth_file}"
    xauth_records=$(xauth -f "${host_xauthority}" nlist "${DISPLAY}") || fail "could not read X11 authorization"
    [[ -n "${xauth_records}" ]] || fail "Xauthority has no cookie for ${DISPLAY}"
    # FamilyWild lets the single copied cookie authenticate the container hostname. The host's
    # complete Xauthority database is never exposed to the container.
    printf '%s\n' "${xauth_records}" | sed 's/^..../ffff/' | xauth -f "${gui_auth_file}" nmerge -
    [[ -s "${gui_auth_file}" ]] || fail "could not create container X11 authorization"
fi

umask 077
if [[ -z "${requested_run_dir}" ]]; then
    default_run_root="${repo_root}/datasets/autonomous_runs"
    mkdir -p -- "${default_run_root}"
    run_dir=$(mktemp -d -- "${default_run_root}/run.XXXXXXXX")
else
    require_safe_mount_path "run directory" "${requested_run_dir}"
    [[ ! -L "${requested_run_dir}" ]] || fail "run directory cannot be a symbolic link"
    if [[ -e "${requested_run_dir}" && ! -d "${requested_run_dir}" ]]; then
        fail "run directory exists and is not a directory"
    fi
    mkdir -p -- "${requested_run_dir}"
    run_dir=$(realpath --canonicalize-existing -- "${requested_run_dir}")
fi
require_safe_mount_path "resolved run directory" "${run_dir}"
[[ -d "${run_dir}" && -w "${run_dir}" && -x "${run_dir}" ]] || fail "run directory is not writable"
[[ "$(stat --format='%u' -- "${run_dir}")" == "${host_uid}" ]] || fail "run directory must be owned by the invoking user"
[[ "${run_dir}" != "/" && "${run_dir}" != "${repo_root}" ]] || fail "run directory is too broad"
case "${repo_root}/" in
    "${run_dir}/"*) fail "run directory cannot contain the repository" ;;
esac

task_copy="${run_dir}/task.yaml"
if [[ "${task_path}" != "${task_copy}" ]]; then
    [[ ! -e "${task_copy}" && ! -L "${task_copy}" ]] || fail "run directory already contains task.yaml"
    install --mode 0400 -- "${task_path}" "${task_copy}"
    cmp --silent -- "${task_path}" "${task_copy}" || fail "copied task YAML did not match its source"
else
    [[ -f "${task_copy}" && -r "${task_copy}" ]] || fail "run-directory task.yaml must be a readable regular file"
    [[ "$(stat --format='%u' -- "${task_copy}")" == "${host_uid}" ]] \
        || fail "run-directory task.yaml must be owned by the invoking user"
fi

printf 'Run directory: %s\n' "${run_dir}"

image_digest="${image_id#sha256:}"
cache_namespace="isaac-autodata-${ISAAC_CACHE_VERSION}-${curobo_api}-${image_digest:0:12}-u${host_uid}"
docker_args=(
    run
    --rm
    --gpus all
    --shm-size 8g
    --security-opt no-new-privileges:true
    --user 0:0
    --entrypoint "${CONTAINER_REPO}/docker/autonomous_entrypoint.sh"
    --workdir "${CONTAINER_RUN_ROOT}"
    --env ACCEPT_EULA
    --env "DOCKER_RUN_USER_ID=${host_uid}"
    --env "DOCKER_RUN_GROUP_ID=${host_gid}"
    --env "PYTHONDONTWRITEBYTECODE=1"
    --env "PYTHONPATH=${CONTAINER_REPO}:${CONTAINER_REPO}/submodules/IsaacLab-Arena"
    --mount "type=bind,src=${repo_root},dst=${CONTAINER_REPO},readonly"
    --mount "type=bind,src=${run_dir},dst=${CONTAINER_RUN_ROOT}"
    --mount "type=bind,src=${task_copy},dst=${CONTAINER_TASK},readonly"
    --mount "type=volume,src=${cache_namespace}-kit,dst=/isaac-sim/kit/cache"
    --mount "type=volume,src=${cache_namespace}-ov,dst=${CONTAINER_HOME}/.cache/ov"
    --mount "type=volume,src=${cache_namespace}-warp,dst=${CONTAINER_HOME}/.cache/warp"
    --mount "type=volume,src=${cache_namespace}-gl,dst=${CONTAINER_HOME}/.cache/nvidia/GLCache"
    --mount "type=volume,src=${cache_namespace}-compute,dst=${CONTAINER_HOME}/.nv/ComputeCache"
)

runner_args=(
    /isaac-sim/python.sh
    "${CONTAINER_REPO}/isaac_autodata_examples/generate_task_dataset.py"
    "${CONTAINER_TASK}"
)
if ${gui}; then
    docker_args+=(
        --env "DISPLAY=${DISPLAY}"
        --env "XAUTHORITY=${CONTAINER_XAUTHORITY}"
        --env "QT_X11_NO_MITSHM=1"
        --mount "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly"
        --mount "type=bind,src=${gui_auth_file},dst=${CONTAINER_XAUTHORITY},readonly"
    )
    runner_args+=(--gui)
fi

docker "${docker_args[@]}" "${image_id}" "${runner_args[@]}"
