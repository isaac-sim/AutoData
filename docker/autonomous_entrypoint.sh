#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly AUTODATA_HOME="/autodata-home"
readonly AUTODATA_RUN_ROOT="/autonomous-run"
readonly AUTODATA_RUNTIME_ROOT="/tmp/autodata-runtime"
readonly AUTODATA_KIT_CACHE="/isaac-sim/kit/cache"
readonly AUTODATA_OV_CACHE="${AUTODATA_HOME}/.cache/ov"
readonly AUTODATA_WARP_CACHE="${AUTODATA_HOME}/.cache/warp"
readonly AUTODATA_GL_CACHE="${AUTODATA_HOME}/.cache/nvidia/GLCache"
readonly AUTODATA_COMPUTE_CACHE="${AUTODATA_HOME}/.nv/ComputeCache"

fail() {
    echo "error: $*" >&2
    exit 2
}

require_positive_id() {
    local label="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${label} must be a positive integer"
    ((value > 0 && value <= 4294967294)) || fail "${label} is outside the supported range"
}

[[ "${EUID}" -eq 0 ]] || fail "container entrypoint must start as root"
[[ "$#" -gt 0 ]] || fail "container command is required"

host_uid="${DOCKER_RUN_USER_ID:-}"
host_gid="${DOCKER_RUN_GROUP_ID:-}"
require_positive_id "DOCKER_RUN_USER_ID" "${host_uid}"
require_positive_id "DOCKER_RUN_GROUP_ID" "${host_gid}"

for required_command in chown cut getent groupadd install setpriv useradd; do
    command -v "${required_command}" >/dev/null || fail "required command is unavailable: ${required_command}"
done

autodata_group="autodata_g${host_gid}"
if ! getent group "${host_gid}" >/dev/null 2>&1; then
    getent group "${autodata_group}" >/dev/null 2>&1 && fail "image group-name collision: ${autodata_group}"
    groupadd --key GID_MAX=4294967294 --gid "${host_gid}" "${autodata_group}"
fi

autodata_user="autodata_u${host_uid}"
passwd_entry=$(getent passwd "${host_uid}" || true)
if [[ -z "${passwd_entry}" ]]; then
    getent passwd "${autodata_user}" >/dev/null 2>&1 && fail "image user-name collision: ${autodata_user}"
    useradd \
        --key UID_MAX=4294967294 \
        --no-create-home \
        --no-log-init \
        --uid "${host_uid}" \
        --gid "${host_gid}" \
        --home-dir "${AUTODATA_HOME}" \
        --shell /bin/bash \
        "${autodata_user}"
else
    autodata_user="${passwd_entry%%:*}"
fi

passwd_entry=$(getent passwd "${host_uid}" || true)
[[ -n "${passwd_entry}" ]] || fail "container passwd lookup failed for host UID"
isaac_group_entry=$(getent group isaac-sim || true)
[[ -n "${isaac_group_entry}" ]] || fail "image does not define the isaac-sim group"
isaac_group_gid=$(printf '%s\n' "${isaac_group_entry}" | cut -d: -f3)
require_positive_id "isaac-sim group ID" "${isaac_group_gid}"

install -d --mode 0700 --owner "${host_uid}" --group "${host_gid}" \
    "${AUTODATA_HOME}" \
    "${AUTODATA_HOME}/.cache" \
    "${AUTODATA_HOME}/.cache/nvidia" \
    "${AUTODATA_HOME}/.nv" \
    "${AUTODATA_RUNTIME_ROOT}"

cache_roots=(
    "${AUTODATA_KIT_CACHE}"
    "${AUTODATA_OV_CACHE}"
    "${AUTODATA_WARP_CACHE}"
    "${AUTODATA_GL_CACHE}"
    "${AUTODATA_COMPUTE_CACHE}"
)
for cache_root in "${cache_roots[@]}"; do
    [[ -d "${cache_root}" ]] || fail "cache mount is missing: ${cache_root}"
    chown "${host_uid}:${host_gid}" "${cache_root}"
done

[[ -d "${AUTODATA_RUN_ROOT}" ]] || fail "run-directory mount is missing: ${AUTODATA_RUN_ROOT}"
chown "${host_uid}:${host_gid}" "${AUTODATA_RUN_ROOT}"

exec setpriv \
    --reuid "${host_uid}" \
    --regid "${host_gid}" \
    --groups "${isaac_group_gid}" \
    --no-new-privs \
    env \
        -u DOCKER_RUN_USER_ID \
        -u DOCKER_RUN_GROUP_ID \
        HOME="${AUTODATA_HOME}" \
        USER="${autodata_user}" \
        LOGNAME="${autodata_user}" \
        XDG_RUNTIME_DIR="${AUTODATA_RUNTIME_ROOT}" \
        "$@"
