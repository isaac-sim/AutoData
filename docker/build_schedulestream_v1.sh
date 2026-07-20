#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXPECTED_SCHEDULESTREAM_COMMIT="f6351b8db8d7da9cb6ddd6854dbfc3123ab048f5"
SCHEDULESTREAM_VERSION="0.0.0.dev0+f6351b8"
BASE_IMAGE="${BASE_IMAGE:-isaac_autodata:curobo}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-isaac_autodata:schedulestream-v1}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
schedulestream_source="${SCHEDULESTREAM_SOURCE:-${repo_root}/../nvplan}"

if [ ! -d "${schedulestream_source}/.git" ]; then
    echo "error: ScheduleStream checkout not found at ${schedulestream_source}" >&2
    echo "set SCHEDULESTREAM_SOURCE to the reviewed local checkout" >&2
    exit 2
fi

actual_commit=$(git -C "${schedulestream_source}" rev-parse HEAD)
if [ "${actual_commit}" != "${EXPECTED_SCHEDULESTREAM_COMMIT}" ]; then
    echo "error: ScheduleStream checkout is ${actual_commit}" >&2
    echo "expected reviewed commit ${EXPECTED_SCHEDULESTREAM_COMMIT}" >&2
    exit 3
fi
if [ -n "$(git -C "${schedulestream_source}" status --porcelain)" ]; then
    echo "error: ScheduleStream checkout has local changes; refusing a non-reproducible image" >&2
    exit 4
fi

base_image_id=$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE}")
if [[ ! "${base_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "error: could not resolve ${BASE_IMAGE} to an immutable local image ID" >&2
    exit 5
fi
base_image_pin="isaac-autodata-base-pin:${base_image_id#sha256:}-$$"
build_context=""
cleanup() {
    if [[ -n "${build_context:-}" && -d "${build_context}" ]]; then
        rm -rf -- "${build_context}"
    fi
    if [[ -n "${base_image_pin:-}" ]]; then
        docker image rm "${base_image_pin}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
docker image tag "${base_image_id}" "${base_image_pin}"
build_context=$(mktemp -d "${TMPDIR:-/tmp}/isaac-autodata-schedulestream.XXXXXXXX")
mkdir -p "${build_context}/schedulestream"
# Materialize only paths tracked by the reviewed commit. The checkout was proven clean above, so
# reading those paths from the worktree preserves any reviewed Git-LFS smudge results while a
# NUL-delimited tree listing excludes ignored files and handles arbitrary tracked path names.
git -C "${schedulestream_source}" ls-tree \
    -r \
    --name-only \
    -z \
    "${actual_commit}" \
    -- \
    pyproject.toml \
    LICENSE \
    src \
    | tar \
        --create \
        --file=- \
        --directory="${schedulestream_source}" \
        --null \
        --files-from=- \
    | tar -xf - -C "${build_context}/schedulestream"

echo "Building ${OUTPUT_IMAGE} from ${BASE_IMAGE} (${base_image_id})"
echo "ScheduleStream commit: ${actual_commit}"
docker build \
    --build-arg "BASE_IMAGE=${base_image_pin}" \
    --build-arg "BASE_IMAGE_ID=${base_image_id}" \
    --build-arg "SCHEDULESTREAM_COMMIT=${actual_commit}" \
    --build-arg "SCHEDULESTREAM_VERSION=${SCHEDULESTREAM_VERSION}" \
    --build-context "schedulestream=${build_context}/schedulestream" \
    --file "${script_dir}/Dockerfile.schedulestream_v1" \
    --tag "${OUTPUT_IMAGE}" \
    "${repo_root}"
