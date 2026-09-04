#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the AutoData test suite inside the repo's GPU Docker image.
# This is the single entry point shared by local runs and CI: it wraps
# ./docker/run_docker.sh so a developer reproduces a CI failure with one command:
#
#   ./scripts/ci/run_tests.sh
#
# Requires a GPU host with the NVIDIA container runtime and nvidia-smi (used by
# run_docker.sh -c to detect the GPU arch for the cuRobo build).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
# Test path(s) collected by pytest, relative to the repo root. May be a single
# path or several space-separated paths. Premerge scopes this to the correctness
# suites; the nightly leaves it at the default (the whole test tree).
TEST_PATH="${TEST_PATH:-autodata_tests/}"
# Optional pytest marker expression. Empty (the default) applies no marker
# filter and runs everything collected under TEST_PATH. Note: use `-` (not `:-`)
# so an explicitly empty value from a caller is honored rather than defaulted.
PYTEST_MARK="${PYTEST_MARK-}"
# Set to "true" to force an image rebuild (used by the nightly workflow).
FORCE_REBUILD="${FORCE_REBUILD:-false}"
# Per-subprocess wall-clock timeout (seconds) for the data-generation child.
# run_docker.sh gives the container a fresh environment, so this is forwarded
# explicitly on the in-container command line rather than exported on the host.
SUBPROCESS_TIMEOUT="${AUTODATA_SUBPROCESS_TIMEOUT:-1200}"
# Container-local cache directory. run_docker.sh bind-mounts the host's
# $HOME/.cache into the container; on the CI runner that path is root-owned (or
# auto-created as root), so the recreated non-root container user cannot write
# it -- warp fails to create ~/.cache/warp with a PermissionError. Pointing the
# cache env vars at a writable, container-local /tmp path sidesteps the mounted
# host cache entirely. The tools create these dirs themselves (makedirs).
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/tmp/autodata_ci_cache}"
# Optional JUnit report dir (repo-relative). The repo is bind-mounted, so a report
# written here lands on the host for CI to collect.
RESULTS_DIR="${AUTODATA_RESULTS_DIR-}"
# ---------------------------------------------------------------------------

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

# cuRobo (-c) is required for the SkillGen tests; one image covers the whole suite.
RUN_DOCKER_ARGS=(-c)
if [ "${FORCE_REBUILD}" = "true" ]; then
    RUN_DOCKER_ARGS+=(-r)
fi

# Build the in-container command. `env VAR=...` forwards config into the
# container (run_docker.sh passes trailing args through verbatim, and the values
# are inherited by the data-generation subprocess pytest spawns). Call
# /isaac-sim/python.sh -m pytest explicitly so it does not depend on the
# in-container `pytest` alias expanding. XDG_CACHE_HOME/WARP_CACHE_PATH steer all
# caches away from the mounted host $HOME/.cache (see CONTAINER_CACHE_DIR above).
PYTEST_ARGS=(
    env
    "AUTODATA_SUBPROCESS_TIMEOUT=${SUBPROCESS_TIMEOUT}"
    "XDG_CACHE_HOME=${CONTAINER_CACHE_DIR}"
    "WARP_CACHE_PATH=${CONTAINER_CACHE_DIR}/warp"
    /isaac-sim/python.sh -m pytest -sv --durations=0
)
if [ -n "${RESULTS_DIR}" ]; then
    # Pre-create on the host so the container can write into the bind-mounted dir.
    mkdir -p "${RESULTS_DIR}"
    PYTEST_ARGS+=("--junitxml=${RESULTS_DIR}/junit.xml")
fi
if [ -n "${PYTEST_MARK}" ]; then
    PYTEST_ARGS+=(-m "${PYTEST_MARK}")
fi
# TEST_PATH may list multiple paths; split on whitespace into separate args.
read -r -a TEST_PATHS <<< "${TEST_PATH}"
PYTEST_ARGS+=("${TEST_PATHS[@]}")

echo ">>> Running E2E tests (mark='${PYTEST_MARK:-<all>}', paths='${TEST_PATH}', rebuild=${FORCE_REBUILD})"
exec ./docker/run_docker.sh "${RUN_DOCKER_ARGS[@]}" "${PYTEST_ARGS[@]}"
