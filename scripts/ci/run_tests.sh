#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the Isaac AutoData test suite, locally or in CI, inside the repo's GPU
# Docker image with one command:
#
#   ./scripts/ci/run_tests.sh
#
# Thin wrapper: prepares the container, then runs
# scripts/ci/run_tests_in_container.sh for the test logic.
#
# Environment overrides:
#   TEST_PATH                          space-separated test paths (default: all)
#   PYTEST_MARK                        pytest -m marker (single token; optional)
#   FORCE_REBUILD=true                 force an image rebuild (nightly)
#   ISAAC_AUTODATA_ISOLATED            1 (default here) drops host cache/X11 mounts
#   ISAAC_AUTODATA_SUBPROCESS_TIMEOUT  data-generation child timeout, seconds
#
# Requires a GPU host with the NVIDIA container runtime and nvidia-smi.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

# cuRobo (-c) is required for the SkillGen tests; one image covers the suite.
RUN_DOCKER_ARGS=(-c)
if [ "${FORCE_REBUILD:-false}" = "true" ]; then
    RUN_DOCKER_ARGS+=(-r)
fi

# Run the container isolated (no host $HOME/.cache or X11 bind-mounts) so CI does
# not inherit the runner's home-directory ownership. Overridable for local dev.
export ISAAC_AUTODATA_ISOLATED="${ISAAC_AUTODATA_ISOLATED:-1}"

# run_docker.sh gives the container a fresh environment and forwards its trailing
# arguments verbatim as the in-container command. Pass test paths positionally
# (space-safe) and the marker/timeout via `env`; the in-container script reads them.
read -r -a TEST_PATHS <<< "${TEST_PATH:-isaac_autodata_tests/}"

exec ./docker/run_docker.sh "${RUN_DOCKER_ARGS[@]}" \
    env \
    "PYTEST_MARK=${PYTEST_MARK-}" \
    "ISAAC_AUTODATA_SUBPROCESS_TIMEOUT=${ISAAC_AUTODATA_SUBPROCESS_TIMEOUT:-1200}" \
    ./scripts/ci/run_tests_in_container.sh "${TEST_PATHS[@]}"
