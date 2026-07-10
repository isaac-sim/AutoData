#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the Isaac AutoData test suite. Environment-independent: it assumes no host
# paths and runs anywhere the dependencies exist (the container or a conda env).
#
# Test paths are positional arguments (default: the whole tree). Behavior is
# tuned via environment variables:
#   PYTEST_MARK                        pytest -m marker expression (optional)
#   ISAAC_AUTODATA_SUBPROCESS_TIMEOUT  data-generation child timeout, seconds
#   ISAAC_AUTODATA_CACHE_DIR           writable cache root (default /tmp/...)
#   ISAAC_AUTODATA_PYTHON              python launcher (default Isaac Sim's)

set -euo pipefail

# Default to the whole test tree when no paths are given.
if [ "$#" -eq 0 ]; then
    set -- isaac_autodata_tests/
fi

# Run-local caches, to avoid the ownership collision on a bind-mounted host
# $HOME/.cache (warp's kernel cache lives under ~/.cache/warp).
CACHE_DIR="${ISAAC_AUTODATA_CACHE_DIR:-/tmp/isaac_autodata_cache}"
export XDG_CACHE_HOME="${CACHE_DIR}"
export WARP_CACHE_PATH="${CACHE_DIR}/warp"

# Inherited by the data-generation subprocess pytest spawns.
export ISAAC_AUTODATA_SUBPROCESS_TIMEOUT="${ISAAC_AUTODATA_SUBPROCESS_TIMEOUT:-1200}"

PYTHON="${ISAAC_AUTODATA_PYTHON:-/isaac-sim/python.sh}"

PYTEST=("${PYTHON}" -m pytest -sv --durations=0)
if [ -n "${PYTEST_MARK:-}" ]; then
    PYTEST+=(-m "${PYTEST_MARK}")
fi
PYTEST+=("$@")

echo ">>> ${PYTEST[*]}  (XDG_CACHE_HOME=${XDG_CACHE_HOME})"
exec "${PYTEST[@]}"
