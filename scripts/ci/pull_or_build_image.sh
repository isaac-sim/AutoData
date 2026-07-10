#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Pull the prebuilt E2E image from GHCR (keyed by content hash) and tag it
# isaac_autodata:curobo so run_docker.sh reuses it. On any miss, do nothing and
# let run_docker.sh build locally, so correctness never depends on the cache.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOCAL_TAG="isaac_autodata:curobo"

REMOTE=$("${SCRIPT_DIR}/image_tag.sh")

echo ">>> Prebuilt image: ${REMOTE}"
if docker pull "${REMOTE}"; then
    docker tag "${REMOTE}" "${LOCAL_TAG}"
    echo ">>> Tagged as ${LOCAL_TAG}; run_docker.sh will reuse it (no build)."
else
    echo ">>> Prebuilt image unavailable (cache miss); run_docker.sh will build locally."
fi
