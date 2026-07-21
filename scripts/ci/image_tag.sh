#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Print the GHCR reference for the prebuilt Isaac AutoData test image.
#
# The tag is a content hash of the inputs that determine the image: the docker/
# tree, the pinned IsaacLab-Arena submodule (which transitively pins IsaacLab),
# packaging metadata, and the CUDA arch. The build workflow and the E2E pull step
# both call this, so an unchanged PR resolves to the tag main published (a cache
# hit). Ids come from the git tree, so no submodule checkout is needed.

set -euo pipefail

# GHCR repo for the prebuilt image (must be lowercase).
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/isaac-sim/isaac-autodata}"
# cuRobo is compiled for this arch and baked into the image (L40S = 8.9).
# Keep in sync with TORCH_CUDA_ARCH_LIST in .github/workflows/build-image.yml.
IMAGE_CUDA_ARCH="${IMAGE_CUDA_ARCH:-8.9+PTX}"

inputs=$(git rev-parse \
    "HEAD:docker" \
    "HEAD:.gitmodules" \
    "HEAD:submodules/IsaacLab-Arena" \
    "HEAD:setup.py" \
    "HEAD:pyproject.toml")

hash=$(printf '%s\n%s\n' "${inputs}" "${IMAGE_CUDA_ARCH}" | sha256sum | cut -c1-16)
echo "${IMAGE_REPO}:curobo-${hash}"
