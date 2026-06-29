#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CUROBO_GIT_COMMIT="ebb71702f3f70e767f40fd8e050674af0288abe8"

echo ">>> Installing CUDA 12.8 toolkit (needed to compile cuRobo's CUDA kernels)"

# Map the base OS to NVIDIA's CUDA apt repo name.
. /etc/os-release
case "${ID}" in
  ubuntu)
    case "${VERSION_ID}" in
      "20.04") cuda_repo="ubuntu2004" ;;
      "22.04") cuda_repo="ubuntu2204" ;;
      "24.04") cuda_repo="ubuntu2404" ;;
      *) echo "error: unsupported Ubuntu ${VERSION_ID} for CUDA toolkit install" >&2; exit 1 ;;
    esac ;;
  *) echo "error: unsupported base OS '${ID}' for CUDA toolkit install" >&2; exit 1 ;;
esac

apt-get update
apt-get install -y --no-install-recommends wget gnupg ca-certificates

base_url="https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64"
wget -q "${base_url}/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
# Pin the CUDA repo so its packages take priority.
wget -q "${base_url}/cuda-${cuda_repo}.pin"
mv "cuda-${cuda_repo}.pin" /etc/apt/preferences.d/cuda-repository-pin-600

apt-get update
apt-get install -y --no-install-recommends \
    cuda-toolkit-12-8 \
    libcudnn9-cuda-12 \
    libcusparselt0 \
    libnccl2 \
    libnccl-dev \
    libnvjitlink-12-8
apt-get -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*

# Make the CUDA runtime libs discoverable system-wide (via the linker cache) instead of relying on
# LD_LIBRARY_PATH. Covers both the build below and cuRobo at runtime.
echo "${CUDA_HOME:-/usr/local/cuda-12.8}/lib64" > /etc/ld.so.conf.d/cuda-12-8.conf
ldconfig

echo ">>> Building cuRobo @ ${CUROBO_GIT_COMMIT} for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
# --no-build-isolation so the build uses the image's already-installed torch.
/isaac-sim/python.sh -m pip install --no-build-isolation \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@${CUROBO_GIT_COMMIT}"

echo ">>> cuRobo install complete"
