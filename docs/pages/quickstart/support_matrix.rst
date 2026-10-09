..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Support Matrix
==============

This matrix describes the supported AutoData v0.1.0 stack. Upstream projects may support
additional platforms, but configurations outside this matrix are not validated by AutoData. Use
the recursively pinned submodules from the AutoData repository instead of independently selecting
Isaac Lab or Isaac Lab-Arena revisions.

.. list-table:: AutoData platform and resource support
   :widths: 25 75
   :header-rows: 1

   * - Area
     - Supported or required configuration
   * - Host operating system
     - Linux x86_64; Ubuntu 22.04 or 24.04
   * - GPU
     - NVIDIA RTX GPU with RT cores and at least 16 GB VRAM.
       See `Isaac Sim requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - NVIDIA driver
     - NVIDIA lists Linux 595.58.03 as a tested driver. See the current
       `Isaac Sim requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - CPU and RAM
     - Minimum: 4 CPU cores and 32 GB RAM. Recommended: 8 or more cores and 64 GB RAM.
       See `Isaac Sim requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - Disk
     - Minimum: 100 GB SSD. Recommended: 500 GB SSD
   * - Docker installation
     - Docker Engine with the NVIDIA Container Toolkit; recommended AutoData installation
   * - Conda installation
     - Optional: Python 3.12, conda, uv >= 0.12.21, CMake, and a C++ compiler;
       ``conda_installer.sh`` installs Arena's locked environment. See :doc:`installation`.
   * - Isaac Sim
     - ``nvcr.io/nvidia/isaac-sim:6.1.0`` (Docker) or ``isaacsim==6.1.0.0`` (conda/pip)
   * - Isaac Lab
     - 3.0.0 at commit ``28a386f063e41c04c07f50e63eefb83fd8408fbe``
   * - Isaac Lab-Arena
     - 0.3.1 at commit ``481f5ae5f19df7bc5d24a17240fc0d96ac81eaf7``
   * - Python and PyTorch
     - Python 3.12 and PyTorch 2.11.0 with CUDA 12.8
   * - NumPy, Warp, and Newton
     - NumPy 2.3.1, Warp 1.16.0, and Newton 1.5.2; see
       :autodata_code_link:`<docker/runtime-constraints.txt>`
   * - Git and Git LFS
     - Git with recursive submodule support and Git LFS
   * - cuRobo and SkillGen
     - Optional; cuRobo commit ``ebb71702f3f70e767f40fd8e050674af0288abe8`` and CUDA Toolkit 12.8
