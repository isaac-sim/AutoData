..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Support Matrix
==============

This matrix describes the supported Isaac AutoData v0.1.0 stack. Upstream projects may support
additional platforms, but configurations outside this matrix are not validated by AutoData. Use
the recursively pinned submodules from the AutoData repository instead of independently selecting
Isaac Lab or Isaac Lab Arena revisions.

.. list-table:: Isaac AutoData platform and resource support
   :widths: 25 75
   :header-rows: 1

   * - Area
     - Supported or required configuration
   * - Host operating system
     - Linux x86_64; Ubuntu 22.04 or 24.04
   * - GPU
     - NVIDIA RTX GPU with RT cores and at least 16 GB VRAM.
       See `Isaac Sim 6.0.1 requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - NVIDIA driver
     - Linux 595.58.03.
       See `Isaac Sim 6.0.1 requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - CPU and RAM
     - Minimum: 4 CPU cores and 32 GB RAM. Recommended: 8 or more cores and 64 GB RAM.
       See `Isaac Sim 6.0.1 requirements
       <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
   * - Disk
     - Minimum: 100 GB SSD. Recommended: 500 GB SSD
   * - Docker installation
     - Docker Engine with the NVIDIA Container Toolkit; recommended AutoData installation
   * - Conda installation
     - Optional Linux installation using conda, ``uv``, and Python 3.12
   * - Isaac Sim
     - 6.0.1 container image; 6.0.1.0 Python package
   * - Isaac Lab
     - 3.0.0 at commit ``ffff603eafc6b74264a5261cc0183d6a65390d78``
   * - Isaac Lab Arena
     - 0.2.0 at commit ``8b82dca224f2b5af08f339f987613c59ce9cdbaa``
   * - Python and PyTorch
     - Python 3.12 and PyTorch 2.10.0 with CUDA 12.8
   * - Git and Git LFS
     - Git with recursive submodule support and Git LFS
   * - cuRobo and SkillGen
     - Optional; cuRobo commit ``ebb71702f3f70e767f40fd8e050674af0288abe8`` and CUDA Toolkit 12.8
