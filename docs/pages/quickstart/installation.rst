Installation
============

Docker is the recommended way to install AutoData. The dev container setup includes Isaac Sim,
Isaac Lab, Isaac Lab-Arena, and AutoData, providing a reproducible environment
without modifying the host Python installation. The repository is bind-mounted into the container,
so edits on the host are live inside it.

This checkout uses Isaac Sim 6.1.0, Isaac Lab 3.0.0, and Arena 0.3.0.
Use the pinned Docker environment or the optional conda installation below for the workflows on this site.

Before installing, review the :doc:`support_matrix` for the complete supported software stack,
hardware requirements, optional cuRobo and XR dependencies, and resource guidance.


Common Prerequisites
--------------------

On the host you need:

* An **NVIDIA GPU and driver**. Verify that the GPU is visible with ``nvidia-smi``.
* **Git LFS** installed (``sudo apt-get install git-lfs``) — the example datasets are stored
  with LFS.


Cloning the Repository
----------------------

Isaac Lab and Isaac Lab-Arena are nested git submodules, so clone recursively:

:autodata_git_clone_code_block:

If you already cloned without ``--recurse-submodules``, run:

.. code-block:: bash

   git submodule update --init --recursive

Then pull the LFS-stored datasets:

.. code-block:: bash

   git lfs install
   git lfs pull

If a dataset remains a small text pointer or cannot be opened as HDF5, see
:ref:`troubleshooting-lfs`.


Recommended Docker Installation
-------------------------------


Docker Prerequisites
^^^^^^^^^^^^^^^^^^^^

In addition to the common prerequisites, install Docker and the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
(provides ``--runtime=nvidia`` / ``--gpus all``).

The Isaac Sim base image lives on ``nvcr.io``, so log in to NGC before pulling it:

.. code-block:: bash

   docker login nvcr.io      # username: $oauthtoken   password: <your NGC API key>


Starting the Container
^^^^^^^^^^^^^^^^^^^^^^

From the repo root, build (first run) and enter the base dev container:

:docker_run_default:

The first run builds the default development image, then drops you into a shell inside the container
in the mounted repo. This build process may take up to 30 minutes. Subsequent runs reuse the versioned
image and are fast.

For **SkillGen** workflows, use the cuRobo image instead. cuRobo compiles CUDA kernels for
your GPU architecture (auto-detected via ``nvidia-smi``), so this build is slower and kept in
a separate versioned image tag that coexists with the default one:

:docker_run_curobo:

.. note::

   Inside the container the repo is mounted at ``/workspaces/autodata`` and ``python`` /
   ``pytest`` are aliased to Isaac Sim's interpreter (``/isaac-sim/python.sh``). Run simulation
   commands in these docs from that directory inside the container. Host commands are marked
   explicitly.

Useful flags of ``./docker/run_docker.sh``:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Description
   * - ``-c``
     - Include cuRobo (required for SkillGen). Auto-detects the GPU arch; override with the
       ``TORCH_CUDA_ARCH_LIST`` env var.
   * - ``-d <dir>``
     - Host datasets directory to mount at ``/datasets`` (default ``$HOME/datasets``).
   * - ``-r`` / ``-R``
     - Force rebuild of the image (``-R`` additionally disables the Docker cache).
   * - ``-v``
     - Verbose.
   * - ``-h``
     - Show all options.

For a missing Kit window or an X11 error, see :ref:`troubleshooting-display`.


Updating an Existing Checkout
------------------------------

After updating AutoData, synchronize the nested submodules and rebuild the image from the host:

.. code-block:: bash

   git submodule update --init --recursive
   ./docker/run_docker.sh -r

For SkillGen, use ``./docker/run_docker.sh -c -r`` instead. The current image tags are
``autodata:sim-6.1.0`` and ``autodata:sim-6.1.0-curobo``. A rebuild does not replace an already
running container. Finish its work, stop the matching container with
``docker stop autodata-sim-6.1.0`` or ``docker stop autodata-sim-6.1.0-curobo``, and rerun the
launcher. The launcher reports an error if a running container still uses the previous image.

Source edits are live through the repository bind mount. Dependency changes require rebuilding.
The checked-in example datasets are under ``./datasets/`` relative to the repository. The separate
``/datasets`` mount comes from the host directory selected by ``-d``; it is not the repository's
dataset directory.


Optional Conda Installation
---------------------------

The :autodata_code_link:`<conda_installer.sh>` installer supports Linux x86_64 and Python 3.12.
It installs Isaac Sim **6.1.0.0** (the pip version of Sim 6.1.0), PyTorch **2.11.0+cu128**,
Isaac Lab, and Arena from the pinned Arena ``uv.lock``, then installs AutoData with its test tools.
Lab and Arena are editable installations from the nested submodules.

On the host, install `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ **0.12.21 or newer**.
Git, CMake, and a C++ compiler must also be available. On Ubuntu:

.. code-block:: bash

   sudo apt-get install build-essential cmake git git-lfs

From a terminal outside any Python virtual environment, run these commands at the repository root:

.. code-block:: bash

   git submodule update --init --recursive
   ./conda_installer.sh -c -i
   conda activate autodata

The environment defaults to ``autodata``. Use ``-n`` to choose a different name, including when
installing into your existing ``isaac_autodata`` environment:

.. code-block:: bash

   ./conda_installer.sh -i -n isaac_autodata
   conda activate isaac_autodata

``-c`` creates an environment and refuses to replace one that already exists. ``-i`` installs or
updates packages in the selected environment, which must use Python 3.12. To keep an older setup
available, create a separate environment with ``./conda_installer.sh -c -i -n autodata_lab3``.
After updating the checkout and its submodules, rerun ``-i`` with the same environment name.
Use a fresh shell if the old environment has Isaac Sim binary-install activation hooks; this
installer uses pip-distributed Sim and does not configure a downloaded Sim binary.

The installer exports Arena's lockfile to ``.cache/conda-installer/pylock.toml``. This preserves
the pinned wheel URLs and hashes, including CUDA-enabled PyTorch, without rewriting the submodules.
It retains unrelated installed packages and checks the locked stack again after installing AutoData.
The generated ``.cache/conda-installer/runtime-constraints.txt`` pins the locked dependencies for
later optional installations, including cuRobo. Rerun the installer to refresh both exports after
an Arena update.
Arena's lock includes upstream dependency overrides. The installed stack matches that lock, but
``pip check`` still reports these five discrepancies with wheel metadata:

.. list-table:: Upstream package metadata versus the Arena lock
   :widths: 30 35 35
   :header-rows: 1

   * - Package
     - Declared requirement
     - Locked installation
   * - ``openpi-client``
     - NumPy < 2
     - NumPy 2.3.1
   * - ``isaacsim-core``
     - Newton 1.5.0
     - Newton 1.5.2
   * - ``isaacsim-kernel``
     - websockets < 15
     - websockets 16.1.1
   * - ``ovphysx``
     - packaging < 24
     - packaging 26.0
   * - ``isaacsim-robot``
     - onnxruntime-gpu 1.26.0
     - CPU onnxruntime 1.26.0

Keep the locked versions and use the runtime checks below. GPU ONNX inference is not provided
by this installation; AutoData's tested GPU generation workflows use PyTorch and CUDA.

Review and accept the Isaac Sim EULA on first launch. For unattended launches after accepting it,
set ``export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y``. See the official
`Isaac Sim pip installation guide
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html>`_.
Run workflow commands with ``python`` from the repository root while the selected conda environment
is active. Substitute your checkout path for the Docker-only ``/workspaces/autodata`` path.


Optional cuRobo for Conda
^^^^^^^^^^^^^^^^^^^^^^^^^

SkillGen additionally needs cuRobo and a **CUDA Toolkit 12.8** installation containing ``nvcc``.
Install the toolkit using NVIDIA's
`CUDA installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_, then
run the following in the activated AutoData environment. Adjust ``CUDA_HOME`` if your toolkit
is installed elsewhere:

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export TORCH_CUDA_ARCH_LIST="$(python -c \
       'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}+PTX")')"
   export MAX_JOBS=4
   uv pip install --python "$CONDA_PREFIX/bin/python" \
       --constraint .cache/conda-installer/runtime-constraints.txt ninja wheel
   uv pip install --python "$CONDA_PREFIX/bin/python" \
       --constraint .cache/conda-installer/runtime-constraints.txt \
       --no-build-isolation \
       'nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8'
   uv pip install --python "$CONDA_PREFIX/bin/python" --check \
       --requirements .cache/conda-installer/pylock.toml

The GPU must be visible when detecting its compute capability. If building for another machine,
set ``TORCH_CUDA_ARCH_LIST`` explicitly to that GPU's compute capability instead. Rebuild cuRobo
when changing PyTorch or the target GPU architecture. The base conda installer does not install cuRobo.


Documentation Environment
-------------------------

Building the documentation only needs a separate Python virtual environment and
``docs/requirements.txt``. It does not require Isaac Sim, Arena, cuRobo, or a conda environment.
See :autodata_code_link:`<docs/README.md>` for local build instructions.


Verifying the Installation
--------------------------

Run these commands from the repository root in the activated conda environment, or from
``/workspaces/autodata`` inside the container. The full suite includes SkillGen tests and requires cuRobo.

Run the fast unit tests (a few seconds, no Isaac Sim launch):

.. code-block:: bash

   pytest autodata_tests -m "not with_subprocess"

Then, to verify the full stack end-to-end, run one data-generation test (launches Isaac Sim
as a subprocess; several minutes):

.. code-block:: bash

   pytest -s autodata_tests/e2e/test_mimicgen_data_generation.py::test_franka_cube_stack_mimicgen_data_generation_single_env_cuda

See :doc:`../advanced/testing_and_ci` for the full test-suite layout.
