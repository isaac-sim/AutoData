Installation
============

Docker is the recommended way to install Isaac AutoData. The dev container setup includes Isaac Sim,
Isaac Lab, Isaac Lab Arena, and Isaac AutoData, providing a reproducible environment
without modifying the host Python installation. The repository is bind-mounted into the container,
so edits on the host are live inside it.

An optional conda installation is also available for users who want direct control over their
Python environment and installed packages. See `Optional Conda Installation`_ below.


Common Prerequisites
--------------------

On the host you need:

* An **NVIDIA GPU and driver**. Verify that the GPU is visible with ``nvidia-smi``.
* **Git LFS** installed (``sudo apt-get install git-lfs``) — the example datasets are stored
  with LFS.


Cloning the Repository
----------------------

Isaac Lab and Isaac Lab Arena are nested git submodules, so clone recursively:

:isaac_autodata_git_clone_code_block:

If you already cloned without ``--recurse-submodules``, run:

.. code-block:: bash

   git submodule update --init --recursive

Then pull the LFS-stored datasets:

.. code-block:: bash

   git lfs install
   git lfs pull


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

   Inside the container the repo is mounted at ``/workspaces/isaac_autodata`` and ``python`` /
   ``pytest`` are aliased to Isaac Sim's interpreter (``/isaac-sim/python.sh``). Unless you chose
   the optional conda installation, run commands in these docs from that directory inside the
   container.

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


Optional Conda Installation
---------------------------

Use the conda route if you want to manage the environment and its packages directly. Docker remains
the recommended route because it provides the project's reproducible, preconfigured environment.

The conda installation requires ``conda`` and `uv <https://docs.astral.sh/uv/>`_ on your ``PATH``.
From the repository root, create the ``isaac_autodata`` environment with Python 3.12:

.. code-block:: bash

   ./conda_installer.sh -c

Activate the environment and install Isaac Sim, CUDA-enabled PyTorch, Isaac Lab, Isaac Lab Arena,
and Isaac AutoData:

.. code-block:: bash

   conda activate isaac_autodata
   ./conda_installer.sh -i

You can create the environment and install the packages in one command:

.. code-block:: bash

   ./conda_installer.sh -c -i

Run ``./conda_installer.sh -h`` to see all installer options. Activate the environment before running
commands from the rest of the documentation:

.. code-block:: bash

   conda activate isaac_autodata


Installing cuRobo for SkillGen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SkillGen additionally requires cuRobo. Before installing it, review the
`NVIDIA cuRobo license <https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses/dependencies/cuRobo-license.txt>`_.

.. warning::

   Install cuRobo from a clean shell that has not sourced Isaac Sim environment scripts such as
   ``setup_conda_env.sh``. Those scripts set ``PYTHONHOME`` and ``PYTHONPATH`` to use Kit's bundled
   packages, which can cause conda to fail during the cuRobo installation.

Activate the AutoData environment, install the CUDA 12.8 toolkit, and configure the build for your
GPU's compute capability:

.. code-block:: bash

   conda activate isaac_autodata
   conda install -c nvidia cuda-toolkit=12.8 -y
   export CUDA_HOME="$CONDA_PREFIX"
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
   export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)+PTX"

Install the cuRobo commit tested with Isaac Lab and used by the AutoData cuRobo container:

.. code-block:: bash

   pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" \
     --no-build-isolation

The editable installation clones cuRobo into ``src/nvidia-curobo`` beneath the current directory.
Run the command from the directory where you want to keep that source checkout.

Verify the installation:

.. code-block:: bash

   python -c "import curobo; print('cuRobo installed successfully')"

.. tip::

   If the import fails because ``libstdc++.so.6`` does not provide ``GLIBCXX_3.4.30``, update the
   environment's C++ runtime libraries:

   .. code-block:: bash

      conda config --env --set channel_priority strict
      conda config --env --add channels conda-forge
      conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"


Verifying the Installation
--------------------------

Docker users should run these commands inside the container. Conda users should run them from the
repository root after activating the ``isaac_autodata`` environment.

Run the fast unit tests (a few seconds, no Isaac Sim launch):

.. code-block:: bash

   pytest isaac_autodata_tests -m "not with_subprocess"

Then, to verify the full stack end-to-end, run one data-generation test (launches Isaac Sim
as a subprocess; several minutes):

.. code-block:: bash

   pytest -s isaac_autodata_tests/e2e/test_mimicgen_data_generation.py::test_franka_cube_stack_mimicgen_data_generation_single_env_cuda

See :doc:`../advanced/testing_and_ci` for the full test-suite layout.
