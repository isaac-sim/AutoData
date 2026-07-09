Installation
============

Isaac AutoData is developed and run inside a Docker dev container that ships Isaac Sim,
Isaac Lab, Isaac Lab Arena, and Isaac AutoData pre-installed. The repository is bind-mounted
into the container, so edits on the host are live inside it — think of
``./docker/run_docker.sh`` as the Docker equivalent of activating a conda environment.

Prerequisites
-------------

On the host you need:

* An **NVIDIA GPU and driver**, Docker, and the
  `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
  (provides ``--runtime=nvidia`` / ``--gpus all``). Verify with:

  .. code-block:: bash

     docker run --rm --gpus all nvcr.io/nvidia/isaac-sim:6.0.0-dev2 nvidia-smi

* An **NGC login** — the Isaac Sim base image lives on ``nvcr.io``:

  .. code-block:: bash

     docker login nvcr.io      # username: $oauthtoken   password: <your NGC API key>

* **Git LFS** installed (``sudo apt-get install git-lfs``) — the example datasets are stored
  with LFS.

Cloning the Repository
----------------------

Isaac Lab Arena and Isaac Lab are nested git submodules, so clone recursively:

:isaac_autodata_git_clone_code_block:

If you already cloned without ``--recurse-submodules``, run:

.. code-block:: bash

   git submodule update --init --recursive

Then pull the LFS-stored datasets:

.. code-block:: bash

   git lfs install
   git lfs pull

Starting the Container
----------------------

From the repo root, build (first run) and enter the base dev container:

:docker_run_default:

The first run builds the image (``isaac_autodata:latest``) — pulling the Isaac Sim base and
installing Isaac Lab takes a while — then drops you into a shell inside the container, as your
host user, in the mounted repo. Subsequent runs reuse the image and are fast.

For **SkillGen** workflows, use the cuRobo image instead. cuRobo compiles CUDA kernels for
your GPU architecture (auto-detected via ``nvidia-smi``), so this build is slower and kept in
a separate image tag (``isaac_autodata:curobo``) that coexists with the default one:

:docker_run_curobo:

.. note::

   Inside the container the repo is mounted at ``/workspaces/isaac_autodata`` and ``python`` /
   ``pytest`` are aliased to Isaac Sim's interpreter (``/isaac-sim/python.sh``). All commands
   in these docs are run from that directory inside the container.

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
     - Verbose (``set -x``).
   * - ``-h``
     - Show all options.

Environment overrides: ``BASE_IMAGE`` changes the Isaac Sim base tag;
``TORCH_CUDA_ARCH_LIST`` pins the GPU arch for the cuRobo build instead of auto-detecting.

Any trailing arguments are run as a one-off command inside the container, which then exits:

.. code-block:: bash

   ./docker/run_docker.sh python isaac_autodata_examples/generate_dataset.py --help

.. note::

   Rendering a window from the container (``--viz kit``) needs a display: ``run_docker.sh``
   forwards ``DISPLAY`` and the X11 socket automatically. On a headless host, pass
   ``--viz none`` to the scripts instead (the test suite already does).

Verifying the Installation
--------------------------

Run the fast unit tests inside the container (a few seconds, no Isaac Sim launch):

.. code-block:: bash

   pytest isaac_autodata_tests -m "not with_subprocess"

Then, to verify the full stack end-to-end, run one data-generation test (launches Isaac Sim
as a subprocess; several minutes):

.. code-block:: bash

   pytest -s isaac_autodata_tests/e2e/test_mimicgen_data_generation.py::test_franka_cube_stack_mimicgen_data_generation_single_env_cuda

See :doc:`../advanced/testing_and_ci` for the full test-suite layout.
