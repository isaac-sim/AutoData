..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Troubleshooting
===============

Use the table to identify the likely cause, then follow the linked check or fix. Commands
marked **host** run outside the development container. Commands marked **container** run from
``/workspaces/autodata`` inside the container. Replace values in angle brackets with values
from your run. Issues are ordered by workflow area: runtime and teleoperation, dataset handling,
then SkillGen.

.. list-table:: Symptom, cause, and fix
   :widths: 31 29 40
   :header-rows: 1

   * - Symptom
     - Likely cause
     - Check or fix
   * - ``--viz kit`` opens no window, or reports an X11/display error.
     - ``DISPLAY`` or the X11 socket was not available when the container started, or the container
       lacks X server permission.
     - Check the host and container display state, then restart the container or use ``--viz none``.
       See :ref:`troubleshooting-display`.
   * - The Apple Vision Pro cannot connect, or hand tracking does not reach the recording script.
     - CloudXR is not running, its ports are blocked, or the recording shell did not source the
       runtime environment.
     - Verify the firewall, start CloudXR first, and then source its environment in the recording
       shell. See :ref:`troubleshooting-cloudxr`.
   * - Isaac Sim prints protobuf, MaterialX, render-interval, or shutdown warnings.
     - Isaac Sim and its plug-ins emit known startup and shutdown noise.
     - Use the exit status, final traceback, and output validation to distinguish noise from a real
       failure. See :ref:`troubleshooting-warnings`.
   * - An example HDF5 file is only a few bytes, or reports ``file signature not found``.
     - Git checked out an LFS pointer instead of the file contents.
     - Check whether the file is an LFS pointer, then pull LFS objects. See
       :ref:`troubleshooting-lfs`.
   * - Dataset validation reports a truncated file, an unreadable file, or missing HDF5 groups.
     - Recording or generation was interrupted, storage filled up, or the wrong file was supplied.
     - Validate the file and restore, re-copy, or regenerate it; there is no general safe repair for
       a truncated HDF5 file. See :ref:`troubleshooting-hdf5`.
   * - Replay diverges from generation, objects move unexpectedly, or an episode fails on replay.
     - The replay environment or device differs from the recorded metadata, or PhysX replay has
       diverged nondeterministically.
     - Validate the metadata, match the environment and device, and reproduce with one environment.
       See :ref:`troubleshooting-replay`.
   * - ``import curobo`` fails, or cuRobo reports an incompatible CUDA kernel.
     - The base container was used, or cuRobo was built for a different GPU compute capability.
     - Start the cuRobo container; for a GPU change, rebuild it without cache. See
       :ref:`troubleshooting-curobo`.
   * - SkillGen fails while downloading the Franka URDF from Nucleus.
     - The container cannot reach the Nucleus asset server because of its network, DNS, proxy, or
       VPN configuration.
     - Test the exact asset download independently. See :ref:`troubleshooting-nucleus`.
   * - SkillGen repeatedly reports that motion planning failed.
     - The goal is unreachable, annotations or robot frames are inconsistent, the collision world
       is wrong, or parallel planners exhausted GPU memory.
     - Reproduce with one environment and visualize the plan before changing planner parameters.
       See :ref:`troubleshooting-planning`.

.. _troubleshooting-display:

Display and X11
---------------

Check the display on the **host** before starting the container:

.. code-block:: bash

   echo "$DISPLAY"
   ls -ld /tmp/.X11-unix
   xhost +local:docker

Both ``DISPLAY`` and ``/tmp/.X11-unix`` must be present. Check the same values inside the
**container**:

.. code-block:: bash

   echo "$DISPLAY"
   ls -ld /tmp/.X11-unix

The run script captures ``DISPLAY`` when it creates the container. If the container was created
before the display was available, stop it and start it again with ``./docker/run_docker.sh`` (or
``./docker/run_docker.sh -c`` for SkillGen). On a machine without a display, use ``--viz none``.

.. _troubleshooting-cloudxr:

CloudXR and Apple Vision Pro
----------------------------

On the **host**, check that the CloudXR firewall rules are active:

.. code-block:: bash

   sudo ufw status

The required TCP and UDP rules are listed in the :ref:`humanoid recording workflow
<record_humanoid_cloudxr>`. In the first **container** shell, start CloudXR and leave it running:

.. code-block:: bash

   python -m isaacteleop.cloudxr --cloudxr-env-config=avp.env

In the second **container** shell, wait until the runtime has created its environment file, then
source it before starting the recording script:

.. code-block:: bash

   test -f ~/.cloudxr/run/cloudxr.env
   source ~/.cloudxr/run/cloudxr.env

If the first command returns nonzero, the runtime is not ready. Also confirm that the CloudXR EULA
was accepted in the runtime shell and that the headset and host satisfy the network requirements.

.. _troubleshooting-warnings:

Noisy Nonfatal Warnings
-----------------------

Warnings such as the following can be nonfatal when Isaac Sim starts or shuts down:

* ``File already exists in database: grpc/health/v1/health.proto``
* ``Enable omni.materialx.libs extension to use MaterialX``
* a render interval smaller than the environment decimation
* a plug-in interface that ``was already released`` during shutdown

Immediately after the command exits, run ``echo $?``. Exit status ``0``, no final traceback, and a
dataset that passes validation indicate that these warnings did not stop the run. Do not ignore a
warning followed by a Python traceback, CUDA error, missing articulation, nonzero exit status, or
missing output file; the final error is the failure to diagnose.

.. _troubleshooting-lfs:

Git LFS Pointer Files
---------------------

From the repo root on the **host**, check a file that should contain HDF5 data:

.. code-block:: bash

   git lfs pointer --check --file=./datasets/annotated_datasets/dataset_franka_annotated.hdf5
   echo $?

Exit status ``0`` means the working-tree file is still an LFS pointer. Pull the file contents and
check LFS status:

.. code-block:: bash

   git lfs install
   git lfs pull
   git lfs status

Run the dataset validator after the pull. If LFS still cannot download the object, check repository
access, available disk space, and any proxy configuration.

.. _troubleshooting-hdf5:

Invalid or Truncated HDF5
-------------------------

Run structural validation inside the **container**:

.. code-block:: bash

   python scripts/validate_dataset.py <dataset.hdf5>
   echo $?

The validator exits nonzero for unreadable files, empty datasets, and episodes missing ``actions``,
``initial_state``, or ``obs``. ``file signature not found`` can indicate an LFS pointer; check
:ref:`troubleshooting-lfs` first for repository datasets.

For a file produced by an interrupted run, check storage with ``df -h .`` and preserve the file for
diagnosis. HDF5 has no general safe repair for a truncated file: restore or re-copy a known-good
source dataset, or regenerate into a new output file. After structural validation succeeds, replay
a small sample to check task semantics.

.. _troubleshooting-replay:

Replay Failures and Divergence
------------------------------

Validate the dataset first and note its ``Env ID`` and ``sim_args``:

.. code-block:: bash

   python scripts/validate_dataset.py <dataset.hdf5>

Replay with that environment, the recorded device, and one environment while diagnosing the issue:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
       --task <env-id-from-validator> \
       --device <device-from-sim-args> \
       --num_envs 1 \
       --viz kit \
       --dataset_file <dataset.hdf5>

If this works, increase ``--num_envs`` gradually. Even with matching settings, PhysX is not fully
deterministic across resets, so action replay can diverge from the original successful generation.
A structurally valid dataset can therefore contain an episode that does not reproduce success on
replay; inspect multiple episodes before concluding that the dataset is invalid.

.. _troubleshooting-curobo:

cuRobo Build and GPU Compatibility
----------------------------------

Verify cuRobo inside the **container**:

.. code-block:: bash

   python -c "import curobo; print(curobo.__file__)"

If the module is missing, leave the base container and start the cuRobo image from the **host**:

.. code-block:: bash

   ./docker/run_docker.sh -c

If cuRobo was built on a different GPU generation, query the active GPU and rebuild without the
Docker cache:

.. code-block:: bash

   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)+PTX"
   ./docker/run_docker.sh -c -R

The rebuild is required for errors such as ``no kernel image is available for execution on the
device``. For ordinary source changes, a no-cache rebuild is not necessary.

.. _troubleshooting-nucleus:

Nucleus Asset Download
-----------------------

SkillGen initialization retrieves its Franka URDF from the Nucleus asset server. Test that exact
download inside the **container**:

.. code-block:: bash

   python -c "from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path; path = f'{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf'; print(retrieve_file_path(path, force_download=True))"

Success prints a local path. If it fails, resolve the container's DNS, proxy, VPN, or outbound
network access before debugging cuRobo or planner settings.

.. _troubleshooting-planning:

SkillGen Planning Failures
--------------------------

An occasional planning failure is expected: when ``guarantee_success: true``, AutoData resets the
scene and retries. If every attempt fails, first rerun the same generation command with these
diagnostic options:

.. code-block:: bash

   --num_envs 1 \
   --visualize_plan \
   --result_file ./datasets/planning_debug.json

Check that the Rerun viewer shows the expected robot start pose, target pose, and collision world.
If no Rerun window appears, run ``command -v rerun`` inside the container and also follow
:ref:`troubleshooting-display`.

Persistent failures usually come from one of four sources: subtask-start annotations that produce
an unreachable goal, an incorrect end-effector frame or offset in the embodiment, the wrong planner
selected by the environment profile, or missing/incorrect collision objects. CUDA out-of-memory
errors are a scaling problem rather than a planning problem; keep ``--num_envs 1`` until the run is
valid, then increase it gradually.
