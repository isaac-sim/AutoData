..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

.. _skillgen-preflight:

Franka SkillGen Preflight
=========================

Complete this preflight before annotating demonstrations or starting a long SkillGen generation
run. It verifies the cuRobo image, GPU build, source dataset, Nucleus asset download, and one complete
end-to-end generation run. Repeat it after changing GPUs or rebuilding the development image.

This page assumes the base :doc:`AutoData installation <../../quickstart/installation>` is complete
and the host satisfies the :doc:`support matrix <../../quickstart/support_matrix>`.

Requirements at a Glance
------------------------

.. list-table:: Franka SkillGen requirements
   :widths: 27 73
   :header-rows: 1

   * - Requirement
     - What is needed
   * - Container
     - The cuRobo development image started with ``./docker/run_docker.sh -c``.
   * - GPU memory
     - At least 24 GB VRAM is recommended for 1–2 environments; use a 48 GB GPU for approximately
       5 environments.
   * - GPU build
     - cuRobo kernels compiled for the compute capability of the GPU running generation.
   * - Network
     - Outbound access to the Nucleus asset server during planner initialization so AutoData can
       retrieve the Franka URDF.
   * - Source data
     - Git LFS objects pulled and a structurally valid SkillGen-annotated HDF5 dataset.
   * - Display
     - Optional. Use ``--viz none`` for headless generation. A display is required for manual
       annotation, Kit visualization, and the Rerun plan viewer.

1. Start the cuRobo Container
-----------------------------

On the **host**, confirm which GPU will run SkillGen and start the cuRobo image:

.. code-block:: bash

   nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
   git lfs pull
   ./docker/run_docker.sh -c

The first build compiles cuRobo for the compute capability reported by ``nvidia-smi`` and can take
considerably longer than starting the base container. If the image was built for a different GPU,
rebuild it from the host:

.. code-block:: bash

   ./docker/run_docker.sh -c -r

Use ``-R`` instead of ``-r`` only when a cached build remains incompatible. See
:ref:`troubleshooting-curobo` for missing-module and CUDA-kernel errors.

2. Verify cuRobo and the Source Dataset
---------------------------------------

Run these checks inside the **cuRobo container**:

.. code-block:: bash

   python -c "import curobo, torch; print(curobo.__file__); print(torch.cuda.get_device_name(0))"
   python scripts/validate_dataset.py ./datasets/annotated_datasets/dataset_franka_skillgen_annotated.hdf5

The first command must print the installed cuRobo path and expected GPU. The validator must report
the source dataset as ``VALID``. If it reports ``file signature not found`` or the file is an LFS
pointer, follow :ref:`troubleshooting-lfs`.

3. Run One End-to-End Planning Check
------------------------------------

Still inside the **cuRobo container**, generate one successful cube-stacking demonstration:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg skillgen \
       --task_descriptor autodata_examples/tasks/franka_cube_stack_skillgen.yaml \
       --embodiment autodata_examples/embodiments/franka_ik_rel_skillgen.yaml \
       --input_file ./datasets/annotated_datasets/dataset_franka_skillgen_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_skillgen_preflight.hdf5 \
       --result_file ./datasets/generated_dataset_skillgen_preflight.json \
       --generation_num_trials 1 \
       --num_envs 1 \
       --viz none

This check exercises the Nucleus URDF download, cuRobo initialization, collision-world update,
motion planning, execution, and HDF5 recording. Validate the result:

.. code-block:: bash

   python scripts/validate_dataset.py ./datasets/generated_dataset_skillgen_preflight.hdf5

The preflight passes when generation records one successful demonstration and the validator reports
the output as ``VALID``. A Nucleus download failure is a setup problem; follow
:ref:`troubleshooting-nucleus` before investigating planner configuration.

.. _skillgen-preflight-failures:

Expected Planning Failures
--------------------------

SkillGen plans against a newly randomized scene on every attempt. Some configurations have no path
within the planner's search budget, so an individual planning failure is expected. Both shipped
SkillGen descriptors set ``guarantee_success: true``. AutoData counts the failed attempt, resets the
scene, and retries until it records the requested number of successful demonstrations.

.. list-table:: Interpreting SkillGen failures
   :widths: 42 18 40
   :header-rows: 1

   * - Observation
     - Expected?
     - Response
   * - An occasional attempt reports no collision-free path, then generation continues.
     - Yes
     - No action is required. Monitor the overall success rate.
   * - Bin stacking rejects more attempts and runs longer than plain cube stacking.
     - Yes
     - Narrow bin clearances make planning harder.
   * - Every attempt fails planning from the same subtask.
     - No
     - Check its start annotation, end-effector frame and offset, planner profile, and collision
       world. See :ref:`troubleshooting-planning`.
   * - CUDA reports out-of-memory while planners initialize or run.
     - No
     - Reduce ``--num_envs``. Validate with one environment before scaling.
   * - cuRobo import, CUDA kernel, or Nucleus download fails before planning begins.
     - No
     - Treat it as a preflight failure and use the corresponding troubleshooting section.
