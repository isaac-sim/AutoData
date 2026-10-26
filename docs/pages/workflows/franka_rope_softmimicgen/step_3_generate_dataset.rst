.. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Step 3: Generate the Dataset
----------------------------

SoftMimicGen generates new demonstrations by randomizing the rope's starting configuration,
selecting a source demonstration by nodal registration cost, and warping the source
end-effector trajectory to the current rope shape. It then executes the adapted trajectory
and evaluates task success.

The commands below use the annotated files from Steps 1–2. To skip those steps, replace
``--input_file`` with the pre-annotated dataset for your backend:

.. tabs::

   .. group-tab:: Newton

      `autodata_tests/test_data/annotated_dataset_franka_rope_softmimicgen_newton.hdf5
      <https://github.com/isaac-sim/AutoData/blob/main/autodata_tests/test_data/annotated_dataset_franka_rope_softmimicgen_newton.hdf5>`_

   .. group-tab:: PhysX

      `autodata_tests/test_data/annotated_dataset_franka_rope_softmimicgen.hdf5
      <https://github.com/isaac-sim/AutoData/blob/main/autodata_tests/test_data/annotated_dataset_franka_rope_softmimicgen.hdf5>`_

These source datasets are Git LFS assets. Use the dataset that matches ``--presets``.


Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with two parallel environments in a Kit window to inspect the generated motions:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --alg softmimicgen \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/annotated_dataset_franka_rope_newton.hdf5 \
             --output_file ./datasets/softmimicgen/generated_dataset_franka_rope_newton_softmimicgen_small.hdf5 \
             --generation_num_trials 10 \
             --num_envs 2 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Parallel SoftMimicGen rope data generation with Newton.

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --alg softmimicgen \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/annotated_dataset_franka_rope_physx.hdf5 \
             --output_file ./datasets/softmimicgen/generated_dataset_franka_rope_physx_softmimicgen_small.hdf5 \
             --generation_num_trials 10 \
             --num_envs 2 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Parallel SoftMimicGen rope data generation with PhysX.

Failed attempts are normal. The task descriptor sets ``guarantee_success: true`` and
``keep_failed: false``, so generation retries until it reaches the target of 10 successful
demonstrations and exports only successes. The console reports progress and success rate.
The attempt count and runtime can be higher than the requested demonstration count.


Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

After checking the small run, increase the target and run without the Kit window.
This example keeps two parallel environments; increase ``--num_envs`` gradually according
to GPU memory and observed throughput. Deformable simulation and camera recording can be
expensive, so the cube-stacking workflow's environment count is not a sizing guide for this task.

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --alg softmimicgen \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/annotated_dataset_franka_rope_newton.hdf5 \
             --output_file ./datasets/softmimicgen/generated_dataset_franka_rope_newton_softmimicgen.hdf5 \
             --generation_num_trials 1000 \
             --num_envs 2 \
             --device cuda \
             --enable_cameras \
             --viz none

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --alg softmimicgen \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/annotated_dataset_franka_rope_physx.hdf5 \
             --output_file ./datasets/softmimicgen/generated_dataset_franka_rope_physx_softmimicgen.hdf5 \
             --generation_num_trials 1000 \
             --num_envs 2 \
             --device cuda \
             --enable_cameras \
             --viz none

To generate state-only datasets, omit ``--enable_cameras``. ``--viz none`` alone does not
disable image recording.


Key Parameters
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``--presets``
     - Physics backend: ``newton`` or ``physx``. Match the source dataset's backend.
   * - ``--alg``
     - Use ``softmimicgen`` for this deformable-object task.
   * - ``--generation_num_trials``
     - Generation target. With this descriptor's ``guarantee_success: true``, counts successful demonstrations.
   * - ``--num_envs``
     - Number of parallel simulation environments.
   * - ``--enable_cameras``
     - Render and record the fixed-view and wrist RGB observations.
   * - ``--viz``
     - ``kit`` opens the simulation window; ``none`` runs headless.

The task descriptor marks both subtasks with ``algo_params.object_soft: true`` and uses
``selection_strategy: registration_cost``. TPS regularization is controlled by ``bend_coef``
and ``rot_coef``; ``use_rotation_transform`` also adapts end-effector orientations.
``reset_settling_steps: 10`` lets the scene settle after a randomized reset before generation
starts. Begin with the supplied settings before tuning the descriptor for your demonstrations.

Continue to :doc:`step_4_validate_dataset`.
