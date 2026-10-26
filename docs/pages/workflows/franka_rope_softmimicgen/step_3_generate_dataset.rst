.. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Step 3: Generate the Dataset
----------------------------

SoftMimicGen generates new demonstrations by randomizing the rope's starting configuration,
selecting a source demonstration by nodal registration cost, and warping the source
end-effector trajectory to the current rope shape. It then executes the adapted trajectory
and evaluates task success.

The commands below use the pre-annotated source datasets that ship with the repository.
Use the source dataset that matches your physics backend:

.. tabs::

   .. group-tab:: Newton

      :autodata_code_link:`<datasets/annotated_datasets/dataset_franka_rope_newton_annotated.hdf5>`

      If you ran Steps 1–2, set ``--input_file`` to
      ``./datasets/dataset_franka_rope_newton_annotated.hdf5`` instead.

   .. group-tab:: PhysX

      :autodata_code_link:`<datasets/annotated_datasets/dataset_franka_rope_physx_annotated.hdf5>`

      If you ran Steps 1–2, set ``--input_file`` to
      ``./datasets/dataset_franka_rope_physx_annotated.hdf5`` instead.


Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with two parallel environments in a Kit window to inspect the generated motions:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --viz kit \
             --device cuda \
             --enable_cameras \
             --num_envs 2 \
             --alg softmimicgen \
             --generation_num_trials 5 \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/annotated_datasets/dataset_franka_rope_newton_annotated.hdf5 \
             --output_file ./datasets/generated_dataset_softmimicgen_franka_rope_newton_small.hdf5

      .. todo::

         Image placeholder: Parallel SoftMimicGen rope data generation with Newton.

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --viz kit \
             --device cuda \
             --enable_cameras \
             --num_envs 2 \
             --alg softmimicgen \
             --generation_num_trials 5 \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/annotated_datasets/dataset_franka_rope_physx_annotated.hdf5 \
             --output_file ./datasets/generated_dataset_softmimicgen_franka_rope_physx_small.hdf5

      .. todo::

         Image placeholder: Parallel SoftMimicGen rope data generation with PhysX.

Failed attempts are normal. The task descriptor sets ``guarantee_success: true`` 
so generation retries until it reaches the target of 5 successful
demonstrations and exports only successes. The console reports progress and success rate.


Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

After checking the small run, generate 100 successful demonstrations with 10 parallel
environments and run without the Kit window. Adjust ``--num_envs`` according to GPU memory
and observed throughput. Deformable simulation and camera recording can be expensive.

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --viz none \
             --device cuda \
             --enable_cameras \
             --num_envs 10 \
             --alg softmimicgen \
             --generation_num_trials 100 \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/annotated_datasets/dataset_franka_rope_newton_annotated.hdf5 \
             --output_file ./datasets/generated_dataset_softmimicgen_franka_rope_newton.hdf5

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/generate_dataset.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --viz none \
             --device cuda \
             --enable_cameras \
             --num_envs 10 \
             --alg softmimicgen \
             --generation_num_trials 100 \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/annotated_datasets/dataset_franka_rope_physx_annotated.hdf5 \
             --output_file ./datasets/generated_dataset_softmimicgen_franka_rope_physx.hdf5

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

Continue to :doc:`step_4_validate_dataset`.
