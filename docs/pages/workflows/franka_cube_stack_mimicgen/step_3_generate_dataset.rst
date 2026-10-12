Step 3: Generate the Dataset
----------------------------

With annotated source demonstrations in hand, MimicGen can synthesize new ones. For each attempt,
the scene is randomized, a source segment is selected per subtask, and its end-effector
trajectory is rigidly transformed to the new object poses and replayed.

.. figure:: ../../../images/franka_mimicgen_datagen.jpg
   :width: 90%
   :align: center
   :alt: Parallel data generation for the Franka cube-stacking task

   Parallel data generation for the Franka cube-stacking task.

The commands below use the pre-annotated source dataset that ships with the repository. If you ran
Steps 1–2, point ``--input_file`` at your own ``dataset_franka_annotated.hdf5`` instead.


Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with a small-scale run in a Kit window (``--viz kit``) to sanity-check the setup:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name IsaacContrib-Stack-Cube-Franka-IK-Rel \
       --viz kit \
       --num_envs 5 \
       --alg mimicgen \
       --generation_num_trials 10 \
       --task_descriptor autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_mimicgen_franka_small.hdf5

You will see the robot repeatedly attempt the task under new cube placements. Failed
attempts are normal and are not exported.

While it runs, the console reports the number of source episodes loaded into the pool and a
running tally after every attempt to generate a new demonstration:

.. code-block:: text

   Loaded 10 source episodes into the datagen pool

   **************************************************
   5/10 (50.0%) successful demos generated
   **************************************************

The script shuts down automatically after 10 successful demonstrations are generated.


Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

For dataset-scale generation, run headless (``--viz none``) with parallel environments:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name IsaacContrib-Stack-Cube-Franka-IK-Rel \
       --viz none \
       --num_envs 100 \
       --alg mimicgen \
       --generation_num_trials 1000 \
       --task_descriptor autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_mimicgen_franka.hdf5

Progress is printed after every attempt (successful demonstrations, attempts, and the running success rate).

.. note::

  **Upgrade validation:** the 100-environment GPU benchmark achieved 38.0% success
  (501 successes in 1,319 attempts) using the repository's test dataset. This is a reference
  measurement, not a timing or success guarantee. Start with a small run and increase
  ``--num_envs`` according to available GPU memory and observed throughput.


Key Parameters
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``--alg``
     - Generation algorithm: ``mimicgen``, ``dexmimicgen``, or ``skillgen``.
   * - ``--env_name``
     - Env id; if omitted, read from the source dataset's metadata.
   * - ``--generation_num_trials``
     - Overrides the task descriptor's ``generation_policy.num_trials``.
   * - ``--num_envs``
     - Number of parallel environments, each generating independently.
   * - ``--viz``
     - Visualization mode (``kit`` opens a Kit window; ``none`` runs headless).

Continue to :doc:`step_4_validate_dataset`.
