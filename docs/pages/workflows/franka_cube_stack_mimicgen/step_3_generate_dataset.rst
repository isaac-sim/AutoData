Step 3: Generate the Dataset
----------------------------

With annotated source demonstrations in hand, MimicGen can synthesize new ones. For each trial,
the scene is randomized, a source segment is selected per subtask, and its end-effector
trajectory is rigidly transformed to the new object poses and replayed.

.. figure:: ../../../images/franka_mimicgen_datagen.jpg
   :width: 90%
   :align: center
   :alt: Parallel data generation for the Franka cube stacking task

   Parallel data generation for the Franka cube stacking task.

The commands below use the pre-annotated source dataset that ships with the repository. If you ran
Steps 1–2, point ``--input_file`` at your own ``dataset_franka_annotated.hdf5`` instead.


Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with a small scale run with simulation viewer enabled to sanity-check the setup:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --viz kit \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --num_envs 20 \
       --alg mimicgen \
       --generation_num_trials 10 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5 \
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

The script will automatically shutdown after 10
successful demonstrations are generated.


Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

For dataset-scale generation, run without the simulation viewer and with parallel environments:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --viz none \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --num_envs 1000 \
       --alg mimicgen \
       --generation_num_trials 1000 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5 \
       --output_file ./datasets/generated_dataset_mimicgen_franka.hdf5

Progress is printed after every attempt (successes / attempts and the running success rate).

.. note::

  **Expected data generation success rate and time**

  * Data generation success rate: ~40%
  * Data generation time: ~15 mins

  *Numbers are based on using an RTX PRO 6000 Blackwell GPU with the provided command.*


Key Parameters
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``--alg``
     - Generation algorithm: ``mimicgen``, ``dexmimicgen``, or ``skillgen``.
   * - ``--task``
     - Env id; if omitted, read from the source dataset's metadata.
   * - ``--generation_num_trials``
     - Overrides the task descriptor's ``generation_policy.num_trials``.
   * - ``--num_envs``
     - Number of parallel environments, each generating independently.
   * - ``--viz``
     - Visualizer backend (``kit`` for an Isaac Sim window, ``none`` for headless).

Continue to :doc:`step_4_validate_dataset`.
