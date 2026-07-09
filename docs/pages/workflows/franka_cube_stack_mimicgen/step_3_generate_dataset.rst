Step 3: Generate the Dataset
----------------------------

With annotated source demonstrations in hand, MimicGen can synthesize new ones: for each trial
the scene is randomized, a source segment is selected per subtask, and its end-effector
trajectory is rigidly transformed to the new object poses and replayed.

Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with a small run in the viewer to sanity-check the setup:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_dataset.hdf5 \
       --output_file ./datasets/generated_dataset_small.hdf5 \
       --generation_num_trials 10 \
       --num_envs 1 \
       --viz kit

You should see the robot repeatedly attempt the task under new cube placements. Failed
attempts are normal — they are simply not exported (see the generation policy below).

Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

For dataset-scale generation, run without rendering and with parallel environments:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_dataset.hdf5 \
       --output_file ./datasets/generated_dataset.hdf5 \
       --generation_num_trials 1000 \
       --num_envs 10 \
       --viz none

.. figure:: ../../../images/franka_datagen.jpg
   :width: 100%
   :align: center
   :alt: Parallel data generation for the Franka cube-stacking task

   Parallel data generation for the Franka cube-stacking task.

Progress is printed after every attempt (successes / attempts and the running success rate).

.. todo::

   State expected throughput and success rate on reference hardware.

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
     - Overrides the descriptor's ``generation_policy.num_trials``.
   * - ``--num_envs``
     - Number of parallel environments, each generating independently.
   * - ``--pause_subtask``
     - Pause after every subtask for interactive debugging.
   * - ``--viz``
     - Visualizer backend (``kit`` for a window, ``none`` for headless).

How the trial count is interpreted is controlled by the descriptor's ``generation_policy``:
with ``guarantee_success: true`` (this task's default) generation retries until
``num_trials`` *successful* demos are exported; with ``false`` it stops after ``num_trials``
attempts regardless of outcome. ``keep_failed: true`` additionally exports failed trials to a
separate file — useful when debugging a low success rate. Runs are seeded
(``generation_policy.seed``) for reproducibility.

Continue to :doc:`step_4_validate_dataset`.
