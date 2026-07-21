Step 3: Generate the Dataset
----------------------------

With annotated source demonstrations in hand, DexMimicGen can synthesize new ones. For each trial,
the scene is randomized, a source segment is selected per arm for each subtask, and each arm's
end-effector trajectory is transformed to the new object poses and replayed. By default both arms
reuse the source demo chosen by the first arm, keeping their motions mutually consistent.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: ../../../images/gr1_pick_place_datagen.png
          :width: 100%
          :align: center
          :alt: GR-1 parallel data generation

          The GR-1 pick-and-place task data generation.
     - .. figure:: ../../../images/g1_pick_place_datagen.png
          :width: 100%
          :align: center
          :alt: G1 pick-and-place task data generation

          The G1 parallel data generation.

The commands below use the pre-annotated source dataset that ships with the repository. If you ran
Steps 1–2, point ``--input_file`` at your own ``dataset_gr1_annotated.hdf5`` /
``dataset_g1_annotated.hdf5`` instead.


Small-Scale Generation
^^^^^^^^^^^^^^^^^^^^^^

Start with a small run with the simulation viewer enabled to sanity-check the setup:

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python scripts/generate_dataset.py \
            --env_name Isaac-PickPlace-GR1T2-Abs-v0 \
            --viz kit \
            --device cpu \
            --num_envs 5 \
            --alg dexmimicgen \
            --generation_num_trials 10 \
            --task_descriptor isaac_autodata_examples/tasks/gr1_pick_place.yaml \
            --embodiment isaac_autodata_examples/embodiments/gr1_ik_abs.yaml \
            --input_file ./datasets/annotated_datasets/dataset_gr1_annotated.hdf5 \
            --output_file ./datasets/generated_dataset_dexmimicgen_gr1_small.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python scripts/generate_dataset.py \
            --env_name Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
            --viz kit \
            --device cpu \
            --num_envs 5 \
            --alg dexmimicgen \
            --generation_num_trials 10 \
            --task_descriptor isaac_autodata_examples/tasks/g1_pick_place.yaml \
            --embodiment isaac_autodata_examples/embodiments/g1_ik_abs.yaml \
            --input_file ./datasets/annotated_datasets/dataset_g1_annotated.hdf5 \
            --output_file ./datasets/generated_dataset_dexmimicgen_g1_small.hdf5

You will see the humanoid repeatedly attempt the task under new object placements. Failed attempts
are normal and are not exported.

While it runs, the console reports the number of source episodes loaded into the pool and a running
tally after every attempt to generate a new demonstration:

.. code-block:: text

   Loaded 5 source episodes into the datagen pool

   **************************************************
   5/10 (50.0%) successful demos generated
   **************************************************

The script shuts down automatically after 10 successful demonstrations are generated.


Full-Scale Generation
^^^^^^^^^^^^^^^^^^^^^

For dataset-scale generation, run without the simulation viewer and with parallel environments:

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python scripts/generate_dataset.py \
            --env_name Isaac-PickPlace-GR1T2-Abs-v0 \
            --viz none \
            --device cpu \
            --num_envs 50 \
            --alg dexmimicgen \
            --generation_num_trials 1000 \
            --task_descriptor isaac_autodata_examples/tasks/gr1_pick_place.yaml \
            --embodiment isaac_autodata_examples/embodiments/gr1_ik_abs.yaml \
            --input_file ./datasets/annotated_datasets/dataset_gr1_annotated.hdf5 \
            --output_file ./datasets/generated_dataset_dexmimicgen_gr1.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python scripts/generate_dataset.py \
            --env_name Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
            --viz none \
            --device cpu \
            --num_envs 50 \
            --alg dexmimicgen \
            --generation_num_trials 1000 \
            --task_descriptor isaac_autodata_examples/tasks/g1_pick_place.yaml \
            --embodiment isaac_autodata_examples/embodiments/g1_ik_abs.yaml \
            --input_file ./datasets/annotated_datasets/dataset_g1_annotated.hdf5 \
            --output_file ./datasets/generated_dataset_dexmimicgen_g1.hdf5

Progress is printed after every attempt (successes / attempts and the running success rate).

.. note::

  **Expected data generation success rate and time**

  * Data generation success rate: ~70%
  * Data generation time: ~40 mins

  *Numbers are based on using an RTX PRO 6000 Blackwell GPU with the provided command.*


Key Parameters
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``--alg``
     - Generation algorithm: ``mimicgen``, ``dexmimicgen``, or ``skillgen``. Use ``dexmimicgen``
       for these two-arm humanoids.
   * - ``--env_name``
     - Env id; if omitted, read from the source dataset's metadata.
   * - ``--generation_num_trials``
     - Overrides the task descriptor's ``generation_policy.num_trials``.
   * - ``--num_envs``
     - Number of parallel environments, each generating independently.
   * - ``--viz``
     - Visualizer backend (``kit`` for an Isaac Sim window, ``none`` for headless).

The descriptor's ``generation_policy.select_src_per_arm`` (``false`` by default) controls whether
each arm may draw from a different source demo; keeping it ``false`` keeps the arms coordinated
without subtask timing constraints. See :doc:`../../concepts/task_descriptors` for the full set of per-arm generation knobs.

Continue to :doc:`step_4_validate_dataset`.
