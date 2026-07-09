Your First Data Generation
==========================

This page runs a complete MimicGen data generation on the Franka cube-stacking task, starting
from a small annotated source dataset that ships with the repository (via Git LFS). It is the
fastest way to verify your setup end-to-end and to see what Isaac AutoData produces.

For the full pipeline — recording your own demonstrations, annotating them, and generating at
scale — see the :doc:`Franka cube stack workflow <../workflows/franka_cube_stack_mimicgen/index>`.

Prerequisites
-------------

* The dev container is running (see :doc:`installation`).
* Git LFS data is pulled — the source dataset lives at
  ``isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5``.

Generate a Dataset
------------------

From ``/workspaces/isaac_autodata`` inside the container:

.. code-block:: bash

   mkdir -p datasets

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5 \
       --output_file datasets/generated_dataset.hdf5 \
       --generation_num_trials 10 \
       --num_envs 10 \
       --viz none

While it runs, the console reports the number of source episodes loaded into the pool and a
running tally after every attempt:

.. code-block:: text

   Loaded 10 source episodes into the datagen pool

   **************************************************
   8/10 (80.0%) successful demos generated
   **************************************************

To watch the generation live, replace ``--viz none`` with ``--viz kit`` — note that rendering
slows generation down considerably, and a window requires a display.

.. figure:: ../../images/franka_datagen.jpg
   :width: 100%
   :align: center
   :alt: Parallel environments generating cube-stacking demonstrations

   Parallel environments generating cube-stacking demonstrations.

.. todo::

   State expected wall-clock time on reference hardware.

What Just Happened?
-------------------

Each generation trial walks the same loop:

1. **Reset & randomize** — the scene resets and object poses are randomized by the
   environment.
2. **Select** — for each subtask declared in the task descriptor
   (:doc:`../concepts/task_descriptors`), a source demonstration segment is chosen by that
   subtask's selection strategy (here: nearest-neighbor on the reference object's pose).
3. **Transform** — the segment's end-effector trajectory, expressed relative to its reference
   object, is rigidly transformed to the object's *current* pose.
4. **Execute** — the robot interpolates to the transformed segment's start, then replays it,
   with a small amount of action noise for diversity (:doc:`../concepts/algorithms`).
5. **Record** — the trial is checked against the task's success condition. With the example
   descriptor's ``generation_policy`` (``guarantee_success: true``, ``keep_failed: false``),
   generation keeps attempting until 10 *successful* demos are recorded, and only those are
   exported.

``--generation_num_trials`` overrides the descriptor's ``num_trials``; ``--num_envs`` runs
that many environments in parallel, each generating independently.

Inspect the Output
------------------

Validate the generated dataset's structure and metadata:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated_dataset.hdf5

The tool prints a summary table — episode count, the env id recorded in the file, and the
simulation args — followed by any structural issues (each episode must contain ``actions``,
``initial_state``, and ``obs``). Pass ``--strict`` to make it exit non-zero on any invalid
file, which is useful in scripts and CI.

Next Steps
----------

* Record and annotate your own demonstrations: :doc:`../workflows/franka_cube_stack_mimicgen/index`.
* Generate for a humanoid with two end-effectors: :doc:`../workflows/humanoid_dexmimicgen/index`.
* Use motion-planned transit for collision-aware generation: :doc:`../workflows/skillgen/index`.
