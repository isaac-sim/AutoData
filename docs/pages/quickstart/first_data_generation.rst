Your First Data Generation
==========================

This page guides you through a quickstart to run data generation on a Franka robot for a cube-stacking task,
starting from a source dataset of pre-annotated human demonstrations that ships with the repository.

For the full pipeline — recording your own demonstrations, annotating them, and generating at
scale — see the :doc:`Franka cube-stacking workflow <../workflows/franka_cube_stack_mimicgen/index>`.


Prerequisites
-------------

* Installation is complete — repo cloned with submodules, NGC login, Docker with the NVIDIA
  Container Toolkit (see :doc:`installation`).
* Git LFS data is pulled — the source dataset lives at
  ``/datasets/annotated_datasets``.


Start the Dev Container
-----------------------

.. note::
   See :doc:`installation` for more details about the development container.

:docker_run_default:

It will take a few minutes to build the container the first time you run it.
Once built, the script will automatically drop you into a shell at ``/workspaces/autodata``
inside the container. All following commands are run from there.

Generate a Dataset
------------------

From ``/workspaces/autodata`` inside the container, run:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --viz kit \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --generation_num_trials 10 \
       --num_envs 10 \
       --task_descriptor autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_franka_quickstart.hdf5


While it runs, the console reports the number of source episodes loaded into the pool and a
running tally after every attempt to generate a new demonstration:

.. code-block:: text

   Loaded 10 source episodes into the datagen pool

   **************************************************
   8/10 (80.0%) successful demos generated
   **************************************************

The ``--viz kit`` option opens a Kit window showing the generation process.
If no window appears, follow :ref:`troubleshooting-display` or switch to ``--viz none``.

Wait for the script to complete after 10 successful demonstrations are recorded in an HDF5 dataset.
The Kit window closes automatically when generation is complete.

.. figure:: ../../images/franka_mimicgen_datagen.jpg
   :width: 100%
   :align: center
   :alt: Parallel environments generating cube-stacking demonstrations

   Parallel environments generating cube-stacking demonstrations.


What Just Happened?
-------------------

Each generation attempt runs the following steps:

1. **Reset & randomize** — the scene resets and object poses are randomized by the
   environment.
2. **Select** — for each subtask declared in the task descriptor
   (:doc:`../concepts/task_descriptors`), a source demonstration segment is chosen by that
   subtask's selection strategy (here: nearest-neighbor on the reference object's pose).
3. **Transform** — the segment's end-effector trajectory, expressed relative to its reference
   object, is rigidly transformed to the object's *current* pose.
4. **Execute** — the robot interpolates to the transformed segment's start, then replays it,
   with a small amount of action noise for diversity (:doc:`../concepts/algorithms`).
5. **Record** — the attempt is checked against the task's success condition.
   Generation keeps attempting until 10 *successful* demonstrations are recorded, and only those are
   exported.


Validate the HDF5 Dataset
-------------------------

Validate the generated HDF5 dataset's structure and metadata:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated_dataset_franka_quickstart.hdf5

The tool prints a summary table — episode count, the env id recorded in the file, and the
simulation args — followed by any structural issues detected.


Replay the Generated Demonstrations
-----------------------------------

Run the command to replay the generated demonstrations:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
       --viz kit \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --dataset_file datasets/generated_dataset_franka_quickstart.hdf5

Each episode resets the scene to its recorded initial state and steps through its recorded
actions — you should see the robot stack the cubes under a different object placement every
episode.


Next Steps
----------

* Record and annotate your own demonstrations: :doc:`../workflows/franka_cube_stack_mimicgen/index`.
* Generate for a humanoid with two end-effectors: :doc:`../workflows/humanoid_dexmimicgen/index`.
* Use motion-planned transit for collision-aware generation: :doc:`../workflows/skillgen/index`.
