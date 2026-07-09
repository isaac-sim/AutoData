Step 4: Validate the Generated Dataset
--------------------------------------

Structural Validation
^^^^^^^^^^^^^^^^^^^^^

Check the generated HDF5 for structural problems and summarize its contents:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated_dataset.hdf5

The validator prints one summary row per file — episode count, the env id recorded in the
file's metadata, and the simulation args — followed by per-file issues. Every episode is
checked for the required fields (``actions``, ``initial_state``, ``obs``). Multiple files and
globs work too, and ``--strict`` makes the exit code non-zero if any file is invalid:

.. code-block:: bash

   python scripts/validate_dataset.py --strict datasets/*.hdf5

Visual Validation
^^^^^^^^^^^^^^^^^

Replay generated episodes through the robot to inspect them visually, using Isaac Lab's
replay tool:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --dataset_file datasets/generated_dataset.hdf5

A good generated demonstration looks like a plausible human one: a direct approach, a clean
grasp, and a controlled place. Common artifacts worth watching for are jerky transitions at
subtask boundaries (interpolation too short — raise ``num_interpolation_steps``) and grasps
that only just succeed (consider tightening ``subtask_term_offset_range`` or reducing
``action_noise`` in the task descriptor).

Training a Policy
^^^^^^^^^^^^^^^^^

The generated HDF5 is a standard Isaac Lab dataset, so it plugs into the imitation-learning
pipelines documented by Isaac Lab (e.g. robomimic behavior cloning) and Arena (e.g. GR00T
fine-tuning) unchanged. See the
`Isaac Lab imitation-learning documentation
<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_
for the robomimic route.

.. todo::

   Once the recommended training route for AutoData-generated datasets is settled, document
   it here end-to-end (training command, checkpoint evaluation, expected success rates).
