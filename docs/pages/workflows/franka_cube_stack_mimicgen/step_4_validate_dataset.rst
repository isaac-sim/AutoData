Step 4: Validate the Generated Dataset
--------------------------------------

Structural Validation
^^^^^^^^^^^^^^^^^^^^^

Check the generated HDF5 for structural problems and summarize its contents:

.. code-block:: bash

   python scripts/validate_dataset.py ./datasets/generated_dataset_mimicgen_franka.hdf5

The validator prints one summary row per file (episode count, the env id recorded in the
file's metadata, and the simulation args) followed by per-file issues. Every episode is
checked for the required fields (``actions``, ``initial_state``, ``obs``):


Visual Validation
^^^^^^^^^^^^^^^^^

Replay generated episodes through the robot to inspect them visually, using Isaac Lab's
replay tool:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --viz kit \
       --num_envs 20 \
       --dataset_file ./datasets/generated_dataset_mimicgen_franka.hdf5

A good generated demonstration looks like a plausible human one (clean and firm
grasp, stable placement, etc). Common artifacts worth watching for are jerky transitions at
subtask boundaries (interpolation too short — raise ``num_interpolation_steps``) and grasps
that only just succeed (consider tightening ``subtask_term_offset_range`` or reducing
``action_noise`` in the task descriptor).

.. figure:: ../../../images/franka_mimicgen_replay.gif
   :width: 90%
   :align: center
   :alt: Replaying generated demonstrations

.. note::

   **Isaac Lab replay is PhysX non-deterministic.** Isaac Lab PhysX is not deterministically reproducible
   across an ``env.reset``, so replaying an episode's recorded actions from its saved initial
   state can diverge from the original. Some episodes may fail to reproduce success during replay
   even though **every** episode in the dataset was a successful demonstration at generation time.
   All recorded episode data in the HDF5 are still valid successes following the environment's success criterion.
