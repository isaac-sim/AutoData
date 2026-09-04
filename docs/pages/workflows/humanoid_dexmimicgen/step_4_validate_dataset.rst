Step 4: Validate the Generated Dataset
--------------------------------------

Structural Validation
^^^^^^^^^^^^^^^^^^^^^

Check the generated HDF5 for structural problems and summarize its contents:

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python scripts/validate_dataset.py ./datasets/generated_dataset_dexmimicgen_gr1.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python scripts/validate_dataset.py ./datasets/generated_dataset_dexmimicgen_g1.hdf5

The validator prints one summary row per file (episode count, the env id recorded in the file's
metadata, and the simulation args) followed by per-file issues. Every episode is checked for the
required fields (``actions``, ``initial_state``, ``obs``).

For unreadable, invalid, or truncated files, see :ref:`troubleshooting-hdf5`.


Visual Validation
^^^^^^^^^^^^^^^^^

Replay generated episodes to inspect them visually using Isaac Lab's replay
tool. CPU simulation matches how the humanoid demonstrations were recorded:

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
            --task Isaac-PickPlace-GR1T2-Abs-v0 \
            --viz kit \
            --device cpu \
            --dataset_file ./datasets/generated_dataset_dexmimicgen_gr1.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
            --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
            --viz kit \
            --device cpu \
            --dataset_file ./datasets/generated_dataset_dexmimicgen_g1.hdf5

A good generated demonstration looks like a plausible human one — a clean, firm grasp and a stable
place, with both arms moving coherently. Common artifacts worth watching for are jerky transitions
at subtask boundaries (interpolation too short — raise ``num_interpolation_steps``) and grasps that
only just succeed (consider tightening ``subtask_term_offset_range`` or reducing ``action_noise`` in
the task descriptor).

.. tabs::

   .. group-tab:: GR-1

      .. figure:: ../../../images/gr1_pick_place.gif
         :width: 90%
         :align: center
         :alt: Replaying generated GR1 demonstrations

   .. group-tab:: G1

      .. figure:: ../../../images/g1_pick_place.gif
         :width: 90%
         :align: center
         :alt: Replaying generated G1 demonstrations

.. note::

   **Isaac Lab replay is PhysX non-deterministic.** Isaac Lab PhysX is not deterministically
   reproducible across an ``env.reset``, so replaying an episode's recorded actions from its saved
   initial state can diverge from the original. Some episodes may fail to reproduce success during
   replay even though **every** episode in the dataset was a successful demonstration at generation
   time. Each episode satisfied the environment's success condition when it was generated.
