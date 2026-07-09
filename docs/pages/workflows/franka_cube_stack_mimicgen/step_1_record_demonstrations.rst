Step 1: Record Source Demonstrations
------------------------------------

Isaac AutoData consumes source demonstrations recorded as Isaac Lab HDF5 datasets (per-episode
actions, initial state, and observations). This step collects a small set of successful
teleoperated demonstrations of the cube-stacking task.

.. note::

   To skip recording (and annotation), use the pre-annotated source dataset that ships with
   the repository — ``isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5``
   — and jump to :doc:`step_3_generate_dataset`.

Recording with Teleoperation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Demonstrations are recorded with Isaac Lab's recording tool, available inside the container
through the pinned submodule:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --teleop_device keyboard \
       --dataset_file ./datasets/dataset.hdf5 \
       --num_demos 10

For keyboard teleoperation, the controls are printed at startup (move with ``W/S``, ``A/D``,
``Q/E``; rotate with ``Z/X``, ``T/G``, ``C/V``; toggle the gripper with ``K``; reset with
``R``). For smoother trajectories, a SpaceMouse is recommended — set
``--teleop_device spacemouse``. See the
`Isaac Lab teleoperation documentation
<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_
for device setup details (SpaceMouse permissions, XR devices).

Stack in the order the task's success condition expects: **red on blue, then green on red**.
A demonstration is auto-concluded as successful after the success condition holds for
``--num_success_steps`` consecutive steps (default 10). If a demonstration goes wrong, press
``R`` to discard it and reset to a new starting configuration.

Tips for demonstrations that generate (and train) well:

* **Keep them short.** Fewer decisions for the downstream policy, and shorter segments for
  the generator to transform.
* **Take a direct path.** Move straight toward the goal instead of following axes.
* **Do not pause.** Smooth, continuous motion is easier to learn than unexplained stops.

About 10 successful demonstrations are enough for this task.

Expected Output
^^^^^^^^^^^^^^^

A ``datasets/dataset.hdf5`` file containing the recorded episodes. Check it:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/dataset.hdf5

Continue to :doc:`step_2_annotate_demonstrations`.
