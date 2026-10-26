.. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Step 1: Record Source Demonstrations
------------------------------------

Record a small set of successful rope-manipulation demonstrations as Isaac Lab HDF5
datasets. Each episode includes the robot's actions, observations, and initial state,
including the rope's deformable state.

.. note::

   To skip recording and annotation, use the backend-specific pre-annotated source dataset
   in :doc:`step_3_generate_dataset`.


Recording with a SpaceMouse (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the SpaceMouse on the host using the device setup instructions in
:doc:`../franka_cube_stack_mimicgen/step_1_record_demonstrations`, then start the dev container:

:docker_run_default:

Run the recording command for your backend:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
             --external_callback autodata_examples.envs.register_environments \
             --task Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --teleop_device spacemouse \
             --no-reset_sim_buffer_each_episode \
             --dataset_file ./datasets/softmimicgen/recorded_dataset_franka_rope_newton.hdf5 \
             --num_demos 3 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Teleoperating the Franka arm to bring the rope ends together with Newton.

   .. group-tab:: PhysX

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
             --external_callback autodata_examples.envs.register_environments \
             --task Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --teleop_device spacemouse \
             --dataset_file ./datasets/softmimicgen/recorded_dataset_franka_rope_physx.hdf5 \
             --num_demos 3 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Teleoperating the Franka arm to bring the rope ends together with PhysX.

``--external_callback`` registers AutoData's Arena rope environment before Isaac Lab creates
it. Keep this option when recording or replaying with Isaac Lab's tools.

.. note::

   Newton recording uses ``--no-reset_sim_buffer_each_episode`` to preserve simulation
   buffers between attempts. PhysX uses the recorder's default buffer reset behavior.

The SpaceMouse controls are the same as in the cube-stacking workflow:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Input
     - Action
   * - Tilt / push / pull
     - Translate the end-effector
   * - Twist
     - Rotate the end-effector
   * - Left button
     - Toggle gripper (open / close)
   * - Right button
     - Discard the current demonstration and reset


Performing the Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Approach one end of the rope, close the gripper around it, then move it toward the other
end. Hold the ends close together until the recorder recognizes success. The recorder
exports successful demonstrations and exits automatically after three have been collected.

Use smooth, direct motions and avoid extended pauses. Confirm that the rope stays in the
gripper as it moves. If a grasp slips or the rope falls off the table, discard the attempt
and reset. Record several demonstrations across the randomized starting rope shapes.


Recording with a Keyboard (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replace ``--teleop_device spacemouse`` with ``--teleop_device keyboard`` in the command for
your backend. No additional device setup is required. Keep the other arguments, including
Newton's ``--no-reset_sim_buffer_each_episode``.

Use the keyboard controls printed by the recorder: ``W/S``, ``A/D``, ``Q/E`` translate the
arm; ``Z/X``, ``T/G``, ``C/V`` rotate it; ``K`` toggles the gripper; and ``R`` discards the
current demonstration and resets.


Expected Output
^^^^^^^^^^^^^^^

Check that the recorded file contains the successful episodes:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/validate_dataset.py \
             ./datasets/softmimicgen/recorded_dataset_franka_rope_newton.hdf5

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/validate_dataset.py \
             ./datasets/softmimicgen/recorded_dataset_franka_rope_physx.hdf5

Continue to :doc:`step_2_annotate_demonstrations`.
