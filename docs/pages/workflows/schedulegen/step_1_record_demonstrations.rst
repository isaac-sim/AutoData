..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Step 1: Record Source Demonstrations
====================================

Record successful demonstrations of the complete nut-pouring task. ScheduleGen will
reuse the contact-rich parts and replace the connecting motions with motion plans.
The recorder saves actions, observations and each episode's initial state in HDF5.

.. tip::

   No teleop hardware? Use the :ref:`included annotated dataset <schedulegen-source-data>`
   and continue to :doc:`step_3_generate_dataset`. Annotation of existing recordings
   requires only a keyboard and a Kit window, not a headset.

Before Recording
----------------

GR1T2 uses the existing IsaacTeleop wrist-pose and dexterous-hand retargeting pipeline.
Follow :ref:`the humanoid CloudXR setup <record_humanoid_cloudxr>` to start the runtime
and connect an Apple Vision Pro. These instructions do not add a new device integration.
Use the current IsaacTeleop hardware and network requirements linked there.

The task accepts 36 action values: two absolute end-effector poses (position [m] and
an XYZW quaternion), followed by 22 hand-joint targets [rad] in the embodiment's declared
order. A single-arm keyboard or SpaceMouse command is not a replacement for that bimanual
pose-and-finger stream.

Start the cuRobo development container from the repository root on the host:

.. code-block:: bash

   ./docker/run_docker.sh -c

Run the following commands inside the container. If CloudXR is already running in another
terminal, first load its environment as described in the humanoid workflow:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env

Record with Your Selected Backend
---------------------------------

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
             --task AutoData-NutPour-GR1T2-ScheduleGen-v0 \
             --external_callback autodata_examples.gr1t2_nut_pour.register \
             --physics_backend newton \
             --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
             --device cuda:0 \
             --viz kit \
             --step_hz 20 \
             --xr \
             --cloudxr_env avp \
             --no-auto_launch_cloudxr \
             --no-reset_sim_buffer_each_episode \
             --num_demos 5 \
             --dataset_file datasets/gr1t2_schedulegen/source_newton.hdf5

   .. group-tab:: PhysX

      .. code-block:: bash

         python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
             --task AutoData-NutPour-GR1T2-ScheduleGen-v0 \
             --external_callback autodata_examples.gr1t2_nut_pour.register \
             --physics_backend physx \
             --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
             --device cuda:0 \
             --viz kit \
             --step_hz 20 \
             --xr \
             --cloudxr_env avp \
             --no-auto_launch_cloudxr \
             --num_demos 5 \
             --dataset_file datasets/gr1t2_schedulegen/source_physx.hdf5

Keep the callback and profile in the command. They select the task, object layout,
controller settings and backend-specific hand/contact configuration. This example uses
``--physics_backend``; the Arena rope workflow's ``--presets`` flag is not interchangeable
with it. The recorder enables external cameras by default; the old Isaac Lab
``--enable_cameras`` recording flag is not needed.

For Newton, keep ``--no-reset_sim_buffer_each_episode``: restoring the scene and flushing
the solver's buffers are different operations. Use the same backend and profile for
recording, annotation and replay of your own demonstrations.

What to Demonstrate
-------------------

1. Grasp the red beaker with the left hand.
2. Pour into the yellow bowl and move the beaker clear.
3. Return the beaker to the bin and release it.
4. Pick up the bowl with the right hand, then place and release it on the scale.

The bowl hand can begin after pouring and clearance; it need not wait for the beaker to
reach the bin. Record clear skill boundaries, complete finger closures and releases,
and smooth motion. Avoid pushing the bowl during pouring or moving an object before
the grasp is secure. Discard failed grasps instead of treating them as source skills.

Start XR and recording using the controls in the humanoid workflow. The recorder exports
episodes only when the live task-success condition holds for its required number of steps.

Hardware-Free Checks
--------------------

A scripted stream of the same wrist-pose/finger commands can exercise the controller,
physics, reset and recorder without a headset. Such a check must use real task-success
evaluation and must not restore recorded states each step or supply motion-planner posture
targets. It checks the software after retargeting, not live hand tracking, calibration,
CloudXR networking or headset rendering. Those still require a hardware session.

Validate the recorded file, using the filename for your selected backend:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/gr1t2_schedulegen/source_newton.hdf5

Continue to :doc:`step_2_annotate_demonstrations`.
