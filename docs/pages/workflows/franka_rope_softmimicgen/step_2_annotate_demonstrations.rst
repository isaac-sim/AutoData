.. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Step 2: Annotate Demonstrations
-------------------------------

Annotation replays the recorded actions and adds the data SoftMimicGen needs: end-effector
poses, rope simulation-node positions, and the ``grasp`` subtask termination signal.
The second subtask, bringing the rope ends together, ends with the trajectory and needs
no explicit termination signal.

Use the same physics backend that recorded the source dataset. The AutoData annotation
script registers the rope environment internally, so it does not need the external callback
used by Isaac Lab's recording tool.


Automatic Annotation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rope environment publishes a ``grasp`` observation. With ``--auto``, annotation uses its
first rising edge as the boundary between grasping and manipulating the rope.

Start the dev container, then run the command for your backend:

:docker_run_default:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/recorded_dataset_franka_rope_newton.hdf5 \
             --output_file ./datasets/softmimicgen/annotated_dataset_franka_rope_newton.hdf5 \
             --auto \
             --device cuda \
             --enable_cameras \
             --viz none

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/recorded_dataset_franka_rope_physx.hdf5 \
             --output_file ./datasets/softmimicgen/annotated_dataset_franka_rope_physx.hdf5 \
             --auto \
             --device cuda \
             --enable_cameras \
             --viz none

The script exports only episodes that complete the task during replay and have a valid
``grasp`` boundary with enough steps left for the task descriptor's offset range. Skipped
episodes are reported in the console. To watch automatic annotation, use ``--viz kit``.

.. note::

   Annotation automatically disables simulation-buffer resets for Newton and enables them
   for PhysX. The explicit ``--no-reset_sim_buffer_each_episode`` flag used during Newton
   recording is unnecessary here.


Manual Annotation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To mark the boundary yourself, omit ``--auto`` and use ``--viz kit``:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets newton \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/recorded_dataset_franka_rope_newton.hdf5 \
             --output_file ./datasets/softmimicgen/annotated_dataset_franka_rope_newton.hdf5 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Marking the rope grasp boundary during Newton annotation.

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name Isaac-Rope-Franka-Arena-IK-Rel-v0 \
             --presets physx \
             --task_descriptor autodata_examples/tasks/franka_rope_softmimicgen.yaml \
             --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
             --input_file ./datasets/softmimicgen/recorded_dataset_franka_rope_physx.hdf5 \
             --output_file ./datasets/softmimicgen/annotated_dataset_franka_rope_physx.hdf5 \
             --device cuda \
             --enable_cameras \
             --viz kit

      .. todo::

         Image placeholder: Marking the rope grasp boundary during PhysX annotation.

Each episode starts paused. Use these controls:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Key
     - Action
   * - ``N``
     - Begin / resume playback
   * - ``B``
     - Pause playback
   * - ``S``
     - Mark the ``grasp`` boundary at the current step
   * - ``Q``
     - Skip the current episode

Press ``S`` once per episode, just after the gripper has grasped the rope and before it
starts moving the rope toward the other end. Pause and resume as needed to place the mark.
Do not mark the end of the trajectory. Only fully annotated episodes that pass the replay
success check are exported.


Expected Output
^^^^^^^^^^^^^^^

Check the annotated dataset:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/validate_dataset.py \
             ./datasets/softmimicgen/annotated_dataset_franka_rope_newton.hdf5

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/validate_dataset.py \
             ./datasets/softmimicgen/annotated_dataset_franka_rope_physx.hdf5

In each exported episode, ``obs/datagen_info/object_nodal_position/object`` contains the
rope node positions, and ``obs/datagen_info/subtask_term_signals/grasp`` contains the
annotated boundary signal. Structural validation checks the general dataset layout;
SoftMimicGen checks the required nodal annotations when loading the source pool.

Continue to :doc:`step_3_generate_dataset`.
