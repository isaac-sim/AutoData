Step 1: Record Source Demonstrations
------------------------------------

Autodata consumes source demonstrations recorded as Isaac Lab HDF5 datasets (per-episode
actions, initial state, and observations). This step collects a small set of successful
teleoperated demonstrations of the cube-stacking task.

.. figure:: ../../../images/franka_cube_stacking_teleop.gif
   :width: 90%
   :align: center
   :alt: Teleoperating the Franka arm to stack cubes

.. note::

   To skip recording (and annotation), use the pre-annotated source dataset that ships with
   the repository (``datasets/annotated_datasets/dataset_franka_annotated.hdf5``)
   and jump to :doc:`step_3_generate_dataset`.

We recommend recording with a **SpaceMouse**: its smooth, off-axis 6-DoF input produces cleaner
demonstrations than a keyboard. If a SpaceMouse is unavailable, a keyboard also works — see
:ref:`franka-record-keyboard` below.


Recording with a SpaceMouse (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab supports the **SpaceMouse Wireless** and **SpaceMouse Compact** from
`3Dconnexion <https://3dconnexion.com/>`_.

**Set up the device (outside the dev container):**

#. Plug the SpaceMouse into the host machine.

#. Grant read/write access to its ``hidraw`` node. This one-liner locates the SpaceMouse by its
   device name (in ``/sys``) and ``chmod``\ s the matching ``/dev/hidraw*`` node(s) — no manual
   lookup needed:

   .. code-block:: bash

      grep -ilE '3dconnexion|spacemouse' /sys/class/hidraw/hidraw*/device/uevent \
          | grep -oE 'hidraw[0-9]+' \
          | xargs -r -I{} sudo chmod 666 /dev/{}

.. note::

   If the command above grants access to nothing, the device name may differ. Find the node manually by listing all
   available hidraw nodes with ``ls -l /dev/hidraw*``. Inspect each node's device name by running
   ``cat /sys/class/hidraw/hidraw<N>/device/uevent`` (for each ``<N>`` that was listed).
   Then ``sudo chmod 666 /dev/hidrawN`` on the one whose ``HID_NAME`` is the SpaceMouse.

**Start the dev container:**

:docker_run_default:

**Run the recording script**:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
      --viz kit \
      --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
      --teleop_device spacemouse \
      --dataset_file ./datasets/dataset_franka.hdf5 \
      --num_demos 10

The SpaceMouse controls the arm as follows:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Input
     - Action
   * - Tilt the SpaceMouse
     - Move arm along x / y axis
   * - Push / pull the SpaceMouse
     - Move arm along z axis
   * - Twist the SpaceMouse
     - Rotate arm
   * - Left button
     - Toggle gripper (open / close)
   * - Right button
     - Discard the current demonstration and reset

**Performing the Demonstrations**:

Stack in the following order: **red on blue, then green on red**.
A demonstration is auto-concluded as successful once the success condition is met
(cubes stacked in the correct order). If a demonstration goes wrong midway through,
discard it and reset by pressing the right button and start again.

Tips for demonstrations that generate (and train) well:

* **Keep them short.** Fewer decisions for the downstream policy, and shorter segments for
  the generator to transform.
* **Take a direct path.** Move straight toward the goal instead of following axes.
* **Don't have extended pauses.** Smooth, continuous motion is easier to learn than unexplained stops.

Collect 10 successful demonstrations of the cube-stacking task. The recording script shuts
down automatically after all 10 are recorded.

.. _franka-record-keyboard:

Recording with a Keyboard (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A keyboard can be used as an alternative if no SpaceMouse is available. No extra device setup is required.

**Start the dev container:**

:docker_run_default:

**Run the recording script**:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py \
      --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
      --viz kit \
      --teleop_device keyboard \
      --dataset_file ./datasets/dataset_franka.hdf5 \
      --num_demos 10

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Key
     - Action
   * - ``W/S``, ``A/D``, ``Q/E``
     - Move arm along x / y / z axis
   * - ``Z/X``, ``T/G``, ``C/V``
     - Rotate arm about x / y / z axis
   * - ``K``
     - Toggle gripper (open / close)
   * - ``R``
     - Discard the current demonstration and reset


Expected Output
^^^^^^^^^^^^^^^

Verify the ``datasets/dataset_franka.hdf5`` file contains the recorded episodes using:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/dataset_franka.hdf5

Continue to :doc:`step_2_annotate_demonstrations`.
