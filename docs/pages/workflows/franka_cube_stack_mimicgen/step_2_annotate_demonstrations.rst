Step 2: Annotate Demonstrations
-------------------------------

Before Isaac AutoData generation, each source demonstration must be annotated with **subtask termination
signals**: the action indices where one subtask ends and the next begins. The subtasks and
their termination signal names are declared by the task descriptor (see
:doc:`../../concepts/task_descriptors`). For this task, the subtasks are ``grasp_1``, ``stack_1``, and
``grasp_2`` (the final subtask ends with the trajectory and needs no explicit signal).

Isaac AutoData supports two annotation modes:

* **Manual** — replay each episode in the simulator viewer and mark boundaries with the keyboard.
* **Automatic** (``--auto``) — sample the environment's boolean subtask-term observations
  during replay. each signal's first rising edge becomes the boundary. Runs headless. Requires
  the environment to publish per-subtask boolean observation terms.

Automatic annotation is recommended for the Franka cube stacking task as the environment supports it.


Automatic Annotation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Start the dev container:**

:docker_run_default:

**Run the annotation script in automatic mode:**

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --viz none \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/dataset_franka.hdf5 \
       --output_file ./datasets/dataset_franka_annotated.hdf5 \
       --auto

Each replay step samples the observation terms named by the task descriptor's
``subtask_term_signal`` entries (a signal's first rising edge becomes the subtask boundary).
Episodes are skipped (with a printed reason) if a signal never fires, fires out of subtask
order, or violates the descriptor's ``subtask_term_offset_range`` spacing.


Manual Annotation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotations can also be performed manually by replaying each episode in the simulator viewer
and marking boundaries with the keyboard.

.. note::

  Manual annotation is not required for this workflow if
  automatic annotation is used.

**Start the dev container:**

:docker_run_default:

**Run the annotation script in manual mode:**

.. code-block:: bash

   python scripts/annotate_demos.py \
      --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
      --viz kit \
      --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
      --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
      --input_file ./datasets/dataset_franka.hdf5 \
      --output_file ./datasets/dataset_franka_annotated.hdf5

Each episode replays in the viewer and is paused at the start. Control playback and mark boundaries
with the keyboard:

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
     - Mark a subtask signal at the current step
   * - ``Q``
     - Skip the current episode

For this task, press ``S`` three times per episode: the moment after the red cube is grasped
(``grasp_1``), the moment after it rests on the blue cube (``stack_1``), and the moment after the green
cube is grasped (``grasp_2``). Pause with ``B`` and resume with ``N`` to place marks
precisely.

If the number of marks does not match the expected count, the episode replays again for
re-marking. The task's success condition is also verified during replay — episodes that fail
it are not exported. Only fully annotated, successful episodes end up in the output file.


Expected Output
^^^^^^^^^^^^^^^

Verify the ``datasets/dataset_franka_annotated.hdf5`` file contains the annotated episodes using:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/dataset_franka_annotated.hdf5

Continue to :doc:`step_3_generate_dataset`.
