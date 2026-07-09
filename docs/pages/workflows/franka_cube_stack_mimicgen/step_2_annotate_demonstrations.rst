Step 2: Annotate Demonstrations
-------------------------------

Before generation, each source demonstration must be annotated with **subtask termination
signals**: the action indices where one subtask ends and the next begins. The subtasks and
their signal names are declared by the task descriptor (see
:doc:`../../concepts/task_descriptors`) — for this task, ``grasp_1``, ``stack_1``, and
``grasp_2`` (the final subtask ends with the trajectory and needs no signal).

Isaac AutoData supports two annotation modes:

* **Manual** — replay each episode in the viewer and mark boundaries with the keyboard.
* **Automatic** (``--auto``) — sample the environment's boolean subtask-term observations
  during replay; each signal's first rising edge becomes the boundary. Runs headless.

Manual Annotation
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/dataset.hdf5 \
       --output_file ./datasets/annotated_dataset.hdf5

Each episode replays in the viewer, paused at the start. Control playback and mark boundaries
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

For this task, press ``S`` three times per episode: the moment the red cube is grasped
(``grasp_1``), the moment it rests on the blue cube (``stack_1``), and the moment the green
cube is grasped (``grasp_2``). Pause with ``B`` and resume with ``N`` to place marks
precisely.

If the number of marks does not match the expected count, the episode replays again for
re-marking. The task's success condition is also verified during replay — episodes that fail
it are not exported. Only fully annotated, successful episodes end up in the output file.

.. note::

   ``--env_name`` may be omitted, in which case the env id recorded in the source dataset is
   used. Manual annotation needs a window and therefore cannot run with ``--headless``.

Automatic Annotation
^^^^^^^^^^^^^^^^^^^^

If the environment publishes per-subtask boolean observation terms (this task's env does, in
the observation group ``subtask_terms``), annotation runs without a human in the loop:

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/dataset.hdf5 \
       --output_file ./datasets/annotated_dataset.hdf5 \
       --auto --headless

Each replay step samples the observation terms named by the descriptor's
``subtask_term_signal`` entries; a signal's first rising edge becomes the subtask boundary.
Episodes are skipped (with a printed reason) if a signal never fires, fires out of subtask
order, or violates the descriptor's ``subtask_term_offset_range`` spacing. The observation
group can be overridden with ``--signal_obs_group`` (default: ``subtask_terms``).

.. note::

   Only termination signals are auto-annotated. SkillGen datasets additionally need subtask
   *start* signals, which currently require manual annotation — see
   :doc:`../skillgen/index`.

Expected Output
^^^^^^^^^^^^^^^

An ``annotated_dataset.hdf5`` containing the successfully annotated episodes. Each episode
now carries one boolean ramp per signal under
``obs/datagen_info/subtask_term_signals/<name>`` — ``False`` until the subtask completes,
``True`` from that step onward — plus the per-step poses the generator needs
(``object_pose``, ``eef_pose``, ``target_eef_pose``).

Continue to :doc:`step_3_generate_dataset`.
