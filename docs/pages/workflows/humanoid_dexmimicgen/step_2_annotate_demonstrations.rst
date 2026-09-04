Step 2: Annotate Demonstrations
-------------------------------

Before AutoData generation, each source demonstration must be annotated with **subtask
termination signals**: the action indices where one subtask ends and the next begins. The subtasks
and their termination signal names are declared **per end-effector** by the task descriptor (see
:doc:`../../concepts/task_descriptors`).

For the humanoid pick-and-place task, the split is:

* **Left arm** — a single subtask spanning the whole trajectory, so it needs **no marks** (the end
  of the final subtask is always implicit).
* **Right arm** — two subtasks, so it needs **one mark**: the ``idle_right`` boundary, placed where
  the right arm finishes idling and begins moving toward the object. The final subtask (place) ends
  with the trajectory and needs no explicit signal.

Unlike the Franka cube-stacking task, the humanoid tasks are annotated **manually**: the episode
replays **once per end-effector**, the tool announces which arm's signals are being marked, and you
mark that arm's boundaries only. Arms whose subtask list needs no marks (the left arm here) are
skipped automatically.

Manual Annotation
^^^^^^^^^^^^^^^^^

**Start the dev container:**

:docker_run_default:

**Run the annotation script:**

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python scripts/annotate_demos.py \
            --env_name Isaac-PickPlace-GR1T2-Abs-v0 \
            --viz kit \
            --device cpu \
            --task_descriptor autodata_examples/tasks/gr1_pick_place.yaml \
            --embodiment autodata_examples/embodiments/gr1_ik_abs.yaml \
            --input_file ./datasets/dataset_gr1.hdf5 \
            --output_file ./datasets/dataset_gr1_annotated.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python scripts/annotate_demos.py \
            --env_name Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
            --viz kit \
            --device cpu \
            --task_descriptor autodata_examples/tasks/g1_pick_place.yaml \
            --embodiment autodata_examples/embodiments/g1_ik_abs.yaml \
            --input_file ./datasets/dataset_g1.hdf5 \
            --output_file ./datasets/dataset_g1_annotated.hdf5

Each episode replays in the Kit window and is paused at the start. The tool prints the arm currently
being annotated and its expected signals. Control playback and mark boundaries with the keyboard:

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

For each episode, press ``S`` **once** when the **right** arm finishes idling and
starts moving toward the object (the ``idle_right`` boundary).
The left-arm pass takes no marks and advances automatically. Pause with ``B`` and resume with ``N``
to place the mark precisely.

.. figure:: ../../../images/gr-1_pick_place_annotation.jpg
   :width: 100%
   :align: center
   :alt: Marking the idle_right subtask boundary during annotation

   A correct ``idle_right`` mark. The right arm has finished idling and begins moving toward the
   object.

If the number of marks does not match the expected count, the episode replays again for
re-marking. The task's success condition is also verified during replay — episodes that fail it are
not exported. Only fully annotated, successful episodes end up in the output file.

.. note::

   Automatic annotation (``--auto``) samples per-subtask boolean observation terms and is used for
   environments that publish them (like the Franka cube-stacking task). The humanoid pick-and-place
   environments do not publish automatic annotation observations so they are annotated manually as shown above.

Expected Output
^^^^^^^^^^^^^^^

Verify the annotated dataset contains the annotated episodes:

.. tabs::

   .. group-tab:: GR-1

      .. code-block:: bash

         python scripts/validate_dataset.py ./datasets/dataset_gr1_annotated.hdf5

   .. group-tab:: G1

      .. code-block:: bash

         python scripts/validate_dataset.py ./datasets/dataset_g1_annotated.hdf5

Continue to :doc:`step_3_generate_dataset`.
