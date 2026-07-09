Humanoid Pick & Place with DexMimicGen
======================================

This example demonstrates multi-end-effector data generation with **DexMimicGen** for humanoid
robots — the GR1T2 and the G1 — performing a pick-and-place task. DexMimicGen extends MimicGen
with per-arm subtask sequences and optional cross-arm coordination constraints.

.. todo::

   Add a task GIF (GR1/G1 pick-and-place).

Task Overview
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Algorithm**
     - DexMimicGen (two end-effectors)
   * - **Task IDs**
     - ``Isaac-PickPlace-GR1T2-Abs-v0`` (GR1),
       ``Isaac-PickPlace-Locomanipulation-G1-Abs-v0`` (G1)
   * - **Embodiments**
     - Whole-body IK with absolute-pose actions and 11-DOF dexterous hands per arm
   * - **Task descriptors**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/gr1_pick_place.yaml>`,
       :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/g1_pick_place.yaml>`
   * - **Embodiment configs**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/gr1_ik_abs.yaml>`,
       :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/g1_ik_abs.yaml>`
   * - **Pre-annotated source datasets**
     - ``isaac_autodata_tests/test_data/annotated_dataset_gr1_pick_place_dexmimicgen.hdf5``,
       ``isaac_autodata_tests/test_data/annotated_dataset_g1_pick_place_dexmimicgen.hdf5``

In the GR1 task, the right arm has two subtasks — approach and grasp the object
(``idle_right`` termination signal), then transport and place it (end of trajectory) — while
the left arm has a single full-trajectory subtask providing support motion. Segments are
selected and transformed **per arm**.

Generate a Dataset
------------------

Both humanoids use the same command shape; only the task id, descriptor, embodiment, and
source dataset change.

**GR1:**

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-PickPlace-GR1T2-Abs-v0 \
       --alg dexmimicgen \
       --task_descriptor isaac_autodata_examples/tasks/gr1_pick_place.yaml \
       --embodiment isaac_autodata_examples/embodiments/gr1_ik_abs.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_gr1_pick_place_dexmimicgen.hdf5 \
       --output_file datasets/generated_gr1_pick_place.hdf5 \
       --generation_num_trials 10 \
       --num_envs 1 \
       --viz none

**G1:**

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
       --alg dexmimicgen \
       --task_descriptor isaac_autodata_examples/tasks/g1_pick_place.yaml \
       --embodiment isaac_autodata_examples/embodiments/g1_ik_abs.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_g1_pick_place_dexmimicgen.hdf5 \
       --output_file datasets/generated_g1_pick_place.hdf5 \
       --generation_num_trials 10 \
       --num_envs 1 \
       --viz none

Annotating Multi-EEF Demonstrations
-----------------------------------

Annotation works exactly as in the :doc:`single-arm workflow
<../franka_cube_stack_mimicgen/step_2_annotate_demonstrations>`, except the episode replays
**once per end-effector**: the tool announces which arm's signals are being marked, and you
mark that arm's boundaries only. Arms whose subtask list needs no marks (like the GR1's left
arm, whose single subtask spans the whole trajectory) are skipped automatically.

What Makes Multi-EEF Generation Different?
------------------------------------------

* **Per-arm subtask sequences.** Each end-effector declares its own subtask list in the task
  descriptor (``subtasks.right``, ``subtasks.left``), each with its own reference objects,
  boundary signals, and generation knobs. Segment selection and transformation happen
  independently per arm; by default all arms reuse the demo picked by the first arm to
  choose (``generation_policy.select_src_per_arm: false``), keeping the arms' motions
  mutually consistent.
* **Coordination constraints.** For tasks where arms must interact — handovers, bimanual
  lifts — the descriptor's ``constraints`` section synchronizes specific subtask pairs across
  arms at runtime (``sequential`` ordering or ``coordination`` with a configurable scheme).
  The pick-and-place examples here need none. See :doc:`../../concepts/task_descriptors`.
* **Absolute-pose embodiments.** The humanoid adapters expose one pose slice and one set of
  hand-joint indices per arm (see :doc:`../../concepts/embodiments`); the action vector is
  reassembled from per-arm targets each step.
