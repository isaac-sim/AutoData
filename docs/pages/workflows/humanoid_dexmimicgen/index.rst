Humanoid Pick & Place with DexMimicGen
======================================

This example demonstrates the Isaac AutoData workflow using DexMimicGen to generate a
synthetic pick-and-place dataset for a bimanual humanoid robot. It covers recording source
demonstrations by Apple Vision Pro teleoperation, annotating their per-arm subtask boundaries,
generating a large dataset with DexMimicGen, and validating the result.

This tutorial supports two humanoids through the same workflow — the **Fourier GR-1** and the
**Unitree G1** (shown below). Throughout this tutorial, use the tabs to switch every
command on the page to the desired robot.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: ../../../images/gr1_pick_place_static.png
          :width: 100%
          :align: center
          :alt: GR-1 humanoid performing the pick-and-place task

          The GR-1 pick-and-place task.
     - .. figure:: ../../../images/g1_pick_place_static.png
          :width: 100%
          :align: center
          :alt: G1 humanoid performing the pick-and-place task

          The G1 pick-and-place task.


Task Overview
-------------

A humanoid grasps a steering wheel with its left hand, transfers it to its right hand, and places it
into a bin. Both arms are driven together (two end-effectors), which is
what makes this a DexMimicGen rather than a MimicGen task.

.. tabs::

   .. group-tab:: GR-1

      .. list-table::
         :widths: 30 70
         :header-rows: 1

         * - Property
           - Value
         * - **Algorithm**
           - DexMimicGen (two end-effectors)
         * - **Environment name**
           - ``Isaac-PickPlace-GR1T2-Abs-v0``
         * - **Embodiment**
           - Upper-body IK with absolute-pose actions and dexterous hands per arm
         * - **Task descriptor**
           - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/gr1_pick_place.yaml>`
         * - **Embodiment config**
           - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/gr1_ik_abs.yaml>`
         * - **Subtasks**
           - Right arm: idle then grasp (``idle_right``) → transport & place (``end of trajectory``).
             Left arm: grasp and transport (``end of trajectory``).
         * - **Pre-annotated source dataset**
           - ``isaac_autodata_tests/test_data/annotated_dataset_gr1_pick_place_dexmimicgen.hdf5``

   .. group-tab:: G1

      .. list-table::
         :widths: 30 70
         :header-rows: 1

         * - Property
           - Value
         * - **Algorithm**
           - DexMimicGen (two end-effectors)
         * - **Environment name**
           - ``Isaac-PickPlace-Locomanipulation-G1-Abs-v0``
         * - **Embodiment**
           - Upper-body IK with absolute-pose actions and dexterous hands per arm. Lower body balancing policy.
         * - **Task descriptor**
           - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/g1_pick_place.yaml>`
         * - **Embodiment config**
           - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/g1_ik_abs.yaml>`
         * - **Subtasks**
           - Right arm: idle then grasp (``idle_right``) → transport & place (``end of trajectory``).
             Left arm: grasp and transport (``end of trajectory``).
         * - **Pre-annotated source dataset**
           - ``isaac_autodata_tests/test_data/annotated_dataset_g1_pick_place_dexmimicgen.hdf5``


Why Multi-End-Effector Generation Differs
-----------------------------------------

Follow the :doc:`Franka cube-stacking workflow <../franka_cube_stack_mimicgen/index>` for an
overview of the single arm (MimicGen) case. This section highlights what changes for the
humanoid (DexMimicGen) case.

* **Per-arm subtask sequences.** Each end-effector declares its own subtask list in the task
  descriptor (``subtasks.right``, ``subtasks.left``), each with its own reference objects,
  boundary signals, and generation knobs. Segment selection and transformation happen
  independently per arm; by default all arms reuse the demo picked by the first arm to choose
  (``generation_policy.select_src_per_arm: false``), keeping the arms' motions mutually
  consistent.
* **Coordination constraints.** For tasks where the arms must interact (handovers, bimanual
  lifts) the task descriptor's ``constraints`` section synchronizes specific subtask pairs across
  arms at runtime. The pick-and-place examples here need none. See
  :doc:`../../concepts/task_descriptors`.
* **Absolute-pose embodiments.** The humanoid embodiment adapters expose one pose slice and one set of
  hand-joint indices per arm (see :doc:`../../concepts/embodiments`). The action vector is
  reassembled from per-arm targets each step.


Workflow
--------

The tutorial covers the pipeline to go from a handful of human demonstrations to a large
synthetically generated dataset ready for policy training. You can follow the whole pipeline, or
skip directly to :doc:`step_3_generate_dataset` using the pre-annotated dataset that ships with the repository.


Prerequisites
^^^^^^^^^^^^^

Start the dev container (see :doc:`../../quickstart/installation`):

:docker_run_default:

Recording humanoid demonstrations additionally requires an **Apple Vision Pro** and the CloudXR
runtime. :doc:`step_1_record_demonstrations` covers the setup and links the Isaac Lab teleop
guides. If you don't have an Apple Vision Pro, skip to :doc:`step_3_generate_dataset` and use the
pre-annotated source dataset.


Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_record_demonstrations`
- :doc:`step_2_annotate_demonstrations`
- :doc:`step_3_generate_dataset`
- :doc:`step_4_validate_dataset`

.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_record_demonstrations
   step_2_annotate_demonstrations
   step_3_generate_dataset
   step_4_validate_dataset
