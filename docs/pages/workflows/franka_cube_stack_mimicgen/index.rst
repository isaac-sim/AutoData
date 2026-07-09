Franka Cube Stacking with MimicGen
==================================

This example demonstrates the complete Isaac AutoData workflow for the **Franka cube-stacking
task**: recording source demonstrations by teleoperation, annotating their subtask boundaries,
generating a large dataset with MimicGen, and validating the result.

.. figure:: ../../../images/franka_mimic_imitation_learning.jpg
   :width: 100%
   :align: center
   :alt: Franka robot performing the cube-stacking task

   The Franka cube-stacking task.

Task Overview
-------------

**Task ID:** ``Isaac-Stack-Cube-Franka-IK-Rel-v0``

**Task Description:** A Franka arm stacks three cubes on a table — red on blue, then green
on red.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Algorithm**
     - MimicGen (single end-effector)
   * - **Embodiment**
     - Franka, relative IK task-space actions (7-D: delta pose + gripper)
   * - **Task descriptor**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack.yaml>`
   * - **Embodiment config**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel.yaml>`
   * - **Subtasks**
     - Grasp red cube (``grasp_1``) → stack red on blue (``stack_1``) → grasp green cube
       (``grasp_2``) → place green on red (end of trajectory)
   * - **Pre-annotated source dataset**
     - ``isaac_autodata_tests/test_data/annotated_dataset_franka_stack_mimicgen.hdf5``

Workflow
--------

The pipeline goes from a handful of human demonstrations to a generated dataset ready for
policy training. You can follow the whole pipeline, or skip the first two steps by using the
pre-annotated dataset that ships with the repository.

Prerequisites
^^^^^^^^^^^^^

Start the dev container (see :doc:`../../quickstart/installation`):

:docker_run_default:

Create a folder for the datasets:

.. code-block:: bash

   mkdir -p datasets

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
