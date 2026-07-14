Franka Cube Stacking with MimicGen
==================================

This example demonstrates the Isaac AutoData workflow using MimicGen to generate a synthetic dataset
for a Franka robot performing a cube stacking task. The workflow will cover recording source demonstrations by teleoperation,
annotating their subtask boundaries, generating a large dataset with MimicGen, and validating the result.

.. figure:: ../../../images/franka_mimic_imitation_learning.jpg
   :width: 100%
   :align: center
   :alt: Franka robot performing the cube stacking task

   The Franka cube stacking task.

Task Overview
-------------

**Environment name:** ``Isaac-Stack-Cube-Franka-IK-Rel-v0``

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
     - Franka, relative IK task-space actions (7-D: delta pose (xyz, rpy) + binary gripper open/close)
   * - **Task descriptor**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack.yaml>`
   * - **Embodiment config**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel.yaml>`
   * - **Subtasks**
     - Grasp red cube (``grasp_1``) → stack red on blue (``stack_1``) → grasp green cube
       (``grasp_2``) → place green on red (``end of trajectory``)
   * - **Pre-annotated source dataset**
     - ``datasets/annotated_datasets/dataset_annotated_franka.hdf5``

Workflow
--------

The tutorial covers the pipeline to go from a handful of human demonstrations to a large synthetically
generated dataset ready for policy training. You can follow the whole pipeline, or skip the first two steps by using the
pre-annotated dataset that ships with the repository.

Prerequisites
^^^^^^^^^^^^^

Start the dev container (see :doc:`../../quickstart/installation`):

:docker_run_default:

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
