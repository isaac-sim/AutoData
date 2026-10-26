.. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Franka Rope Manipulation with SoftMimicGen
==========================================

This example demonstrates the AutoData workflow using SoftMimicGen to generate a synthetic
dataset for a Franka robot manipulating a deformable rope. It covers recording source
demonstrations by teleoperation, annotating their subtask boundaries, generating a dataset,
and validating the result.

The same workflow supports **Newton** and **PhysX**. Use the tabs throughout the tutorial to
select a physics backend. Keep that backend consistent for recording, annotation, generation,
and replay. The commands use separate dataset filenames for each backend.

.. tabs::

   .. group-tab:: Newton

      .. todo::

         Image placeholder: Franka rope task overview with Newton.

   .. group-tab:: PhysX

      .. todo::

         Image placeholder: Franka rope task overview with PhysX.


Task Overview
-------------

**Environment name:** ``Isaac-Rope-Franka-Arena-IK-Rel-v0``

**Task Description:** A Franka arm grasps a deformable rope on a table and manipulates it
until its two ends are close together.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Algorithm**
     - SoftMimicGen (single end-effector in this example)
   * - **Physics backends**
     - Newton (``--presets newton``) or PhysX (``--presets physx``)
   * - **Embodiment**
     - Franka, relative IK task-space actions (7-D: delta position, axis-angle rotation, and gripper open/close)
   * - **Task descriptor**
     - :autodata_code_link:`<autodata_examples/tasks/franka_rope_softmimicgen.yaml>`
   * - **Embodiment config**
     - :autodata_code_link:`<autodata_examples/embodiments/franka_ik_rel.yaml>`
   * - **Subtasks**
     - Approach and grasp the rope (``grasp``) → bring its ends together (``end of trajectory``)
   * - **Newton source dataset**
     - :autodata_code_link:`<datasets/annotated_datasets/dataset_franka_rope_newton_annotated.hdf5>`
   * - **PhysX source dataset**
     - :autodata_code_link:`<datasets/annotated_datasets/dataset_franka_rope_physx_annotated.hdf5>`


How SoftMimicGen Adapts Demonstrations
--------------------------------------

In the :doc:`cube-stacking workflow <../franka_cube_stack_mimicgen/index>`, MimicGen transforms
each source segment relative to a rigid object's pose. A rope can change shape, so its pose
alone does not describe the geometry the gripper needs to follow.

SoftMimicGen uses the rope's **simulation-node positions** to compare the source and current
rope shapes. It selects a source demonstration using registration cost, then fits a
**thin-plate-spline (TPS)** transform to adapt the source end-effector trajectory to the new
shape. Annotation records the nodal positions needed for this transform along with the
subtask signals. The source data and live rope must have matching node counts and topology.


Workflow
--------

Follow the whole pipeline, or skip recording and annotation by using the backend-specific
pre-annotated source dataset described in :doc:`step_3_generate_dataset`.

Prerequisites
^^^^^^^^^^^^^

Complete :doc:`../../quickstart/installation`, including downloading the Git LFS assets, and
start the dev container:

:docker_run_default:

Run the commands below from the repository root inside the container. Both backends use
``--device cuda``. A SpaceMouse is recommended for recording; a keyboard also works.

The examples enable cameras to include the task's fixed-view and wrist RGB observations.
``--viz kit`` opens the simulation window; ``--viz none`` runs without it. Camera recording
is controlled separately by ``--enable_cameras``.

Workflow Steps
^^^^^^^^^^^^^^

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
