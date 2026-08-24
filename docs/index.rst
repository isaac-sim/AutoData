Welcome to Isaac AutoData!
==========================

``Isaac AutoData`` is a trajectory data-generation framework built on top of
`Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_ and
`Isaac Lab-Arena <https://github.com/isaac-sim/IsaacLab-Arena>`_.
Given a handful of annotated human demonstrations, it uses parallel simulation environments to
synthesize large datasets of new demonstrations by transforming and recombining the human
demonstration segments.

.. figure:: images/autodata.gif
   :width: 100%
   :align: center
   :alt: isaac_autodata

   Isaac AutoData


The Problem
===========

Imitation-learning policies are data hungry. They need large, diverse datasets of successful
demonstrations, and collecting those by human teleoperation is slow and expensive. Yet most of
what a policy needs to learn from a thousand demonstrations is already contained in ten: the
same skill repeated under different object placements.

Isaac AutoData exploits that redundancy. A human demonstration is split into **subtasks** (each
a contiguous segment in which the robot's end-effector motion is driven by a single reference
object). Because each segment is object-relative, it can be *transformed* to a new scene
configuration and replayed. Stitching transformed segments together turns a
handful of demonstrations into an arbitrarily large dataset.


Isaac AutoData
==============

Four pieces cooperate to generate data:

* **Task descriptor** (YAML) — declares the task's subtasks per end-effector, the boundary
  signals that separate them, cross-arm constraints, and the generation policy (generation targets,
  seeding, export behavior). See :doc:`pages/concepts/task_descriptors`.
* **Embodiment** (YAML + adapter) — describes the robot from the generator's point of view:
  where to read end-effector poses and how to convert between target poses and the
  environment's action vector. See :doc:`pages/concepts/embodiments`.
* **Datastream** — the single read interface the generator uses to observe the world: object
  poses, end-effector poses, subtask signals, and the pool of annotated source demonstrations.
  See :doc:`pages/concepts/datastream`.
* **Generation algorithms** — MimicGen (single-arm), DexMimicGen (multi-arm), and SkillGen (motion-planned transit).
  See :doc:`pages/concepts/algorithms`.
  All three plug into one data generator — see :doc:`pages/concepts/data_generator`.

.. figure:: images/System_Architecture.svg
   :width: 100%
   :align: center
   :target: _images/System_Architecture.svg
   :alt: Isaac AutoData system architecture — contracts, typed data flow, modular generation and execution

   The Isaac AutoData architecture: declarative contracts feed the Datastream read interface,
   which the data generator and its algorithm plug-ins consume to produce waypoints, actions,
   and finally recorded HDF5 episodes.

.. .. todo::
..
..    Add an architecture diagram (source dataset -> pool -> algorithm -> env -> output dataset).


Usage Example
=============

Generating a dataset from annotated source demonstrations is a single command:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --viz kit \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output_file ./datasets/generated_dataset.hdf5 \
       --generation_num_trials 100 \
       --num_envs 10

To get started, follow the instructions in :doc:`pages/quickstart/installation` and run your
first generation with :doc:`pages/quickstart/first_data_generation`.


License
=======

Isaac AutoData is licensed under the `Apache License 2.0
<https://github.com/isaac-sim/Isaac-AutoData/blob/main/LICENSE.md>`_.


Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Set Up

   pages/quickstart/support_matrix
   pages/quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   pages/quickstart/first_data_generation

.. toctree::
   :maxdepth: 1
   :caption: Example Workflows

   pages/workflows/franka_cube_stack_mimicgen/index
   pages/workflows/humanoid_dexmimicgen/index
   pages/workflows/skillgen/index

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   pages/concepts/concept_overview
   pages/concepts/task_descriptors
   pages/concepts/environment_profiles
   pages/concepts/embodiments
   pages/concepts/datastream
   pages/concepts/algorithms
   pages/concepts/data_generator

.. toctree::
   :maxdepth: 1
   :caption: Migration from Isaac Lab Mimic

   pages/workflows/migrate_isaac_lab_mimic

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   pages/advanced/motion_planners
   pages/advanced/testing_and_ci

.. toctree::
   :maxdepth: 1
   :caption: References

   pages/references/troubleshooting
   pages/references/release_notes
