Welcome to Isaac AutoData!
==========================

``Isaac AutoData`` is a trajectory data-generation framework built on top of
`Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_ and
`Isaac Lab Arena <https://github.com/isaac-sim/IsaacLab-Arena>`_.
Given a handful of annotated human demonstrations, it synthesizes large datasets of new,
successful demonstrations by recombining and transforming demonstration segments —
using algorithms from the MimicGen family (MimicGen, DexMimicGen) and motion-planner-backed
generation (SkillGen).

.. todo::

   Add a hero figure/GIF here showing a generated demonstration
   (e.g. Franka cube stacking or a humanoid pick-and-place).

The Problem
===========

Imitation-learning policies are data hungry: they need large, diverse datasets of successful
demonstrations, and collecting those by human teleoperation is slow and expensive. Yet most of
what a policy needs to learn from a thousand demonstrations is already contained in ten — the
same skill, repeated under different object placements.

Isaac AutoData exploits that redundancy. A human demonstration is split into **subtasks**, each
a contiguous segment in which the robot's end-effector motion is driven by a single reference
object (reach the red cube, stack it on the blue cube, ...). Because each segment is
object-relative, it can be *transformed* to a new scene configuration and replayed. Stitching
transformed segments together — and keeping only the trials that actually succeed — turns a
handful of demonstrations into an arbitrarily large dataset.

The ideas come from the MimicGen line of work. What Isaac AutoData adds is an implementation
with explicit boundaries: the generation machinery is separated from the simulator and from the
robot by small, declarative interfaces (a task-descriptor YAML, an embodiment YAML, and a read
interface over the environment), so new tasks, robots, and generation algorithms can be added
independently of one another.

Isaac AutoData
==============

Four pieces cooperate to generate data:

* **Task descriptor** (YAML) — declares the task's subtasks per end-effector, the boundary
  signals that separate them, cross-arm constraints, and the generation policy (trial counts,
  seeding, export behavior). See :doc:`pages/concepts/task_descriptors`.
* **Embodiment** (YAML + adapter) — describes the robot from the generator's point of view:
  where to read end-effector poses and how to convert between target poses and the
  environment's action vector. See :doc:`pages/concepts/embodiments`.
* **Datastream** — the single read interface the generator uses to observe the world: object
  poses, end-effector poses, subtask signals, and the pool of annotated source demonstrations.
  See :doc:`pages/concepts/datastream`.
* **Generation algorithms** — MimicGen (single arm), DexMimicGen (two arms with coordination
  constraints), and SkillGen (motion-planned transit). See :doc:`pages/concepts/algorithms`.
  All three plug into one data generator — see :doc:`pages/concepts/data_generator`.

.. todo::

   Add an architecture diagram (source dataset -> pool -> algorithm -> env -> output dataset).

Usage Example
=============

Generating a dataset from annotated source demonstrations is a single command:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg mimicgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/annotated_dataset.hdf5 \
       --output_file ./datasets/generated_dataset.hdf5 \
       --generation_num_trials 100 \
       --num_envs 10 \
       --viz none

To get started, follow the instructions in :doc:`pages/quickstart/installation` and run your
first generation with :doc:`pages/quickstart/first_data_generation`.

License
=======

.. todo::

   Link the repository license once finalized.

TABLE OF CONTENTS
=================

.. toctree::
   :maxdepth: 1
   :caption: Set Up

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
   pages/concepts/embodiments
   pages/concepts/datastream
   pages/concepts/algorithms
   pages/concepts/data_generator

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   pages/advanced/motion_planners
   pages/advanced/testing_and_ci

.. toctree::
   :maxdepth: 1
   :caption: References

   pages/references/release_notes
