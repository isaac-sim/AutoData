SkillGen: Motion-Planned Data Generation
========================================

**SkillGen** augments MimicGen-style generation with collision-aware, GPU-accelerated motion
planning (`cuRobo <https://curobo.org/>`_). Instead of interpolating the end-effector between
subtasks, SkillGen plans a collision-free *transit* motion to each skill segment's start, then
replays the segment. This produces higher-quality data in cluttered scenes and enables tasks
where naive interpolation would collide.

What is SkillGen?
-----------------

MimicGen bridges between subtasks by linearly interpolating the end-effector pose — fast, but
blind: the interpolated path can sweep through obstacles, and its quality degrades the further
the new object poses are from the source demonstration's. SkillGen splits every subtask into
two phases instead:

1. **Transit** — a cuRobo-planned, collision-free motion from the current end-effector pose
   to the (transformed) start of the subtask's skill segment. Grasped objects are attached to
   the robot's kinematic chain during planning, so a held cube is itself collision-checked.
2. **Skill** — the human demonstration segment, transformed and replayed as in MimicGen.

Because the planner needs to know exactly where skills begin and end, SkillGen requires
**subtask start signals** in addition to termination signals — including a termination signal
on the *final* subtask (unlike MimicGen, where the final subtask simply ends with the
trajectory). Cross-subtask constraints, if any, apply only during the skill phase; transit is
constraint-free.

Prerequisites
-------------

SkillGen requires the cuRobo container image. cuRobo's CUDA kernels are compiled for your
GPU architecture at image build time (auto-detected via ``nvidia-smi``; override with
``TORCH_CUDA_ARCH_LIST``):

:docker_run_curobo:

.. note::

   cuRobo kernels are compiled for the GPU you build on. If you later run on a different GPU
   generation, rebuild with ``./docker/run_docker.sh -c -r``.

The Task Descriptor for SkillGen
--------------------------------

Compare :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml>`
with the MimicGen version of the same task. The differences are characteristic:

* ``algo: skillgen`` and ``generation_policy.use_skillgen: true``.
* **Every** subtask has a ``subtask_term_signal`` — including the final one (``stack_2``) —
  so the planner knows when the last skill segment finishes. Each subtask's start signal is
  keyed by the same name.
* ``num_interpolation_steps: 0`` — transit replaces interpolation.
* Per-subtask ``algo_params`` may set ``subtask_start_offset_range`` to randomize where the
  skill segment is entered.

Annotating for SkillGen
-----------------------

SkillGen datasets need a start *and* a termination mark per subtask. Annotate in manual mode
— with ``use_skillgen: true`` in the descriptor, the tool expects the marks to interleave:
start, termination, start, termination, ... for each subtask in order:

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel_skillgen.yaml \
       --input_file ./datasets/dataset.hdf5 \
       --output_file ./datasets/annotated_dataset_skillgen.hdf5

Mark a subtask's *start* where the skill's contact-rich part begins (e.g. the final approach
to the cube) and its *termination* where that interaction completes (e.g. the cube is grasped
or released) — everything between a termination and the next start becomes plannable transit.

.. note::

   Automatic annotation (``--auto``) only produces termination signals and therefore cannot
   be used for SkillGen datasets yet. A pre-annotated SkillGen source dataset ships with the
   repository (see below).

.. note::

   The SkillGen embodiment config differs from the MimicGen one only in ``eef_offset``
   (``[0, 0, 0.1034]``): the SkillGen source dataset is annotated in the Franka
   ``panda_hand`` frame rather than the inter-fingertip frame, and the offset reconciles the
   two. See :doc:`../../concepts/embodiments`.

Generating with SkillGen
------------------------

Using the pre-annotated source dataset from the repository:

.. code-block:: bash

   python isaac_autodata_examples/generate_dataset.py \
       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --alg skillgen \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel_skillgen.yaml \
       --input_file isaac_autodata_tests/test_data/annotated_dataset_franka_stack_skillgen.hdf5 \
       --output_file datasets/generated_skillgen.hdf5 \
       --generation_num_trials 10 \
       --num_envs 1 \
       --viz none

One cuRobo planner is created per environment (they are auto-wired when ``--alg skillgen``
is selected), so ``--num_envs`` trades GPU memory for throughput. When motion planning fails
for a trial — no collision-free path to the skill start — the trial is abandoned and counted
as a failure; with ``guarantee_success: true`` generation simply retries with a new scene
configuration.

Motion Planner Configuration
----------------------------

The cuRobo planner (robot config, collision world, attached-object handling, planning seeds)
is resolved per task and is fully configurable — see :doc:`../../advanced/motion_planners`.
