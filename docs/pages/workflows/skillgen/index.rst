SkillGen: Motion-Planned Data Generation
========================================

**SkillGen** augments MimicGen-style generation with collision-aware, GPU-accelerated motion
planning (`cuRobo <https://curobo.org/>`_). Instead of interpolating the end-effector between
subtasks, SkillGen plans a collision-free *transit* motion to each skill segment's start, then
replays the segment. Compared to plain MimicGen this gives:

* **Motion quality** — transitions are smooth, kinematically feasible robot motions instead of
  straight-line pose interpolations.
* **Collision awareness** — planned transits avoid the scene *and* whatever the gripper is
  holding; interpolation is blind to both.
* **Adaptability** — because transitions are planned fresh each trial, the same annotated
  demonstrations keep working when object placements move far from the source demonstration —
  or into cluttered scenes where interpolation would collide.

.. figure:: ../../../images/cube_stack_data_gen_skillgen.gif
   :width: 75%
   :align: center
   :alt: Franka cube stacking demonstrations generated with SkillGen

   Cube stacking demonstrations generated with SkillGen.

Task Overview
-------------

**Environment name:** ``Isaac-Stack-Cube-Franka-IK-Rel-v0``

**Task description:** A Franka arm stacks three cubes on a table — red on blue, then green on
red. This is the same environment as the :doc:`MimicGen workflow
<../franka_cube_stack_mimicgen/index>`; the SkillGen-specific end-effector frame is carried
entirely by the embodiment config, not by a dedicated task.

**Key specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Algorithm**
     - SkillGen (single end-effector, cuRobo motion planning)
   * - **Embodiment**
     - Franka, relative IK task-space actions (7-D: delta pose (xyz, rpy) + binary gripper
       open/close)
   * - **Task descriptor**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml>`
   * - **Embodiment config**
     - :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel_skillgen.yaml>`
   * - **Subtasks**
     - Grasp red cube (``grasp_1``) → stack red on blue (``stack_1``) → grasp green cube
       (``grasp_2``) → stack green on red (``stack_2``); each subtask also carries a start
       signal keyed by the same name
   * - **Pre-annotated source dataset**
     - ``isaac_autodata_tests/test_data/annotated_dataset_franka_stack_skillgen.hdf5``

How SkillGen Works
------------------

MimicGen bridges between subtasks by linearly interpolating the end-effector pose — fast, but
blind: the interpolated path can sweep through obstacles, and its quality degrades the further
the new object poses are from the source demonstration's. SkillGen splits every subtask into
two phases instead:

1. **Transit** — a cuRobo-planned, collision-free motion from the current end-effector pose
   to the (transformed) start of the subtask's skill segment. Grasped objects are attached to
   the robot's kinematic chain during planning, so a held cube is itself collision-checked.
2. **Skill** — the human demonstration segment, transformed and replayed as in MimicGen.

Each generation trial then runs this pipeline:

1. **Reset.** The environment resets and randomizes the cube placements; the initial state is
   snapshot for the output episode.
2. **Select and transform.** For the current subtask, a source skill segment is chosen
   (nearest-neighbor over reference-object poses by default) and rigidly transformed to the
   reference object's current pose. The segment's entry point is randomized within the
   descriptor's offset ranges.
3. **Plan.** The planner syncs its collision world to the live scene — attaching the held
   object, if the subtask expects one — and plans a collision-free transit from the current
   end-effector pose to the segment's start. If planning fails, the trial is abandoned and
   counted as a failure.
4. **Execute.** The transit waypoints are executed, then the skill segment replays with
   per-step action noise for diversity.
5. **Record.** The task's success condition is checked every step; successful trials are
   exported to the output dataset.

Because the planner needs to know exactly where skills begin, SkillGen requires **subtask
start signals** in addition to termination signals. In the task descriptor this means every
subtask — including the final one — must name a ``subtask_term_signal``: the name does double
duty as the key for that subtask's start signal. The final subtask itself still ends with the
trajectory, as in MimicGen, and never needs a termination mark. Cross-subtask constraints, if
any, apply only during the skill phase; transit is constraint-free.

Prerequisites
-------------

SkillGen requires the cuRobo container image. cuRobo's CUDA kernels are compiled for your
GPU architecture at image build time (auto-detected via ``nvidia-smi``; override with
``TORCH_CUDA_ARCH_LIST``):

:docker_run_curobo:

.. note::

   cuRobo kernels are compiled for the GPU you build on. If you later run on a different GPU
   generation, rebuild with ``./docker/run_docker.sh -c -r``.

A pre-annotated source dataset ships with the repository (see the table above), so you can
run the whole workflow without recording anything. To start from your own demonstrations
instead, record them exactly as in the MimicGen workflow
(:doc:`../franka_cube_stack_mimicgen/step_1_record_demonstrations`), then annotate them as
described below.

The Task Descriptor for SkillGen
--------------------------------

Compare :isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml>`
with the MimicGen version of the same task. The differences are characteristic:

* ``algo: skillgen`` and ``generation_policy.use_skillgen: true``.
* **Every** subtask names a ``subtask_term_signal`` — including the final one (``stack_2``).
  The name keys the subtask's start signal, which is why the final subtask needs one even
  though it still ends with the trajectory.
* ``num_interpolation_steps: 0`` — transit replaces interpolation.
* Per-subtask ``algo_params`` may set ``subtask_start_offset_range`` to randomize where the
  skill segment is entered.

Annotating for SkillGen
-----------------------

SkillGen datasets need a start mark for every subtask and a termination mark for every
subtask *except the last* (which ends with the trajectory). A skill segment should cover the
contact-rich part of a subtask — the final approach, the grasp, the placement — and
everything between a termination and the next start becomes plannable transit.

Annotation is manual: each episode replays in the simulator viewer, and you mark signals with
the keyboard.

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel_skillgen.yaml \
       --input_file ./datasets/dataset.hdf5 \
       --output_file ./datasets/annotated_dataset_skillgen.hdf5

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

With ``use_skillgen: true`` in the descriptor, the tool expects the marks to interleave —
start, termination, start, termination, ... — ending with the final subtask's start. For the
four-subtask cube-stack task that is **7 marks** per episode, in this order:

1. Start of ``grasp_1`` — the gripper begins its final approach to the red cube.
2. Termination of ``grasp_1`` — the red cube is securely grasped.
3. Start of ``stack_1`` — the placement onto the blue cube begins.
4. Termination of ``stack_1`` — the red cube rests on the blue cube, gripper released.
5. Start of ``grasp_2`` — the final approach to the green cube begins.
6. Termination of ``grasp_2`` — the green cube is securely grasped.
7. Start of ``stack_2`` — the placement onto the red cube begins.

.. tip::

   Pause with ``B`` a few steps before the skill begins, mark the start with ``S``, and
   resume with ``N``; once the interaction completes, pause again a few steps later and mark
   the termination. If the number of marks does not match the expected count, the episode
   simply replays for re-marking.

.. note::

   Automatic annotation (``--auto``) only produces termination signals and therefore cannot
   be used for SkillGen datasets yet. To skip annotation entirely, use the pre-annotated
   dataset that ships with the repository.

.. note::

   The SkillGen embodiment config differs from the MimicGen one only in ``eef_offset``
   (``[0, 0, 0.1034]``): the SkillGen source dataset is annotated in the Franka
   ``panda_hand`` frame rather than the inter-fingertip frame, and the offset reconciles the
   two. See :doc:`../../concepts/embodiments`.

Generating with SkillGen
------------------------

Start small to verify the setup, using the pre-annotated source dataset from the repository:

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
configuration until the trial target is met.

For a full-scale run, raise ``--generation_num_trials`` (hundreds to thousands for policy
training) and keep ``--viz none`` — rendering slows generation considerably. Expect SkillGen
to be slower per trial than MimicGen: every subtask transition is a motion-planning problem
solved at generation time.

Validate the generated dataset the same way as any other:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated_skillgen.hdf5

Visualizing and Debugging Plans
-------------------------------

The cuRobo backend can stream its planned trajectories and collision-sphere model to
`Rerun <https://rerun.io/>`_ — useful for diagnosing planning failures, unexpected detours,
or collision-world mismatches before committing to a long generation run.

.. figure:: ../../../images/rerun_cube_stack.gif
   :width: 80%
   :align: center
   :alt: Rerun visualization of planned trajectories and collision spheres

   Rerun visualization: planned end-effector trajectories with collision spheres.

Visualization is controlled by the ``visualize_plan`` and ``visualize_spheres`` flags of the
planner configuration (``CuroboPlannerCfg``); the shipped cube-stack preset enables plan
visualization by default. During multi-env generation only env 0 is visualized, to keep the
simulation responsive. The visualizer can also save the session as a ``.rrd`` recording for
offline inspection. Since visualization is independent of the simulator window, it works
together with ``--viz none``.

Motion Planner Configuration
----------------------------

The cuRobo planner (robot config, collision world, attached-object handling, planning seeds)
is resolved per task and is fully configurable — see :doc:`../../advanced/motion_planners`.
