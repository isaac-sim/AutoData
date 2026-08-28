Datastream
==========

The ``Datastream`` is the **single read interface** through which the data generator — and
every other consumer, such as motion planners and the annotation tool — observes the world.
It composes four things behind one object:

* the live **environment**,
* the **task descriptor** (subtasks, signals, constraints, generation policy),
* the **embodiment adapter** (pose reads and pose ↔ action transforms),
* the **source demonstration pool** (the annotated demonstrations to generate from).

See :isaac_autodata_code_link:`<isaac_autodata_interfaces/datastream/datastream.py>` for the
full API.

Why a Read Facade?
------------------

The generation machinery should not know the simulator or the robot directly. By funneling
every read through one object, the generator stays free of ``isaaclab`` imports, motion
planners stay simulator-agnostic, and there is exactly one place to look when a value seems
wrong. Writes deliberately stay out: the Datastream never steps, resets, or records.

Construction
------------

This is how the generation entry point composes one (from ``scripts/generate_dataset.py``):

.. code-block:: python

   from isaac_autodata_interfaces.datastream import Datastream
   from isaac_autodata_interfaces.embodiments import embodiment_adapter_from_yaml
   from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor

   task_descriptor = TaskDescriptor.from_yaml("franka_cube_stack.yaml")
   embodiment_adapter = embodiment_adapter_from_yaml("franka_ik_rel.yaml")

   datastream = Datastream(
       env=env,                                    # the live, already-created env
       task_descriptor=task_descriptor,
       embodiment_adapter=embodiment_adapter,
       source_dataset_path="annotated_dataset.hdf5",
       uses_start_signals=False,                   # True for SkillGen datasets
   )

The constructor is where the pieces are checked against each other, so mismatches surface
immediately instead of mid-generation:

* The task descriptor and embodiment adapter are **bound to the same env** as the Datastream
  (both are bound automatically if not already).
* Both must declare **the same end-effector names** — a task descriptor written for
  ``franka`` cannot run on an embodiment exposing ``left``/``right``.
* Exactly one source of demonstrations is given: ``source_dataset_path`` (an HDF5 file the
  Datastream loads into a pool) **or** a pre-built ``source_pool``.
* ``uses_start_signals`` tells the pool how to parse subtask boundaries: from start *and*
  termination signals (SkillGen) or from termination signals alone. The CLI derives it from
  the chosen algorithm.

Frames and Conventions
----------------------

Every consumer of the Datastream sees the same conventions; when a pose looks wrong, check
against this list first:

* **Poses are 4×4 homogeneous matrices** (``torch.Tensor``), on the environment's compute
  device (``datastream.device``).
* **Object and robot-root poses are env-relative**: each environment's origin is subtracted,
  so poses are comparable across parallel envs and match the frame recorded in datasets.
  Motion planners exchange poses with the framework in this frame too.
* **Quaternions are (x, y, z, w)** wherever they appear — see the conventions note in
  :doc:`embodiments`.
* **Reads are batched**: methods take ``env_ids`` (a sequence of env indices) and return
  tensors shaped ``(len(env_ids), ...)``; passing ``None`` reads all envs.

What It Exposes
---------------

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - Query group
     - Representative methods
   * - Task semantics
     - ``get_subtasks``, ``get_task_constraints``, ``get_generation_policy``,
       ``get_object_refs``, ``get_term_signal_names``, ``get_start_signal_names``,
       ``get_eef_names``, ``get_expected_attached_object`` (delegated to the task
       descriptor).
   * - Embodiment
     - ``get_robot_eef_pose``, ``get_robot_joint_positions``, ``get_robot_joint_names``,
       ``target_eef_pose_to_action``, ``action_to_target_eef_pose``,
       ``actions_to_passthrough_actions`` (delegated to the
       :doc:`embodiment adapter <embodiments>`).
   * - Live scene reads
     - ``get_object_poses`` (env-relative rigid-object poses), ``get_robot_root_pose``,
       ``get_scene_state`` (the snapshot recorded as an episode's initial state),
       ``get_subtask_term_signals`` (per-subtask boolean terms from the observation buffer —
       what automatic annotation samples), ``device``.
   * - Collision-world source
     - ``get_usd_stage``, ``get_env_prim_path``, ``get_robot_prim_path``. Motion planners
       build their collision world from these instead of reaching into ``env.scene``, and
       pose-sync obstacles through ``get_object_poses``.
   * - Source demonstration pool
     - ``source_pool``, ``datagen_infos``, ``subtask_boundaries``, ``num_source_demos``,
       ``add_episode``.

The Source Demonstration Pool
-----------------------------

The pool (``DataGenInfoPool``) holds the annotated source demonstrations in the form the
generator consumes. For each episode it keeps one **per-step record** with these fields:

.. list-table::
   :widths: 30 26 44
   :header-rows: 1

   * - Field
     - Shape (per entry)
     - Meaning
   * - ``eef_pose``
     - ``{eef_name: [T, 4, 4]}``
     - Recorded end-effector poses, step by step.
   * - ``target_eef_pose``
     - ``{eef_name: [T, 4, 4]}``
     - The controller *target* poses recovered from the recorded actions (via the
       embodiment adapter's inverse transform).
   * - ``object_poses``
     - ``{object_name: [T, 4, 4]}``
     - Recorded object poses; subtask segments are re-expressed relative to their reference
       object using these.
   * - ``subtask_term_signals``
     - ``{subtask_name: [T]}``
     - Boolean per-step completion flags, written by annotation.
   * - ``subtask_start_signals``
     - ``{subtask_name: [T]}``
     - Boolean per-step start flags — present only in SkillGen datasets.
   * - ``passthrough_action``
     - ``{channel_name: [T, D]}``
     - The non-pose action channels (grippers, hands, extra channels), replayed verbatim.

These records are produced by the annotation tool, which replays each episode and records
them under ``obs/datagen_info`` in the output HDF5 — see the
:doc:`annotation step <../workflows/franka_cube_stack_mimicgen/step_2_annotate_demonstrations>`.

Alongside the records, the pool stores each episode's **subtask boundaries** — per
end-effector, a ``(start, end)`` index pair per subtask. Boundaries are derived from the
signals: a termination signal's rising edge ends its subtask (the final subtask ends with the
trajectory), and with start signals enabled (SkillGen), each subtask's start comes from its
own start-signal edge instead of the previous subtask's end. At load time the pool checks
that the boundaries stay valid under the descriptor's worst-case randomization offsets, so a
mis-annotated episode is rejected immediately with a named reason.

The pool can also **grow during generation** (``add_episode``): an asyncio lock guards
concurrent growth across the per-env generation tasks, and it is skipped entirely for the
common case of a static, fully pre-loaded pool.

What It Deliberately Does Not Wrap
----------------------------------

Controller-side operations stay on the env and are reached through the ``get_env()`` escape
hatch:

* ``env.step`` / ``env.reset`` — stepping is the simulator loop's job.
* ``env.recorder_manager`` — success flags and episode export are controller-side.
* The async action/reset queues used by multi-env generation.

Keeping mutation off the facade keeps the read surface honest: any code that changes the
world is easy to find, because it must go through ``get_env()``.

For Algorithm and Planner Authors
---------------------------------

If you are writing a generation algorithm or a motion-planner backend, treat the Datastream
as your only window into the world:

* **Read through the facade.** Task structure, robot state, object poses, and source demonstrations
  are all available through the query groups above — an algorithm that sticks to them runs
  unchanged if the simulator wiring changes.
* **``get_env()`` is an escape hatch, not a convenience.** Its legitimate uses are
  controller-side: the data generator reaches the recorder and reset queue through it.
  Reading world state through ``get_env()`` defeats the boundary.
* **Missing a read? Extend the facade.** When motion planners needed collision geometry,
  the answer was three new Datastream methods (``get_usd_stage``,
  ``get_env_prim_path``, ``get_robot_prim_path``) — not planners reaching into
  ``env.scene``. A new consumer with a new need should follow that pattern: add a narrow,
  documented read method instead of importing the simulator.

Who Reads It
------------

* The :doc:`data generator <data_generator>` reads all task, embodiment, and scene state
  through it — it never imports the simulator.
* **Motion planners** (SkillGen) read their collision geometry, obstacle poses, robot root
  pose, and joint start state through it — see :doc:`../advanced/motion_planners`.
* The **annotation tool** (``scripts/annotate_demos.py``) records per-step datagen info
  through it, and in ``--auto`` mode samples ``get_subtask_term_signals`` each replay step
  to find subtask boundaries without a human in the loop.
