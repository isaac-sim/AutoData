Datastream
==========

The ``Datastream`` is the **single read interface** through which the data generator — and
every other consumer, such as motion planners and the annotation tool — observes the world.
It composes four things behind one object:

* the live **environment**,
* the **task descriptor** (subtasks, signals, constraints, generation policy),
* the **embodiment adapter** (pose reads and pose ↔ action transforms),
* the **source-demo pool** (the annotated demonstrations to generate from).

On construction it validates the pieces against each other: everything must be bound to the
same env, and the task descriptor and embodiment adapter must declare the same end-effector
names. It builds the source pool itself when given ``source_dataset_path`` (an HDF5 file), or
adopts a pre-built pool.

See :isaac_autodata_code_link:`<isaac_autodata_interfaces/datastream/datastream.py>` for the
full API.

Why a Read Facade?
------------------

The generation machinery should not know the simulator or the robot directly. By funneling
every read through one object, the generator stays free of ``isaaclab`` imports, motion
planners stay simulator-agnostic, and there is exactly one place to look when a value seems
wrong. Writes deliberately stay out: the Datastream never steps, resets, or records.

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
   * - Source-demo pool
     - ``source_pool``, ``datagen_infos``, ``subtask_boundaries``, ``num_source_demos``,
       ``add_episode``.

All poses are 4×4 matrices in the **env-relative frame** (per-env origin subtracted), which
is the one frame convention shared by the generator, the planners, and the recorded datasets.

What It Deliberately Does Not Wrap
----------------------------------

Controller-side operations stay on the env and are reached through the ``get_env()`` escape
hatch:

* ``env.step`` / ``env.reset`` — stepping is the simulator loop's job.
* ``env.recorder_manager`` — success flags and episode export are controller-side.
* The async action/reset queues used by multi-env generation.

Keeping mutation off the facade keeps the read surface honest: any code that changes the
world is easy to find, because it must go through ``get_env()``.

Who Reads It
------------

* The :doc:`data generator <data_generator>` reads all task, embodiment, and scene state
  through it — it never imports the simulator.
* **Motion planners** (SkillGen) read their collision geometry, obstacle poses, robot root
  pose, and joint start state through it — see :doc:`../advanced/motion_planners`.
* The **annotation tool** (``scripts/annotate_demos.py``) records per-step datagen info
  through it, and in ``--auto`` mode samples ``get_subtask_term_signals`` each replay step
  to find subtask boundaries without a human in the loop.
