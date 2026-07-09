Motion Planners
===============

SkillGen plans collision-free transit motions with a pluggable motion-planner backend. The
shipped backend is `cuRobo <https://curobo.org/>`_ — GPU-accelerated, collision-aware
trajectory optimization. Planners live in ``isaac_autodata_interfaces/motion_planners/``.

The Planner Interface
---------------------

A backend implements ``MotionPlannerBase``, whose surface is deliberately small:

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Method
     - Purpose
   * - ``update_world_and_plan_motion(target_pose, ...)``
     - Sync the collision world to the current scene and plan to a target end-effector pose
       (including any expected attached object). Returns success.
   * - ``has_next_waypoint()`` / ``get_next_waypoint_ee_pose()``
     - Iterate the planned trajectory as end-effector poses.
   * - ``get_planned_poses()``
     - The full planned pose sequence.
   * - ``reset_plan()``
     - Discard the current plan.
   * - ``get_planner_info()``
     - Diagnostics for logging.

Planners read all world state — collision geometry, object poses, the robot's joint
configuration — through the shared :doc:`Datastream <../concepts/concept_overview>`; they
never touch the env or robot handles directly. During SkillGen generation, one planner is
constructed per environment (``--num_envs`` planners total).

cuRobo Backend
--------------

``CuroboPlanner`` is configured by ``CuroboPlannerCfg``. Configurations are resolved from
the task id:

.. code-block:: python

   from isaac_autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

   config = CuroboPlannerCfg.from_task_name("Isaac-Stack-Cube-Franka-IK-Rel-v0")

``from_task_name()`` pattern-matches the task id to a named preset (e.g.
``franka_config()``, ``franka_stack_cube_bin_config()``); unknown robots fall back to the
Franka preset with a printed warning. The ``generate_dataset.py`` entry point calls this
automatically when ``--alg skillgen`` is selected — no CLI flags needed.

Key configuration fields:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Field
     - Meaning
   * - ``robot_config_file`` / ``robot_name``
     - cuRobo robot model (kinematics, collision spheres).
   * - ``ee_link_name``
     - The link planned to the target pose.
   * - ``gripper_joint_names`` + ``gripper_open/closed_positions``
     - Gripper state applied during planning.
   * - ``attached_object_link_name`` / ``hand_link_names``
     - Where grasped objects attach on the kinematic chain, and which links to ignore for
       attached-object collision.
   * - ``world_config_file`` / ``static_objects`` / ``world_ignore_substrings``
     - The collision world: base description, scene objects treated as static, and prims to
       exclude.
   * - ``collision_checker_type``
     - Collision representation (mesh by default).
   * - ``num_trajopt_seeds`` / ``num_graph_seeds`` / ``trajopt_tsteps``
     - Planning effort: more seeds cost time but escape more local minima.
   * - ``interpolation_dt``
     - Time step of the returned trajectory.
   * - ``collision_activation_distance``
     - Buffer distance at which collision costs activate.
   * - ``approach_distance``
     - Straight-line approach segment length before the target.

Attached Objects
----------------

When a subtask's skill segment ends with the gripper holding an object, the next transit must
treat that object as part of the robot. The generator tells the planner which object it
expects attached (from the task descriptor's reference objects); the planner snapshots the
object's pose relative to the attach link and collision-checks the combined body throughout
the plan. On release, the attachment is dropped and the object returns to the world model.

Visualization and Debugging
---------------------------

The cuRobo backend can visualize its collision-sphere model and planned trajectories via
`rerun <https://rerun.io/>`_ (``visualize_spheres`` / ``visualize_plan`` config flags).
During multi-env generation these are enabled only for env 0 to keep the simulation
responsive. Planner diagnostics (success rates, timing) are available through
``get_planner_info()``.

Adding a New Backend
--------------------

1. Subclass ``MotionPlannerBase`` and implement the methods above, reading world state
   through the Datastream passed at construction.
2. Construct your planners where SkillGen expects them — one per env id, exposing
   ``update_world_and_plan_motion(...)`` and ``get_planned_poses()`` (this is all
   ``SkillGen`` requires of a planner).
3. Wire construction into your entry point the way ``generate_dataset.py`` builds its cuRobo
   planners (``_build_motion_planners``).
