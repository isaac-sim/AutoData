Task Descriptors
================

A **task descriptor** is a YAML file that declares everything Isaac AutoData needs to know
about a task: its subtasks per end-effector, the signals that mark subtask boundaries,
cross-arm constraints, and the generation policy. It is the single source of truth shared by
the annotation tool and the data generator — both load it, so annotations and generation can
never disagree about what the subtasks are.

Example descriptors live in
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack.yaml>` (MimicGen),
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/gr1_pick_place.yaml>`
(DexMimicGen, two arms), and
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml>`
(SkillGen).

Schema
------

.. code-block:: yaml

   name: <str>
   description: <str>          # optional
   algo: <str>                 # mimicgen | dexmimicgen | skillgen

   subtasks:
     <eef_name>:
       - object_ref: <str>            # reference object driving this subtask
         description: <str>
         subtask_start_signal: <str>  # optional; SkillGen only
         subtask_term_signal: <str>   # omitted on the final subtask (MimicGen/DexMimicGen)

         # algorithm-agnostic generation knobs:
         selection_strategy: <str>
         selection_strategy_kwargs: {<kwarg>: <value>}
         first_subtask_start_offset_range: [<int>, <int>]
         subtask_term_offset_range: [<int>, <int>]
         action_noise: <float>
         num_interpolation_steps: <int>
         num_fixed_steps: <int>
         apply_noise_during_interpolation: <bool>
         algo_params: {<kwarg>: <value>}   # optional, per-algorithm dataclass

   constraints:                # optional, cross-subtask constraints
     - constraint_type: <str>  # "sequential" | "coordination"
       eef_subtask_constraint_tuple: [[<eef>, <int>], [<eef>, <int>]]
       sequential_min_time_diff: <int>            # sequential only
       coordination_scheme: <str>                 # coordination only
       coordination_scheme_pos_noise_scale: <float>
       coordination_scheme_rot_noise_scale: <float>
       coordination_synchronize_start: <bool>

   generation_policy:          # optional
     name: <str>
     seed: <int>
     num_trials: <int>
     guarantee_success: <bool>
     keep_failed: <bool>
     use_skillgen: <bool>
     use_navigation_controller: <bool>
     select_src_per_subtask: <bool>
     select_src_per_arm: <bool>
     transform_first_robot_pose: <bool>
     interpolate_from_last_target_pose: <bool>

Descriptors are validated on load; a violation raises an error naming the offending field.

Subtasks
--------

A **subtask** is a contiguous segment of a demonstration in which the end-effector's motion
is driven by a single reference object (``object_ref``) — reach the red cube, stack it on the
blue cube. A new subtask begins whenever the reference object changes. During generation each
subtask's segment is transformed *in its reference object's frame*, which is what makes
recombining segments under new object placements possible.

Boundary signals:

* ``subtask_term_signal`` names the boolean signal whose rising edge ends the subtask. The
  final subtask of a sequence normally has none — it ends with the trajectory. (SkillGen is
  the exception: every subtask, including the last, needs one; the same name doubles as the
  start-signal key.)
* ``subtask_term_offset_range: [lo, hi]`` randomizes the boundary by a per-trial offset in
  that range, so generated demos do not all switch subtasks at identical steps. The pool
  validates at load time that worst-case offsets keep adjacent subtasks non-empty and
  non-overlapping.

Per-subtask generation knobs:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Field
     - Effect
   * - ``selection_strategy`` (+ ``_kwargs``)
     - How the source segment is picked each trial — see
       :doc:`algorithms`.
   * - ``first_subtask_start_offset_range``
     - Random offset for the very first subtask's start.
   * - ``action_noise``
     - Amplitude of per-step action noise during the segment (diversity knob).
   * - ``num_interpolation_steps``
     - Steps used to interpolate from the current pose to the segment start
       (MimicGen/DexMimicGen; SkillGen sets 0 and plans instead).
   * - ``num_fixed_steps``
     - Extra steps holding the interpolation target before the segment plays.
   * - ``apply_noise_during_interpolation``
     - Whether ``action_noise`` also applies while interpolating.
   * - ``algo_params``
     - Algorithm-specific extras; e.g. SkillGen's ``subtask_start_offset_range``.

Constraints (multi-EEF)
-----------------------

For two-arm tasks, the optional ``constraints`` section relates one arm's subtask to
another's at runtime. Each entry names the two (end-effector, subtask index) pairs it binds:

* ``sequential`` — one subtask must finish before the other may proceed, optionally with a
  minimum time separation (``sequential_min_time_diff``). Use for ordered hand-offs.
* ``coordination`` — the two subtasks execute in lockstep under a configurable scheme
  (with optional pose-noise scales and synchronized start). Use for bimanual moves that must
  stay in phase.

The shipped pick-and-place examples need no constraints — the GR1's left arm simply replays
a single support-motion subtask — but the machinery activates as soon as a ``constraints``
section is present. Constraints apply only during skill segments; SkillGen's planned transit
is constraint-free.

Generation Policy
-----------------

The ``generation_policy`` block configures the run as a whole:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Field
     - Effect
   * - ``seed``
     - Seeds ``random`` / ``numpy`` / ``torch`` for reproducibility.
   * - ``num_trials``
     - Demo target; ``--generation_num_trials`` on the CLI overrides it.
   * - ``guarantee_success``
     - ``true``: retry until ``num_trials`` *successful* demos are exported. ``false``: stop
       after ``num_trials`` attempts, whatever their outcome.
   * - ``keep_failed``
     - Also export failed trials (to a separate file) — useful when debugging a low success
       rate.
   * - ``use_skillgen``
     - Marks the task as SkillGen-annotated (start signals expected). Set automatically to
       match the chosen algorithm at generation time; meaningful for annotation.
   * - ``select_src_per_subtask``
     - ``true``: re-select a source demo for every subtask; ``false``: the first selection
       sticks for the whole episode.
   * - ``select_src_per_arm``
     - ``true``: each end-effector picks its own source demo; ``false``: all arms share the
       first arm's pick.
   * - ``transform_first_robot_pose``
     - Prepend the recorded start pose to every subtask's target sequence, anchoring
       interpolation to the robot's recorded pose instead of the controller target.
   * - ``interpolate_from_last_target_pose``
     - Non-first subtasks interpolate from the last *executed waypoint* rather than the live
       robot pose (default ``true``; suppresses drift accumulation).
