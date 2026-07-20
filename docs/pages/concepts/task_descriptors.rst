Task Descriptors
================

A **task descriptor** is a YAML file that declares everything Isaac AutoData needs to know
about a task: its subtasks per end-effector, the signals that mark subtask boundaries,
cross-arm constraints, and the data generation policy for the task. It is the single source of truth shared by
the annotation tool and the data generator. Both load it, so annotations and generation can
never disagree about what the subtasks are.

Example descriptors live in
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack.yaml>` (MimicGen),
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/gr1_pick_place.yaml>`
(DexMimicGen, two arms), and
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack_skillgen.yaml>`
(SkillGen).

YAML Schema
-----------

.. code-block:: yaml

   name: <str>                 # task name
   description: <str>          # human-readable description of the task
   algo: <str>                 # mimicgen | dexmimicgen | skillgen

   subtasks:
     <eef_name>:
       - object_ref: <str>
         description: <str>
         subtask_start_signal: <str>
         subtask_term_signal: <str>

         # data generation parameters for the subtask:
         selection_strategy: <str>
         selection_strategy_kwargs: {<kwarg>: <value>}
         first_subtask_start_offset_range: [<int>, <int>]
         subtask_term_offset_range: [<int>, <int>]
         action_noise: <float>
         num_interpolation_steps: <int>
         num_fixed_steps: <int>
         apply_noise_during_interpolation: <bool>
         algo_params: {<kwarg>: <value>}   # optional, additional per-algorithm params

       - ...

   # optional, cross-subtask constraints for multi-arm tasks
   constraints:
     - constraint_type: <str>
       eef_subtask_constraint_tuple: [[<eef>, <int>], [<eef>, <int>]]
       sequential_min_time_diff: <int>
       coordination_scheme: <str>
       coordination_scheme_pos_noise_scale: <float>
       coordination_scheme_rot_noise_scale: <float>
       coordination_synchronize_start: <bool>

   # data generation parameters for the task:
   generation_policy:
     name: <str>
     seed: <int>
     num_trials: <int>
     guarantee_success: <bool>
     keep_failed: <bool>
     use_skillgen: <bool>
     select_src_per_subtask: <bool>
     select_src_per_arm: <bool>
     transform_first_robot_pose: <bool>
     interpolate_from_last_target_pose: <bool>

Descriptors are validated on load and a violation raises an error naming the offending field.

Subtasks
--------

A **subtask** is a contiguous segment of a demonstration in which an end-effector's motion is
dictated by a single object, such as reaching for a cube or placing it on another cube. A subtask
ends and the next one begins when the object dictating the motion changes. During
generation, AutoData selects a recorded subtask segment, transforms it into the current scene, connects the
robot's current trajectory to the segment, and replays it.

The ``subtasks`` section of the YAML schema groups subtasks by ``<eef_name>``. The eef name key must exactly match
an end-effector name exposed by the embodiment descriptor, such as ``franka`` or ``left`` / ``right``.
Declaration order is execution order and determines the integer subtask indices used by constraints.

.. list-table::
   :widths: 28 16 56
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``<eef_name>``
     - Required
     - End-effector whose ordered subtask list follows. It must match the embodiment descriptor.
       MimicGen and SkillGen expect one EEF; DexMimicGen expects at least two.
   * - ``object_ref``
     - ``str: ""``
     - Name of the object whose pose defines the subtask's reference frame. The name must match a Datastream
       object-pose key.
   * - ``description``
     - ``str: ""``
     - Human-readable summary exposed to tools and agents. It does not affect generation.
   * - ``subtask_start_signal``
     - ``str: ""``
     - Subtask start-signal name. SkillGen only.
   * - ``subtask_term_signal``
     - ``str: ""``
     - Subtask end-signal name. Required for every non-final subtask.
   * - ``first_subtask_start_offset_range``
     - ``[0, 0]``
     - Inclusive start-index offset range ``[lo, hi]`` for the first subtask of an EEF.
       Starts the first subtask at a random index within the range.
   * - ``subtask_term_offset_range``
     - ``[0, 0]``
     - Inclusive termination-boundary offset range ``[lo, hi]``. Negative values move the
       boundary earlier and positive values move it later. The final subtask must use ``[0, 0]``.
       Offsets are validated to keep segments non-empty and correctly ordered.
   * - ``selection_strategy``
     - ``random``
     - Strategy used to choose the source demonstration for this segment. See the table below.
   * - ``selection_strategy_kwargs``
     - ``{}``
     - Keyword arguments passed directly to ``selection_strategy``. Unknown or unsupported values
       fail during generation. See the table below.
   * - ``action_noise``
     - ``0.0``
     - Scale of independent Gaussian noise applied to pose-action values during data generation.
   * - ``num_interpolation_steps``
     - ``0``
     - Number of bridge steps from the current starting pose to the transformed segment's first pose.
   * - ``num_fixed_steps``
     - ``0``
     - Extra bridge steps that hold the subtasks's first target pose before replay begins.
   * - ``apply_noise_during_interpolation``
     - ``false``
     - Whether ``action_noise`` also applies to interpolation and fixed-hold steps. The replayed
       source segment receives noise regardless of this flag.
   * - ``algo_params``
     - ``{}``
     - Algorithm-specific fields. MimicGen and DexMimicGen accept no additional fields; SkillGen's
       supported field is listed below. Unknown keys are rejected when the descriptor loads.

**Selection strategies**

.. list-table::
   :widths: 29 45 26
   :header-rows: 1

   * - ``selection_strategy``
     - Selection behavior
     - Supported ``selection_strategy_kwargs``
   * - ``random``
     - Selects uniformly from all source demonstrations. This is the only strategy allowed when
       ``object_ref`` is empty.
     - None.
   * - ``nearest_neighbor_object``
     - Compares each source segment's object pose at its first step with the object's current pose,
       then randomly chooses one of the ``nn_k`` nearest sources.
     - ``pos_weight`` (default ``1.0``), ``rot_weight`` (default ``1.0``), ``nn_k`` (default
       ``3``).
   * - ``nearest_neighbor_robot_distance``
     - Transforms each source segment's first end-effector pose into the current object frame,
       compares it with the robot's current end-effector pose, then randomly chooses one of the
       ``nn_k`` nearest sources.
     - ``pos_weight`` (default ``1.0``), ``rot_weight`` (default ``1.0``), ``nn_k`` (default
       ``3``).

Both nearest-neighbor strategies rank sources using
``pos_weight * position_distance + rot_weight * rotation_angle``. Position distance is Euclidean
distance (m), and rotation distance is the angular difference (rad). ``nn_k`` must be at least 1 and
is capped at the number of available source demonstrations. Use ``nn_k: 1`` to always choose the nearest source.

**SkillGen ``algo_params``**

.. list-table::
   :widths: 30 18 52
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``subtask_start_offset_range``
     - ``[0, 0]``
     - Inclusive start-boundary offset range ``[lo, hi]``. Negative values begin the skill
       earlier and positive values begin it later. Worst-case start and termination offsets are
       validated together. On the first subtask, this offset is added to
       ``first_subtask_start_offset_range``.


Constraints (multi-EEF)
-----------------------

The optional ``constraints`` list relates two subtasks in a multi-end-effector task. Each subtask is
identified by its end-effector name and zero-based index in that end-effector's ``subtasks`` list. Constraints
are used to ensure that subtasks are executed in a specific order or in coordination.
Constraints are currently supported only by DexMimicGen.

.. list-table::
   :widths: 30 16 54
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``constraints``
     - ``[]``
     - List of cross-subtask constraint mappings. Omit it when the task has no constraints.
   * - ``constraint_type``
     - Required
     - ``sequential`` or ``coordination``. String values are case-insensitive. See the constraint
       type table below.
   * - ``eef_subtask_constraint_tuple``
     - Required
     - Exactly two ``[eef_name, subtask_index]`` pairs. Names must match the descriptor's EEF keys,
       and indices are zero-based. Pair order matters for ``sequential`` constraints.
   * - ``sequential_min_time_diff``
     - ``-1``
     - Sequential only. ``-1`` holds the second subtask and prevents it from starting until the first finishes.
       A positive value lets the second execute until that many steps remain, then holds it until
       the first finishes. ``0`` reserves no steps and adds no wait.
   * - ``coordination_scheme``
     - ``replay``
     - Coordination only. Controls the shared pose transform applied to both source segments. See
       the coordination scheme table below.
   * - ``coordination_scheme_pos_noise_scale``
     - ``0.0``
     - Coordination only. Half-width of uniform xyz translation noise (m) added to the shared
       transform. Each component is sampled from ``[-scale, scale]``.
   * - ``coordination_scheme_rot_noise_scale``
     - ``0.0``
     - Coordination only. Half-width of uniform per-axis rotation noise (rad) added to the shared
       transform. Each component is sampled from ``[-scale, scale]``.
   * - ``coordination_synchronize_start``
     - ``false``
     - Coordination only. If ``true``, a subtask holds its first waypoint until
       the paired EEF reaches its coordinated subtask. If ``false``, an unmatched leading portion
       may execute before the synchronized window begins.

**Constraint types**

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Type
     - Behavior
   * - ``sequential``
     - The first pair in ``eef_subtask_constraint_tuple`` is the precondition (former) subtask. The
       second is the constrained (latter) subtask, which waits according to
       ``sequential_min_time_diff`` until the first finishes.
   * - ``coordination``
     - The pairs are peers. Both use the same selected source demonstration and shared pose
       transform. Their final ``min(first_length, second_length)`` steps execute in lockstep. Any
       extra prefix on the longer segment may execute first.

**Coordination schemes**

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Scheme
     - Shared transform
   * - ``replay``
     - Identity transform: replay both source segments without adapting them to a moved reference
       object. An ``object_ref`` is not required.
   * - ``transform``
     - Full rigid transform from the source reference-object pose to its current pose, including
       translation and rotation. The coordinated subtask establishing the transform requires an
       ``object_ref``.
   * - ``translate``
     - Translation from the source reference-object position to its current position; recorded
       orientations are unchanged. The coordinated subtask establishing the transform requires an
       ``object_ref``.


Generation Policy
-----------------

The ``generation_policy`` section of the YAML schema defines data generation parameters for the task.
It controls source selection, trajectory stitching, recording,
and when generation stops. All fields use their defaults if omitted from the task descriptor.

.. list-table::
   :widths: 30 16 54
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``name``
     - ``str: "demo"``
     - Identifier for the generation run.
   * - ``seed``
     - ``1``
     - Seeds Python's ``random``, NumPy, and PyTorch before generation begins.
   * - ``num_trials``
     - ``10``
     - Generation target. It counts successful demonstrations when ``guarantee_success`` is
       ``true`` and total attempts when it is ``false``. The ``--generation_num_trials`` CLI arg
       of the data generation script overrides this value.
   * - ``guarantee_success``
     - ``true``
     - If ``true``, keep attempting generation until ``num_trials`` successes are recorded. If
       ``false``, stop after ``num_trials`` attempts regardless of their outcomes.
   * - ``keep_failed``
     - ``false``
     - Whether failed attempts are exported to a separate HDF5 file. Successful demonstrations
       are exported to the main output file regardless of this flag.
   * - ``use_skillgen``
     - ``false``
     - Whether annotation must collect SkillGen subtask start signals. During generation this
       value is overwritten to match the algorithm selected by ``--alg``.
   * - ``select_src_per_subtask``
     - ``false``
     - Whether each subtask may select a new source demonstration. If ``false``, an EEF reuses
       its first selected source for the entire generated episode.
   * - ``select_src_per_arm``
     - ``false``
     - Whether each EEF selects its source demonstration independently. If ``false``, the first
       selection is shared by every EEF and remains cached for the episode.
   * - ``transform_first_robot_pose``
     - ``false``
     - Whether to prepend the source EEF pose to every subtask's target-pose sequence. If
       ``false``, this is done only for the first subtask of each EEF.
   * - ``interpolate_from_last_target_pose``
     - ``true``
     - Whether non-first subtasks seed their interpolation bridge from the last executed waypoint.
       If ``false``, they start from the live EEF pose. SkillGen always bridges from the preceding
       motion plan's endpoint when it joins that plan to the replayed skill.
