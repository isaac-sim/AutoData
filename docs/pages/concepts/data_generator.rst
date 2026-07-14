The Data Generator Interface
============================

The ``DataGenerator`` is the engine that turns annotated source demonstrations into new ones.
It runs the per-trial loop shared by every algorithm — randomize subtask boundaries, select a
source segment, transform it to the current scene, execute waypoints, record — and routes
everything algorithm-specific through a small plug-in interface,
``GenerationAlgorithm``. New algorithms plug in without editing the generator.

The code lives in :isaac_autodata_code_link:`<isaac_autodata_core/data_generator.py>` and
:isaac_autodata_code_link:`<isaac_autodata_core/algorithms.py>`.

Construction
------------

.. code-block:: python

   from isaac_autodata_core import DataGenerator, get_algorithm

   generator = DataGenerator(datastream=datastream, algorithm=get_algorithm("mimicgen"))

The generator reads *all* task, embodiment, and scene state through the
:doc:`Datastream <datastream>` — it never imports the simulator or touches the robot
directly. The constructor validates the setup and fails fast on mismatches:

* the number of end-effectors must match the algorithm's ``expected_eef_count``;
* the descriptor may declare constraints only if the algorithm ``supports_coordination``;
* the final subtask of every end-effector must have ``subtask_term_offset_range: [0, 0]``
  (there is no later boundary to offset against);
* the algorithm gets a last look via ``validate_setup(datastream)`` and may raise.

Generating One Demonstration
----------------------------

``generate()`` is an async method producing one demonstration attempt for one environment:

.. code-block:: python

   result = await generator.generate(
       env_id=0,
       success_term=success_term,          # the task's success condition
       env_reset_queue=reset_queue,        # simulator-loop queues
       env_action_queue=action_queue,
   )
   result.success        # True if the success condition held during the trial
   result.initial_state  # scene snapshot the episode started from

Each call resets the environment (randomizing the scene), builds and executes the per-subtask
trajectories, and — on success — exports the episode through the env's recorder. Whether
failed attempts count against the trial target or are retried is decided by the
:doc:`generation policy <task_descriptors>` (``guarantee_success``, ``keep_failed``).

Asynchronous Multi-Env Generation
---------------------------------

One ``DataGenerator`` serves all parallel environments. The entry point
(``isaac_autodata_examples/generate_dataset.py``) creates one asyncio task per environment,
each looping over ``generate(env_id=...)``; a single synchronous ``env_loop`` drains one
action per env from the shared queue, steps the simulator in batch, and services resets. The
simulator steps in lockstep while each environment's trial logic (including retries) runs
independently.

The GenerationAlgorithm Plug-In
-------------------------------

An algorithm is a subclass of ``GenerationAlgorithm``. Setting the ``name`` class
attribute auto-registers it — ``get_algorithm(name)`` constructs it, and the CLI's ``--alg``
choices follow. Five class attributes declare how the algorithm differs from a vanilla
MimicGen run:

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``name``
     - Registry key; the value passed to ``--alg``.
   * - ``expected_eef_count``
     - End-effector count(s) supported: ``1``, ``2``, or a tuple like ``(1, 2)``.
   * - ``requires_motion_planner``
     - The algorithm needs motion planners at construction (SkillGen takes
       ``motion_planners=``, one per env).
   * - ``uses_subtask_start_signals``
     - Source demos must carry subtask *start* signals; the pool parses boundaries from them.
   * - ``supports_coordination``
     - Cross-arm coordination constraints are honored during generation.

The shipped algorithms — ``mimicgen``, ``dexmimicgen``, ``skillgen`` — are described in
:doc:`algorithms`.

The behavioral hook: ``plan_subtask_trajectory``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generator calls ``plan_subtask_trajectory(...)`` whenever an end-effector needs its next
executable trajectory. The return value tells the generator what to do:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Return
     - Meaning
   * - ``(waypoints, False)``
     - Execute as the actual subtask; constraints apply.
   * - ``(waypoints, True)``
     - Execute as a motion-planned transit; constraints are skipped while it runs. The
       algorithm stashes the real subtask trajectory on the per-EEF state so the follow-up
       call can splice it in after the transit finishes.
   * - ``None``
     - Planning failed; the current ``generate()`` attempt aborts with ``success=False``.

The base-class default implements the MimicGen / DexMimicGen path: generate the subtask
segment and merge an interpolation segment from the previous pose. SkillGen overrides it to
plan a collision-free transit to the segment start first (the two-phase flow above); see
:doc:`algorithms` and the :doc:`SkillGen workflow <../workflows/skillgen/index>`.

Adding a New Algorithm
----------------------

1. Subclass ``GenerationAlgorithm`` in ``isaac_autodata_core/algorithms.py`` (or a new
   module imported from the package ``__init__``).
2. Set ``name``, ``expected_eef_count``, and the behavioral flags; implement
   ``validate_setup()`` for any algorithm-specific invariants.
3. Override ``plan_subtask_trajectory()`` only if the algorithm changes how trajectories are
   built or bridged — the flags alone cover many variants.

Registration is automatic; no CLI edits are needed.
