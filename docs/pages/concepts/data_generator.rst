The Data Generator Interface
============================

The ``DataGenerator`` is the engine that turns annotated source demonstrations into new ones.
It runs the per-attempt loop shared by every algorithm — randomize subtask boundaries, select a
source segment, transform it to the current scene, execute waypoints, record — and routes
everything algorithm-specific through a small plug-in interface, ``GenerationAlgorithm``.
New algorithms plug in without editing the generator.

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

The Life of One Generation Attempt
----------------------------------

``generate()`` is an async method producing one generation attempt for one environment:

.. code-block:: python

   result = await generator.generate(
       env_id=0,
       success_term=success_term,          # the task's success condition
       env_reset_queue=reset_queue,        # simulator-loop queues
       env_action_queue=action_queue,
   )

One call runs this sequence:

1. **Reset and snapshot.** The env is reset through the reset queue (randomizing the scene),
   and the resulting scene state is snapshotted — it becomes the output episode's initial
   state.
2. **Randomize boundaries.** Each source episode's subtask boundaries are perturbed within
   the descriptor's offset ranges, so generated demonstrations do not all switch subtasks at
   identical steps.
3. **Plan the next stretch.** Whenever an end-effector has no waypoints left, the generator
   asks the algorithm (``plan_subtask_trajectory``) for its next executable trajectory.
   For MimicGen-style algorithms that means: select a source segment, transform it to the
   current object pose, and merge an interpolation ramp from the previous pose. For
   SkillGen it means a motion-planned transit first — see the two-phase note below.
4. **Execute in lockstep.** Each tick, every end-effector's next waypoint is assembled into
   **one** env action and stepped (see `The Waypoint Layer`_). Cross-arm constraints are
   enforced here: an end-effector whose constraint is not yet satisfied holds its pose
   instead of advancing.
5. **Latch success.** The task's success condition is evaluated after every step; once it
   fires, the attempt is marked successful (it does not need to stay true).
6. **Record.** When all end-effectors finish their subtasks, the episode's success flag is
   set on the env's recorder and — for successful attempts (and failed attempts too, if
   ``keep_failed`` is set) — the episode is exported.

The returned ``GenerationResult`` carries ``success`` and the ``initial_state`` snapshot.
A ``None`` from the algorithm (e.g. a SkillGen planning failure) aborts the attempt with
``success=False``; whether that counts toward the configured target or is retried is the
:doc:`generation policy's <task_descriptors>` call (``guarantee_success``).

**The two-phase (transit + skill) flow.** An algorithm can answer step 3 with *transit*
waypoints — a motion-planned approach — while stashing the actual skill segment on the
end-effector's state. The generator executes the transit (constraints are skipped during
it), then calls the algorithm again; the algorithm notices the stashed segment and returns
it, merged from the transit's end pose. This is how SkillGen interleaves planned motion with
demonstration replay without the generator knowing anything about motion planning.

The Waypoint Layer
------------------

Algorithms and the generator talk to each other in **waypoints**, not actions:

* A ``Waypoint`` is one 6-DoF target pose plus the passthrough (non-pose) action channels
  active at that tick, and an optional noise amplitude.
* A ``WaypointTrajectory`` is the ordered stretch an end-effector still has to execute;
  merging an interpolation ramp onto a transformed segment happens at this level.
* A ``MultiWaypoint`` is the per-tick bundle: one waypoint per end-effector, executed as a
  single env step. Its ``execute()`` converts all target poses and passthrough channels into
  one action through the Datastream's action codec, enqueues it for the simulator loop, and
  checks the success condition after the step.

Only at the ``MultiWaypoint`` boundary do poses become robot actions — everything upstream
of it is embodiment-agnostic.

Asynchronous Multi-Env Generation
---------------------------------

One ``DataGenerator`` serves all parallel environments. The entry point
(``scripts/generate_dataset.py``) creates one asyncio task per environment, each looping
over ``generate(env_id=...)``; a single synchronous ``env_loop`` steps the simulator::

   generator task (env 0) ──┐                        ┌──> env_loop:
   generator task (env 1) ──┼──> env_action_queue ───┤    - waits until every env
        ...                 │    (env_id, action)    │      has queued one action
   generator task (env N) ──┘                        │    - steps the sim once, batched
             │                                       │    - services reset requests
             └────────────── env_reset_queue ────────┘

The simulator steps in lockstep — one batched step once every env has produced its action —
while each environment's attempt logic (including retries after failures) runs independently.
The generator tasks tally outcomes into a shared ``stats`` dict; ``env_loop`` reads it to
report progress and decide when to stop:

* ``guarantee_success: true`` — run until ``num_trials`` **successful demonstrations** are exported.
* ``guarantee_success: false`` — run until ``num_trials`` total **attempts**, whatever their
  outcome.

Results and Statistics
----------------------

Three layers report what happened:

* **Per attempt** — ``GenerationResult``: ``success`` and ``initial_state``.
* **Per run, in memory** — the ``stats`` counters: ``num_success``, ``num_failures``,
  ``num_attempts`` (``num_success + num_failures = num_attempts``).
* **Per run, on disk** — with ``--result_file``, a JSON sidecar is written atomically at the
  end of a completed run:

  .. code-block:: json

     {
       "algorithm": "skillgen",
       "requested_trials": 1000,
       "num_success": 1000,
       "num_failures": 412,
       "num_attempts": 1412,
       "env_profile": {
         "name": "franka_bin_stack",
         "path": "isaac_autodata_examples/env_profiles/franka_bin_stack.yaml",
         "planner": "franka_stack_cube_bin"
       }
     }

  ``env_profile`` appears only when an :doc:`environment profile <environment_profiles>` was
  applied — the generated dataset itself records only the base env id, so the sidecar is
  what documents that the scene was modified.

The GenerationAlgorithm Plug-In
-------------------------------

An algorithm is a subclass of ``GenerationAlgorithm``. Setting the ``name`` class attribute
auto-registers it — ``get_algorithm(name)`` constructs it, and the CLI's ``--alg`` choices
follow. Five class attributes declare how the algorithm differs from a vanilla MimicGen run:

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
     - Source demonstrations must carry subtask *start* signals; the pool parses boundaries from them.
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
plan a collision-free transit to the segment start first (the two-phase flow above).

Writing Your Own Algorithm
--------------------------

A new algorithm is one subclass — no generator or CLI edits. The minimal shape:

.. code-block:: python

   # isaac_autodata_core/algorithms.py (or a new module imported from the package __init__)

   class MyAlgorithm(GenerationAlgorithm):
       """One-line description of what differs from vanilla MimicGen."""

       name = "my_algorithm"            # becomes the --alg value
       expected_eef_count = 1
       requires_motion_planner = False
       uses_subtask_start_signals = False
       supports_coordination = False

       def validate_setup(self, datastream) -> None:
           # Optional: fail fast on setups the algorithm can't handle.
           for eef_name in datastream.get_eef_names():
               assert datastream.num_subtasks(eef_name) >= 2, (
                   f"{self.name} needs at least two subtasks per end-effector."
               )

Setting ``name`` registers the class automatically; ``--alg my_algorithm`` works as soon as
the module is imported. If the flags alone don't express the behavior, override
``plan_subtask_trajectory`` and honor its return contract (table above) — ``SkillGen`` in
:isaac_autodata_code_link:`<isaac_autodata_core/algorithms.py>` is the worked example,
including the stash-and-resume pattern for two-phase execution. Helpers you'll want are on
the ``data_generator`` argument: ``generate_eef_subtask_trajectory`` (select + transform a
source segment) and ``merge_eef_subtask_trajectory`` (merge it with an approach from the
previous pose).

Checklist for a new algorithm:

1. Subclass, set ``name`` and the flag attributes.
2. Implement ``validate_setup`` for algorithm-specific invariants.
3. Override ``plan_subtask_trajectory`` only if the flags don't cover the behavior.
4. If the algorithm needs external machinery (like SkillGen's planners), accept it in
   ``__init__`` — the CLI forwards keyword arguments through ``get_algorithm``.
