Architecture Overview
=====================

Isaac AutoData is organized around one idea: **the generation machinery should not know the
simulator or the robot directly.** Everything the generator needs is reachable through small,
explicit interfaces — a task-descriptor YAML for the task, an embodiment YAML for the robot,
and a read interface (the Datastream) for the live environment. New tasks, robots, and
algorithms can therefore be added independently of one another.

.. todo::

   Add an architecture diagram (source dataset -> pool -> algorithm -> env -> output dataset).

Packages
--------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Responsibility
   * - ``isaac_autodata_interfaces``
     - The boundary to the simulator and to configuration: task descriptors, embodiment
       adapters, the Datastream, env setup helpers, and motion-planner backends.
   * - ``isaac_autodata_core``
     - The generation machinery: the data generator, generation algorithms, the source-demo
       pool, selection strategies, and waypoint execution.
   * - ``isaac_autodata_utils``
     - Small shared utilities (pose math, tensor helpers).
   * - ``isaac_autodata_examples``
     - The ``generate_dataset.py`` entry point plus example task descriptors and embodiment
       configs.
   * - ``scripts``
     - Dataset tools: annotation (``annotate_demos.py``) and validation
       (``validate_dataset.py``).

Key Abstractions
----------------

Datastream
^^^^^^^^^^

The ``Datastream`` composes the live environment, the task descriptor, the embodiment
adapter, and the source-demo pool into the **single read interface** the generator observes
the world through:

* World state: object poses (env-relative), end-effector poses, robot joint positions, the
  scene state snapshot used to record an episode's initial state.
* Task state: per-subtask boolean termination signals sampled from the env's observation
  buffer, term/start signal names per end-effector.
* Action codec: converting between target end-effector poses and the environment's action
  vector, both directions, via the embodiment adapter.

Writes deliberately stay out: controller-side operations — stepping the env, resetting,
driving the recorder — go through the ``get_env()`` escape hatch, keeping the read surface
honest and the mutation points easy to audit. See :doc:`datastream` for the full surface.

DataGenInfoPool
^^^^^^^^^^^^^^^

The pool loads the annotated source dataset and turns each episode into per-subtask
**boundaries**. For every end-effector it walks the episode's boolean signal ramps: a
termination signal's rising edge ends its subtask (the final subtask ends with the
trajectory); with SkillGen enabled, each subtask's start comes from its start-signal ramp
instead of the previous subtask's end. At load time the pool validates that the worst-case
randomized boundaries (after applying the descriptor's offset ranges) remain non-empty and
non-overlapping, so misannotated episodes fail fast with a named reason rather than
producing broken generations later.

Data Generator and Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DataGenerator`` runs the per-trial loop; the pluggable
:doc:`generation algorithm <algorithms>` decides how each subtask's trajectory is planned
(interpolated transitions for MimicGen/DexMimicGen, motion-planned transit for SkillGen).
Execution is waypoint-based: each step, every end-effector's next waypoint is assembled into
one action through the Datastream's action codec, and the env is stepped once.

Generation is **asynchronous**: one generator task per environment produces actions into a
queue, and a single synchronous ``env_loop`` drains one action per env, steps the simulator
in batch, and services reset requests. This keeps the simulator stepping in lockstep while
each environment's trial logic runs independently — including retries after failed trials.
See :doc:`data_generator` for the generator's interface and the algorithm plug-in surface.

Data Flow
---------

One generation trial, end to end:

1. **Reset.** The environment resets and randomizes the scene; the initial state is snapshot
   for the output episode.
2. **Select.** For the current subtask of each end-effector, a source demo segment is chosen
   by the subtask's selection strategy (random or nearest-neighbor — see
   :doc:`algorithms`), within the scope configured by
   ``generation_policy.select_src_per_subtask`` / ``select_src_per_arm``.
3. **Transform.** The segment's end-effector trajectory, expressed in its source reference
   object's frame, is re-expressed under the object's current pose. Subtask boundaries are
   randomized within the descriptor's offset ranges.
4. **Bridge.** The gap from the robot's current pose to the segment start is closed — by
   interpolation (``num_interpolation_steps``), or by a collision-free planned transit under
   SkillGen.
5. **Execute.** Waypoints are converted to actions and stepped, with per-subtask action
   noise; the task's success condition is evaluated every step and latched.
6. **Record.** Successful trials are exported to the output HDF5 (failed ones too, into a
   separate file, if ``keep_failed`` is set). Generation continues until the policy's trial
   target is met.
