..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Step 2: Annotate Demonstrations
===============================

ScheduleGen uses **SkillGen-style skill windows, separately for each arm**. DexMimicGen
normally marks where consecutive subtasks end. Here, also mark where each reusable skill
starts so motion planning can replace the approach to it. Do not annotate the future
motion-planned transits or scheduler-inserted waits.

The same ``scripts/annotate_demos.py`` tool handles all these annotations. It replays an
episode for each arm and prints the marks expected by the task descriptor. The supplied
descriptor already enables skill-start annotation; no extra annotation mode is needed.

.. important::

   Keep the recording's physics backend and environment profile. Write a new output file;
   do not overwrite the raw demonstration or modify the supplied annotated dataset.

Run Manual Annotation
---------------------

Inside the development container, run:

.. tabs::

   .. group-tab:: Newton

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name AutoData-NutPour-GR1T2-ScheduleGen-v0 \
             --external_callback autodata_examples.gr1t2_nut_pour:register \
             --physics_backend newton \
             --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
             --task_descriptor autodata_examples/tasks/gr1t2_nut_pour_schedulegen.yaml \
             --embodiment autodata_examples/embodiments/gr1t2_schedulegen.yaml \
             --input_file datasets/gr1t2_schedulegen/source_newton.hdf5 \
             --output_file datasets/gr1t2_schedulegen/source_newton_annotated.hdf5 \
             --device cuda:0 \
             --viz kit

   .. group-tab:: PhysX

      .. code-block:: bash

         python scripts/annotate_demos.py \
             --env_name AutoData-NutPour-GR1T2-ScheduleGen-v0 \
             --external_callback autodata_examples.gr1t2_nut_pour:register \
             --physics_backend physx \
             --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
             --task_descriptor autodata_examples/tasks/gr1t2_nut_pour_schedulegen.yaml \
             --embodiment autodata_examples/embodiments/gr1t2_schedulegen.yaml \
             --input_file datasets/gr1t2_schedulegen/source_physx.hdf5 \
             --output_file datasets/gr1t2_schedulegen/source_physx_annotated.hdf5 \
             --device cuda:0 \
             --viz kit

Newton defaults to preserving solver buffers between restores; PhysX keeps its established
reset behavior. Add ``--enable_cameras`` if the annotated output needs RGB observations.
Input images are not simply copied: annotation records observations during replay.
``--auto`` does not provide the skill starts and internal object events required here.

Controls
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Action
   * - ``N``
     - Begin or resume replay.
   * - ``B``
     - Pause replay to inspect a boundary.
   * - ``S``
     - Mark the next printed signal at the current action index.
   * - ``Q``
     - Skip the episode.

Marks are taken before the current action is applied. Use the visible robot/object state,
not a guessed timestamp. Each press of ``S`` accepts one signal. A skill's start must be
at a later action index than the previous skill's end, even when the visible states are
close. For example, if one skill ends at action 100, start the next at action 101 or later.
The tool repeats a pass if the mark count is incomplete and
does not export a replay that fails the task-success check.

Marks for This Task
-------------------

The following order comes from the shipped task descriptor. Names reused between start
and end marks belong to separate HDF5 groups.

.. list-table:: Left arm
   :header-rows: 1
   :widths: 18 32 50

   * - Mark
     - Signal
     - Meaning
   * - Skill start
     - ``grasp_left``
     - Begin the beaker grasp, after the approach.
   * - Object event
     - ``beaker_grasp``
     - The beaker is securely held by the left hand.
   * - Skill end
     - ``grasp_left``
     - Finish the grasp skill, before transport to the bowl.
   * - Skill start
     - ``pour_left``
     - Begin pouring after arriving above the bowl.
   * - Skill end
     - ``pour_clear_left``
     - Finish pouring and move the beaker clear of the bowl.
   * - Skill start
     - ``beaker_release_start``
     - Begin the release skill after arriving at the bin.
   * - Object event
     - ``beaker_release``
     - Release the beaker; it is no longer carried by the hand.

.. list-table:: Right arm
   :header-rows: 1
   :widths: 18 32 50

   * - Mark
     - Signal
     - Meaning
   * - Skill start
     - ``grasp_right``
     - Begin the bowl grasp after the approach.
   * - Object event
     - ``bowl_grasp``
     - The bowl is securely held by the right hand.
   * - Skill end
     - ``grasp_right``
     - Finish the grasp skill, before transport to the scale.
   * - Skill start
     - ``place_bowl_on_scale_right``
     - Begin placement/release after arriving at the scale.
   * - Object event
     - ``bowl_release``
     - Release the bowl onto the scale.

Each arm's final subtask ends implicitly with the episode. Its explicit **start** is still
required. The descriptor also limits release windows around their object events; do not
omit those marks because a final termination signal is implicit.
Record and retain at least two actions after each ``beaker_release`` and ``bowl_release``
mark for the configured release windows. The penultimate action is too late.

Grasp/release marks update the planner's carried-object collision model. They do not weld
objects to the robot in simulation and do not impose task-order constraints. Those remain
in the task YAML. Events at a segment's explicit start/end need no internal event mark.

Expected Output
---------------

The annotated HDF5 contains the recorded data plus aligned signals under:

.. code-block:: text

   obs/datagen_info/subtask_start_signals/
   obs/datagen_info/subtask_term_signals/
   obs/datagen_info/object_event_signals/

There is no separate annotation sidecar. Validate the file and inspect a complete annotated
episode before generation:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/gr1t2_schedulegen/source_newton_annotated.hdf5

Continue to :doc:`step_3_generate_dataset`.
