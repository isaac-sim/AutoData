..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Step 4: Validate and Inspect the Dataset
========================================

A completed process is not enough: check the attempt counters, exported episodes and
physical task outcome before using the data for learning.

Check the Files
---------------

Inside the development container:

.. code-block:: bash

   python scripts/validate_dataset.py datasets/gr1t2_schedulegen/normal_newton.hdf5

The file must contain episodes with actions, observations and initial state. Check that
numeric arrays are finite and observation lengths match the action count. The generation
JSON should agree with the successful exports. Its per-attempt diagnostics distinguish
planning failure, scheduling infeasibility, motion validation and physical task failure.
For nut pouring, inspect the final nut-in-bowl, bowl-on-scale and beaker-in-bin checks.

For the standard metrics report, also pass the matching generation result:

.. code-block:: bash

   python scripts/validate_dataset.py \
       datasets/gr1t2_schedulegen/normal_newton.hdf5 \
       --report \
       --result_file datasets/gr1t2_schedulegen/normal_newton.json \
       --source datasets/annotated_datasets/dataset_gr1t2_nut_pour_schedulegen_annotated.hdf5 \
       --report_file datasets/gr1t2_schedulegen/normal_newton_report.json

Some metrics require per-step inputs. Add ``--record_metric_inputs`` during generation
when those metrics are needed; missing measurements should not be interpreted as zeros.
Use your actual source filename in the report if you recorded your own demonstrations.

Inspect the Recorded Rollout
----------------------------

The video saved during generation shows the execution that produced the dataset. Look for:

* A secure beaker grasp, followed by pouring without pushing the bowl away.
* Clearance above the bowl before its pickup starts.
* Overlapping beaker return and bowl handling when the paths allow it.
* Stable releases, with the nut in the bowl, the bowl on the scale and the beaker in the bin.

The front-view video is separate from the robot's observation camera. To record the
robot's camera instead, add ``--video_observation_path policy.robot_pov_cam`` to a
single-environment generation command. This does not change either camera pose.

Replay with Isaac Lab
---------------------

Use the same backend and profile as the saved episode:

.. code-block:: bash

   python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
       --task AutoData-NutPour-GR1T2-ScheduleGen-v0 \
       --external_callback autodata_examples.gr1t2_nut_pour.register \
       --physics_backend newton \
       --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
       --dataset_file datasets/gr1t2_schedulegen/normal_newton.hdf5 \
       --device cuda:0 \
       --num_envs 1 \
       --viz kit

Use ``--physics_backend physx`` and the PhysX filename for PhysX data. For cross data,
also use the cross environment profile. Leave the replay tool's solver-buffer-reset
option disabled for Newton. Action replay can diverge from recorded physics; it is an
additional inspection, not a replacement for generation's recorded task outcome.

Inspect Plans in Rerun
----------------------

Add ``--visualize_plan`` to the generation command to open a live Rerun viewer of the
planned paths and collision spheres. A display is needed for the viewer; ``--viz none``
only disables the simulator visualizer. To save a Rerun recording instead of opening the
viewer, also add:

.. code-block:: bash

   --rerun_output datasets/gr1t2_schedulegen/plans.rrd

Check robot/object frames, the active arm's spheres, carried-object geometry and the
planned transits. Rerun shows planning geometry; the simulator video is needed to judge
contacts and physical execution.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - First check
   * - Source file cannot be opened
     - Confirm it is the complete Git LFS HDF5, not a pointer or an unrelated pick-and-place file.
   * - Missing skill start or object event
     - Compare the annotation groups with the task's printed mark list. Use manual annotation.
   * - Incorrect hand motion
     - Check the GR1T2 callback, embodiment action layout and recording profile before changing gains.
   * - Planner cannot reach a target
     - Inspect the reset, tool frames, model joint limits and the matching normal/cross planning config.
   * - Scheduler is infeasible
     - Inspect task-order/continuity constraints and collision conflicts; timing cannot repair an unreachable path.
   * - Contacts fail despite a valid schedule
     - Inspect the recorded grasp, pour and release windows, object properties and actual rollout.
   * - GPU memory is exhausted
     - Reduce independent planner contexts or try buffered generation. Do not assume ten contexts fit every GPU.
   * - Few or no successful exports
     - Read the attempt/failure counters and stages before increasing the failure allowance.
