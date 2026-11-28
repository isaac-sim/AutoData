..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Step 3: Generate a Dataset
==========================

Start with one environment and one successful demonstration. Inspect the result before
raising the trial count or enabling parallel generation. Run these commands inside the
cuRobo development container described in :doc:`index`.

.. _schedulegen-source-data:

Choose the Annotated Source
---------------------------

The included source is
:autodata_code_link:`<datasets/annotated_datasets/dataset_gr1t2_nut_pour_schedulegen_annotated.hdf5>`.
It is about 62 MiB, self-contained, and contains five demonstrations with skill boundaries
and internal grasp/release marks. The supplied normal/cross examples share this source;
it has also been exercised with both physics backends. Do not substitute the humanoid
steering-wheel pick-and-place dataset.

If the checkout contains only a Git LFS pointer, fetch the file from the **host**:

.. code-block:: bash

   git lfs pull --include="datasets/annotated_datasets/dataset_gr1t2_nut_pour_schedulegen_annotated.hdf5"

If you completed Steps 1–2, use your new annotated file instead. Keep its recording
backend/profile consistent for your first generation check. There is no events sidecar
or ``--source_event_annotations`` argument.

Configuration Files
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - File
     - Purpose
   * - :autodata_code_link:`<autodata_examples/tasks/gr1t2_nut_pour_schedulegen.yaml>`
     - Subtasks, segment providers, grasp/release events, task order and scheduling options.
   * - :autodata_code_link:`<autodata_examples/embodiments/gr1t2_schedulegen.yaml>`
     - End-effector observations, pose targets, hand joints and the 36-value action layout.
   * - :autodata_code_link:`<autodata_examples/embodiments/gr1t2_planning.yaml>`
     - Arm joint groups, tool frames, robot collision geometry and cuRobo settings.
   * - `Environment profile <https://github.com/isaac-sim/AutoData/blob/main/autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml>`_
     - Object placement/randomization, controller settings, object properties and front view.

Generate the Normal Layout
--------------------------

.. code-block:: bash

   python scripts/generate_dataset.py \
       --alg schedulegen \
       --env_name AutoData-NutPour-GR1T2-ScheduleGen-v0 \
       --external_callback autodata_examples.gr1t2_nut_pour:register \
       --physics_backend newton \
       --task_descriptor autodata_examples/tasks/gr1t2_nut_pour_schedulegen.yaml \
       --embodiment autodata_examples/embodiments/gr1t2_schedulegen.yaml \
       --robot_planning autodata_examples/embodiments/gr1t2_planning.yaml \
       --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen.yaml \
       --input_file datasets/annotated_datasets/dataset_gr1t2_nut_pour_schedulegen_annotated.hdf5 \
       --output_file datasets/gr1t2_schedulegen/normal_newton.hdf5 \
       --result_file datasets/gr1t2_schedulegen/normal_newton.json \
       --generation_num_trials 1 \
       --generation_max_failures 10 \
       --generation_mode per_environment \
       --num_envs 1 \
       --device cuda:0 \
       --viz none \
       --record_metric_inputs \
       --record_video \
       --video_dir datasets/gr1t2_schedulegen/video_newton

For PhysX, use ``--physics_backend physx`` and distinct output/result/video paths, such as
``normal_physx.hdf5``, ``normal_physx.json`` and ``video_physx``. The other configurations
are unchanged. To use your own Newton recording, replace ``--input_file`` with
``datasets/gr1t2_schedulegen/source_newton_annotated.hdf5``.

``--record_video`` saves the profile's front viewport and also renders and records camera
observations in the HDF5 dataset, which can substantially increase its size.
For generation without video, omit both video arguments. Add ``--enable_cameras`` if you
still want the robot's RGB observations in the dataset. Start the container once and
reuse it for subsequent checks; first-use assets, shaders and robot models can take time.
``--record_metric_inputs`` saves the per-step inputs needed by the standard metrics report
in Step 4; omit it when those extra measurements are not needed.

Add ``--visualize_plan`` to open Rerun for environment zero. The default **Arm plans** tab
plays motion-planned transits in preparation order on the ``planning`` timeline; it does not
show live physics or the intervening demonstration segments. For the complete proposed
episode, open **Scheduled robot** and select its ``full_state_schedule_<index>`` timeline.
That preview includes both arms, demonstration segments and scheduled waits. Add
``--rerun_output datasets/plans.rrd`` to save the recording instead of opening a live viewer.

Trials, Failures and Files
--------------------------

The supplied descriptor sets ``guarantee_success: true``. Thus ``--generation_num_trials``
is a target number of **successful demonstrations**, not a fixed attempt count. The
failure allowance bounds additional attempts; reaching that limit can leave a valid but
smaller dataset. Read the JSON counters rather than assuming the target was reached.
The supplied descriptor sets this allowance explicitly. Other workflows keep unlimited
success retries by default; omit ``max_num_failures`` or set it to ``null`` for that behavior.

The descriptor also sets ``keep_failed: true``. Failed physical replays may be exported
to a separate failure HDF5; planning failures can occur before there are steps to export.
Do not mix failure files into successful policy-training data. Set ``keep_failed: false``
in your task configuration if those recordings are not wanted.

The HDF5 writer replaces an existing file at the selected output path. Reusing a disposable
path is convenient for local checks; choose a new path for data you intend to retain.

Try the Cross Layout
--------------------

The fixed-waist cross layout increases overlap between the arms' paths. Use the same
annotated source and task descriptor, replacing these arguments together:

.. code-block:: bash

   --robot_planning autodata_examples/embodiments/gr1t2_cross_planning.yaml \
   --env_profile autodata_examples/env_profiles/gr1t2_nut_pour_schedulegen_cross_overlap.yaml \
   --output_file datasets/gr1t2_schedulegen/cross_newton.hdf5 \
   --result_file datasets/gr1t2_schedulegen/cross_newton.json \
   --video_dir datasets/gr1t2_schedulegen/video_cross_newton

The cross planner checks the moving arm/hand groups rather than the full fixed robot.
This is not whole-body collision coverage. Neither layout's sampled checks guarantee
collision avoidance between samples or successful physical contact; simulation and task
evaluation still matter.

Scale After Inspecting One Episode
----------------------------------

Raise the trial count and choose an appropriate failure allowance only after validating
the first output. The two :ref:`generation modes <schedulegen-generation-modes>` are:

* ``per_environment``: independent planner contexts, with asynchronous synchronized stepping.
* ``buffered``: shared planning context, optional parallel CPU schedule solves, batched replay.

For a small buffered run, replace the corresponding arguments above with:

.. code-block:: bash

   --generation_mode buffered \
   --num_envs 5 \
   --schedule_workers 2 \
   --generation_num_trials 10

Remove ``--record_video`` and ``--video_dir`` when using multiple environments; the video
wrapper currently supports one environment. The requested demo count applies to the whole
run, not each batch. Later batches fill only the missing successful demos, subject to the
same attempt budget. Progress counters remain cumulative across batches.

More environments do not guarantee higher throughput or unchanged success rates. Independent
planner contexts can consume substantial GPU memory; buffered generation shares planner memory
but still allocates simulation state.

Continue to :doc:`step_4_validate_dataset`.
