KPIs
====

The KPI report describes what a generated dataset delivers. It answers four questions:

- how many usable demonstrations a run yields, and at what cost;
- whether they make good training data;
- whether they cover the task's variation;
- why attempts fail.

``scripts/validate_dataset.py --report`` computes the report from files that generation and
annotation already write. It needs no simulator.

Producing the Inputs
--------------------

Every generated dataset carries two per-step entries under ``obs/datagen_info``:

- ``intended_eef_pose``: the noise-free waypoint pose commanded at that step, for end effectors whose
  action commands a pose (tracking-only channels have none);
- ``eef_pose``: the achieved control-link pose, read before that step.

Pass ``--result_file`` to ``generate_dataset.py`` to also write:

- The **result JSON**, with a ``run`` block: seed, stop rule, device, env count, control step
  (``step_dt``), setup and generation time, source dataset path and digest, and the reset events
  that sample object poses from a ``pose_range``.
- The **attempt log**, ``<result_file stem>.attempts.jsonl``. It holds one JSON line per completed
  attempt: env id, start and end time, outcome, ``failure_cause`` (``planner`` or ``execution``), the
  failed subtask per end effector, step count, initial object poses, and one summary per
  motion-planner call.

Two inputs come from elsewhere:

- Pass ``--summary_file`` to ``annotate_demos.py`` to record annotation yield.
- Pass the annotated source dataset as ``--source`` to enable the source baselines. Its
  ``obs/datagen_info/eef_pose`` is in the same control-link frame as the generated data's.

The embodiment YAML is needed only for ``action_saturation``.

Running the Report
------------------

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated.hdf5 --report \
       --result_file datasets/generated.json \
       --source datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --report_file datasets/generated_kpis.json

The command validates the dataset, prints a KPI table, and optionally writes the full report as
JSON. A dataset left empty because every attempt failed is still reported. The report stops with an
error in three cases:

- the dataset's demonstration count differs from the result file's success count;
- the attempt log's line count differs from the result file's attempt count;
- ``--source`` is not the dataset the run generated from.

KPI Definitions
---------------

.. list-table::
   :widths: 26 52 22
   :header-rows: 1

   * - KPI
     - Definition
     - Needs
   * - ``success_rate``
     - Successful attempts over attempts, with a Wilson 95% interval. An attempt succeeds if the
       task is successful at any step. ``raw_rate`` is the uncorrected rate from the result file.
     - Result JSON, attempt log
   * - ``demos_per_hour``
     - Successful demonstrations per hour of generation time. One run uses one GPU, so this is
       also demonstrations per GPU-hour. Setup time is reported separately.
     - Result JSON
   * - ``demos_per_human_demo``
     - Successful demonstrations over source demonstrations. When the run stops at a success
       count, this is the requested count divided by the source count.
     - Result JSON
   * - ``annotation_yield``
     - Exported over processed episodes, with rejection reasons and manual re-marks.
     - Annotation summary
   * - ``tracking_error``
     - p50 and p95 distance [mm] from each achieved end-effector position to the noise-free
       intended path. Position only.
     - Generated dataset
   * - ``action_saturation``
     - Share of pose-action entries at ±1, for adapters that clip pose actions.
     - Embodiment YAML
   * - ``motion_smoothness``
     - p50 and p95 end-effector linear jerk [m/s³].
     - ``step_dt``
   * - ``initial_state_coverage``
     - Per object, grid cells occupied by successful starts over cells occupied by as many
       attempted starts. The worst object is reported.
     - Attempt log, reset ranges
   * - ``success_by_initial_configuration``
     - Number of cells whose success rate is significantly below the overall success rate. Each
       object's reset range has three cells per varying axis (3 for one axis, 9 for two). Only
       cells with at least 10 attempts count. A cell is flagged when the exact binomial
       probability of so few successes at the overall rate is below 0.025 divided by the number
       of counted cells. ``per_asset`` holds every cell's counts.
     - Attempt log, reset ranges
   * - ``failure_share``
     - Share of failed attempts by cause, and counts by end effector and subtask.
     - Attempt log
   * - ``planner_failure_types``
     - Share of failed motion plans by cuRobo status.
     - Attempt log (SkillGen)
   * - ``planning_time``
     - p50 and p95 planning wall-clock time per attempt [s], and its share of generation time.
       Planning blocks the event loop, so this share is time no env steps.
     - Attempt log (SkillGen)

Reading the Values
------------------

- **Source baselines.** Quality KPIs carry a ``source_baseline`` computed on the source
  demonstrations. Compare against it rather than reading absolute values.
- **Tracking error includes noise.** Generated values include the injected action noise by design.
  The source baseline is measured against the executed teleoperation commands.
- **Attempts cancelled at the stop.** When generation stops, attempts still running are cancelled
  and never logged. Long attempts are the likelier to be running, so a raw rate under-counts long
  outcomes. ``success_rate``, ``failure_share`` and the configuration KPIs count only attempts
  that started before the first env finished its last logged attempt; all of those completed. The
  correction needs at least three logged attempts from every env, since with fewer it is biased
  too. Otherwise they use all attempts and mark the value uncorrected.
- **Not applicable.** A KPI that cannot be computed reports ``not_applicable`` with a reason.
  Missing inputs never read as zero.
- **Coverage grid.** The grid is sized to about five successful starts per cell, with at least two
  bins per varying axis. An object with fewer successful starts is skipped.
- **Range frames.** Ranges of ``randomize_object_pose`` are absolute in the env frame. Ranges of
  ``reset_root_state_uniform`` are offsets from the asset's default position. The configuration
  KPIs skip, and list the reason for, other reset functions, ranges without x or y variation, and
  objects whose starts fall outside their range.
- **Failed subtasks.** Execution failures are attributed only when the env publishes the
  termination signal of every non-final subtask in its ``subtask_terms`` observation group.

The report's JSON layout is versioned by ``schema_version``. The functions live in
``autodata_utils/dataset_metrics.py``.
