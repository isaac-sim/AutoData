KPIs
====

``scripts/validate_dataset.py --report`` computes KPIs of a generated dataset from files that
generation writes. It needs no simulator.

Inputs
------

- **Generated dataset.** Generation records ``obs/datagen_info`` before every step with the same
  recorder term as annotation: object poses, the control-link pose (``eef_pose``) and the target
  pose the step's action commands (``target_eef_pose``).
- **Result JSON and attempt log.** ``generate_dataset.py --result_file`` adds a ``run`` block to
  the result JSON (device, GPU and CPU model, env count, generation time, source episode count)
  and writes ``<result_file stem>.attempts.jsonl``, one line per completed attempt: env id, start
  and end time, outcome, ``failure_cause`` (``planner`` or ``execution``), motion-planning time and
  initial object positions.
- **Source dataset** (``--source``): enables the source baselines.
- **Embodiment YAML** (``--embodiment``): needed for ``action_saturation``.
- **Recorded dataset** (``--recorded_dataset``): the teleoperated demonstrations annotated into
  ``--source``; needed for ``annotation_pass_rate``.

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated.hdf5 --report \
       --result_file datasets/generated.json \
       --source datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --report_file datasets/generated_kpis.json

A dataset left empty because every attempt failed is still reported. A KPI that cannot be
computed reports ``not_applicable`` with a reason.

Definitions
-----------

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - KPI
     - Definition
   * - ``success_rate``
     - Successful attempts over attempts, with a Wilson 95% interval. An attempt succeeds if the
       task is successful at any step.
   * - ``demos_per_hour``
     - Successful demonstrations per hour of generation time, labelled with the hardware (the GPU,
       or the CPU for ``--device cpu``) and env count. Compare only on the same hardware and env count.
   * - ``demos_per_human_demo``
     - Successful demonstrations over source demonstrations.
   * - ``annotation_pass_rate``
     - Share of recorded demonstrations that pass annotation into the source dataset.
   * - ``tracking_error``
     - p50 and p95 distance [mm] from each achieved end-effector position to the commanded path.
       Tracking-only channels, which command no pose, are skipped.
   * - ``action_saturation``
     - Share of pose-action entries at ±1, for adapters that clip pose actions.
   * - ``motion_smoothness``
     - p50 and p95 end-effector linear jerk [m/s³].
   * - ``initial_state_coverage``
     - Per object, grid cells occupied by successful starts over cells occupied by as many
       attempted starts, on a grid of about five successes per cell. The worst object is reported.
   * - ``success_by_initial_configuration``
     - Number of cells (three per varying axis, at least 10 attempts each) whose success count is
       improbably low at the overall rate: one-sided exact binomial test at 0.025 divided by the
       number of cells.
   * - ``failure_share``
     - Share of failed attempts by cause.
   * - ``planning_time``
     - p50 and p95 motion-planning wall-clock time per attempt [s], and its share of generation time.

Quality KPIs carry a ``source_baseline`` computed the same way on the source demonstrations.

The configuration KPIs grid each object's start positions along the x and y axes on which the
attempted starts span at least 1 cm, between the smallest and largest attempted start.

When generation stops, attempts still running are cancelled and never logged, and long attempts
are the likelier to be running. ``success_rate``, ``failure_share`` and the configuration KPIs
therefore count only attempts that started before the first env finished its last logged
attempt. This needs at least three logged attempts from every env; otherwise they use all attempts
and say the value is uncorrected.
