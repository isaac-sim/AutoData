KPIs
====

``scripts/validate_dataset.py --report`` computes KPIs of a generated dataset from files that
generation and annotation write. It needs no simulator.

Inputs
------

- **Generated dataset.** Every episode records, per step under ``obs/datagen_info``, the
  noise-free waypoint pose commanded (``intended_eef_pose``, only for end effectors whose action
  commands a pose) and the control-link pose read before the step (``eef_pose``).
- **Result JSON and attempt log.** ``generate_dataset.py --result_file`` adds a ``run`` block to
  the result JSON (device, GPU and CPU model, env count, control step, generation time, source
  episode count, reset events that sample object poses from a ``pose_range``) and writes
  ``<result_file stem>.attempts.jsonl``, one line per completed attempt: env id, start and end
  time, outcome, ``failure_cause`` (``planner`` or ``execution``), failed subtask per end
  effector, initial object poses, and one summary per motion-planner call.
- **Source dataset** (``--source``): enables the source baselines.
- **Embodiment YAML** (``--embodiment``): needed for ``action_saturation``.
- **Annotation summary** (``annotate_demos.py --summary_file``): needed for ``annotation_yield``.

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
   * - ``annotation_yield``
     - Exported over processed episodes, with rejection reasons.
   * - ``tracking_error``
     - p50 and p95 distance [mm] from each achieved end-effector position to the intended path.
       Generated values include the injected action noise.
   * - ``action_saturation``
     - Share of pose-action entries at ±1, for adapters that clip pose actions.
   * - ``motion_smoothness``
     - p50 and p95 end-effector linear jerk [m/s³].
   * - ``initial_state_coverage``
     - Per object, grid cells occupied by successful starts over cells occupied by as many
       attempted starts, on a grid of about five successes per cell. The worst object is reported.
   * - ``success_by_initial_configuration``
     - Number of reset-range cells (three per varying axis, at least 10 attempts each) whose
       success count is improbably low at the overall rate: one-sided exact binomial test at
       0.025 divided by the number of cells.
   * - ``failure_share``
     - Share of failed attempts by cause, with counts by end effector and subtask. Execution
       failures are attributed to a subtask only when the env publishes every non-final subtask's
       termination signal in its ``subtask_terms`` observation group.
   * - ``planner_failure_types``
     - Share of failed motion plans by cuRobo status.
   * - ``planning_time``
     - p50 and p95 planning wall-clock time per attempt [s], and its share of generation time.

Quality KPIs carry a ``source_baseline`` computed the same way on the source demonstrations.

When generation stops, attempts still running are cancelled and never logged, and long attempts
are the likelier to be running. ``success_rate``, ``failure_share`` and the configuration KPIs
therefore count only attempts that started before the first env finished its last logged
attempt. This needs at least three logged attempts from every env; otherwise they use all attempts
and say the value is uncorrected.

Ranges of ``randomize_object_pose`` are absolute in the env frame; ranges of
``reset_root_state_uniform`` are offsets from the asset's default position. The configuration KPIs
skip other reset functions and say why.
