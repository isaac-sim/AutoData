Metrics
=======

``scripts/validate_dataset.py --report`` computes the metrics of a dataset and, given its result
file, of the generation run that produced it. Metrics are computed from recorded files and need no
simulator. Generate, annotate or retarget with ``--record_metric_inputs`` to record what they read;
without it, most metrics report that their inputs are missing.

.. code-block:: bash

   python scripts/validate_dataset.py datasets/generated.hdf5 --report \
       --result_file datasets/generated.json \
       --source datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --report_file datasets/generated_report.json

The command prints a summary; ``--report_file`` writes the full report, with every detail, as JSON.

.. code-block:: text

   AutoData metrics · gr1.hdf5 · run e79e37b1 · source dataset_gr1_annotated.hdf5
   6/6 episodes valid · 7 computed · 3 not computed · gate failed on 2 of 4

                                         Value   Source  Based on
   VALIDITY
   ✗ action_rotation_error          p95 0.0345  3.58e-7  1,612 poses
   ✓ correctness_pass_rate                100%           6 episodes
   ✓ final_success_rate                   100%           6 episodes
     image_stream_pass_rate               100%           6 episodes
   VIABILITY
     demos_per_gpu_hour       65.8 demos/GPU-h           6 demos
   ✗ generation_success_rate             21.4%           28 attempts · 95% CI 10.2–39.5%
     time_to_dataset                6 min 11 s           1 run

   NOT COMPUTED
     not reported by this algorithm: planning_failure_rate, planning_time
     needs a --source annotated with --record_metric_inputs: annotation_pass_rate

- The second line is the verdict: valid episodes, how many metrics were computed, and, with ``--gate``,
  the gate, whose metrics are marked ✓ or ✗.
- **Value** is the metric's value; a metric with percentiles shows the one a gate checks by default.
  **Source** is the same metric on the ``--source`` dataset's valid episodes. **Based on** is the sample
  count, with a rate's 95% interval and, when some checks did not apply, how many checks a pass rate rests
  on.
- **Not computed** groups the metrics that do not apply by what would make them apply. A metric that
  fails is listed under **Errors**; the other metrics are still computed.

The report reads the run named by the dataset's run id. A result file from another run, such as the
earlier result of an interrupted rerun, is set aside with a warning, and the run is reported from its own
attempt log. The report refuses a dataset, result file and attempt log whose episode, attempt or success
counts disagree, as they do after datasets are merged. When the result file records the run's source, the
report also refuses a different ``--source``; without that record, the source is not verified.

What the report trusts
----------------------

The report checks every record it reads against the shape it expects before computing anything: the
dataset's ``env_args`` and ``autodata`` entry, the result file, each attempt-log line, and the gate. A
record with a missing field, a wrong type or a negative count is refused with a message naming the file
and the field, so no metric is computed from part of a malformed record. Fields the report does not read
are not checked. A metric loaded with ``--metrics`` runs as trusted code, but its ``Value`` is checked when
it is created: a malformed one makes that metric an error, and the other metrics are still computed.

Gating
------

``--gate`` reads metric bounds from YAML and makes the command fail when one is violated:

.. code-block:: yaml

   generation_success_rate: {min: 0.3, version: 1}
   planning_time: {max: 5.0, key: p95}

A gated metric also violates the gate if it is not applicable, fails, or reports a different
``version``. A gate entry may set ``min``, ``max``, ``key`` and ``version``; any other field, or a bound
that is not a finite number, is an error.

Episode checks
--------------

Before any metric, every episode goes through the checks below. An episode that fails one is counted in
``correctness_pass_rate`` or ``image_stream_pass_rate`` and left out of every dataset metric; the report lists
the failures per check, with examples.

:autodata_checks_table:

Built-in metrics
----------------

:autodata_metrics_table:

Diversity metrics
-----------------

The diversity metrics describe how differently the demonstrations behave: the space the hands move
through, how they approach what they lift, and where objects start. The four spread metrics each average,
over pairs of episodes or lifts, how much the two differ, and the two lift metrics then average over the
lifted objects. That estimates how different two demonstrations from the same generator are, so it does
not grow with the number of demonstrations and is defined from two of them on; fewer only make it noisier.
The source column holds the same average over the seeds. With many episodes, the averages use 2,000 pairs
drawn with a fixed seed.

They follow each hand's grasp point, from the hand and object poses that ``obs/datagen_info`` records in
one frame before each step; generation records the object poses with ``--record_metric_inputs``. An
object is lifted when ``states`` first shows it 1 cm above its start, and lift-off is the last
observation before that; the hand that carries it is the one in whose frame it moves least while it
rises. The hand's grasp point is where the objects it carries sit in its frame at lift-off, the median
over the dataset. Recorded hand frames differ, from a fingertip point to a flange or a wrist, so the
point is learned from each dataset; a hand that carries nothing keeps its own frame. An object's approach
is the grasp point's path in the object's frame from the last time it was 15 cm from the object until
lift-off.

- ``workspace_coverage``: the share of the 5 cm voxels two episodes' grasp points visit that only one of
  them visits, ``1 - |A ∩ B| / |A ∪ B|``. Two episodes that sweep the same region in another order count
  as alike.
- ``approach_spread``: the angle between the directions two lifts of an object come from, averaged over
  the lifted objects.
- ``behavior_diversity``: the root-mean-square distance between two lifts' approach paths, each resampled
  to 16 points along its length, averaged over the lifted objects. Each object's novelty detail is the
  median, over its lifts, of each path's distance to the nearest seed path.
- ``start_spread``: the root-mean-square distance between two episodes' start positions in the horizontal
  plane, over the objects that start in more than one place. Its details list each object, with the seeds'
  spread beside it, so an object the seeds move but the dataset pins shows.

``exact_duplicates`` is the share of episodes whose actions repeat an earlier episode's bit for bit.

Adding a metric
---------------

A metric is a function registered with ``autodata_metrics.metric``. It names the evidence it
requires. Without that evidence, the report says which key is missing and does not call the function.

.. code-block:: python

   from autodata_metrics import Value, metric


   @metric(
       "slowest_plan",
       area="diagnostics",
       signal="generation performance",
       scope="run",
       unit="s",
       sample="attempts",
       better="lower",
       requires=["attempts.extras.planning_time_s"],
   )
   def slowest_plan(ev):
       """Longest motion-planning time of any attempt."""
       times = [attempt["extras"]["planning_time_s"] for attempt in ev.attempts]
       return Value(max(times), n=len(times))

An evidence key is one of:

- an HDF5 path present in every episode, such as ``step_success``;
- ``env_args.<path>`` or ``run.<path>``;
- ``attempts`` or ``attempts.<path>``;
- ``source`` or ``source.<key>``.

Built-in metrics live in ``autodata_metrics/``. Load your own with
``--metrics my_package.my_metrics``. Raise a metric's ``version`` whenever its definition changes.

When you add an algorithm or an embodiment:

- An algorithm reports per-attempt facts by overriding ``attempt_extras()``. Metrics that require
  those facts then apply to its runs.
- An embodiment's adapter describes its pose actions through ``metric_context()``. Add datasheet
  ``hardware_limits`` to its YAML to compute ``joint_limit_ratio``.
- List the workflow's expected metrics in its end-to-end test, so a metric that stops applying fails
  the test.
