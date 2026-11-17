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
       --gate datasets/gate.yaml \
       --report_file datasets/generated_report.json

The command prints a summary; ``--report_file`` writes the full report as JSON, including inputs, checks,
warnings and metric results.

.. code-block:: text

   AutoData metrics · generated.hdf5 · run cc273b3f · source dataset_franka_annotated.hdf5
   20/20 episodes valid · 11 computed · 5 not computed · gate failed on 1 of 2

                                        Value   Source  Based on
   VALIDITY
     action_rotation_error        p95 8.34e-7  4.77e-7  4,803 poses
     action_saturation                  4.06%       0%  4,803 steps
   ✓ correctness_pass_rate               100%           20 episodes
     final_success_rate                   95%           20 episodes
     joint_limit_ratio           velocity 40%      10%  20 episodes
     motion_smoothness           p95 142 m/s³  40 m/s³  4,743 poses
   DIVERSITY
     start_layout_spread               217 mm   222 mm  20 episodes
     workspace_spread                 46.9 mm  44.7 mm  20 episodes
   VIABILITY
     demos_per_gpu_hour       863 demos/GPU-h           20 demos
   ✗ generation_success_rate              40%           50 attempts · 95% CI 27.6–53.8%
     time_to_dataset                    105 s           1 run

   NOT COMPUTED
     not reported by this algorithm: planning_failure_rate, planning_time
     no camera streams: image_stream_pass_rate
     no pose action is recorded as absolute: tracking_error
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

``--gate`` reads metric bounds from YAML and makes the command fail when one is violated.
The example report uses these bounds in ``datasets/gate.yaml``:

.. code-block:: yaml

   correctness_pass_rate: {min: 1.0, version: 1}
   generation_success_rate: {min: 0.5, version: 1}

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

The diversity metrics are supporting dispersion diagnostics. They describe variation within the recorded
positions, not coverage of a target distribution, object-relative behavioral novelty or policy learning value.
Each averages a distance over pairs of episodes, so it needs at least two:

- ``start_layout_spread`` [mm]: how far apart two episodes' starting scenes are. Each object's horizontal
  start distance is combined with the others' as a root-sum-square, so an object that never moves adds nothing
  and one that moves more adds more. Robots count by their root, deformable objects by their nodes' centroid.
- ``workspace_spread`` [mm]: how far one episode's hands pass from where the other's went. A hand is each
  recorded end effector, followed by the origin of its frame as the embodiment records it (a whole-body
  controller records its hands relative to the robot's body), from the episode's first recorded step. Its
  path, sampled every 2.5 mm between recorded steps, keeps one point per 5 mm cell, so retracing a path adds
  nothing; the mean distance from one path's kept points to the other's nearest kept points is averaged over
  the two directions, then over hands. Beyond 63 episodes, it measures 2,000 pairs drawn with a fixed seed
  among the episodes it reads.

An average over pairs estimates how different two demonstrations from the same producer are, so it does not
grow with the number of episodes. The spreads read episodes in a ranking by a digest of each episode's actions
and how many earlier episodes repeat them, at most the first 1,000, so the order of episodes in the file does
not matter unless two episodes repeat each other's actions exactly.

Interpreting spread
~~~~~~~~~~~~~~~~~~~

A larger spread is not necessarily better. Two distant groups of demonstrations can have a large spread
while leaving a gap between them. Moving objects can widen hand workspaces without adding new
object-relative behaviors. Use the values as context or scoped regression diagnostics, not as data-quality
acceptance scores. Coverage requires a declared target distribution and task-relevant tolerances; these
metrics do not use a target or identify uncovered conditions.

Comparing datasets
~~~~~~~~~~~~~~~~~~

Two spreads compare only on the same task and scene objects, and two workspaces only with the same hands
recorded in the same frames. Each metric's details list the reported dataset's objects or hands.
The Source column contains only the baseline value, without its breakdown or reason for being unavailable.

On the Franka stack's test datasets and runs, moving a hand's frame between its MimicGen and SkillGen
frames, 10.34 cm along the gripper, moved the workspace by up to 3.3%. Moving it the same distance across
the gripper changed the workspace by up to 42.2%. Workspace values assume each path starts at its
episode's first recorded step, the reset pose. On the test datasets and
generated runs, leaving that step out moved them by up to 7.1%, and halving the recording rate moved them by
under 1%, except for the GR1 pick-and-place seeds and run and the fridge seeds, which it made not applicable.
A hand that moves far between recorded steps blurs its path, so the workspace is not applicable when any hand
moves more than 8 cm in one recorded step.

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
