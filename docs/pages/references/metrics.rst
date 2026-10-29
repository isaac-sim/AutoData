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

Each metric reports one of three things:

- a value, with its unit and sample count;
- ``not_applicable``, with the reason, such as a missing recorded input;
- ``error``, when it fails. The other metrics are still computed.

The report reads the run named by the dataset's run id. A result file from another run, such as the
earlier result of an interrupted rerun, is set aside with a warning, and the run is reported from its own
attempt log. The report refuses a dataset, result file and attempt log whose episode, attempt or success
counts disagree, as they do after datasets are merged. When the result file records the run's source, the
report also refuses a different ``--source``; without that record, the source is not verified.

Gating
------

``--gate`` reads metric bounds from YAML and makes the command fail when one is violated:

.. code-block:: yaml

   generation_success_rate: {min: 0.3, version: 1}
   planning_time: {max: 5.0, key: p95}

A gated metric also violates the gate if it is not applicable, fails, or reports a different
``version``. A bound that is not a finite number is an error.

Built-in metrics
----------------

:autodata_metrics_table:

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
- An embodiment's adapter describes its pose actions through ``metric_context()``.
- List the workflow's expected metrics in its end-to-end test, so a metric that stops applying fails
  the test.
