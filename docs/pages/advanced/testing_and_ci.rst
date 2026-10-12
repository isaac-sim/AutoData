Testing and CI
==============

Test Suite
----------

The test suite lives in ``autodata_tests/`` and runs inside the dev container. It is
organized in tiers of increasing cost:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Directory
     - Contents
   * - ``interfaces/``, ``core/``, ``utils/``
     - Fast unit tests for the pure-Python layers (task descriptors, embodiment adapters,
       datastream, selection strategies, ...). No Isaac Sim launch; run in seconds.
   * - ``e2e/``
     - Recording and annotation checks, plus data-generation runs for MimicGen, DexMimicGen,
       and SkillGen in single- and multi-env variants. Marked ``with_subprocess``: they
       launch a CLI as an Isaac Sim child process. Minutes per test. Recording tests use
       simulated SpaceMouse input; physical-device and XR interaction need manual testing.
   * - ``datagen_perf/``
     - Success-rate benchmarks (hundreds of attempts per run) asserting generation quality
       thresholds per task. Nightly-scale.
   * - ``test_data/``
     - Pre-annotated source datasets (Git LFS) used by the tests — and handy as quickstart
       inputs.

Common invocations:

.. code-block:: bash

   # Fast unit tests only (seconds)
   pytest autodata_tests -m "not with_subprocess"

   # One end-to-end generation test, streaming Isaac Sim output live
   pytest -s autodata_tests/e2e/test_mimicgen_data_generation.py::test_franka_cube_stack_mimicgen_data_generation_single_env_cuda

   # Everything that launches Isaac Sim (long)
   pytest autodata_tests -m with_subprocess

   # All tests, including the success-rate benchmarks (use the cuRobo image)
   pytest -s autodata_tests

.. note::

   The SkillGen end-to-end tests require the cuRobo image
   (``./docker/run_docker.sh -c``).

Debugging Isaac Sim Runs
------------------------

The end-to-end tests launch the generation CLI as a child process with a wall-clock timeout —
env var ``AUTODATA_SUBPROCESS_TIMEOUT``, default 1200 seconds. On expiry the child's
process group is killed and the test fails with ``TimeoutExpired``.

When a run dies silently or hangs, work down this list:

1. **Stream the console.** Run pytest with ``-s``; without it, pytest swallows the child's
   output unless the test fails.
2. **Bypass pytest.** Copy the CLI invocation from the relevant test file and run it directly.
   For generation, this also lets you shrink the run
   (``--num_envs 1``, ``--generation_num_trials 1``).
3. **Capture Kit's own log.** Isaac Sim messages from the C++/USD/PhysX layers often never
   reach Python stdout. Forward settings to Kit through the AppLauncher:

   .. code-block:: bash

      python scripts/generate_dataset.py ... \
          --kit_args "--/log/file=/workspaces/autodata/kit.log --/log/level=verbose --/log/async=false"

   ``--/log/async=false`` is the important one for crashes: asynchronous logging loses the
   final messages when the process dies.
4. **Unmute USD diagnostics.** Isaac Sim mutes USD warnings by default; add
   ``--/persistent/app/usd/muteUsdDiagnostics=false`` to ``--kit_args``.


Linting
-------

Pre-commit hooks enforce the style guide (black, flake8, isort, pyupgrade, codespell, and
RST checks). Run them inside the development container before committing:

.. code-block:: bash

   pre-commit run --all-files

Documentation
-------------

Build the current working tree with warnings treated as errors using the dedicated docs
environment described in :autodata_code_link:`<docs/README.md>`:

.. code-block:: bash

   cd docs
   source venv_docs/bin/activate
   make html SPHINXOPTS="-W --keep-going"

Open ``docs/_build/current/html/index.html`` to test local edits. ``make multi-docs`` builds
committed Git revisions, so it does not preview uncommitted workflow changes.
