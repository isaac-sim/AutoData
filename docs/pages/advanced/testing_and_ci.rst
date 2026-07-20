Testing and CI
==============

Test Suite
----------

The test suite lives in ``isaac_autodata_tests/`` and runs inside the dev container. It is
organized in tiers of increasing cost:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Directory
     - Contents
   * - ``interfaces/``, ``core/``
     - Fast unit tests for the pure-Python layers (task descriptors, embodiment adapters,
       datastream, selection strategies, ...). No Isaac Sim launch; run in seconds.
   * - ``e2e/``
     - End-to-end data-generation runs, one file per algorithm (MimicGen, DexMimicGen,
       SkillGen), each in single- and multi-env variants. Marked ``with_subprocess``: they
       launch the generation CLI as an Isaac Sim child process. Minutes per test.
   * - ``datagen_perf/``
     - Success-rate benchmarks (hundreds of trials per run) asserting generation quality
       thresholds per task. Nightly-scale.
   * - ``test_data/``
     - Pre-annotated source datasets (Git LFS) used by the tests — and handy as quickstart
       inputs.

Common invocations:

.. code-block:: bash

   # Fast unit tests only (seconds)
   pytest isaac_autodata_tests -m "not with_subprocess"

   # One end-to-end generation test, streaming Isaac Sim output live
   pytest -s isaac_autodata_tests/e2e/test_mimicgen_data_generation.py::test_franka_cube_stack_mimicgen_data_generation_single_env_cuda

   # Everything that launches Isaac Sim (long)
   pytest isaac_autodata_tests -m with_subprocess

.. note::

   The SkillGen end-to-end tests require the cuRobo image
   (``./docker/run_docker.sh -c``).

Debugging Isaac Sim Runs
------------------------

The end-to-end tests launch the generation CLI as a child process with a wall-clock timeout —
env var ``ISAAC_AUTODATA_SUBPROCESS_TIMEOUT``, default 1200 seconds. On expiry the child's
process group is killed and the test fails with ``TimeoutExpired``.

When a run dies silently or hangs, work down this list:

1. **Stream the console.** Run pytest with ``-s``; without it, pytest swallows the child's
   output unless the test fails.
2. **Bypass pytest.** Every e2e test is just a ``generate_dataset.py`` invocation — copy the
   command from the test file and run it directly, which also lets you shrink it
   (``--num_envs 1``, ``--generation_num_trials 1``).
3. **Capture Kit's own log.** Isaac Sim messages from the C++/USD/PhysX layers often never
   reach Python stdout. Forward settings to Kit through the AppLauncher:

   .. code-block:: bash

      python isaac_autodata_examples/generate_dataset.py ... \
          --kit_args "--/log/file=/workspaces/isaac_autodata/kit.log --/log/level=verbose --/log/async=false"

   ``--/log/async=false`` is the important one for crashes: asynchronous logging loses the
   final messages when the process dies.
4. **Unmute USD diagnostics.** Isaac Sim mutes USD warnings by default; add
   ``--/persistent/app/usd/muteUsdDiagnostics=false`` to ``--kit_args``.


Linting
-------

Pre-commit hooks enforce the style guide (black, flake8, isort, pyupgrade, codespell, and
RST checks). Run them on the host before committing:

.. code-block:: bash

   pre-commit run --all-files
