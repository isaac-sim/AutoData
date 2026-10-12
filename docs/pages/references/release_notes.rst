Release Notes
=============

Arena 0.3 / Isaac Lab 3.0 Upgrade
-----------------------------------

This checkout pins Arena 0.3.1 at ``481f5ae5f19df7bc5d24a17240fc0d96ac81eaf7`` and its nested
Isaac Lab commit ``28a386f063e41c04c07f50e63eefb83fd8408fbe`` (v3.0.0-EA plus 3 commits),
with Isaac Sim 6.1.0 Docker images. See the
:doc:`support matrix <../quickstart/support_matrix>` for the complete version set.

* Workflow commands, environment profiles, and source dataset metadata use Lab's
  ``IsaacContrib-*`` task names directly.
* Generation and annotation default to state observations. Use AutoData's ``--enable_cameras``
  option to include a task's configured image observations.
* Annotation signal lengths now match action lengths. Generation and annotation preserve nonzero
  exit status when they fail.
* GR1 generation shares the USD-to-URDF conversion across its independent IK controllers.
* Docker constrains runtime dependencies, checks package consistency, and supports the pinned
  cuRobo release with Warp 1.16.
* The conda installer uses Arena's locked Sim 6.1.0.0/Lab/PyTorch packages, supports named
  environments with ``-n``, and includes AutoData's test tools. It checks prerequisites before
  creating an environment and refuses to replace an existing environment with ``-c``.

Rebuild the Docker image when upgrading, or rerun ``./conda_installer.sh -i -n <environment-name>``
for a Python 3.12 conda environment. See :doc:`../quickstart/installation` for both setup paths.

v0.1.0
------

This initial release of AutoData provides a standalone framework for transforming a small
set of annotated demonstrations into larger robot-learning datasets. It brings data-generation
algorithm parity with Isaac Lab Mimic without depending on the Isaac Lab Mimic package and covers
the workflow from demonstration annotation through parallel generation, validation, and replay.

Key features of this release include:

- **Generation algorithms:** MimicGen for single-arm tasks, DexMimicGen for coordinated bimanual
  tasks, and SkillGen for collision-aware transit planning with cuRobo.
- **Declarative task integration:** Reusable YAML task descriptors and embodiment configurations
  define subtask signals, source-selection strategies, generation policies, observations, and
  task-space action mappings without requiring an Isaac Lab Mimic environment configuration.
- **Isaac Lab and Arena interoperability:** A unified datastream and embodiment-adapter layer works
  with standard Isaac Lab and Isaac Lab-Arena environments. Environment profiles create scene,
  reset-randomization, and planner variants without registering duplicate environments.
- **Dataset tooling:** Tools for manual and automatic demonstration annotation,
  parallel data generation, structural HDF5 validation, and replay using Isaac Lab-compatible
  datasets.
- **Example workflows:** Complete tutorials and pre-annotated source datasets for Franka cube
  stacking with MimicGen, Fourier GR-1 and Unitree G1 pick-and-place with DexMimicGen, and Franka
  cube and bin stacking with SkillGen.
- **Reproducible installation:** A Docker development environment for MimicGen and DexMimicGen, an
  opt-in cuRobo image for SkillGen, and an optional conda installation with pinned Isaac Sim,
  Isaac Lab, and Isaac Lab-Arena revisions.
- **Migration and developer documentation:** A migration guide for moving Isaac Lab Mimic tasks to
  AutoData, a published platform support matrix, and unit, end-to-end, and data-generation
  performance tests.

Known limitations:

- **Platform support:** AutoData currently supports Linux x86_64 systems with an NVIDIA RTX GPU.
  See the :doc:`support matrix <../quickstart/support_matrix>` for the complete hardware and
  software requirements.
- **SkillGen:** SkillGen is currently single-arm only and requires the optional cuRobo
  installation. Bimanual collision-aware generation is not supported.
- **Dataset validation:** The validation tool checks HDF5 structure and metadata only. Semantic
  correctness, task success, and action reproducibility must be confirmed by replaying the dataset
  in simulation.
