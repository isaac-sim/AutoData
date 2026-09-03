Release Notes
=============

v0.1.0
------

This initial release of Autodata provides a standalone framework for transforming a small
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
  Autodata, a published platform support matrix, and unit, end-to-end, and data-generation
  performance tests.

Known limitations:

- **Platform support:** Autodata currently supports Linux x86_64 systems with an NVIDIA RTX GPU.
  See the :doc:`support matrix <../quickstart/support_matrix>` for the complete hardware and
  software requirements.
- **SkillGen:** SkillGen is currently single-arm only and requires the optional cuRobo
  installation. Bimanual collision-aware generation is not supported.
- **Dataset validation:** The validation tool checks HDF5 structure and metadata only. Semantic
  correctness, task success, and action reproducibility must be confirmed by replaying the dataset
  in simulation.
