<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# YAML Configuration Guide

Isaac AutoData runs are driven by declarative YAML. Start from the shipped examples in
`isaac_autodata_examples/` (`tasks/`, `embodiments/`, `env_profiles/`) and adapt them.

## Task descriptor

Top-level fields (see `assets/franka_cube_stack.example.yaml`):

- `name` — task name.
- `description` — human-readable task description.
- `algo` — the generation algorithm this descriptor is written for (`mimicgen`, `dexmimicgen`,
  or `skillgen`).
- `generation_policy` — run-level policy:
  - `name` — policy name.
  - `seed` — RNG seed for reproducibility.
  - `num_trials` — number of demos to attempt.
  - `guarantee_success` — keep generating until `num_trials` successes are collected.
  - `keep_failed` — whether failed trials are written out.
  - `source_dataset_path`, `generation_path`, `task_name` — usually `null` (set via CLI).
  - `use_skillgen`, `use_navigation_controller` — algorithm toggles.
  - `select_src_per_subtask`, `select_src_per_arm` — source-demo selection granularity.
  - `transform_first_robot_pose`, `interpolate_from_last_target_pose` — transform behavior.
- `subtasks` — a map keyed by arm/robot name (e.g. `franka:`) to an ordered list of subtasks.

### Subtask fields

Each subtask entry:

- `object_ref` — the scene object this subtask acts on (e.g. `cube_2`).
- `description` — human-readable subtask description.
- `subtask_term_signal` — the annotation signal that marks this subtask complete
  (e.g. `grasp_1`, `stack_1`). **Omit on the final subtask** — the last subtask ends the
  trajectory and has no termination signal.
- `selection_strategy` — how a source segment is chosen (e.g. `nearest_neighbor_object`).
- `selection_strategy_kwargs` — strategy parameters (e.g. `nn_k: 3`).
- `subtask_term_offset_range` — `[min, max]` step offset applied to the termination point.
  Use `[0, 0]` for the final subtask.
- `action_noise` — action-space noise injected for diversity (e.g. `0.03`).
- `num_interpolation_steps` — steps used to interpolate between segments.
- `num_fixed_steps` — fixed steps held at the segment boundary.
- `apply_noise_during_interpolation` — whether `action_noise` applies during interpolation.

## Embodiment config

Robot kinematics adapter (e.g. `franka_ik_rel.yaml`, `gr1_ik_abs.yaml`, `g1_ik_abs.yaml`).
Match the embodiment to the algorithm: bimanual embodiments (GR-1, G1) pair with `dexmimicgen`;
Franka pairs with `mimicgen` or, for the SkillGen variants, `franka_ik_rel_skillgen.yaml`.

## Environment profile

Optional. Apply a profile (e.g. `franka_bin_stack.yaml`) to create a task variant without
defining a new simulation environment. Passed via `--env_profile`.

## Editing tips

- Keep the subtask order consistent with the annotated `subtask_term_signal`s in the source
  dataset.
- When adding a subtask, copy an existing block and change `object_ref`, `description`, and
  `subtask_term_signal`; keep the interpolation/noise fields unless you have a reason to change.
- Preserve the final-subtask convention (no term signal, `[0, 0]` offset).
