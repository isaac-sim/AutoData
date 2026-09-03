---
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: autodata-generate-datasets
description: Generate amplified robot-demonstration datasets with NVIDIA Autodata. Record and annotate demonstrations, author task-descriptor / embodiment / environment-profile YAML, choose the right generation algorithm (MimicGen for single-arm, DexMimicGen for bimanual, SkillGen for single-arm, collision-aware motion-planned transitions), run parallel simulation generation with generate_dataset.py, and validate the resulting HDF5 datasets. Use when working with Autodata, MimicGen/DexMimicGen/SkillGen demonstration generation, task descriptors, or HDF5 robot-demo datasets in the Isaac Lab / Isaac Sim stack. Do not use for general Isaac Lab RL policy training, teleoperation hardware setup, or non-Isaac data pipelines.
---

# Autodata: Generate Datasets

## When to Use This Skill

Use this skill when a user wants to produce robot-demonstration training data with
**NVIDIA Autodata** — the framework that amplifies a small set of annotated human
demonstrations into large, diverse HDF5 datasets using parallel Isaac Lab simulation.

Trigger this skill when the user asks to:

- Generate or amplify a dataset with Autodata / MimicGen / DexMimicGen / SkillGen.
- Record or annotate source demonstrations for data generation.
- Author or edit a task-descriptor, embodiment, or environment-profile YAML.
- Choose which generation algorithm or container an embodiment/task needs.
- Validate a generated HDF5 dataset.

Do **not** trigger this skill for: training an RL policy in Isaac Lab (PPO/cartpole/etc.),
setting up teleoperation hardware (Apple Vision Pro, VR), building/debugging cuRobo CUDA
kernels, generic host setup (NVIDIA Container Toolkit install), or non-Isaac data pipelines
(e.g. converting a ROS bag to HDF5).

## Prerequisites (state these before starting)

Autodata runs **only inside its Docker dev container** on a
Linux host with an NVIDIA GPU. Commands in this skill assume you are inside the container,
where the repo is mounted at `/workspaces/autodata`. Use `/isaac-sim/python.sh` for Python commands rather than host aliases.
If no container is running, tell the user to start one first:

```bash
./docker/run_docker.sh        # default dev container
./docker/run_docker.sh -c     # include cuRobo — REQUIRED for SkillGen
```

The first build can take up to ~30 minutes and requires an NGC login (`docker login nvcr.io`).
Do not attempt to provision hardware or bypass the container — surface the requirement and let
the user start it.

## Workflow

Autodata follows a four-stage loop: **record -> annotate -> generate -> validate**.
Most requests touch the last two stages (a pre-annotated dataset already exists).

### Step 1 — Pick the generation algorithm and container

Choose based on the robot embodiment and task. See `references/algorithm-selection.md` for the
full decision guide.

| Algorithm | Use for | Container |
|-----------|---------|-----------|
| `mimicgen` | Single-arm tasks (e.g. Franka cube stacking) | default (`./docker/run_docker.sh`) |
| `dexmimicgen` | Coordinated multi-arm / bimanual tasks (e.g. GR-1, G1) | default |
| `skillgen` | Single-arm Franka collision-aware, motion-planned transitions (current support) | cuRobo (`./docker/run_docker.sh -c`) |

SkillGen currently supports only single-arm Franka tasks with shipped cuRobo planner profiles. For bimanual tasks, use `dexmimicgen`; bimanual collision-aware planning is not available in the current release.

SkillGen requires the cuRobo container; the other two do not.

### Step 2 — Assemble the configuration

A generation run is driven by declarative YAML:

- **Task descriptor** — subtask order, `subtask_term_signal`, `selection_strategy` (+ kwargs),
  `action_noise`, interpolation, and a `generation_policy` block (`seed`, `num_trials`,
  `guarantee_success`, `keep_failed`, ...). See `assets/franka_cube_stack.example.yaml` and
  `references/yaml-config-guide.md`.
- **Embodiment** — robot kinematics config (e.g. `franka_ik_rel.yaml`, `gr1_ik_abs.yaml`,
  `g1_ik_abs.yaml`).
- **Environment profile** (optional) — apply a profile to create a task variant without
  defining a new simulation environment.

Start from the shipped examples in `autodata_examples/` (`tasks/`, `embodiments/`,
`env_profiles/`) and adapt rather than writing from scratch.

### Step 3 — Generate

Use the bundled helper for generation; it invokes `generate_dataset.py` with the verified container interpreter. Required helper flags are `--alg`, `--task`, `--embodiment`, and `--input`.
Use `--env-name`, `--output`, `--trials`, and `--num-envs` for a reproducible run. Full direct-CLI
flag reference is in `references/cli-reference.md`.

Quickstart happy path (10 Franka cube-stacking demos with MimicGen):

```bash
skills/autodata-generate-datasets/scripts/run_generation.sh \
    --viz none \
    --env-name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --alg mimicgen \
    --trials 10 \
    --num-envs 10 \
    --task autodata_examples/tasks/franka_cube_stack.yaml \
    --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
    --input ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
    --output ./datasets/generated_dataset_franka_quickstart.hdf5
```

`--num_envs` controls how many Isaac Lab environments run in parallel; increase it (within GPU
memory) to speed up generation. For SkillGen, remember to launch with the `-c` (cuRobo) container.

The bundled `skills/autodata-generate-datasets/scripts/run_generation.sh` helper resolves the Isaac Sim interpreter and validates
the completed output automatically (see its `--help`).

### Step 4 — Validate

Confirm the generated HDF5 is well-formed before using it for training:

```bash
HDF5_USE_FILE_LOCKING=FALSE /isaac-sim/python.sh scripts/validate_dataset.py ./datasets/generated_dataset_franka_quickstart.hdf5
```

The script exits non-zero if any file is invalid, so an agent can gate on the result.
`validate_dataset.py` accepts multiple files / globs and prints a per-file summary (episode
count, env id, sim args) followed by any issues.

### Step 5 — Replay (optional)

To view a generated Franka dataset in Kit, use the bundled helper:

```bash
bash skills/autodata-generate-datasets/scripts/replay_franka_generation.sh \
    datasets/generated_dataset_franka_quickstart.hdf5
```


### Step 6 (optional, earlier stages) — Record and annotate

If no annotated source dataset exists yet:

- **Record** teleop demonstrations (see the product's workflow docs for the recording step).
- **Annotate** subtask signals with `scripts/annotate_demos.py` (`--env_name`,
  `--task_descriptor`, `--embodiment`, `--input_file`, `--output_file`). Interactive controls:
  `N` play/resume, `B` pause, `S` mark a subtask signal, `Q` skip episode. Pass `--auto`
  (with `--headless` supported) to annotate termination signals without keyboard input;
  `--signal_obs_group` selects the observation group (default `subtask_terms`). SkillGen start
  signals still require manual mode.

## Common Pitfalls

- **Wrong container for SkillGen** — SkillGen needs the cuRobo container (`-c`); the default
  container will fail. See `references/algorithm-selection.md`.
- **Bimanual task with `mimicgen`** — use `dexmimicgen` for coordinated multi-arm embodiments.
- **Bimanual collision-aware planning** — not supported in the current release; do not select `skillgen`.
- **Skipping validation** — always run `validate_dataset.py` before treating a dataset as ready.
- **Malformed final subtask** — the last subtask in a task descriptor has no `subtask_term_signal`
  and a `[0, 0]` term offset (it ends the trajectory). Preserve that convention when editing.

## References

- `references/algorithm-selection.md` — MimicGen vs DexMimicGen vs SkillGen decision guide.
- `references/cli-reference.md` — condensed reference for the three CLI scripts and all flags.
- `references/yaml-config-guide.md` — annotated task-descriptor / embodiment / env-profile fields.
- `assets/franka_cube_stack.example.yaml` — copy-adaptable task-descriptor fixture.
