<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Reference

All commands run inside the AutoData container (repo mounted at
`/workspaces/autodata`, `python` aliased to Isaac Sim's interpreter). Grounded in the
argparse definitions in `scripts/`.

## `scripts/generate_dataset.py`

Data-generation entrypoint. Composes a `Datastream` from the task-descriptor YAML, embodiment
YAML, live env, and HDF5 source dataset, then runs the selected generation algorithm.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--alg {mimicgen,dexmimicgen,skillgen}` | yes | — | Generation algorithm. `skillgen` needs a motion planner (auto-wired from upstream cuRobo) and the `-c` container. |
| `--input_file <path.hdf5>` | yes | — | Source (annotated) dataset. |
| `--env_name <env_id>` | no | from dataset | Environment name; overrides the env recorded in the source dataset. |
| `--task_descriptor <yaml>` | yes | — | Task semantics: subtask order, signals, object refs, algo params. |
| `--embodiment <yaml>` | yes | — | Robot kinematics adapter config. |
| `--env_profile <yaml>` | no | — | Environment profile for a task variant. |
| `--output_file <path.hdf5>` | no | `./datasets/output_dataset.hdf5` | Destination for generated demos. |
| `--generation_num_trials <N>` | no | `None` | Number of demos to generate. |
| `--num_envs <N>` | no | `1` | Number of parallel simulation environments. |
| `--result_file <path>` | no | — | Where to write the generation result summary. |
| `--pause_subtask` | no | off | Pause at subtask boundaries (debugging). |
| `--visualize_plan` | no | off | Visualize the motion plan (SkillGen debugging). |

The launcher also accepts Isaac Sim `AppLauncher` flags (e.g. `--viz kit`, `--viz none`).

## `scripts/annotate_demos.py`

Annotate source demonstrations with subtask signals.

| Flag | Description |
|------|-------------|
| `--env_name <env_id>` | Env id; read from the source dataset if omitted. |
| `--task_descriptor <yaml>` | Task descriptor. |
| `--embodiment <yaml>` | Embodiment config. |
| `--input_file <path.hdf5>` | Source dataset. |
| `--output_file <path.hdf5>` | Annotated output. |
| `--auto` | Annotate termination signals without keyboard input (`--headless` supported). |
| `--signal_obs_group <name>` | Observation group holding per-subtask boolean terms (default `subtask_terms`). |

Interactive controls: `N` play/resume, `B` pause, `S` mark subtask signal, `Q` skip episode.
SkillGen start signals still require manual mode.

## `scripts/validate_dataset.py`

Validate generated HDF5 datasets. Prints a per-file summary (episode count, env id, sim args)
followed by any issues.

```bash
python scripts/validate_dataset.py <file.hdf5> [<file2.hdf5> ...]
python scripts/validate_dataset.py directory/*.hdf5
```

Required demo fields checked: `actions` (dataset), `initial_state` (group), `obs` (group).
The command exits non-zero if any file is invalid, so workflows can gate on a clean exit code.
