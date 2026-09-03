<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BENCHMARK: autodata-generate-datasets

Measures whether the skill improves an agent's ability to drive Autodata correctly,
compared to the same agent without the skill.

## Agents Used

| Harness | LLM version | Role |
|---------|-------------|------|
| Claude Code | `<pending — record model version at run time>` | Primary evaluation harness |
| Codex | `<pending — record model version at run time>` | Portability / cross-harness check |

## Metrics

- **Task completion rate** — did the agent produce a runnable, correct `generate_dataset.py`
  invocation (or valid YAML edit) for the task?
- **Algorithm correctness** — did the agent pick the correct `--alg` (mimicgen / dexmimicgen /
  skillgen) for the embodiment/task?
- **Container correctness** — did the agent identify when the cuRobo (`-c`) container is
  required (SkillGen) vs the default container?
- **Token consumption** — total tokens used to reach the answer.
- **Wall-clock time** — time to produce the answer.

## Test Tasks

Drawn from `evals/evals.json`:

1. `pos-mimicgen-franka` — generate 10 Franka cube-stack demos + validate.
2. `pos-dexmimicgen-gr1` — bimanual GR-1 pick-and-place generation.
3. `pos-skillgen-container` — SkillGen Franka bin-stacking (cuRobo container).
4. `pos-yaml-edit` — add a subtask to a task descriptor.
5. `neg-rl-training` — RL training prompt (skill should stay silent).
6. `neg-teleop-setup` — teleop hardware prompt (skill should stay silent).
7. `neg-generic-hdf5` — non-Isaac data pipeline prompt (skill should stay silent).

## Results: With Skill vs Without Skill

| Task | Metric | Without skill | With skill |
|------|--------|---------------|------------|
| pos-mimicgen-franka | Task completion | `<pending — run on hardware>` | `<pending — run on hardware>` |
| pos-mimicgen-franka | Algorithm correct | `<pending — run on hardware>` | `<pending — run on hardware>` |
| pos-dexmimicgen-gr1 | Algorithm correct | `<pending — run on hardware>` | `<pending — run on hardware>` |
| pos-skillgen-container | Container correct | `<pending — run on hardware>` | `<pending — run on hardware>` |
| pos-yaml-edit | Task completion | `<pending — run on hardware>` | `<pending — run on hardware>` |
| Negative cases (3) | Correct silence rate | `<pending — run on hardware>` | `<pending — run on hardware>` |
| Aggregate | Task completion rate | `<pending — run on hardware>` | `<pending — run on hardware>` |
| Aggregate | Tokens (mean) | `<pending — run on hardware>` | `<pending — run on hardware>` |
| Aggregate | Wall-clock (mean) | `<pending — run on hardware>` | `<pending — run on hardware>` |

## Notes on Live Execution

Command-correctness, algorithm/container-choice, and YAML-authoring rows can be graded doc-only
(no GPU). Rows that require **executing** a generation run (verifying a dataset is actually
produced and passes `validate_dataset.py`) need a Linux host with an NVIDIA GPU and the Isaac Sim
container; those are deferred until such a host is available. See `TESTING.md`.
