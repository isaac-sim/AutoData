<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Algorithm Selection Guide

Autodata ships three generation algorithms. Pick based on the robot embodiment and the
task's motion requirements.

## Decision tree

1. **Does the embodiment have two coordinated arms (bimanual / humanoid)?**
   - Yes -> `dexmimicgen` (two-arm MimicGen with subtask coordination constraints). Examples:
     Fourier GR-1, Unitree G1. Default container. Bimanual collision-aware motion planning is not
     available in the current release; do not select `skillgen` for that combination.
   - No -> continue.
2. **Do transitions between skill segments need collision-aware motion planning?**
   - Yes -> `skillgen` (single-arm Franka only in the current release). Requires the **cuRobo** container
     (`./docker/run_docker.sh -c`). SkillGen depends on a motion-planner interface; the CLI
     wires in the upstream Arena `CuroboPlanner`.
   - No -> `mimicgen` (single-arm MimicGen). Default container. Example: Franka cube stacking.

## Algorithm summary

| Algorithm | Arms | Motion planning | Container | Example task / embodiment |
|-----------|------|-----------------|-----------|---------------------------|
| `mimicgen` | Single | No | default | `Isaac-Stack-Cube-Franka-IK-Rel-v0`, `franka_cube_stack.yaml` / `franka_ik_rel.yaml` |
| `dexmimicgen` | Multi / bimanual | No | default | `gr1_pick_place.yaml` / `gr1_ik_abs.yaml`; `g1_pick_place.yaml` / `g1_ik_abs.yaml` |
| `skillgen` | Single-arm Franka (current release) | Yes (cuRobo) | `-c` (cuRobo) | `franka_bin_stack_skillgen.yaml` / `franka_ik_rel_skillgen.yaml` |

## Container reminder

- Default container: `./docker/run_docker.sh` — covers `mimicgen` and `dexmimicgen`.
- cuRobo container: `./docker/run_docker.sh -c` — **required** for `skillgen`. cuRobo compiles
  CUDA kernels for the detected GPU arch, so the first `-c` build is slower and kept as a
  separate image tag.

Choosing `skillgen` without the `-c` container is the most common failure mode.
