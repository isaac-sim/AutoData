#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Replay a generated Franka dataset in Isaac Sim.
# Run this script from the repo root, inside the AutoData container or with the repo's uv environment.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset.hdf5>" >&2
  exit 1
fi

dataset_file="$1"
if [[ ! -f "$dataset_file" ]]; then
  echo "ERROR: Dataset file does not exist: $dataset_file" >&2
  exit 1
fi

if [[ -x /isaac-sim/python.sh ]]; then
  python_cmd=(/isaac-sim/python.sh)
elif python -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("isaaclab_arena") is None)' 2>/dev/null; then
  python_cmd=(python)
elif [[ -x .venv/bin/python ]]; then
  python_cmd=(.venv/bin/python)
else
  echo "ERROR: Isaac Lab-Arena was not found. Start ./docker/run_docker.sh, or run uv sync --locked." >&2
  exit 1
fi

exec "${python_cmd[@]}" submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
    --viz kit \
    --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
    --num_envs 1 \
    --dataset_file "$dataset_file"
