#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Replay a generated Franka dataset in Isaac Sim.
# Run this script from inside the AutoData container.

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

if [[ ! -x /isaac-sim/python.sh ]]; then
  echo "ERROR: Isaac Sim Python was not found. Start ./docker/run_docker.sh first." >&2
  exit 1
fi

exec /isaac-sim/python.sh submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tools/replay_demos.py \
    --viz kit \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --num_envs 1 \
    --dataset_file "$dataset_file"
