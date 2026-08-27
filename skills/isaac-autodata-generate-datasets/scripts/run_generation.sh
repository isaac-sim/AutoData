#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_generation.sh — assemble and run an Isaac AutoData generate_dataset.py invocation,
# then validate the output. Run this INSIDE the Isaac AutoData container, from the repo root
# (/workspaces/isaac_autodata). For SkillGen, start the container with `-c` (cuRobo) first.
#
# Usage:
#   skills/isaac-autodata-generate-datasets/scripts/run_generation.sh \
#     --alg mimicgen \
#     --env-name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
#     --task isaac_autodata_examples/tasks/franka_cube_stack.yaml \
#     --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
#     --input ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
#     --output ./datasets/generated_dataset_franka_quickstart.hdf5 \
#     --trials 10 --num-envs 10
#
# --alg, --task, --embodiment, and --input are required by generate_dataset.py.
# --no-validate skips the post-generation validation step.
set -euo pipefail

ALG=""
ENV_NAME=""
TASK=""
EMBODIMENT=""
ENV_PROFILE=""
INPUT=""
OUTPUT=""
TRIALS=""
NUM_ENVS=""
VIZ=""
VALIDATE=1

usage() {
  printf "%s\n" \
    "Usage: skills/isaac-autodata-generate-datasets/scripts/run_generation.sh [options]" \
    "" \
    "Required options:" \
    "  --alg {mimicgen|dexmimicgen|skillgen}" \
    "  --task <task_descriptor.yaml>" \
    "  --embodiment <embodiment.yaml>" \
    "  --input <annotated_dataset.hdf5>" \
    "" \
    "Optional options:" \
    "  --env-name <gym_environment_id>" \
    "  --env-profile <environment_profile.yaml>" \
    "  --output <generated_dataset.hdf5>" \
    "  --trials <count>" \
    "  --num-envs <count>" \
    "  --viz {kit|none}" \
    "  --no-validate"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alg) ALG="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --embodiment) EMBODIMENT="$2"; shift 2 ;;
    --env-profile) ENV_PROFILE="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --viz) VIZ="$2"; shift 2 ;;
    --no-validate) VALIDATE=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$ALG" || -z "$TASK" || -z "$EMBODIMENT" || -z "$INPUT" ]]; then
  echo "ERROR: --alg, --task, --embodiment, and --input are required." >&2
  usage 1
fi

if [[ "$VALIDATE" -eq 1 && -z "$OUTPUT" ]]; then
  echo "ERROR: --output is required unless --no-validate is set." >&2
  usage 1
fi

case "$ALG" in
  mimicgen|dexmimicgen|skillgen) ;;
  *) echo "ERROR: --alg must be one of mimicgen, dexmimicgen, skillgen (got '$ALG')." >&2; exit 1 ;;
esac

if ! submodule_status=$(git submodule status -- submodules/IsaacLab-Arena); then
  echo "ERROR: Could not determine submodule status." >&2
  exit 1
fi
if [[ -z "$submodule_status" ]] || grep -qE '^[+-U]' <<<"$submodule_status"; then
  echo "ERROR: IsaacLab-Arena is not at the commit pinned by this repository." >&2
  echo "       Run git submodule update --init --recursive, then restart the container." >&2
  exit 1
fi


if [[ "$ALG" == "skillgen" ]]; then
  echo "NOTE: SkillGen requires the cuRobo container (./docker/run_docker.sh -c)." >&2
fi

if [[ ! -x /isaac-sim/python.sh ]]; then
  echo "ERROR: Isaac Sim Python was not found. Start ./docker/run_docker.sh first." >&2
  exit 1
fi
python_cmd=(/isaac-sim/python.sh)

cmd=("${python_cmd[@]}" scripts/generate_dataset.py --alg "$ALG" --input_file "$INPUT")
[[ -n "$VIZ" ]]         && cmd+=(--viz "$VIZ")
[[ -n "$ENV_NAME" ]]    && cmd+=(--env_name "$ENV_NAME")
[[ -n "$TASK" ]]        && cmd+=(--task_descriptor "$TASK")
[[ -n "$EMBODIMENT" ]]  && cmd+=(--embodiment "$EMBODIMENT")
[[ -n "$ENV_PROFILE" ]] && cmd+=(--env_profile "$ENV_PROFILE")
[[ -n "$OUTPUT" ]]      && cmd+=(--output_file "$OUTPUT")
[[ -n "$TRIALS" ]]      && cmd+=(--generation_num_trials "$TRIALS")
[[ -n "$NUM_ENVS" ]]    && cmd+=(--num_envs "$NUM_ENVS")

echo "+ ${cmd[*]}"
"${cmd[@]}"

if [[ "$VALIDATE" -eq 1 ]]; then
  echo "+ HDF5_USE_FILE_LOCKING=FALSE ${python_cmd[*]} scripts/validate_dataset.py $OUTPUT"
  HDF5_USE_FILE_LOCKING=FALSE "${python_cmd[@]}" scripts/validate_dataset.py "$OUTPUT"
fi
