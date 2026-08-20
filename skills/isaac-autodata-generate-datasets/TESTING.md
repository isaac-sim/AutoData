<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing: isaac-autodata-generate-datasets

The skill passed its trigger evaluation and a live Franka MimicGen acceptance run using the base Isaac Lab environment. The comparative
with-skill/without-skill benchmark remains unmeasured; `BENCHMARK.md` intentionally retains those pending cells.

## Required environment

Isaac AutoData needs real hardware and infrastructure:

- **Linux host** with an **NVIDIA GPU** and a driver meeting the Isaac Sim requirements.
- **Docker** + the **NVIDIA Container Toolkit**.
- An **NGC account** (`docker login nvcr.io`) to pull the Isaac Sim base image.
- **Git LFS** to pull the example datasets.
- For SkillGen tasks: the **cuRobo** container variant (`./docker/run_docker.sh -c`).

The first container build can take ~30 minutes. All generation commands run **inside** the
container, where the repo is mounted at `/workspaces/isaac_autodata`.

## Recorded live acceptance run

Executed inside the default `./docker/run_docker.sh` container on Linux with an NVIDIA RTX 3090:

- Helper: `skills/isaac-autodata-generate-datasets/scripts/run_generation.sh`
- Algorithm/environment: `mimicgen` / `Isaac-Stack-Cube-Franka-IK-Rel-v0`
- Source: `datasets/annotated_datasets/dataset_franka_annotated.hdf5`
- Parameters: 10 trials, 10 parallel environments, `--viz none`
- Output: `datasets/generated_dataset_franka_base_env_smoke.hdf5`
- Result: strict validation passed — 5 successful episodes; the expected base environment ID and simulation arguments were present.

The validator was run with `HDF5_USE_FILE_LOCKING=FALSE`, which is required after Isaac Sim has held the HDF5 file open.

## Trigger evaluation (can run doc-only, no GPU)

The 4 positive and 3 negative cases in `evals/evals.json` test whether the skill activates on
the right prompts and stays silent on distractors. These can be graded without a GPU by checking
the agent's routing and its proposed commands/YAML against each case's `expected_behavior`.

Suggested prompts (from `evals/evals.json`):

- Positive: "Use Isaac AutoData to generate 10 Franka cube-stacking demonstrations ... and
  validate the output." → expect mimicgen + franka YAML + validate_dataset.py --strict.
- Positive: "bimanual pick-and-place demonstrations for the Fourier GR-1" → expect dexmimicgen.
- Positive: "collision-aware Franka bin-stacking trajectories with SkillGen" → expect skillgen
  + cuRobo container.
- Negative: "Train a PPO policy in Isaac Lab on cartpole" → skill stays silent.

## Live execution (needs GPU host)

To verify end-to-end generation:

1. Provision a Linux + NVIDIA GPU host and clone Isaac AutoData with submodules + LFS.
2. Start the container: `./docker/run_docker.sh` (or `-c` for SkillGen).
3. Run the full helper invocation for the `pos-mimicgen-franka` case:

   ```bash
   skills/isaac-autodata-generate-datasets/scripts/run_generation.sh \
       --alg mimicgen \
       --env-name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --task isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output datasets/generated_dataset_franka_quickstart.hdf5 \
       --trials 10 --num-envs 10 --viz none
   ```

4. Confirm the output HDF5 passes `HDF5_USE_FILE_LOCKING=FALSE /isaac-sim/python.sh scripts/validate_dataset.py --strict <output>`.
5. Repeat with and without the skill to fill the `BENCHMARK.md` result table (task completion
   rate, algorithm/container correctness, token consumption, wall-clock time).

## Populating BENCHMARK.md

Replace each `<pending — run on hardware>` cell with the measured value from the with-skill and
without-skill runs, and record the exact LLM model versions in the "Agents Used" table.
