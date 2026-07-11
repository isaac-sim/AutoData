# Isaac Auto Data

## Dependencies

This repo depends on [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), which in turn depends on [IsaacLab](https://github.com/isaac-sim/IsaacLab). Both are pulled in automatically via nested Git submodules.

```
IsaacAutoData
└── submodules/
    └── IsaacLab-Arena/
        └── submodules/
            └── IsaacLab/
```

## Cloning

Clone the repo and initialize all submodules:

```bash
git clone --recurse-submodules git@github.com:isaac-sim/Isaac-AutoData.git
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### Datasets (Git LFS)

Dataset files are stored in Git LFS. Install Git LFS and pull the files:

```bash
git lfs install
git lfs pull
```

## Installation Option 1 (conda env)

Use `installer.sh` at the repo root. It will create a conda env and install Isaac Sim, Isaac Lab, Arena, and Isaac Auto Data.

Prerequisites: `conda` and [`uv`](https://docs.astral.sh/uv/) must be on your PATH.

### 1. Create the conda env

```bash
./installer.sh -c
```

This creates a conda env named `isaac_autodata` with Python 3.12.

### 2. Install everything into the env

```bash
conda activate isaac_autodata
./installer.sh -i
```

`-i` installs Isaac Sim 6.0.0, CUDA-enabled PyTorch, Isaac Lab, Isaac Lab - Arena, and this repo (editable).

You can also combine both steps in one call:

```bash
./installer.sh -c -i
```

Run `./installer.sh -h` to see all options.

## Installation Option 2 (docker)

### 1. Build and launch into the docker container (all deps are automatically set up in the container)

```bash
./docker/run_docker.sh
```

Use `-R` to force a rebuild.

## Verify the installation

Verify the Isaac Lab installation:

```bash
python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/tutorials/00_sim/create_empty.py --viz kit
```

Verify that Isaac Lab Mimic runs:

```bash
python submodules/IsaacLab-Arena/submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--viz kit \
--num_envs 3 \
--generation_num_trials 5 \
--input_file ./datasets/annotated_dataset.hdf5 \
--output_file ./datasets/generated_dataset_small.hdf5
```

Verify that Isaac Lab Arena runs:

```bash
python submodules/IsaacLab-Arena/isaaclab_arena/scripts/imitation_learning/replay_demos.py \
--viz kit \
--device cpu \
--enable_cameras \
--dataset_file "./datasets/ranch_bottle_into_fridge_annotated.hdf5" \
put_item_in_fridge_and_close_door \
--object ranch_dressing_hope_robolab \
--embodiment gr1_pink
```

## Contributing

### Formatting

Run the pre-commit formatter using:

```bash
pre-commit run --all-files
```

## Continuous Integration

GitHub Actions runs on every pull request and on pushes to `main`
(`.github/workflows/ci.yml`):

- **`pre_commit`** — runs `pre-commit run --all-files` (same hooks as local).
- **`test_e2e`** — runs the correctness suites inside the repo Docker image:
  `isaac_autodata_tests/e2e/` (GPU/Sim data generation) plus
  `isaac_autodata_tests/interfaces/` and `isaac_autodata_tests/core/` (CPU unit
  tests). The heavier `datagen_perf/` success-rate benchmarks are left to the
  nightly.

The test job and the manual nightly (`.github/workflows/nightly.yml`,
`workflow_dispatch`) both call one script, which is also the way to reproduce a
CI failure locally on a GPU host:

```bash
./scripts/ci/run_tests.sh
```

It wraps `./docker/run_docker.sh -c` (cuRobo enabled) and runs pytest inside the
container. Override behavior with environment variables: `TEST_PATH` (one or
more space-separated test paths; defaults to the whole tree), `PYTEST_MARK`
(marker filter; empty runs everything), and `FORCE_REBUILD=true` (image rebuild,
used by the nightly).

Test datasets are pulled from Git LFS in CI; no external credentials are
required. The jobs run on `[self-hosted, gpu]` runners.
