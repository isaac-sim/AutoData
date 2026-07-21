<div align="center">

# Isaac AutoData

### Scalable Robot Demonstration Generation for Imitation Learning

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/isaac-sim/Isaac-AutoData)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.0.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://www.linux.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE)

[Documentation](docs/index.rst) · [Getting Started](docs/pages/quickstart/first_data_generation.rst) · [Report a Bug](https://github.com/isaac-sim/Isaac-AutoData/issues) · [Discussions](https://github.com/isaac-sim/Isaac-AutoData/discussions)

</div>

---

## Overview

**Isaac AutoData** is a trajectory data-generation framework built on
[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and
[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena). Given a small set of annotated human
demonstrations, it uses parallel simulation environments to generate diverse datasets of successful robot
demonstrations for imitation learning.

AutoData splits demonstrations into object-relative skill segments. During generation, it transforms those segments
to new scene configurations, connects them into complete trajectories, executes them in simulation, and records the
successful trials as HDF5 datasets.

<p align="center">
  <img src="docs/images/autodata.gif" alt="Isaac AutoData generating robot demonstrations in parallel" width="100%">
</p>

## Why Isaac AutoData?

Imitation-learning policies require large and diverse collections of successful demonstrations. Gathering all of
that data through human teleoperation is slow and expensive, even though a small set of demonstrations often already
contains the task's essential skills.

Isaac AutoData scales those demonstrations across randomized object placements and scene configurations, reducing
the amount of manual collection needed to produce datasets for policy training.

## Key Features

- **Demonstration amplification** — Transform and recombine a small number of annotated demonstrations into larger,
  more diverse datasets.
- **Three generation algorithms** — Use MimicGen for single-arm tasks, DexMimicGen for coordinated multi-arm tasks,
  or SkillGen for collision-aware, motion-planned transitions.
- **Parallel simulation** — Generate demonstrations across multiple Isaac Lab environments at once.
- **Robot and task configuration** — Describe embodiments, tasks, subtasks, and generation policies in
  reusable YAML files.
- **Adaptable environments** — Apply environment profiles to create task variants without defining a new simulation
  environment.
- **End-to-end dataset tools** — Record, annotate, generate, validate, and replay HDF5 demonstrations.

## Quick Start

### Prerequisites

- Linux with an NVIDIA GPU and driver that meet the
  [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Docker](https://docs.docker.com/engine/install/) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Git and [Git LFS](https://git-lfs.com/)
- An [NGC](https://catalog.ngc.nvidia.com/) account for the Isaac Sim container image

### Installation

Clone the repository with its nested Isaac Lab-Arena and Isaac Lab submodules, then pull the example datasets:

```bash
git clone --recurse-submodules git@github.com:isaac-sim/Isaac-AutoData.git
cd Isaac-AutoData
git lfs install
git lfs pull
```

Log in to NGC and launch the development container:

```bash
docker login nvcr.io
./docker/run_docker.sh
```

The first launch builds the development image and opens a shell in the repository at
`/workspaces/isaac_autodata`. Subsequent launches reuse the image. Use `./docker/run_docker.sh -c` to include cuRobo
for SkillGen workflows.

Docker is the recommended setup. An optional conda installation and additional container options are described in
the [installation guide](docs/pages/quickstart/installation.rst).

### Generate Your First Dataset

Inside the container, generate ten Franka cube-stacking demonstrations from the pre-annotated dataset:

```bash
python scripts/generate_dataset.py \
    --viz kit \
    --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --alg mimicgen \
    --generation_num_trials 10 \
    --num_envs 10 \
    --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
    --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
    --input_file ./datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
    --output_file ./datasets/generated_dataset_franka_quickstart.hdf5
```

Validate the generated dataset:

```bash
python scripts/validate_dataset.py ./datasets/generated_dataset_franka_quickstart.hdf5
```

See [Your First Data Generation](docs/pages/quickstart/first_data_generation.rst) for an explanation of each step
and instructions for replaying the result.

## Example Workflows

| Workflow | Description |
|----------|-------------|
| [Franka Cube Stacking](docs/pages/workflows/franka_cube_stack_mimicgen/index.rst) | Record, annotate, and expand single-arm demonstrations with MimicGen. |
| [Humanoid Pick and Place](docs/pages/workflows/humanoid_dexmimicgen/index.rst) | Generate bimanual demonstrations for Fourier GR-1 and Unitree G1 with DexMimicGen. |
| [Motion-Planned Generation](docs/pages/workflows/skillgen/index.rst) | Use SkillGen and cuRobo to create collision-aware trajectories, including task variants built with environment profiles. |

## Project Structure

```text
Isaac-AutoData/
├── isaac_autodata_core/        # Data generation algorithms and execution
├── isaac_autodata_interfaces/  # Task, embodiment, datastream, and planner interfaces
├── isaac_autodata_utils/       # Shared utilities
├── isaac_autodata_examples/    # Example task, embodiment, and environment-profile configs
├── isaac_autodata_tests/       # Unit, end-to-end, and performance tests
├── scripts/                    # Dataset generation, annotation, and validation tools
├── datasets/                   # Example and test datasets stored with Git LFS
├── docker/                     # Reproducible development containers
├── docs/                       # Sphinx documentation
└── submodules/                 # Isaac Lab-Arena and Isaac Lab
```

## Contributing

Bug reports, feature suggestions, documentation improvements, and pull requests are welcome. Before opening a pull
request, run the repository's pre-commit checks:

```bash
pre-commit run --all-files
```

For test-suite details and common commands, see [Testing and CI](docs/pages/advanced/testing_and_ci.rst).

## Support

- **Questions and ideas** — [GitHub Discussions](https://github.com/isaac-sim/Isaac-AutoData/discussions)
- **Bug reports** — [GitHub Issues](https://github.com/isaac-sim/Isaac-AutoData/issues)
- **Isaac Sim questions** — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67)

## License

Isaac AutoData is released under the [Apache License 2.0](LICENSE).

Isaac AutoData depends on Isaac Sim, which includes components distributed under proprietary licensing terms. See
the [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
for details.

## Citation

If you use Isaac AutoData in your research, please cite:

```bibtex
@misc{isaacautodata2026,
    title  = {Isaac AutoData: Scalable Robot Demonstration Generation for Imitation Learning},
    author = {{NVIDIA Isaac AutoData Contributors}},
    year   = {2026},
    url    = {https://github.com/isaac-sim/Isaac-AutoData}
}
```

Depending on the generation algorithm used, please also cite the original
[MimicGen](https://arxiv.org/abs/2310.17596), [DexMimicGen](https://arxiv.org/abs/2410.24185), or
[SkillMimicGen](https://arxiv.org/abs/2410.18907) work. Isaac Lab users should also cite the
[Isaac Lab paper](https://arxiv.org/abs/2511.04831).

## Acknowledgements

Isaac AutoData builds on NVIDIA Isaac Sim, Isaac Lab, and Isaac Lab-Arena. Its data-generation workflows incorporate
ideas from MimicGen, DexMimicGen, and SkillMimicGen, with cuRobo providing GPU-accelerated motion planning for
SkillGen workflows.

We thank the authors and contributors of these projects, along with the broader robotics community, for their
foundational work.

---

<div align="center">

**Isaac AutoData** · [Documentation](docs/index.rst) · [GitHub](https://github.com/isaac-sim/Isaac-AutoData)

</div>
