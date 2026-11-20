# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


class _TestPaths:
    def __init__(self):
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        self.tests_dir = os.path.dirname(utils_dir)
        self.repo_root = os.path.dirname(self.tests_dir)

        # Test data directory
        self.test_data_dir = os.path.join(self.tests_dir, "test_data")

        # Config directories
        self.examples_dir = os.path.join(self.repo_root, "autodata_examples")
        self.scripts_dir = os.path.join(self.repo_root, "scripts")
        self.generate_dataset_script = os.path.join(self.scripts_dir, "generate_dataset.py")
        self.tasks_dir = os.path.join(self.examples_dir, "tasks")
        self.embodiments_dir = os.path.join(self.examples_dir, "embodiments")
        self.env_profiles_dir = os.path.join(self.examples_dir, "env_profiles")

        self.python_path = sys.executable


TestPaths = _TestPaths()

# Metrics a run recorded with --record_metric_inputs must compute, per workflow, besides start_layout_spread and
# workspace_spread, which need two valid episodes. No workflow here records a camera, so no image checks.
MIMICGEN_FRANKA_METRICS = {
    "generation_success_rate",
    "demos_per_gpu_hour",
    "time_to_dataset",
    "correctness_pass_rate",
    "final_success_rate",
    "action_rotation_error",
    "action_saturation",
    "motion_smoothness",
    "joint_limit_ratio",
}
# Absolute pose actions are not clipped (no action_saturation) and the GR1 and G1 configs give no hardware_limits
# (no joint_limit_ratio).
DEXMIMICGEN_METRICS = {
    "generation_success_rate",
    "demos_per_gpu_hour",
    "time_to_dataset",
    "correctness_pass_rate",
    "final_success_rate",
    "action_rotation_error",
    "tracking_error",
    "motion_smoothness",
}
# One hand step of over 8 cm in any of a benchmark's hundreds of episodes makes workspace_spread not applicable.
BENCHMARK_OPTIONAL_METRICS = frozenset({"workspace_spread"})
