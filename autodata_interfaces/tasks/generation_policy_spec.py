# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationPolicy:
    """Configuration settings for data generation processes.

    General parameters:
        name: Identifier for the data-generation run.
        seed: RNG seed applied to ``random``/``numpy``/``torch`` for reproducibility.
        num_trials: Number of demos to generate.
        guarantee_success: If True, retry until ``num_trials`` successful demos; if False,
            stop after ``num_trials`` attempts regardless of success.
        keep_failed: If True, also export failed trials (useful for debugging low success rates).
        source_dataset_path: Path to the source dataset of human demos.
        generation_path: Path the generated dataset is written to.
        task_name: Name of the task being generated.
        use_skillgen: Whether SkillGen is used to generate motion trajectories.
        use_navigation_controller: Whether a navigation controller generates loco-manipulation
            trajectories.

    Segment stitching parameters:
        select_src_per_subtask: If True, re-select a source demo for every subtask. If False, the
            first subtask's selection sticks for the rest of the episode.
        select_src_per_arm: If True, each EEF picks its own source demo. If False, all EEFs share
            the source demo selected by the first arm to choose.
        transform_first_robot_pose: If True, prepend the recorded EEF pose to every subtask's
            target-pose sequence (anchors interpolation to the robot's recorded pose, not the
            controller target). If False, only the first subtask uses this anchoring.
        interpolate_from_last_target_pose: If True, non-first subtasks interpolate from the last
            executed waypoint instead of from the live robot pose.
    """

    # --- general parameters ---
    name: str = "demo"
    seed: int = 1
    num_trials: int = 10
    guarantee_success: bool = True
    keep_failed: bool = False
    source_dataset_path: str | None = None
    generation_path: str | None = None
    task_name: str | None = None
    use_skillgen: bool = False
    use_navigation_controller: bool = False

    # --- segment stitching ---
    select_src_per_subtask: bool = False
    select_src_per_arm: bool = False
    transform_first_robot_pose: bool = False
    interpolate_from_last_target_pose: bool = True
