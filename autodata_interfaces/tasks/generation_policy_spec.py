# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
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
        max_num_failures: Maximum additional failed attempts tolerated when ``guarantee_success``
            is True. Together with ``num_trials``, this defines a finite attempt budget. It does
            not extend fixed-attempt runs.
        keep_failed: If True, also export failed trials (useful for debugging low success rates).
        source_dataset_path: Path to the source dataset of human demos.
        generation_path: Path the generated dataset is written to.
        task_name: Name of the task being generated.
        use_skillgen: Whether SkillGen is used to generate motion trajectories.
        use_navigation_controller: Whether a navigation controller generates loco-manipulation
            trajectories.
        reset_settling_steps: Number of ordinary batched environment steps to run after reset
            before generation begins. During these steps, the resetting environment receives
            ``env.cfg.idle_action`` (zeros when absent) while other environments execute their generated actions.
        max_navigation_hold_steps: Maximum consecutive simulation steps spent waiting for a
            navigation goal at the end of its recorded segment. Reaching the limit fails the attempt.

    Segment stitching parameters:
        select_src_per_subtask: If True, re-select a source demo for every subtask. If False, the
            first subtask's selection sticks for the rest of the episode.
        select_src_per_arm: If True, each EEF picks its own source demo. If False, all EEFs share
            the source demo selected by the first arm to choose.
        source_selection_anchor: Explicit ``"<eef>.<subtask>"`` used to rank one shared episode
            source independently of EEF declaration order. ``None`` retains ordinary selection.
        source_demo_keys: Ordered subset of loaded source episodes eligible for selection. ``None``
            keeps every loaded episode eligible.
        source_candidate_count: Maximum eligible source demos tried on one reset when complete-path
            joint-trajectory solving fails the IK checks. Defaults to one, preserving ordinary generation behavior.
        transform_first_robot_pose: If True, prepend the recorded EEF pose to every subtask's
            target-pose sequence (anchors interpolation to the robot's recorded pose, not the
            controller target). If False, only the first subtask uses this anchoring.
        interpolate_from_last_target_pose: If True, non-first subtasks interpolate from the last
            executed waypoint instead of from the live robot pose.
        skill_passthrough_delay_steps: Samples to hold the previous passthrough command after a
            major source-skill transition.
        skill_passthrough_interpolation_steps: Samples used to blend a delayed passthrough
            transition to its new value.
        max_joint_step_rad: Maximum joint change between consecutive validated controller
            targets [rad]. ``None`` preserves the computed paths unchanged.
        final_hold_steps: Additional synchronized holds after the scheduled paths finish.
        terminal_convergence_steps: Maximum additional commands of the validated terminal target
            used to reach the planner pose thresholds.
    """

    # --- general parameters ---
    name: str = "demo"
    seed: int = 1
    num_trials: int = 10
    guarantee_success: bool = True
    max_num_failures: int = 50
    keep_failed: bool = False
    source_dataset_path: str | None = None
    generation_path: str | None = None
    task_name: str | None = None
    use_skillgen: bool = False
    use_navigation_controller: bool = False
    reset_settling_steps: int = 0
    max_navigation_hold_steps: int = 1000

    # --- segment stitching ---
    select_src_per_subtask: bool = False
    select_src_per_arm: bool = False
    transform_first_robot_pose: bool = False
    interpolate_from_last_target_pose: bool = True
    skill_passthrough_delay_steps: int = 0
    skill_passthrough_interpolation_steps: int = 0
    max_joint_step_rad: float | None = None
    final_hold_steps: int = 0
    terminal_convergence_steps: int = 0
    source_selection_anchor: str | None = None
    source_demo_keys: list[str] | None = None
    source_candidate_count: int = 1

    def remaining_attempts(self, num_attempts: int, num_success: int, num_pending: int = 0) -> int:
        """Return how many episodes may start without exceeding the trial or failure limit.

        Pending episodes reserve one possible success each when successful demos are requested.

        Args:
            num_attempts: Completed episodes, including failures.
            num_success: Successful completed episodes.
            num_pending: Episodes already being generated.

        Returns:
            Number of additional episodes that may start.
        """

        assert 0 <= num_success <= num_attempts and num_pending >= 0
        completed = num_success if self.guarantee_success else num_attempts
        budget = self.num_trials + (self.max_num_failures if self.guarantee_success else 0)
        return max(0, min(self.num_trials - completed, budget - num_attempts) - num_pending)

    def __post_init__(self) -> None:
        """Validate trial budgets, source selection and motion-execution settings."""
        assert self.max_navigation_hold_steps > 0, "max_navigation_hold_steps must be positive."
        assert type(self.num_trials) is int and self.num_trials > 0, "num_trials must be a positive integer"
        assert isinstance(self.guarantee_success, bool), "guarantee_success must be a bool"
        assert (
            type(self.max_num_failures) is int and self.max_num_failures >= 0
        ), "max_num_failures must be a non-negative integer"
        assert self.source_selection_anchor is None or (
            isinstance(self.source_selection_anchor, str) and self.source_selection_anchor
        ), "source_selection_anchor must be a non-empty string or None"
        if self.source_demo_keys is not None:
            assert isinstance(self.source_demo_keys, list), "source_demo_keys must be a list or None"
            assert self.source_demo_keys, "source_demo_keys must be non-empty or None"
            assert all(
                isinstance(key, str) and key for key in self.source_demo_keys
            ), "source_demo_keys must contain non-empty strings"
            assert len(self.source_demo_keys) == len(set(self.source_demo_keys)), "source_demo_keys must be unique"
        assert (
            type(self.source_candidate_count) is int and self.source_candidate_count >= 1
        ), "source_candidate_count must be a positive integer"
        for name in (
            "skill_passthrough_delay_steps",
            "skill_passthrough_interpolation_steps",
            "final_hold_steps",
            "terminal_convergence_steps",
        ):
            value = getattr(self, name)
            assert type(value) is int and value >= 0, f"{name} must be a non-negative integer"
        if self.max_joint_step_rad is not None:
            assert (
                type(self.max_joint_step_rad) is float
                and math.isfinite(self.max_joint_step_rad)
                and self.max_joint_step_rad > 0.0
            ), "max_joint_step_rad must be a finite positive float or None"
