# Copyright (c) 2026, The Isaac Auto Data Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationPolicy:
    """Cross-cutting generation flags that shape how the data generator stitches segments.

    These were ``MimicEnvCfg.datagen_config`` fields upstream; they live on the task descriptor
    now because they describe how the task should be generated, not how the robot or env behaves.

    Args:
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

    select_src_per_subtask: bool = False
    select_src_per_arm: bool = False
    transform_first_robot_pose: bool = False
    interpolate_from_last_target_pose: bool = True
