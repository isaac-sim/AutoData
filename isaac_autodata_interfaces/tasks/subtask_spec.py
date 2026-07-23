# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any


@dataclass
class SubtaskAlgoParams:
    """Base class for algorithm-specific subtask parameters.

    Each data generation algorithm (e.g. DexMimicGen, SkillGen) defines its own
    subclass with the fields it needs.
    """

    pass


@dataclass
class MimicGenSubtaskAlgoParams(SubtaskAlgoParams):
    """MimicGen has no subtask parameters beyond the shared ones on :class:`Subtask`."""

    pass


@dataclass
class DexMimicGenSubtaskAlgoParams(SubtaskAlgoParams):
    """DexMimicGen has no subtask parameters beyond the shared ones on :class:`Subtask`."""

    pass


@dataclass
class SkillGenSubtaskAlgoParams(SubtaskAlgoParams):
    """SkillGen-specific subtask parameters.

    Args:
        subtask_start_offset_range: Random offset range for the subtask's
            start, in steps.
    """

    subtask_start_offset_range: tuple[int, int] = (0, 0)


@dataclass
class SoftMimicGenSubtaskAlgoParams(SubtaskAlgoParams):
    """SoftMimicGen-specific subtask parameters.

    Args:
        object_soft: Whether the reference object is deformable.
        use_rotation_transform: Whether to transform EEF rotations with the local TPS Jacobian.
        bend_coef: TPS bending regularization coefficient.
        rot_coef: TPS affine-rotation regularization coefficient.
    """

    object_soft: bool = False
    use_rotation_transform: bool = True
    bend_coef: float = 0.1
    rot_coef: float = 1e-3


@dataclass(kw_only=True)
class Subtask:
    """Configuration object used to specify subtasks used in data generation.

    Args:
        object_ref: Reference object involved in the subtask. Empty string
            if no object is involved.
        description: Human/agent readable description of the subtask.
        subtask_start_signal: Boolean start signal name. Empty string if the
            subtask has no explicit start signal.
        subtask_term_signal: Boolean termination signal name. Empty string
            for the final subtask in a sequence.
        selection_strategy: Source-segment selection strategy name. One of
            ``"random"``, ``"nearest_neighbor_object"``,
            ``"nearest_neighbor_robot_distance"``.
        selection_strategy_kwargs: Extra arguments to the selection strategy.
        first_subtask_start_offset_range: Random offset range for the first
            subtask's start, in steps.
        subtask_term_offset_range: Random offset range applied to termination
            boundaries during generation, in steps.
        action_noise: Amplitude of action noise applied during this subtask.
        num_interpolation_steps: Steps used to interpolate to the start of
            this subtask's segment.
        num_fixed_steps: Additional fixed steps the robot holds before
            executing this subtask's segment.
        apply_noise_during_interpolation: Whether to apply ``action_noise``
            during the interpolation phase as well.
        algo_params: Algorithm-specific parameters (e.g.
            :class:`SkillGenSubtaskAlgoParams`).
    """

    object_ref: str = ""
    description: str = ""
    subtask_start_signal: str = ""
    subtask_term_signal: str = ""
    selection_strategy: str = "random"
    selection_strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    first_subtask_start_offset_range: tuple[int, int] = (0, 0)
    subtask_term_offset_range: tuple[int, int] = (0, 0)
    action_noise: float = 0.0
    num_interpolation_steps: int = 0
    num_fixed_steps: int = 0
    apply_noise_during_interpolation: bool = False
    algo_params: SubtaskAlgoParams = MISSING


ALGO_PARAMS_REGISTRY: dict[str, type[SubtaskAlgoParams]] = {
    "mimicgen": MimicGenSubtaskAlgoParams,
    "dexmimicgen": DexMimicGenSubtaskAlgoParams,
    "skillgen": SkillGenSubtaskAlgoParams,
    "softmimicgen": SoftMimicGenSubtaskAlgoParams,
}
"""Maps the ``algo:`` discriminator in a YAML task config to the
corresponding :class:`SubtaskAlgoParams` subclass.

New algorithms register themselves here.
"""
