# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation algorithm registry.

:class:`GenerationAlgorithm` is the plug-in surface :class:`DataGenerator` consults whenever a
behavior diverges between Mimic, DexMimicGen, SkillGen, etc.
Subclasses self-register at import time via :class:`_AlgorithmMeta`.

The main behavioral hook is :meth:`GenerationAlgorithm.plan_subtask_trajectory`, called by
``DataGenerator`` every time an EEF needs a new executable trajectory. Object-state adaptation is
routed through :meth:`GenerationAlgorithm.transform_source_eef_poses. Rigid algorithms use the
default object-pose transform while SoftMimicGen overrides it with nodal TPS registration.
"""

from __future__ import annotations

import torch
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaac_autodata_core.data_generator import DataGenerator, _EEFGenerationState
    from isaac_autodata_core.waypoint import Waypoint
    from isaac_autodata_interfaces.datastream.datastream import Datastream
    from isaac_autodata_interfaces.tasks.subtask_spec import Subtask

REGISTERED_ALGORITHMS: dict[str, type[GenerationAlgorithm]] = {}


def get_algorithm(name: str, **kwargs: Any) -> GenerationAlgorithm:
    """Construct the algorithm registered under ``name``, forwarding ``kwargs`` to its ``__init__``.

    Mimic and DexMimicGen take no kwargs. SkillGen requires ``motion_planners=`` (per-env dict).
    """
    if name not in REGISTERED_ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}. Registered: {sorted(REGISTERED_ALGORITHMS)}")
    return REGISTERED_ALGORITHMS[name](**kwargs)


def iter_algorithms() -> Iterator[type[GenerationAlgorithm]]:
    """Iterate the registered algorithm classes; CLI uses this to populate ``--alg`` choices."""
    yield from REGISTERED_ALGORITHMS.values()


class _AlgorithmMeta(type):
    """Auto-registers concrete :class:`GenerationAlgorithm` subclasses by their ``name`` attribute."""

    def __new__(mcs, name: str, bases: tuple, class_dict: dict) -> type:
        cls = super().__new__(mcs, name, bases, class_dict)
        if bases and class_dict.get("name"):
            REGISTERED_ALGORITHMS[class_dict["name"]] = cls
        return cls


class GenerationAlgorithm(metaclass=_AlgorithmMeta):
    """Plug-in describing how an algorithm differs from a vanilla Mimic run.

    Concrete subclasses set the class attributes below and may override
    :meth:`plan_subtask_trajectory` to customize per-subtask transition logic.
    """

    name: str = ""
    expected_eef_count: int | tuple[int, ...] = 1
    requires_motion_planner: bool = False
    uses_subtask_start_signals: bool = False
    supports_coordination: bool = False

    def validate_setup(self, datastream: Datastream) -> None:
        """Algorithm-specific validation against the composed datastream. Default: no-op.

        Subclasses can read ``datastream.get_subtasks(eef)``, ``datastream.get_task_constraints()``,
        or ``datastream.get_env()`` (escape hatch) to enforce algorithm-specific invariants.
        """

    def is_deformable_subtask(self, subtask: Subtask) -> bool:
        """Return whether ``subtask`` uses deformable nodal state instead of a rigid pose."""

        return False

    def transform_source_eef_poses(
        self,
        *,
        data_generator: DataGenerator,
        eef_name: str,
        subtask_ind: int,
        subtask_object_name: str | None,
        subtask_object_pose: torch.Tensor | None,
        src_subtask_object_pose: torch.Tensor | None,
        subtask_object_nodal_positions: torch.Tensor | None,
        src_subtask_object_nodal_positions: torch.Tensor | None,
        src_eef_poses: torch.Tensor,
        use_delta_transform: torch.Tensor | None,
        coord_transform_scheme: Any,
        runtime_subtask_constraints_dict: dict,
    ) -> torch.Tensor:
        """Adapt source EEF poses to the current subtask object state.

        The default implementation performs the existing rigid-object transform. Algorithms using
        another object representation override this method.
        """

        return data_generator._apply_subtask_transform(
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            subtask_object_name=subtask_object_name,
            subtask_object_pose=subtask_object_pose,
            src_subtask_object_pose=src_subtask_object_pose,
            src_eef_poses=src_eef_poses,
            use_delta_transform=use_delta_transform,
            coord_transform_scheme=coord_transform_scheme,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
        )

    def plan_subtask_trajectory(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        eef_name: str,
        eef_state: _EEFGenerationState,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> tuple[list[Waypoint], bool] | None:
        """Return the next trajectory to execute, paired with ``is_motion_plan_phase``.

        Return value semantics:

        * ``(waypoints, False)`` — execute these waypoints as the actual subtask. The default
          implementation produces this by calling ``generate_eef_subtask_trajectory`` followed by
          ``merge_eef_subtask_trajectory``.
        * ``(waypoints, True)`` — execute these waypoints as a motion-planned transit. The
          algorithm is responsible for stashing the actual subtask trajectory on ``eef_state``
          (typically in ``pending_subtask_trajectory``) so that the follow-up call can splice it in.
        * ``None`` — the algorithm failed to plan (e.g. SkillGen motion planning failure). The
          caller aborts the current ``generate()`` attempt and reports ``success=False``.

        Args:
            data_generator: The owning :class:`DataGenerator`; exposes ``generate_eef_subtask_trajectory``,
                ``merge_eef_subtask_trajectory``, ``datastream``, ``src_demo_datagen_info_pool``.
            env_id: Env index this trajectory belongs to.
            eef_name: End-effector key.
            eef_state: Mutable per-EEF state container.
            all_randomized_subtask_boundaries: Subtask boundary table for the current pool snapshot.
            runtime_subtask_constraints_dict: Mutable runtime constraint table.
            selected_src_demo_inds: Mutable per-EEF selected source demo index map.
        """
        # Resume path: a previous call returned an MP transit and stashed the actual subtask.
        # Splice the stashed subtask now using the MP-end pose as the interpolation seed.
        if eef_state.pending_subtask_trajectory is not None:
            pending = eef_state.pending_subtask_trajectory
            prev_executed_traj = eef_state.current_trajectory
            merged = data_generator.merge_eef_subtask_trajectory(
                env_id=env_id,
                eef_name=eef_name,
                subtask_index=eef_state.current_subtask_index,
                prev_executed_traj=prev_executed_traj,
                subtask_trajectory=pending,
                force_use_prev_traj=True,
            )
            eef_state.pending_subtask_trajectory = None
            eef_state.is_in_motion_plan = False
            return merged, False

        # Default path: generate subtask trajectory, merge an interpolation segment from the
        # robot's current pose (or last executed target pose) onto the front.
        subtask_traj = data_generator.generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=eef_state.current_subtask_index,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        merged = data_generator.merge_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_index=eef_state.current_subtask_index,
            prev_executed_traj=eef_state.current_trajectory,
            subtask_trajectory=subtask_traj,
        )
        return merged, False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class MimicGen(GenerationAlgorithm):
    """Vanilla single-arm MimicGen — interpolation transitions, no coordination."""

    name = "mimicgen"
    expected_eef_count = 1
    requires_motion_planner = False
    uses_subtask_start_signals = False
    supports_coordination = False


class DexMimicGen(GenerationAlgorithm):
    """Two-arm MimicGen with subtask coordination constraints."""

    name = "dexmimicgen"
    expected_eef_count = 2
    requires_motion_planner = False
    uses_subtask_start_signals = False
    supports_coordination = True


class SoftMimicGen(GenerationAlgorithm):
    """MimicGen for deformable reference objects.

    Deformable subtasks select source segments using nodal state and warp their EEF trajectories
    with a thin-plate-spline transform. Rigid subtasks retain the normal MimicGen transform.
    """

    name = "softmimicgen"
    expected_eef_count = (1, 2)
    requires_motion_planner = False
    uses_subtask_start_signals = False
    supports_coordination = False

    def is_deformable_subtask(self, subtask: Subtask) -> bool:
        return bool(getattr(subtask.algo_params, "object_soft", False))

    def validate_setup(self, datastream: Datastream) -> None:
        live_nodal_positions = datastream.get_object_nodal_positions()
        for eef_name in datastream.get_eef_names():
            for subtask_index, subtask in enumerate(datastream.get_subtasks(eef_name)):
                if not self.is_deformable_subtask(subtask):
                    continue
                assert (
                    subtask.object_ref
                ), f"SoftMimicGen deformable subtask {eef_name}[{subtask_index}] requires object_ref"
                assert subtask.object_ref in live_nodal_positions, (
                    f"Deformable object {subtask.object_ref!r} is not present in the live scene; "
                    f"available: {sorted(live_nodal_positions)}"
                )
                live_node_count = live_nodal_positions[subtask.object_ref].shape[-2]
                for demo_index, datagen_info in enumerate(datastream.source_pool.datagen_infos):
                    assert (
                        datagen_info.object_nodal_positions is not None
                    ), f"Source demo {demo_index} lacks object_nodal_position annotations"
                    assert (
                        subtask.object_ref in datagen_info.object_nodal_positions
                    ), f"Source demo {demo_index} lacks nodal positions for {subtask.object_ref!r}"
                    source_node_count = datagen_info.object_nodal_positions[subtask.object_ref].shape[-2]
                    assert source_node_count == live_node_count, (
                        f"Source demo {demo_index} has {source_node_count} nodes for "
                        f"{subtask.object_ref!r}, live object has {live_node_count}"
                    )

    def transform_source_eef_poses(
        self,
        *,
        data_generator: DataGenerator,
        eef_name: str,
        subtask_ind: int,
        subtask_object_name: str | None,
        subtask_object_pose: torch.Tensor | None,
        src_subtask_object_pose: torch.Tensor | None,
        subtask_object_nodal_positions: torch.Tensor | None,
        src_subtask_object_nodal_positions: torch.Tensor | None,
        src_eef_poses: torch.Tensor,
        use_delta_transform: torch.Tensor | None,
        coord_transform_scheme: Any,
        runtime_subtask_constraints_dict: dict,
    ) -> torch.Tensor:
        subtask = data_generator.datastream.get_subtask(eef_name, subtask_ind)
        if not self.is_deformable_subtask(subtask):
            return super().transform_source_eef_poses(
                data_generator=data_generator,
                eef_name=eef_name,
                subtask_ind=subtask_ind,
                subtask_object_name=subtask_object_name,
                subtask_object_pose=subtask_object_pose,
                src_subtask_object_pose=src_subtask_object_pose,
                subtask_object_nodal_positions=subtask_object_nodal_positions,
                src_subtask_object_nodal_positions=src_subtask_object_nodal_positions,
                src_eef_poses=src_eef_poses,
                use_delta_transform=use_delta_transform,
                coord_transform_scheme=coord_transform_scheme,
                runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            )

        assert subtask_object_nodal_positions is not None, "current deformable nodal state is missing"
        assert src_subtask_object_nodal_positions is not None, "source deformable nodal state is missing"
        from isaac_autodata_core.deformable_transforms import transform_source_data_segment_using_nodal_registration

        return transform_source_data_segment_using_nodal_registration(
            src_eef_poses=src_eef_poses,
            src_obj_nodal_pos=src_subtask_object_nodal_positions,
            tgt_obj_nodal_pos=subtask_object_nodal_positions,
            use_rotation_transform=subtask.algo_params.use_rotation_transform,
            bend_coef=subtask.algo_params.bend_coef,
            rot_coef=subtask.algo_params.rot_coef,
        )


class SkillGen(GenerationAlgorithm):
    """Single-arm SkillGen — motion-planned transit followed by skill replay per subtask.

    Each subtask runs as two phases: a motion-planned transit to the subtask start, then the
    skill trajectory itself. Constraints (sequential or coordination) only apply during the skill
    phase; the transit is constraint-free.
    """

    name = "skillgen"
    expected_eef_count = 1
    requires_motion_planner = True
    uses_subtask_start_signals = True
    supports_coordination = False

    def __init__(self, motion_planners: dict[int, Any]) -> None:
        """
        Args:
            motion_planners: One planner per env_id. Planners must expose
                ``update_world_and_plan_motion(target_pose, expected_attached_object, env_id, ...)``
                and ``get_planned_poses() -> list[torch.Tensor[4, 4]]``. ``config.motion_noise_scale``
                is read if present.
        """
        assert motion_planners, "SkillGen requires at least one motion planner"
        self.motion_planners = motion_planners

    def plan_subtask_trajectory(
        self,
        *,
        data_generator: DataGenerator,
        env_id: int,
        eef_name: str,
        eef_state: _EEFGenerationState,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> tuple[list[Waypoint], bool] | None:
        # Resume path — same as the base class. Splice the stashed skill segment now.
        if eef_state.pending_subtask_trajectory is not None:
            pending = eef_state.pending_subtask_trajectory
            prev_executed_traj = eef_state.current_trajectory
            merged = data_generator.merge_eef_subtask_trajectory(
                env_id=env_id,
                eef_name=eef_name,
                subtask_index=eef_state.current_subtask_index,
                prev_executed_traj=prev_executed_traj,
                subtask_trajectory=pending,
                force_use_prev_traj=True,
            )
            eef_state.pending_subtask_trajectory = None
            eef_state.is_in_motion_plan = False
            return merged, False

        # New subtask — generate skill trajectory, then plan an MP transit to its first pose.
        subtask_traj = data_generator.generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=eef_state.current_subtask_index,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )

        target_pose = subtask_traj[0].pose
        target_passthrough_action = subtask_traj[0].passthrough_action

        # Which object (if any) the EEF carries during this subtask, so the planner can attach it
        # to the robot's collision model before planning transit. Derived from the task
        # descriptor's subtask metadata via the Datastream interface.
        expected_attached_object = data_generator.datastream.get_expected_attached_object(
            eef_name, eef_state.current_subtask_index
        )

        planner = self.motion_planners[env_id]
        planning_success = planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=expected_attached_object,
            env_id=env_id,
            step_size=getattr(planner, "step_size", None),
            enable_retiming=(hasattr(planner, "step_size") and planner.step_size is not None),
        )
        if not planning_success:
            return None

        mp_waypoints = self._convert_planned_trajectory_to_waypoints(planner, target_passthrough_action)

        # Stash the skill segment so the follow-up call merges it from the MP end pose.
        eef_state.pending_subtask_trajectory = subtask_traj
        eef_state.is_in_motion_plan = True
        return mp_waypoints, True

    @staticmethod
    def _convert_planned_trajectory_to_waypoints(
        motion_planner: Any,
        passthrough_action: dict,
    ) -> list[Waypoint]:
        """Wrap each planner pose into a :class:`Waypoint` with the supplied passthrough channels.

        Reads ``motion_planner.config.motion_noise_scale`` if present; defaults to 0.0.
        """
        from isaac_autodata_core.waypoint import Waypoint

        motion_noise_scale = getattr(motion_planner.config, "motion_noise_scale", 0.0)
        return [
            Waypoint(pose=pose, passthrough_action=passthrough_action, noise=motion_noise_scale)
            for pose in motion_planner.get_planned_poses()
        ]
