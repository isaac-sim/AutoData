# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy
from autodata_interfaces.tasks.schedule_constraint_spec import (
    NonPreemptiveConstraintSpec,
    PrecedenceConstraintSpec,
    ScheduleSpec,
    SegmentAnchorPoint,
    SegmentId,
)
from autodata_interfaces.tasks.segment_spec import (
    SegmentEventPoint,
    SegmentEventType,
    SegmentKind,
    SegmentProvider,
    SegmentRobotContactSpec,
    SegmentWorldContactSpec,
)
from autodata_interfaces.tasks.subtask_constraint_spec import SubtaskConstraint
from autodata_interfaces.tasks.subtask_spec import ALGO_PARAMS_REGISTRY, Subtask, SubtaskAlgoParams
from autodata_interfaces.tasks.task_descriptor_utils import (
    build_constraint,
    build_schedule,
    build_subtask,
    validate_task_dict,
)

if TYPE_CHECKING:
    from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter


@dataclass
class TaskDescriptor:
    """Class used to describe a task for data generation.

    Args:
        name: Task identifier.
        description: Human-readable task description.
        subtasks: Per-end-effector ordered subtask lists. Keys are eef names;
            each value is the ordered sequence of :class:`Subtask` for that eef.
        constraints: Cross-subtask coordination/sequential constraints (multi-eef tasks).
        generation_policy: Data generation policy parameters.
        attempt_diagnostics: Optional ``"module.path:function"`` callback for task-specific
            terminal diagnostics. The generation entrypoint resolves it at composition time.
    """

    name: str = MISSING
    description: str = ""
    subtasks: dict[str, list[Subtask]] = field(default_factory=dict)
    constraints: list[SubtaskConstraint] = field(default_factory=list)
    schedule: ScheduleSpec | None = None
    generation_policy: GenerationPolicy = field(default_factory=GenerationPolicy)
    env: Any = None
    attempt_diagnostics: str | None = None

    def bind_env(self, env: Any) -> None:
        """Attach the Isaac Lab env after construction.

        Asserts the env was not previously bound — call exactly once.
        """

        assert self.env is None, "env already bound"
        self.env = env

    def validate_embodiment(self, embodiment_adapter: EmbodimentAdapter) -> None:
        """Validate task/embodiment compatibility without starting the simulator.

        Args:
            embodiment_adapter: Adapter whose channels and navigation configuration must match this task.
        """
        task_eefs = set(self.get_eef_names())
        adapter_eefs = set(embodiment_adapter.get_eef_names())
        assert task_eefs == adapter_eefs, (
            f"EEF mismatch: task descriptor declares {sorted(task_eefs)}, "
            f"embodiment adapter exposes {sorted(adapter_eefs)}"
        )
        has_navigation = embodiment_adapter.get_navigation_eef_name() is not None
        assert (
            self.generation_policy.use_navigation_controller == has_navigation
        ), "Set generation_policy.use_navigation_controller exactly when the embodiment declares navigation."

    def get_eef_names(self) -> list[str]:
        """Return the end-effector names declared by this task."""

        return list(self.subtasks.keys())

    def get_subtasks(self, eef_name: str) -> list[Subtask]:
        """Return the list of subtasks for the given end-effector name."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return self.subtasks[eef_name]

    def get_object_refs(self, eef_name: str) -> list[str]:
        """Return all object reference names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.object_ref for st in self.subtasks[eef_name]]

    def get_start_signal_names(self, eef_name: str) -> list[str]:
        """Return subtask start-signal names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.subtask_start_signal for st in self.subtasks[eef_name]]

    def get_term_signal_names(self, eef_name: str) -> list[str]:
        """Return subtask termination-signal names for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.subtask_term_signal for st in self.subtasks[eef_name]]

    def get_subtask_descriptions(self, eef_name: str) -> list[str]:
        """Return subtask descriptions for the given eef, in subtask order. Empty string if unset."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.description for st in self.subtasks[eef_name]]

    def get_expected_attached_object(self, eef_name: str, subtask_index: int) -> str | None:
        """Return the object the EEF is expected to carry during a subtask, or ``None``.

        SkillGen plans collision-aware transit while the gripper holds a grasped object, so it
        needs the identity of the carried object for each subtask. A *stack* subtask is treated as
        carrying the object grasped in the immediately preceding *grasp* subtask; subtasks are
        classified by the ``"grasp"`` / ``"stack"`` substrings in their termination-signal names,
        so the result is derived purely from the descriptor's subtask metadata.

        Args:
            eef_name: End-effector to query.
            subtask_index: Index of the subtask whose carried object is requested.

        Returns:
            The held object's reference name, or ``None`` when the subtask carries nothing (a
            grasp/approach subtask, an out-of-range index, or an unknown eef).
        """

        if eef_name not in self.subtasks:
            return None
        subtasks = self.subtasks[eef_name]
        if not 0 <= subtask_index < len(subtasks):
            return None
        current = subtasks[subtask_index]
        if "stack" in current.subtask_term_signal.lower() and subtask_index > 0:
            prev = subtasks[subtask_index - 1]
            if "grasp" in prev.subtask_term_signal.lower():
                return prev.object_ref or None
        return None

    def get_subtask_algo_params(self, eef_name: str) -> list[SubtaskAlgoParams]:
        """Return subtask algorithm parameters for the given eef, in subtask order."""

        assert eef_name in self.subtasks, f"Unknown eef name: {eef_name}"
        return [st.algo_params for st in self.subtasks[eef_name]]

    def get_task_constraints(self) -> list[SubtaskConstraint]:
        """Return the task's cross-subtask constraints, in declaration order.

        Each :class:`SubtaskConstraint` exposes ``generate_runtime_subtask_constraints()``, the
        verbatim expansion the data generator consumes — mirroring ``MimicEnvCfg.task_constraint_configs``.
        """

        return self.constraints

    def get_schedule_spec(self) -> ScheduleSpec | None:
        """Return the episode-level schedule policy, if configured."""

        return self.schedule

    def get_segment_world_contacts(self) -> dict[SegmentId, tuple[SegmentWorldContactSpec, ...]]:
        """Return task-defined object/fixture contacts keyed by exact segment identity."""

        return {
            SegmentId(eef_name, subtask.name, segment.name): segment.world_contacts
            for eef_name, subtasks in self.subtasks.items()
            for subtask in subtasks
            for segment in subtask.segments
            if segment.world_contacts
        }

    def get_segment_robot_contacts(self) -> dict[SegmentId, tuple[SegmentRobotContactSpec, ...]]:
        """Return task-defined robot/object contacts keyed by exact segment identity."""

        return {
            SegmentId(eef_name, subtask.name, segment.name): segment.robot_contacts
            for eef_name, subtasks in self.subtasks.items()
            for subtask in subtasks
            for segment in subtask.segments
            if segment.robot_contacts
        }

    def get_segment_resource_ids(self) -> dict[SegmentId, tuple[str, ...]]:
        """Return additional task-local resource claims keyed by exact segment identity."""

        return {
            SegmentId(eef_name, subtask.name, segment.name): segment.resource_ids
            for eef_name, subtasks in self.subtasks.items()
            for subtask in subtasks
            for segment in subtask.segments
            if segment.resource_ids
        }

    def get_path_build_order(self) -> tuple[str, ...]:
        """Order path construction by cross-arm object-pose dependencies.

        Returns:
            End-effector names, with each object-pose producer before its consumers.

        Raises:
            AssertionError: A dependency names an unknown end effector or contains a cycle.
        """

        dependencies = {eef_name: set() for eef_name in self.subtasks}
        for consumer_eef, subtasks in self.subtasks.items():
            for subtask in subtasks:
                for segment in subtask.segments:
                    goal = segment.motion_goal
                    if goal is None or goal.object_pose_after_segment_id is None:
                        continue
                    producer_eef = goal.object_pose_after_segment_id.eef_name
                    assert producer_eef in dependencies, f"unknown producer EEF {producer_eef!r}"
                    if producer_eef != consumer_eef:
                        dependencies[consumer_eef].add(producer_eef)

        result: list[str] = []
        pending = dict(dependencies)
        while pending:
            ready = [eef_name for eef_name in self.subtasks if eef_name in pending and not pending[eef_name]]
            assert ready, "object-pose dependencies require a cycle-free path construction order"
            result.extend(ready)
            for eef_name in ready:
                pending.pop(eef_name)
            for predecessors in pending.values():
                predecessors.difference_update(ready)
        return tuple(result)

    def requires_source_demos(self) -> bool:
        """Return whether generation requires a source-demonstration pool."""

        subtasks = tuple(subtask for eef_subtasks in self.subtasks.values() for subtask in eef_subtasks)
        return (
            not subtasks
            or any(not subtask.segments for subtask in subtasks)
            or any(
                segment.provider == SegmentProvider.DEMONSTRATION
                for subtask in subtasks
                for segment in subtask.segments
            )
        )

    def get_generation_policy(self) -> GenerationPolicy:
        """Return the cross-cutting generation flags."""

        return self.generation_policy

    def get_attempt_diagnostics(self) -> str | None:
        """Return the optional task-diagnostics callable path."""

        return self.attempt_diagnostics

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskDescriptor:
        """Build a TaskDescriptor from a YAML config file.

        Expected schema:

            name: <str>
            description: <str>          # optional
            algo: <str>                 # key into ALGO_PARAMS_REGISTRY
            subtasks:
              <eef_name>:
                - object_ref: <str>            # optional
                  description: <str>           # optional
                  subtask_start_signal: <str>  # optional
                  subtask_term_signal: <str>   # optional

                  # algorithm-agnostic generation knobs:
                  selection_strategy: <str>
                  selection_strategy_kwargs: {<kwarg>: <value>}
                  first_subtask_start_offset_range: [<int>, <int>]
                  subtask_term_offset_range: [<int>, <int>]
                  action_noise: <float>
                  num_interpolation_steps: <int>
                  num_fixed_steps: <int>
                  apply_noise_during_interpolation: <bool>
                  algo_params:                 # optional, fields match the
                    <kwarg>: <value>           # chosen algo's dataclass
                  segments:                    # ScheduleGen
                    - name: <str>
                      kind: <str>
                      provider: <str>
                      goal_segment_id: <eef>.<subtask>.<segment>  # motion planner only
                      resource_ids: [<task_local_resource>, ...]  # optional shared mutexes
                      world_contacts:          # optional exact object/fixture pairs
                        - object_ref: <str>
                          fixture_ref: <str>
                - ...
              <other_eef>: [...]
            constraints:                        # optional, cross-subtask constraints
              - constraint_type: <str>          # "sequential" | "coordination"
                eef_subtask_constraint_tuple: [[<eef>, <int>], [<eef>, <int>]]
                sequential_min_time_diff: <int>            # sequential only
                coordination_scheme: <str>                 # coordination only
                coordination_scheme_pos_noise_scale: <float>
                coordination_scheme_rot_noise_scale: <float>
                coordination_synchronize_start: <bool>
              - ...
            generation_policy:                  # optional; full parity with MimicEnvCfg.datagen_config
              name: <str>
              seed: <int>
              num_trials: <int>
              guarantee_success: <bool>
              keep_failed: <bool>
              max_num_failures: <int>
              source_dataset_path: <str>
              generation_path: <str>
              task_name: <str>
              use_skillgen: <bool>
              use_navigation_controller: <bool>
              select_src_per_subtask: <bool>
              select_src_per_arm: <bool>
              source_selection_anchor: <eef>.<subtask>
              source_demo_keys: [<str>, ...]
              source_candidate_count: <int>
              transform_first_robot_pose: <bool>
              interpolate_from_last_target_pose: <bool>
              skill_passthrough_delay_steps: <int>
              skill_passthrough_interpolation_steps: <int>
              max_joint_step_rad: <float>
              final_hold_steps: <int>
              terminal_convergence_steps: <int>
            attempt_diagnostics: <str>           # optional module.path:function
        """

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskDescriptor:
        """Build a TaskDescriptor from a parsed config dict.

        Schema is validated up-front; on a violation, raises
        :class:`AssertionError` with a message naming the offending field.
        """

        validate_task_dict(data)
        algo_cls = ALGO_PARAMS_REGISTRY[data["algo"]]

        subtasks: dict[str, list[Subtask]] = {
            eef_name: [build_subtask(st, algo_cls) for st in eef_subtasks]
            for eef_name, eef_subtasks in data["subtasks"].items()
        }
        constraints = [build_constraint(c) for c in data.get("constraints", [])]
        schedule = build_schedule(data["schedule"]) if "schedule" in data else None
        generation_policy = GenerationPolicy(**data.get("generation_policy", {}))

        descriptor = cls(
            name=data["name"],
            description=data.get("description", ""),
            subtasks=subtasks,
            constraints=constraints,
            schedule=schedule,
            generation_policy=generation_policy,
            attempt_diagnostics=data.get("attempt_diagnostics"),
        )
        descriptor._validate_segments(require_schedule=data["algo"] == "schedulegen")
        return descriptor

    def _validate_segments(self, *, require_schedule: bool) -> None:
        """Check segment identities, dependencies, event pairing and schedule references.

        Args:
            require_schedule: Whether every subtask must declare named schedule segments.
        """
        segment_ids: set[SegmentId] = set()
        resource_eefs: dict[str, set[str]] = {}
        for eef_name, subtasks in self.subtasks.items():
            names = tuple(subtask.name for subtask in subtasks if subtask.name)
            assert len(names) == len(set(names)), f"duplicate subtask name for {eef_name!r}"
            for subtask_index, subtask in enumerate(subtasks):
                assert require_schedule or not subtask.segments, "explicit segments require the ScheduleGen algorithm"
                if require_schedule:
                    assert (
                        subtask.name and subtask.segments
                    ), f"ScheduleGen subtask {eef_name!r} requires a name and segments"
                    segment_shape = tuple((segment.kind, segment.provider) for segment in subtask.segments)
                    skill_only_shape = ((SegmentKind.SKILL, SegmentProvider.DEMONSTRATION),)
                    motion_only_shape = ((SegmentKind.TRANSIT, SegmentProvider.MOTION_PLANNER),)
                    base_shape = (
                        (SegmentKind.TRANSIT, SegmentProvider.MOTION_PLANNER),
                        (SegmentKind.SKILL, SegmentProvider.DEMONSTRATION),
                    )
                    skill_motion_shape = skill_only_shape + motion_only_shape
                    base_motion_shape = base_shape + motion_only_shape
                    hold_shape = base_shape + ((SegmentKind.HOLD, SegmentProvider.SCRIPTED),)
                    direct_egress_shape = base_shape + ((SegmentKind.TRANSIT, SegmentProvider.SCRIPTED),)
                    hold_egress_shape = hold_shape + ((SegmentKind.TRANSIT, SegmentProvider.SCRIPTED),)
                    is_skill_only = segment_shape == skill_only_shape
                    is_motion_only = segment_shape == motion_only_shape
                    is_base = segment_shape == base_shape
                    is_skill_motion = segment_shape == skill_motion_shape
                    is_base_motion = segment_shape == base_motion_shape
                    is_hold = segment_shape == hold_shape
                    is_direct_egress = segment_shape == direct_egress_shape
                    is_hold_egress = segment_shape == hold_egress_shape
                    is_release = is_hold or is_direct_egress or is_hold_egress
                    assert (
                        is_skill_only or is_motion_only or is_base or is_skill_motion or is_base_motion or is_release
                    ), (
                        "current ScheduleGen runtime requires one motion-planned transit, one demonstration "
                        "skill, an optional incoming and/or outgoing motion-planned transit around that skill, "
                        "or the bounded scripted release suffix "
                        f"for {eef_name}.{subtask.name}; got {segment_shape}"
                    )
                    if is_release:
                        assert (
                            subtask_index == len(subtasks) - 1
                        ), f"the bounded release form is supported only on the final subtask for {eef_name!r}"

                    skill_index = None if is_motion_only else int(is_base or is_base_motion or is_release)
                    for segment_index, segment in enumerate(subtask.segments):
                        assert not segment.events or segment_index == skill_index, (
                            "current ScheduleGen runtime supports events only on the demonstration "
                            f"skill for {eef_name}.{subtask.name}"
                        )
                        assert (
                            not segment.robot_contacts
                            or segment.kind == SegmentKind.SKILL
                            or (segment.provider == SegmentProvider.SCRIPTED)
                        ), (
                            "robot contacts are supported only on demonstration or scripted "
                            f"segments; got {eef_name}.{subtask.name}.{segment.name}"
                        )
                        annotation_keys = {
                            event.annotation_key
                            for event in segment.events
                            if event.point == SegmentEventPoint.SOURCE_ANNOTATION
                        }
                        for contact in segment.robot_contacts:
                            assert (
                                not contact.start_annotation_key or contact.start_annotation_key in annotation_keys
                            ), (
                                "robot-contact start_annotation_key must name a source-annotation "
                                f"event on {eef_name}.{subtask.name}.{segment.name}"
                            )

                    skill = subtask.segments[skill_index] if skill_index is not None else None
                    if is_base or is_base_motion or is_release:
                        assert skill is not None
                        transit = subtask.segments[0]
                        expected_goal_segment_id = SegmentId(eef_name, subtask.name, skill.name)
                        assert transit.goal_segment_id == expected_goal_segment_id, (
                            "current ScheduleGen motion-planned transit must target its same-subtask "
                            f"demonstration skill {expected_goal_segment_id}; got {transit.goal_segment_id}"
                        )
                    if is_motion_only or is_skill_motion or is_base_motion:
                        outgoing_motion = subtask.segments[-1]
                        assert outgoing_motion.motion_goal is not None, (
                            f"outgoing motion-planned segment {eef_name}.{subtask.name}.{outgoing_motion.name} "
                            "requires an independent motion_goal"
                        )
                    if not is_release:
                        assert all(
                            segment.demonstration_source_window is None
                            and segment.duration_steps == 0
                            and not segment.passthrough_preset
                            and not segment.cartesian_offset_m
                            for segment in subtask.segments
                        ), (
                            "legacy ScheduleGen transit/skill pairs and demonstration-only subtasks do not "
                            f"support release-hold provider fields for {eef_name}.{subtask.name}"
                        )
                    else:
                        assert skill is not None
                        hold = subtask.segments[2] if is_hold or is_hold_egress else None
                        egress = subtask.segments[-1] if is_direct_egress or is_hold_egress else None
                        assert (
                            transit.demonstration_source_window is None
                            and transit.duration_steps == 0
                            and not transit.passthrough_preset
                            and not transit.cartesian_offset_m
                        ), f"release entry transit has unexpected provider fields for {eef_name}.{subtask.name}"
                        source_window = skill.demonstration_source_window
                        assert source_window is not None, (
                            "the bounded release form requires a demonstration_source_window on "
                            f"{eef_name}.{subtask.name}.{skill.name}"
                        )
                        assert (
                            skill.duration_steps == 0 and not skill.passthrough_preset and not skill.cartesian_offset_m
                        ), (
                            "release demonstration skill supports only demonstration_source_window "
                            f"provider fields for {eef_name}.{subtask.name}"
                        )
                        matching_anchors = [
                            event
                            for event in skill.events
                            if event.point == SegmentEventPoint.SOURCE_ANNOTATION
                            and event.annotation_key == source_window.anchor_annotation_key
                        ]
                        assert len(matching_anchors) == 1, (
                            "demonstration_source_window anchor must match exactly one declared "
                            f"source-annotation event on {eef_name}.{subtask.name}.{skill.name}"
                        )
                        release_event = matching_anchors[0]
                        assert release_event.event_type == SegmentEventType.DETACH, (
                            "demonstration_source_window anchor must be a detach event on "
                            f"{eef_name}.{subtask.name}.{skill.name}"
                        )
                        assert (
                            len(skill.robot_contacts) == 1
                        ), f"release form requires one robot contact on skill for {eef_name}.{subtask.name}"
                        skill_contact = skill.robot_contacts[0]
                        assert (
                            skill_contact.start_annotation_key == release_event.annotation_key
                            and skill_contact.object_ref == release_event.object_ref
                        ), (
                            "release source event and robot contact must name the same object "
                            f"and annotation on {eef_name}.{subtask.name}.{skill.name}"
                        )
                        previous_contact = skill_contact
                        if hold is not None:
                            assert (
                                len(hold.robot_contacts) == 1
                            ), f"release form requires one robot contact on hold for {eef_name}.{subtask.name}"
                            hold_contact = hold.robot_contacts[0]
                            assert (
                                hold_contact.object_ref == skill_contact.object_ref
                                and hold_contact.robot_link_names == skill_contact.robot_link_names
                                and not hold_contact.start_annotation_key
                                and not hold_contact.clear_at_end
                            ), (
                                "release skill and hold must authorize the same exact "
                                f"robot/object pairs for {eef_name}.{subtask.name}"
                            )
                            assert (
                                hold.demonstration_source_window is None
                                and hold.duration_steps > 0
                                and hold.passthrough_preset
                                and not hold.cartesian_offset_m
                            ), (
                                "scripted hold requires positive duration_steps and a passthrough_preset for "
                                f"{eef_name}.{subtask.name}"
                            )
                            previous_contact = hold_contact
                        if egress is not None:
                            assert (
                                egress.demonstration_source_window is None
                                and egress.duration_steps >= 2
                                and (hold is None or egress.passthrough_preset == hold.passthrough_preset)
                                and len(egress.cartesian_offset_m) == 3
                                and any(value != 0.0 for value in egress.cartesian_offset_m)
                            ), (
                                "scripted Cartesian egress requires at least two steps, any preceding hold's "
                                "passthrough preset, and a nonzero XYZ offset for "
                                f"{eef_name}.{subtask.name}"
                            )
                            assert len(egress.robot_contacts) == 1, (
                                "scripted Cartesian egress requires one terminal-clear robot contact for "
                                f"{eef_name}.{subtask.name}"
                            )
                            egress_contact = egress.robot_contacts[0]
                            assert (
                                egress_contact.object_ref == previous_contact.object_ref
                                and egress_contact.robot_link_names == previous_contact.robot_link_names
                                and not egress_contact.start_annotation_key
                                and egress_contact.clear_at_end
                            ), (
                                "release suffix and egress must authorize the same exact robot/object pairs, "
                                f"with a strictly clear egress endpoint for {eef_name}.{subtask.name}"
                            )
                segment_names = tuple(segment.name for segment in subtask.segments)
                assert len(segment_names) == len(
                    set(segment_names)
                ), f"duplicate segment name under {eef_name}.{subtask.name}"
                for segment_name in segment_names:
                    segment_id = SegmentId(eef_name, subtask.name, segment_name)
                    segment_ids.add(segment_id)
                for segment in subtask.segments:
                    for resource_id in segment.resource_ids:
                        resource_eefs.setdefault(resource_id, set()).add(eef_name)

        if require_schedule:
            assert self.schedule is not None, "ScheduleGen requires a schedule block"
        assert require_schedule or not resource_eefs, "segment resource_ids are currently supported only by ScheduleGen"
        for resource_id, eef_names in resource_eefs.items():
            assert (
                len(eef_names) >= 2
            ), f"resource {resource_id!r} must be claimed by segments on at least two end-effector lanes"
        segments_by_id = {
            SegmentId(eef_name, subtask.name, segment.name): segment
            for eef_name, subtasks in self.subtasks.items()
            for subtask in subtasks
            for segment in subtask.segments
        }
        segment_indices = {
            segment_id: index
            for eef_name in self.subtasks
            for index, segment_id in enumerate(
                segment_id for segment_id in segments_by_id if segment_id.eef_name == eef_name
            )
        }
        for consumer_id, consumer in segments_by_id.items():
            goal = consumer.motion_goal
            if goal is None or goal.object_pose_after_segment_id is None:
                continue
            assert require_schedule, "object_pose_after_segment_id is currently supported only by ScheduleGen"
            producer_id = goal.object_pose_after_segment_id
            assert producer_id in segments_by_id, f"motion goal references unknown effect segment {producer_id}"
            assert producer_id != consumer_id, "a motion goal cannot consume its own object pose"
            producer = segments_by_id[producer_id]
            assert any(
                event.event_type == SegmentEventType.DETACH and event.object_ref == goal.object_ref
                for event in producer.events
            ), (
                f"effect producer {producer_id} must detach motion-goal object {goal.object_ref!r} "
                "before its terminal pose can be consumed"
            )
            if producer_id.eef_name == consumer_id.eef_name:
                assert (
                    segment_indices[producer_id] < segment_indices[consumer_id]
                ), f"effect producer {producer_id} must precede consumer {consumer_id} on its EEF lane"
                continue
            assert self.schedule is not None
            assert any(
                isinstance(constraint, PrecedenceConstraintSpec)
                and constraint.before.segment_id == producer_id
                and constraint.before.point == SegmentAnchorPoint.END
                and constraint.after.segment_id == consumer_id
                and constraint.after.point == SegmentAnchorPoint.START
                for constraint in self.schedule.constraints
            ), (
                f"cross-lane object-pose dependency {producer_id} -> {consumer_id} requires explicit "
                "producer.end -> consumer.start precedence"
            )
        self.get_path_build_order()
        policy = self.generation_policy
        if not self.requires_source_demos():
            assert policy.source_selection_anchor is None, "source_selection_anchor requires demonstration segments"
            assert policy.source_demo_keys is None, "source_demo_keys requires demonstration segments"
        if policy.source_selection_anchor is not None:
            assert require_schedule, "explicit episode source selection is currently supported only by ScheduleGen"
            assert (
                not policy.select_src_per_subtask and not policy.select_src_per_arm
            ), "source_selection_anchor requires one shared source per episode"
            eef_name, separator, subtask_name = policy.source_selection_anchor.partition(".")
            assert (
                separator and eef_name in self.subtasks
            ), "source_selection_anchor must name a known '<eef>.<subtask>'"
            assert any(
                subtask.name == subtask_name for subtask in self.subtasks[eef_name]
            ), f"source_selection_anchor references unknown subtask {policy.source_selection_anchor!r}"
        else:
            assert (
                policy.source_candidate_count == 1
            ), "source_candidate_count greater than one requires source_selection_anchor"
        if self.schedule is not None:
            referenced = {
                segment_id
                for constraint in self.schedule.constraints
                for segment_id in constraint.referenced_segment_ids
            }
            missing = referenced - segment_ids
            assert not missing, f"schedule references unknown segments: {sorted(map(str, missing))}"
            for constraint in self.schedule.constraints:
                if not isinstance(constraint, NonPreemptiveConstraintSpec):
                    continue
                eef_names = {segment_id.eef_name for segment_id in constraint.segments}
                assert len(eef_names) == 1, "non-preemptive segments must use one end-effector lane"
                indices = tuple(segment_indices[segment_id] for segment_id in constraint.segments)
                assert indices == tuple(
                    range(indices[0], indices[0] + len(indices))
                ), "non-preemptive segments must be a contiguous lane sequence in declaration order"
