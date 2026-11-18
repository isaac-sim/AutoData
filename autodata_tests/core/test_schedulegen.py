# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-building and replay invariants."""

import asyncio
import torch
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import autodata_core.schedulegen.algorithm as algorithm_module
from autodata_core.algorithms import get_algorithm
from autodata_core.data_generator import DataGenerator
from autodata_core.datagen_info import DatagenInfo
from autodata_core.schedulegen import ScheduleGen
from autodata_core.schedulegen.collision import _joint_timeline_densify, _path_densify, scheduled_world_validate
from autodata_core.schedulegen.path_builder import PathBuilder
from autodata_core.schedulegen.scheduling import schedule_robot_contacts, schedule_world_contacts
from autodata_core.schedulegen.types import (
    ArmPath,
    EpisodeGenerationError,
    EpisodePaths,
    PathObjectEvent,
    ScheduledObjectEvent,
)
from autodata_core.scheduling import Schedule
from autodata_core.waypoint import Waypoint
from autodata_interfaces.embodiments.embodiment_types import NamedJointTrajectory
from autodata_interfaces.tasks.schedule_constraint_spec import SegmentId
from autodata_interfaces.tasks.segment_spec import (
    SegmentEventPoint,
    SegmentEventSpec,
    SegmentEventType,
    SegmentRobotContactSpec,
    SegmentWorldContactSpec,
)
from autodata_utils import pose_math


@pytest.mark.parametrize("collision_kind", ["self", "world"])
def test_reset_transition_and_world_queries_use_dense_boundary_samples(collision_kind):
    from autodata_core.schedulegen.collision import scheduled_robot_validate

    segment = SegmentId.from_string("left.move.transit")
    path = ArmPath("left", (Waypoint(torch.eye(4), {}),) * 2, torch.full((2, 1), 2.0), {segment: (0, 2)})
    config = SimpleNamespace(ee_link_name="tool", position_threshold=0.01, rotation_threshold=0.01)
    planner = SimpleNamespace(
        active_joint_names=("joint",),
        locked_joint_positions={},
        plan_visualizer=None,
        planning_root_pose_refresh=lambda: None,
        joint_positions_to_link_poses=lambda states, links: {"tool": torch.eye(4).repeat(len(states), 1, 1)},
        joint_positions_self_collision_free=lambda states, margin: (
            (states[:, 0] != 1) if collision_kind == "self" else torch.ones(len(states), dtype=torch.bool)
        ),
        update_world=lambda: None,
        joint_positions_world_collision_free=lambda states, margin, **kwargs: states[:, 0] != 1,
    )
    arguments = (
        SimpleNamespace(get_robot_joint_names=lambda: ["joint"]),
        {"left": SimpleNamespace(active_joint_names=("joint",), config=config)},
        planner,
        {"left": path},
        Schedule({segment: (0, 1)}, 2),
        torch.zeros(1),
    )
    if collision_kind == "self":
        with pytest.raises(EpisodeGenerationError, match="self collision"):
            scheduled_robot_validate(*arguments, collision_margin_m=0, densify_factor=2)
    else:
        robot = scheduled_robot_validate(*arguments, collision_margin_m=0, densify_factor=2)
        assert robot.collision_sample_ticks == (0, 0.5, 1, 1.5, 2)
        with pytest.raises(EpisodeGenerationError, match="schedule tick 0.5"):
            scheduled_world_validate(planner, robot, {}, {}, collision_margin_m=0, object_manipulation_starts={})


@pytest.mark.parametrize("has_release_contact", [False, True])
def test_carried_object_checks_do_not_start_only_at_release(has_release_contact):
    from autodata_core.schedulegen.collision import scheduled_robot_object_validate
    from autodata_core.schedulegen.types import ScheduledRobotContact

    segment = SegmentId.from_string("left.release.skill")
    contacts = (
        (ScheduledRobotContact(segment, SegmentRobotContactSpec("beaker", ("hand",)), 2, 3),)
        if has_release_contact
        else ()
    )
    robot = SimpleNamespace(initial_joint_positions=torch.zeros(1), joint_positions=torch.tensor([[1.0], [2.0]]))
    planner = SimpleNamespace(
        joint_positions_environment_spheres_collision_free=lambda states, *args, **kwargs: states[:, 0] != 1
    )
    with pytest.raises(EpisodeGenerationError, match="boundary tick 1"):
        scheduled_robot_object_validate(
            planner,
            robot,
            {"beaker": torch.ones(3, 1, 4)},
            collision_margin_m=0,
            object_manipulation_starts={"beaker": 1},
            robot_contacts=contacts,
        )


def test_original_mesh_covers_object_until_manipulation_begins():
    from autodata_core.schedulegen.collision import scheduled_robot_object_validate

    robot = SimpleNamespace(initial_joint_positions=torch.zeros(1), joint_positions=torch.tensor([[1.0], [2.0]]))
    query = Mock(side_effect=lambda states, *args: states[:, 0] == 2)
    planner = SimpleNamespace(joint_positions_environment_spheres_collision_free=query)
    assert scheduled_robot_object_validate(
        planner,
        robot,
        {"beaker": torch.ones(3, 1, 4)},
        collision_margin_m=0,
        object_manipulation_starts={"beaker": 2},
        robot_contacts=(),
    ) == (1, 0)
    assert query.call_args.args[0].tolist() == [[2.0]]


def test_projected_objects_are_checked_against_each_other():
    first = torch.tensor([[[-2.0, 0, 0, 0.1]], [[0.0, 0, 0, 0.1]], [[-2.0, 0, 0, 0.1]]])
    second = first.clone()
    second[:, :, 0] *= -1
    robot = SimpleNamespace(
        joint_positions=torch.zeros(2, 1), collision_joint_positions=torch.zeros(3, 1), collision_sample_ticks=(0, 1, 2)
    )
    planner = SimpleNamespace(
        update_world=lambda: None,
        joint_positions_world_collision_free=lambda states, *args, **kwargs: torch.ones(len(states), dtype=torch.bool),
        environment_spheres_world_collision_free=lambda spheres, *args, **kwargs: torch.ones(
            len(spheres), dtype=torch.bool
        ),
    )
    with pytest.raises(EpisodeGenerationError, match="projected objects"):
        scheduled_world_validate(
            planner,
            robot,
            {"first": first, "second": second},
            {name: SimpleNamespace(obstacle_path=f"/{name}") for name in ("first", "second")},
            collision_margin_m=0,
            object_manipulation_starts={"first": 0, "second": 0},
        )


def test_grasp_and_carry_permissions_allow_only_the_carrying_hand():
    grasp, release = (SegmentId.from_string(f"left.{name}.skill") for name in ("grasp", "release"))
    path = ArmPath("left", (Waypoint(torch.eye(4), {}),) * 5, torch.zeros(5, 1), {grasp: (0, 3), release: (3, 5)})
    events = tuple(
        ScheduledObjectEvent(
            tick,
            PathObjectEvent(
                segment,
                0,
                SegmentEventSpec(kind, "beaker", attachment_frame="tool" if kind == SegmentEventType.ATTACH else ""),
                torch.eye(4) if kind == SegmentEventType.ATTACH else None,
            ),
        )
        for tick, segment, kind in (
            (2, grasp, SegmentEventType.ATTACH),
            (4, release, SegmentEventType.DETACH),
        )
    )
    contacts = schedule_robot_contacts(
        {"left": path},
        Schedule({grasp: (0, 1, 2), release: (3, 4)}, 5),
        {},
        events,
        attachment_link_names={"left": ("hand",)},
    )
    assert len(contacts) == 1
    assert (contacts[0].start_tick, contacts[0].stop_tick) == (0, 4)
    assert contacts[0].spec.robot_link_names == ("hand",)


def test_buffered_generation_rejects_ignored_reset_settling():
    from autodata_core.schedulegen.batching import generate_buffered
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    datastream = SimpleNamespace(
        get_env=lambda: None, get_generation_policy=lambda: GenerationPolicy(reset_settling_steps=2)
    )
    with pytest.raises(AssertionError, match="does not support reset_settling_steps"):
        generate_buffered(
            env=None,
            data_generator=SimpleNamespace(datastream=datastream),
            algorithm=SimpleNamespace(planning_contexts={0: None}),
            success_term=None,
            schedule_workers=1,
        )


@pytest.mark.parametrize(
    "point, slot", [(SegmentEventPoint.START, 0), (SegmentEventPoint.END, 6), (SegmentEventPoint.SOURCE_ANNOTATION, 4)]
)
def test_attachment_events_support_boundaries_and_embedded_annotations(point, slot):
    """Boundary events need no extra marks; internal marks keep their exact merged path slot."""
    pool = SimpleNamespace(
        demo_keys=("source",),
        datagen_infos=[
            DatagenInfo(
                target_eef_pose={"left": torch.eye(4).repeat(10, 1, 1)},
                object_poses={"beaker": torch.eye(4).repeat(10, 1, 1)},
            )
        ],
        get_object_event_index=Mock(return_value=3),
    )
    event = SegmentEventSpec(
        SegmentEventType.ATTACH,
        "beaker",
        point=point,
        attachment_frame="hand",
        annotation_key="grasp" if point == SegmentEventPoint.SOURCE_ANNOTATION else "",
    )
    resolved, poses = PathBuilder({})._skill_events_resolve(
        data_generator=SimpleNamespace(src_demo_datagen_info_pool=pool),
        planner=SimpleNamespace(config=SimpleNamespace(ee_link_name="hand")),
        segment_id=SegmentId("left", "pick", "skill"),
        events=(event,),
        source_demo_index=0,
        source_boundary=(1, 5),
        source_length=4,
        merged_skill=tuple(Waypoint(pose=torch.eye(4), passthrough_action={}) for _ in range(6)),
    )
    assert tuple(resolved) == (slot,)
    assert torch.equal(resolved[slot][0].tool_from_object, torch.eye(4))
    assert poses
    if point == SegmentEventPoint.SOURCE_ANNOTATION:
        pool.get_object_event_index.assert_called_once_with(0, "grasp")
    else:
        pool.get_object_event_index.assert_not_called()


@pytest.mark.parametrize("collision_tick", [None, 1, 3])
def test_unmanipulated_object_mesh_remains_a_world_obstacle(collision_tick):
    """Initial reaches cannot ignore an object merely because it is manipulated later."""
    calls = []

    def robot_query(states, margin, *, excluded_obstacle_paths):
        """Report a collision only while the original mesh is active."""
        calls.append((states[:, 0].tolist(), excluded_obstacle_paths))
        free = torch.ones(len(states), dtype=torch.bool)
        if collision_tick is not None and "/bowl" not in excluded_obstacle_paths:
            free &= states[:, 0] != collision_tick
        return free

    planner = SimpleNamespace(
        update_world=lambda: None,
        joint_positions_world_collision_free=robot_query,
        joint_positions_world_collision_witness=lambda *args, **kwargs: ("finger", 0, "/bowl"),
        environment_spheres_world_collision_free=lambda spheres, *args, **kwargs: torch.ones(
            len(spheres), dtype=torch.bool
        ),
    )
    arguments = (
        planner,
        SimpleNamespace(
            joint_positions=torch.arange(4).reshape(4, 1),
            collision_joint_positions=torch.arange(4).reshape(4, 1),
            collision_sample_ticks=(0.0, 1.0, 2.0, 3.0),
        ),
        {"bowl": torch.ones(5, 1, 4)},
        {"bowl": SimpleNamespace(obstacle_path="/bowl")},
    )
    options = dict(collision_margin_m=0.0, object_manipulation_starts={"bowl": 2})
    if collision_tick == 1:
        with pytest.raises(EpisodeGenerationError, match="schedule tick 1/4"):
            scheduled_world_validate(*arguments, **options)
    else:
        assert scheduled_world_validate(*arguments, **options) == (4, 5, 0)
        assert calls == [([0, 1], ()), ([2, 3], ("/bowl",))]


def test_contact_intervals_preserve_initial_inter_segment_and_terminal_holds():
    """Contact permissions follow held occupancy rather than nominal segment endpoints."""
    release = SegmentId.from_string("left.release.skill")
    hold = SegmentId.from_string("left.release.open_hold")
    path = ArmPath(
        eef_name="left",
        waypoints=tuple(Waypoint(torch.eye(4), {}) for _ in range(5)),
        joint_positions=torch.zeros(5, 1),
        segment_bounds={release: (0, 2), hold: (2, 5)},
    )
    schedule = Schedule({release: (3, 5), hold: (8, 10, 12)}, makespan_ticks=15)
    world_spec = SegmentWorldContactSpec("beaker", "bin")
    contacts = schedule_world_contacts({"left": path}, schedule, {release: (world_spec,), hold: (world_spec,)})
    assert [(contact.start_tick, contact.stop_tick) for contact in contacts] == [(0, 8), (8, 16)]

    detach = ScheduledObjectEvent(
        5,
        PathObjectEvent(
            release,
            1,
            SegmentEventSpec(SegmentEventType.DETACH, "beaker", SegmentEventPoint.SOURCE_ANNOTATION, "release"),
        ),
    )
    robot_contacts = schedule_robot_contacts(
        {"left": path},
        schedule,
        {
            release: (SegmentRobotContactSpec("beaker", ("finger",), start_annotation_key="release"),),
            hold: (SegmentRobotContactSpec("beaker", ("finger",), clear_at_end=True),),
        },
        (detach,),
    )
    assert [(contact.start_tick, contact.stop_tick) for contact in robot_contacts] == [(5, 8), (8, 12)]
    with pytest.raises(AssertionError, match="exactly one event"):
        schedule_robot_contacts(
            {"left": path},
            schedule,
            {release: (SegmentRobotContactSpec("beaker", ("finger",), start_annotation_key="missing"),)},
            (detach,),
        )


@pytest.mark.parametrize("shared_model", [False, True])
def test_environment_contexts_and_caches_are_isolated_during_replay(monkeypatch, shared_model):
    contexts = {env_id: SimpleNamespace(group_planners={"left": object()}, full_state=None) for env_id in (0, 1)}
    algorithm = get_algorithm("schedulegen", planning_contexts=contexts)
    contexts.clear()
    assert isinstance(algorithm, ScheduleGen)
    first, second = algorithm._path_builders.values()
    assert first is not second
    assert first._object_geometry_cache is not second._object_geometry_cache
    seen = []

    async def execute(self, *, env_id, **kwargs):
        await asyncio.sleep(0)
        seen.append(env_id)
        return False

    def pose_error(datastream, planner, env_id, pose):
        assert planner is algorithm.planning_contexts[0 if shared_model else env_id].group_planners["left"]
        return 0.0, 0.0

    for context in algorithm.planning_contexts.values():
        context.group_planners["left"] = SimpleNamespace(
            config=SimpleNamespace(position_threshold=0.01, rotation_threshold=0.01)
        )
    monkeypatch.setattr(
        algorithm_module, "_scheduled_waypoints_build", lambda path, schedule: (SimpleNamespace(pose=None),)
    )
    monkeypatch.setattr(algorithm_module.MultiWaypoint, "execute", execute)
    monkeypatch.setattr(algorithm_module, "_terminal_pose_error", pose_error)
    datastream = SimpleNamespace(
        get_generation_policy=lambda: SimpleNamespace(final_hold_steps=0, terminal_convergence_steps=1),
        get_robot_joint_positions=lambda env_ids: torch.zeros(1, 1),
    )

    async def replay_both():
        return await asyncio.gather(*(
            algorithm._schedule_replay(
                data_generator=SimpleNamespace(datastream=datastream),
                env_id=env_id,
                success_term=None,
                env_action_queue=None,
                arm_paths={"left": object()},
                schedule=SimpleNamespace(makespan_ticks=1),
                planning_env_id=0 if shared_model else None,
            )
            for env_id in (0, 1)
        ))

    assert asyncio.run(replay_both()) == [(False, None), (False, None)]
    assert sorted(seen) == [0, 1]


@pytest.mark.parametrize("build_fails", [False, True])
def test_episode_hook_preserves_shared_reset_recording_and_failure_handling(monkeypatch, build_fails):
    context = SimpleNamespace(group_planners={"left": object()}, full_state=None)
    algorithm = ScheduleGen({0: context})
    order = []

    def build(*args, **kwargs):
        order.append("build")
        if build_fails:
            raise EpisodeGenerationError("planning", "no path")
        return object(), ()

    def restore(*args):
        order.append("restore")

    async def replay(**kwargs):
        order.append("replay")
        return True, 0

    monkeypatch.setattr(PathBuilder, "build", build)
    monkeypatch.setattr(ScheduleGen, "_state_restore", staticmethod(restore))
    monkeypatch.setattr(algorithm, "_schedule_replay", replay)
    monkeypatch.setattr(
        algorithm,
        "_episode_validate",
        lambda **kwargs: SimpleNamespace(
            arm_paths={},
            schedule=None,
            replay_joint_trajectory=None,
            diagnostics_before_success=(),
            diagnostics_after_success=(),
        ),
    )
    generator = object.__new__(DataGenerator)
    generator.algorithm = algorithm
    generator.datastream = SimpleNamespace(
        get_eef_names=lambda: ["left"],
        get_schedule_spec=lambda: object(),
        get_scene_state=lambda **kwargs: {},
        get_object_poses=lambda **kwargs: {},
        get_robot_joint_positions=lambda **kwargs: torch.zeros(1, 1),
    )
    generator.src_demo_datagen_info_pool = SimpleNamespace(demo_keys=[])
    generator.randomize_subtask_boundaries = Mock(return_value={})
    generator._reset_environment_for_generation = AsyncMock(return_value=(torch.tensor([0]), {"reset": True}))
    generator._finalize_episode_recording = Mock()
    result = asyncio.run(generator.generate(0, None, env_reset_queue=None, env_action_queue=None, export_demo=False))
    assert result.initial_state == {"reset": True}
    assert result.success is not build_fails
    assert dict(result.diagnostics)["stage"] == ("planning" if build_fails else "complete")
    assert order == (["build", "restore"] if build_fails else ["build", "restore", "replay"])
    generator._reset_environment_for_generation.assert_awaited_once()
    assert generator._finalize_episode_recording.call_args.kwargs["success"] is not build_fails
    assert generator._finalize_episode_recording.call_args.kwargs["export_demo"] is False


@pytest.mark.parametrize("failed_step", [None, "object_events", "object_poses", "world_contacts", "robot_contacts"])
def test_scheduled_validation_preserves_diagnostics_and_failure_stages(monkeypatch, failed_step):
    """Tick mapping retains successful diagnostics and identifies the first rejected step."""
    import autodata_core.schedulegen.scheduling as scheduling_module

    context = SimpleNamespace(group_planners={"left": object()}, full_state=None)
    algorithm = ScheduleGen({0: context})
    segment = SegmentId.from_string("left.pick.skill")
    path = ArmPath("left", (Waypoint(torch.eye(4), {}),), torch.zeros(1, 1), {segment: (0, 1)})
    datastream = SimpleNamespace(get_segment_world_contacts=lambda: {}, get_segment_robot_contacts=lambda: {})
    steps = ("object_events", "object_poses", "world_contacts", "robot_contacts")
    arguments = dict(
        data_generator=SimpleNamespace(datastream=datastream),
        episode_paths=EpisodePaths({"left": path}, {}),
        initial_object_poses={},
        initial_joint_positions=torch.zeros(1),
        schedule_spec=SimpleNamespace(collision_margin_m=0.0),
        conflicts=(),
        schedule=Schedule({segment: (0,)}, makespan_ticks=1),
    )
    if failed_step is not None:
        monkeypatch.setattr(scheduling_module, f"schedule_{failed_step}", Mock(side_effect=AssertionError("invalid")))
        with pytest.raises(EpisodeGenerationError, match="invalid") as caught:
            algorithm._episode_validate_scheduled(**arguments)
        assert caught.value.stage == f"{failed_step.removesuffix('s')}_lowering"
        diagnostics = caught.value.diagnostics
        expected_steps = steps[: steps.index(failed_step)]
    else:
        validated = algorithm._episode_validate_scheduled(**arguments)
        assert validated.replay_joint_trajectory is None
        assert validated.diagnostics_after_success == ()
        diagnostics = validated.diagnostics_before_success
        expected_steps = steps
    assert tuple(name for name, _ in diagnostics) == (
        "object_collision_spheres",
        "collision_conflicts",
        "makespan_ticks",
        "schedule_segments",
        *(f"scheduled_{step}" for step in expected_steps),
    )


def test_buffered_states_stack_without_aliasing_or_extra_environment_rows():
    from autodata_core.schedulegen.batching import _states_stack

    states = [{"robot": {"position": torch.full((1, 2), float(index))}} for index in range(3)]
    stacked = _states_stack(states)
    assert torch.equal(stacked["robot"]["position"], torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    stacked["robot"]["position"].zero_()
    assert states[2]["robot"]["position"].sum() == 4
    with pytest.raises(AssertionError, match="exactly one environment"):
        _states_stack([stacked])


@pytest.mark.parametrize("guarantee_success", [False, True])
def test_buffered_preparation_failures_use_the_finite_attempt_budget_without_empty_exports(guarantee_success):
    from autodata_core.algorithms import AlgorithmEpisodeResult
    from autodata_core.schedulegen.algorithm import _PreparedEpisode
    from autodata_core.schedulegen.batching import generate_buffered
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    env = SimpleNamespace(num_envs=3, recorder_manager=SimpleNamespace(reset=Mock()), reset=Mock())
    datastream = SimpleNamespace(
        device="cpu",
        get_env=lambda: env,
        get_generation_policy=lambda: GenerationPolicy(
            num_trials=5, guarantee_success=guarantee_success, max_num_failures=2
        ),
        get_schedule_spec=lambda: object(),
    )
    generator = SimpleNamespace(datastream=datastream)
    prepare = Mock(
        side_effect=lambda *args: _PreparedEpisode(
            planning_env_id=0,
            initial_state={},
            initial_object_poses={},
            initial_joint_positions=torch.zeros(1),
            selected_source={},
            result=AlgorithmEpisodeResult(False, (("stage", "planning"),)),
        )
    )
    algorithm = SimpleNamespace(planning_contexts={0: object()}, _episode_prepare=prepare)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = generate_buffered(env, generator, algorithm, None, schedule_workers=2)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    count = 7 if guarantee_success else 5
    assert result["completed"] is not guarantee_success
    assert result["stats"] == {"num_success": 0, "num_failures": count, "num_attempts": count}
    assert len(result["attempts"]) == prepare.call_count == count
    assert env.reset.call_count == count
    assert result["execution"]["schedule_workers"] == 2


@pytest.mark.parametrize(
    "lengths, initial_success, initial_attempts, success",
    [
        ((2, 5, 3), 0, 0, True),
        ((2,), 0, 0, True),
        ((2, 5, 3), 3, 5, True),
        ((2,), 3, 5, True),
        ((2,), 9, 9, True),
        ((2,), 3, 59, False),
    ],
)
def test_buffered_replay_exports_unequal_slots_once_without_hold_tails(
    monkeypatch, capsys, lengths, initial_success, initial_attempts, success
):
    import autodata_core.schedulegen.batching as batching
    from autodata_core.algorithms import AlgorithmEpisodeResult
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        cfg=SimpleNamespace(),
        action_space=SimpleNamespace(shape=(3, 1)),
        recorder_manager=SimpleNamespace(reset=Mock()),
        reset_to=Mock(),
        sim=SimpleNamespace(is_stopped=lambda: False),
        ticks=0,
    )
    actions = []

    def step(action):
        actions.append(action.clone())
        env.ticks += 1

    env.step = step
    datastream = SimpleNamespace(
        device="cpu",
        get_env=lambda: env,
        get_generation_policy=GenerationPolicy,
        reset_robot_joint_posture_target=Mock(),
    )
    exported = []
    executed = []
    generator = SimpleNamespace(
        datastream=datastream,
        _finalize_episode_recording=lambda **kwargs: exported.append((int(kwargs["env_id_tensor"][0]), env.ticks)),
    )

    async def send(env_id, queue, value):
        await queue.put((env_id, torch.tensor([float(value)])))
        await queue.join()

    async def execute(generator, prepared, env_id, success_term, queue):
        executed.append(env_id)
        for _ in range(prepared.length):
            await send(env_id, queue, prepared.length)
        return AlgorithmEpisodeResult(success=success)

    async def hold(self, datastream, success_term, env_id, env_action_queue):
        await send(env_id, env_action_queue, -1)

    monkeypatch.setattr(batching.MultiWaypoint, "execute", hold)
    ready = [
        SimpleNamespace(
            length=length,
            start_time_s=0.0,
            initial_state={"robot": {"position": torch.tensor([[float(index)]])}},
            validated=SimpleNamespace(
                arm_paths={"left": SimpleNamespace(waypoints=(object(),))}, replay_joint_trajectory=None
            ),
        )
        for index, length in enumerate(lengths)
    ]
    stats = {
        "num_success": initial_success,
        "num_failures": initial_attempts - initial_success,
        "num_attempts": initial_attempts,
    }
    attempts = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        batching._replay_wave(
            env, generator, SimpleNamespace(_episode_execute=execute), None, ready, loop, stats, attempts
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert sorted(exported) == list(enumerate(lengths))
    assert sorted(executed) == list(range(len(lengths)))
    assert env.ticks == max(lengths)
    assert stats == {
        "num_success": initial_success + success * len(lengths),
        "num_failures": initial_attempts - initial_success + (not success) * len(lengths),
        "num_attempts": initial_attempts + len(lengths),
    }
    assert sorted(attempt["env_id"] for attempt in attempts) == list(range(len(lengths)))
    assert f"{initial_success + success * len(lengths)}/10 successful demos generated" in capsys.readouterr().out
    assert not loop.is_running()
    if len(lengths) > 1:
        assert actions[-1][0].item() == -1


@pytest.mark.parametrize(
    "num_envs, guarantee_success, failure_budget, expected_batches, expected_success, completed",
    [
        (5, True, 5, [5, 2, 1], 5, True),
        (5, True, 2, [5, 2], 4, False),
        (5, False, 5, [5], 3, True),
        (2, True, 5, [2, 2, 2, 1, 1], 5, True),
    ],
)
def test_buffered_retries_keep_run_totals_and_stop_at_the_requested_target(
    monkeypatch, num_envs, guarantee_success, failure_budget, expected_batches, expected_success, completed
):
    """Failed replays refill only missing demos; fixed-attempt runs and finite budgets remain bounded."""
    import autodata_core.schedulegen.batching as batching
    from autodata_core.data_generator import GenerationResult
    from autodata_core.schedulegen.algorithm import _PreparedEpisode
    from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

    env = SimpleNamespace(num_envs=num_envs, recorder_manager=SimpleNamespace(reset=Mock()), reset=Mock())
    policy = GenerationPolicy(num_trials=5, guarantee_success=guarantee_success, max_num_failures=failure_budget)
    generator = SimpleNamespace(
        datastream=SimpleNamespace(
            device="cpu",
            get_env=lambda: env,
            get_generation_policy=lambda: policy,
            get_schedule_spec=lambda: SimpleNamespace(collision_margin_m=0.0),
        )
    )
    prepare = Mock(
        side_effect=lambda *args: _PreparedEpisode(0, {}, {}, torch.zeros(1), {}, paths=SimpleNamespace(arm_paths={}))
    )
    algorithm = SimpleNamespace(
        planning_contexts={0: object()},
        _episode_prepare=prepare,
        _conflicts_find=Mock(return_value=()),
        _episode_schedule_solve=Mock(return_value=object()),
        _state_restore=Mock(),
        _episode_validate_scheduled=Mock(return_value=object()),
    )
    outcomes = iter((True, False, False, True, True, True, False, True))
    batches = []

    def replay(env, generator, algorithm, success_term, ready, loop, stats, attempts, **kwargs):
        batches.append(len(ready))
        for env_id, _ in enumerate(ready):
            batching.record_generation_attempt(GenerationResult({}, next(outcomes)), env_id, stats, attempts)

    monkeypatch.setattr(batching, "_replay_wave", replay)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = batching.generate_buffered(env, generator, algorithm, None, schedule_workers=2)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert batches == expected_batches
    assert result["stats"] == {
        "num_success": expected_success,
        "num_failures": sum(expected_batches) - expected_success,
        "num_attempts": sum(expected_batches),
    }
    assert result["completed"] is completed
    assert prepare.call_count == len(result["attempts"]) == sum(expected_batches)
    assert [attempt["attempt"] for attempt in result["attempts"]] == list(range(1, sum(expected_batches) + 1))


def test_carried_transit_uses_explicit_poses_without_simulator_access():
    captured = {}

    def plan(**kwargs):
        captured.update(kwargs)
        return True

    pose = torch.eye(4)
    carried_pose = pose.clone()
    carried_pose[0, 3] = 0.25
    overrides = {"bowl": pose}
    planner = SimpleNamespace(
        config=SimpleNamespace(planning_group_name="left_arm", motion_noise_scale=0.0, retreat_distance=0.1),
        step_size=None,
        active_joint_names=("joint",),
        get_planned_joint_trajectory=lambda: NamedJointTrajectory(("joint",), torch.zeros(1, 1)),
        get_planned_poses=lambda: [pose],
        update_world_and_plan_motion=plan,
    )
    generator = SimpleNamespace(datastream=SimpleNamespace(passthrough_actions_to_joint_positions=lambda actions: {}))

    waypoints, positions = PathBuilder({"left": planner})._transit_plan(
        generator, planner, 0, pose, {}, torch.zeros(1), "beaker", carried_pose, overrides
    )

    assert captured["object_pose_overrides"]["beaker"] is carried_pose
    assert captured["object_pose_overrides"]["bowl"] is pose
    assert "beaker" not in overrides
    assert captured["force_object_attachment"]
    assert len(waypoints) == len(positions) == 1


@pytest.mark.parametrize("finite", [True, False])
def test_replay_stops_at_first_non_finite_robot_state(monkeypatch, finite):
    monkeypatch.setattr(algorithm_module, "_scheduled_waypoints_build", lambda path, schedule: (object(),))

    async def execute(self, **kwargs):
        return False

    monkeypatch.setattr(algorithm_module.MultiWaypoint, "execute", execute)
    positions = torch.tensor([[0.0 if finite else float("nan")]])
    datastream = SimpleNamespace(
        get_generation_policy=lambda: SimpleNamespace(final_hold_steps=0, terminal_convergence_steps=0),
        get_robot_joint_positions=lambda env_ids: positions,
    )
    context = SimpleNamespace(group_planners={"left": object()}, full_state=None)
    algorithm = ScheduleGen({0: context})
    replay = algorithm._schedule_replay(
        data_generator=SimpleNamespace(datastream=datastream),
        env_id=0,
        success_term=None,
        env_action_queue=None,
        arm_paths={"left": object()},
        schedule=SimpleNamespace(makespan_ticks=1),
    )
    if finite:
        assert asyncio.run(replay) == (False, None)
    else:
        with pytest.raises(AssertionError, match="non-finite robot joints at replay tick 0"):
            asyncio.run(replay)


@pytest.mark.parametrize("factor", [1, 3])
def test_collision_sampling_preserves_ticks_and_waypoint_support(factor):
    path = torch.tensor([[0.0], [1.0], [3.0]])
    dense, ticks = _joint_timeline_densify(path, factor)
    conflict_dense, supports = _path_densify(path, factor)
    assert torch.equal(dense, conflict_dense)
    assert len(ticks) == len(supports) == 2 * factor + 1
    assert ticks[::factor] == (0.0, 1.0, 2.0)
    assert supports[::factor] == ((0,), (1,), (2,))
    assert torch.equal(dense[::factor], path)
    if factor > 1:
        assert supports[1] == (0, 1) and supports[-2] == (1, 2)
        assert ticks[1] == pytest.approx(1 / factor)


def test_pose_errors_agree_for_scalar_and_batched_targets():
    actual, target = torch.eye(4), torch.eye(4)
    target[:3, 3] = torch.tensor([3.0, 4.0, 0.0])
    target[:2, :2] *= -1
    scalar = pose_math.pose_errors(actual, target)
    batched = pose_math.pose_errors(actual.unsqueeze(0), target.unsqueeze(0))
    assert scalar[0].item() == pytest.approx(5.0)
    assert scalar[1].item() == pytest.approx(torch.pi)
    assert all(torch.equal(value.reshape(1), batch) for value, batch in zip(scalar, batched, strict=True))
