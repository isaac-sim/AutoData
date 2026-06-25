# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ScheduleStream generation algorithm (cuStream2 / cuRoboV2 backend).

This plug-in ports the standalone ScheduleStream IsaacLab TAMP agent
(``schedulestream/applications/isaaclab/{tamp_agent,planner,controller}.py``) into the
Isaac AutoData :class:`GenerationAlgorithm` interface, adapted to the cuStream2 application
(cuRoboV2). Unlike MimicGen/SkillGen — which transform *recorded source demonstrations* per
subtask — ScheduleStream plans the **whole task from scratch** from a symbolic goal via
``custream2.example.solve_tamp`` and replays the resulting command sequence.

To fit the per-subtask plug-in contract, the matching task descriptor declares a **single
subtask** per EEF (see ``tasks/franka_cube_stack_schedulestream.yaml``). The single
:meth:`ScheduleStream.plan_subtask_trajectory` call solves TAMP once and returns the entire
trajectory as a flat ``list[Waypoint]``; :class:`DataGenerator` then steps and records it like any
other generated demo.

Heavy dependencies (cuRobo, the ``schedulestream`` package, the IsaacLab stack MDP) are imported
lazily inside methods so importing this module to register the algorithm never requires them until
a run actually selects ``--alg schedulestream``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from isaac_autodata_core.algorithms import GenerationAlgorithm

if TYPE_CHECKING:
    from isaac_autodata_core.data_generator import DataGenerator, _EEFGenerationState
    from isaac_autodata_core.waypoint import Waypoint
    from isaac_autodata_interfaces.datastream.datastream import Datastream

# Gripper action encoding, mirroring the ScheduleStream IsaacLab controller: open is negative,
# close is positive. The Franka IK-Rel binary gripper action is a single trailing dim.
OPEN_ACTION = 1.0
CLOSE_ACTION = -OPEN_ACTION


@contextlib.contextmanager
def _autograd_enabled():
    """Locally re-enable autograd for cuRobo calls.

    The Isaac AutoData ``env_loop`` runs the whole generation under ``torch.inference_mode()``, but
    cuRobo's IK / trajectory optimizers call ``backward()`` and therefore need autograd. Without
    this, the solver fails with "element 0 of tensors does not require grad and does not have a
    grad_fn". ``inference_mode(False)`` lifts inference mode; ``enable_grad`` lifts no_grad.
    """
    import torch

    with torch.inference_mode(False), torch.enable_grad():
        yield


def to_numpy(array: Any) -> Any:
    """Convert a torch tensor (via detach/cpu) or any array-like to a numpy array."""
    import numpy as np

    return array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)


def _reorder_positions(src_names: list[str], src_positions: Any, dst_names: list[str]) -> Any:
    """Reorder ``src_positions`` (indexed by ``src_names``) into ``dst_names`` order."""
    import numpy as np

    from schedulestream.common.utils import apply_mapping

    mapping = dict(zip(src_names, to_numpy(src_positions)))
    return np.asarray(apply_mapping(mapping, dst_names), dtype=np.float32)


def _leaf_commands(command: Any) -> Any:
    """Yield leaf commands in execution order, unwrapping Commands/Composite containers.

    Composite (parallel) commands are linearized; this is correct for the single-arm case
    ScheduleStream targets here.
    """
    from schedulestream.applications.custream2.command import Commands, Composite

    if isinstance(command, (Commands, Composite)):
        for sub in command.commands:
            yield from _leaf_commands(sub)
    else:
        yield command


class ScheduleStream(GenerationAlgorithm):
    """Whole-task TAMP via cuStream2 (cuRoboV2), exposed as a single-subtask generation algorithm.

    The algorithm owns one :class:`_ScheduleStreamPlanner` per ``env_id`` (lazily built, reused
    across ``generate()`` calls). Each invocation re-syncs the planner world from the live env,
    solves TAMP for the task goal, and converts the command sequence into executable waypoints.
    """

    name = "schedulestream"
    expected_eef_count = 1
    requires_motion_planner = False
    uses_subtask_start_signals = False
    supports_coordination = False

    def __init__(self, success_term: Any) -> None:
        """
        Args:
            success_term: The task's success :class:`TerminationTermCfg`; the symbolic goal is
                derived from it (e.g. the cube-stacking termination yields stacking ``Attached``
                relations). It is passed in because ``setup_env_config`` strips it from the env, so
                it cannot be recovered at plan time. All other tuning (collisions, max_time,
                profile, hold, animate) is read from the task descriptor's
                single-subtask ``algo_params`` (:class:`ScheduleStreamSubtaskAlgoParams`) via the
                datastream — see :meth:`_get_planner`.
        """
        self.success_term = success_term
        self._planners: dict[int, _ScheduleStreamPlanner] = {}

    def validate_setup(self, datastream: Datastream) -> None:
        """Require exactly one subtask per EEF — ScheduleStream plans the whole task at once."""
        for eef_name in datastream.get_eef_names():
            num_subtasks = datastream.num_subtasks(eef_name)
            if num_subtasks != 1:
                raise ValueError(
                    "schedulestream plans the whole task in one shot and expects exactly one "
                    f"subtask per EEF, but EEF {eef_name!r} declares {num_subtasks}. Use a "
                    "single-subtask descriptor (e.g. tasks/franka_cube_stack_schedulestream.yaml)."
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
        """Solve TAMP once for the (single) subtask and return the full trajectory.

        Returns ``(waypoints, False)`` to execute the planned trajectory as the subtask.

        On planning failure this RAISES rather than returning ``None``. Returning ``None`` makes
        ``generate()`` report failure without ever enqueuing an action, but ``env_loop`` blocks
        waiting for an action and so never reaches its attempt-count stop check — the failed attempt
        is retried forever (a livelock of repeated tracebacks). Raising propagates out through
        ``generate()`` and the data-gen task, where ``env_loop`` re-raises it and the process
        terminates with the traceback. For multi-trial production with retry-on-failure, this would
        need the upstream env_loop to count no-action attempts; until then, fail loudly.
        """
        planner = self._get_planner(data_generator.datastream, env_id)
        waypoints = planner.plan(env_id)
        if not waypoints:
            raise RuntimeError(
                f"schedulestream planning produced no trajectory for env {env_id} "
                f"(solve_tamp found no plan or returned empty commands)."
            )
        return waypoints, False

    def _get_planner(self, datastream: Datastream, env_id: int) -> _ScheduleStreamPlanner:
        if env_id not in self._planners:
            # Tuning lives on the single subtask's algo_params (ScheduleStreamSubtaskAlgoParams),
            # read here via the datastream so it stays in the task descriptor, not the CLI.
            params = datastream.get_subtask_algo_params(datastream.get_eef_names()[0])[0]
            self._planners[env_id] = _ScheduleStreamPlanner(
                datastream=datastream,
                success_term=self.success_term,
                env_id=env_id,
                collisions=params.collisions,
                max_time=params.max_time,
                profile=params.profile,
                hold=params.hold,
                animate=params.animate,
            )
        return self._planners[env_id]


class _ScheduleStreamPlanner:
    """Bridge between the Isaac AutoData env (via :class:`Datastream`) and the cuStream2 TAMP solver.

    Adapted from ``schedulestream/applications/isaaclab/planner.py`` and ``controller.py``: builds a
    cuStream2 :class:`World` from the live USD scene, derives the symbolic goal from the success
    term, and converts solved ``Commands`` into :class:`Waypoint` EEF targets in the frame the
    embodiment adapter expects. State is read through the Datastream's ``get_env()`` escape hatch
    (the env/scene handles a motion planner legitimately needs).
    """

    def __init__(
        self,
        datastream: Datastream,
        success_term: Any,
        *,
        env_id: int = 0,
        collisions: bool = True,
        max_time: float = 60.0,
        profile: bool = False,
        hold: int | None = None,
        animate: bool = False,
        verbose: bool = False,
    ) -> None:
        self.datastream = datastream
        self.env = datastream.get_env()
        self.collisions = collisions
        self.max_time = max_time
        self.profile = profile
        self.hold = hold
        self.animate = animate
        self.verbose = verbose

        # cuStream2's World (cuRoboV2) self-initializes in its constructor; the cuRoboV1-era
        # set_retract_conf()/initialize() calls from the upstream isaaclab planner don't exist here.
        self.world = self._create_world(env_id=env_id)
        self.goal = self.create_goal(success_term)
        # The IK action's control link + body offset are invariant, so resolve them once here and
        # reuse via self (rather than recomputing/threading them through the waypoint helpers).
        self.body_name, self.body_offset = self._eef_action_info()

    # ------------------------------------------------------------------
    # Env / scene accessors
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> Any:
        from isaaclab.envs import ManagerBasedEnv

        if isinstance(self.env, ManagerBasedEnv):
            return self.env
        return self.env.env

    @property
    def scene(self) -> Any:
        return self.base_env.scene

    @property
    def robot(self) -> str:
        [robot] = self.scene.articulations
        return robot

    @property
    def articulation(self) -> Any:
        return self.scene.articulations[self.robot]

    @property
    def joints(self) -> list[str]:
        return self.world.all_joints

    # ------------------------------------------------------------------
    # Pose helpers (reference frame == robot base, matching the IsaacLab planner)
    # ------------------------------------------------------------------

    def _to_pose(self, root_pose: Any) -> Any:
        """Convert an IsaacLab ``root_pose`` (``[pos(3), quat]``) to a cuRobo :class:`Pose`.

        IsaacLab's warp ``root_pose_w`` quaternion is xyzw; cuRobo wants wxyz.
        """
        from curobo.types import Pose
        from isaaclab.utils.math import convert_quat

        root_pose = self.world.to_device(root_pose)
        root_pose[..., 3:7] = convert_quat(root_pose[..., 3:7], to="wxyz")
        return Pose(position=root_pose[:, :3], quaternion=root_pose[:, 3:])

    def _pose(self, name: str) -> Any:
        state = self.scene.state
        for body_type in state:
            if name in state[body_type]:
                return self._to_pose(state[body_type][name]["root_pose"])
        raise ValueError(name)

    @property
    def reference_pose(self) -> Any:
        return self._pose(self.robot)

    def to_reference(self, pose: Any) -> Any:
        from schedulestream.applications.custream2.utils import multiply_poses

        return multiply_poses(self.reference_pose.inverse(), pose)

    def from_reference(self, pose: Any) -> Any:
        from schedulestream.applications.custream2.utils import multiply_poses

        return multiply_poses(self.reference_pose, pose)

    def pose(self, name: str, reference: bool = True) -> Any:
        pose = self._pose(name)
        if reference:
            pose = self.to_reference(pose)
        return pose

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _convert_objects(self, scene_cfg: Any) -> list[Any]:
        import numpy as np

        from isaaclab.sim import find_matching_prims
        from schedulestream.applications.custream2.object import GraspConfig, MeshObject
        from schedulestream.applications.custream2.utils import position_from_pose, to_pose

        name_from_path: dict[str, str] = {}
        for name, rigid_object in self.scene.rigid_objects.items():
            for prim in find_matching_prims(rigid_object.cfg.prim_path):
                name_from_path[prim.GetPath().pathString] = name

        objects = []
        for i, obstacle in enumerate(scene_cfg.objects):
            path = obstacle.name
            name = path
            for _path, _name in name_from_path.items():
                if path.startswith(_path):
                    name = _name
                    break

            floating = True
            if name in self.scene.rigid_objects:
                rigid_object = self.scene.rigid_objects[name]
                if (rigid_object.cfg.spawn is not None) and (rigid_object.cfg.spawn.rigid_props is not None):
                    floating = not rigid_object.cfg.spawn.rigid_props.kinematic_enabled

            mesh = obstacle.get_trimesh_mesh()
            pose = to_pose(obstacle.pose)
            # Floating objects (cubes) get a top-down grasp config (custream2/scene.py idiom);
            # static objects (e.g. the table) are collision-only.
            grasp_config = GraspConfig(roll_interval="top") if floating else None
            objects.append(MeshObject(name, mesh, pose=pose, grasp_config=grasp_config, surface_config=None))
            if self.verbose:
                print(
                    f"{i}/{len(scene_cfg.objects)}) Name: {name} | Path: {path} | Floating: {floating} "
                    f"| Position: {np.round(position_from_pose(pose), 2)}"
                )
        return objects

    def _create_objects(self, env_id: int = 0) -> list[Any]:
        from curobo._src.util.usd_scene_parser import UsdSceneParser

        usd_parser = UsdSceneParser()
        usd_parser.load_stage(self.scene.stage)

        env_path = self.scene.env_regex_ns.replace(".*", f"{env_id}")
        robot_path = f"{env_path}/Robot"
        ignore_list = [
            f"{env_path}/Robot",
            f"{env_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        scene_cfg = usd_parser.get_obstacles_from_stage(
            only_paths=[env_path],
            reference_prim_path=robot_path,
            ignore_substring=ignore_list,
            timecode=0,
        )
        obstacle_names = [obj.name for obj in scene_cfg.objects]
        assert obstacle_names, f"no obstacles parsed under {env_path}"
        if self.verbose:
            print(f"Obstacles ({len(obstacle_names)}): {obstacle_names}")
        return self._convert_objects(scene_cfg)

    def _create_world(self, env_id: int = 0) -> Any:
        import os

        from schedulestream.applications.custream2.franka import load_franka_config
        from schedulestream.applications.custream2.scene import CAMERA_POSE
        from schedulestream.applications.custream2.world import World

        objects = self._create_objects(env_id=env_id)

        usd_name = os.path.basename(self.articulation.cfg.spawn.usd_path)

        # Franka-stack scope: only the standard panda is supported.
        if usd_name != "panda_instanceable.usd":
            raise NotImplementedError(
                f"schedulestream currently supports only the Franka panda, got robot USD {usd_name!r}"
            )
        robot_config = load_franka_config(base_poses=None)

        # The World and ALL cuRobo operations must run with inference mode OFF. The Isaac AutoData
        # env_loop runs the whole generation under torch.inference_mode(); if the World is built in
        # that context, cuRobo allocates "inference tensors" (e.g. kinematics link_spheres) that can
        # neither be backward()'d (IK) nor updated in-place outside inference mode (grasp sphere
        # context) -- producing "Inplace update to inference tensor outside InferenceMode" errors.
        # Building everything under _autograd_enabled() makes them normal tensors throughout.
        #
        # World creation is profiled separately from solving (see plan()): cProfile when
        # self.profile is set, else a no-op.
        from schedulestream.common.utils import profiler

        if self.profile:
            print(f"{'=' * 30} PROFILE: world creation {'=' * 30}")
        with profiler(field="cumtime" if self.profile else None, num=25), _autograd_enabled():
            world = World(robot_config, objects)

            state_dict = self.scene.state
            positions = state_dict["articulation"][self.robot]["joint_position"][env_id]
            # Reorder env joint positions into the world's joint ordering by name.
            env_joint_names = list(self.articulation.joint_names)
            ordered = _reorder_positions(env_joint_names, positions, world.all_joints)
            world.set_joint_positions(world.all_joints, ordered)
            world.set_camera_pose(CAMERA_POSE)

            # Warm up the IK / motion-planning solvers, which CAPTURES cuRobo's CUDA graphs. Without
            # this, the first solve_tamp replays a never-captured graph and crashes in seed_ik_solver
            # ('NoneType' has no attribute 'replay'). custream2's example.py and the upstream isaaclab
            # planner both warm up before solving (the latter via its now-removed v1 initialize()).
            world.warmup()

        # Diagnostic (outside _autograd_enabled — just toggles collision-active state, no backward):
        # every movable object must be registered in the cuRobo scene_collision_checker, else grasp
        # sampling (grasp.py active_context -> set_object_active) raises KeyError. Toggling active
        # forces the strict container walk (is_object_active defaults True, so a no-op set wouldn't
        # probe membership). Restores state and never raises.
        missing = []
        for name in world.movable_names:
            try:
                world.set_object_active(name, False)
                world.set_object_active(name, True)
            except KeyError:
                missing.append(name)
        if missing:
            print(
                f"[schedulestream] WARNING: movable objects missing from scene_collision_checker: "
                f"{missing} | movable_names={world.movable_names} | fixed_names={world.fixed_names}"
            )
        return world

    # ------------------------------------------------------------------
    # Goal + state sync
    # ------------------------------------------------------------------

    def create_goal(self, success_term: Any) -> Any:
        """Derive the symbolic ScheduleStream goal from the task success term (cube stacking)."""
        from schedulestream.applications.custream2.example import Attached

        if success_term is None:
            # Fall back to stacking the two highest-indexed movable objects.
            self.world.dump()
            obj1, obj2 = sorted(self.world.movable_names, reverse=True)[:2]
            return Attached(obj1) == obj2

        from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked

        if success_term.func == cubes_stacked:
            cubes: list[str | None] = [f"cube_{i}" for i in range(1, 3 + 1)]
            for i, cube in enumerate(cubes):
                cube_cfg = f"{cube}_cfg"
                if cube_cfg not in success_term.params:
                    continue
                if success_term.params[cube_cfg] is None:
                    cubes[i] = None
                else:
                    cubes[i] = success_term.params[cube_cfg].name
            cube1, cube2, cube3 = cubes
            goal = Attached(cube2) == cube1
            if cube3 is not None:
                goal = goal & (Attached(cube3) == cube2)
            return goal
        raise NotImplementedError(f"schedulestream goal not implemented for {success_term.func}")

    def set_world_state(self, env_id: int) -> None:
        """Sync the cuStream2 world (joints + object poses) from the live env scene state."""
        state = self.scene.state
        positions = state["articulation"][self.robot]["joint_position"][env_id]
        ordered = _reorder_positions(list(self.articulation.joint_names), positions, self.joints)
        self.world.set_joint_positions(self.joints, ordered)
        for name in self.scene.rigid_objects:
            self.world.set_object_pose(name, self.pose(name))

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _animate(self, state: Any, commands: Any, **kwargs: Any) -> Any:
        """Animate ``commands`` (``None`` shows the initial state) when ``self.animate`` is set.

        Returns ``animate_commands``'s output (frames), or ``None`` when animation is disabled.
        Mirrors the isaaclab planner.
        """
        if not self.animate:
            return None
        from schedulestream.applications.custream2.example import animate_commands

        return animate_commands(state, commands, **kwargs)

    def _eef_action_info(self) -> tuple[str, Any]:
        """Return ``(ik_body_name, body_offset_pose_or_None)`` for the EEF's IK action term."""
        from curobo.types import Pose
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        action_manager = getattr(self.base_env, "action_manager", None)
        assert action_manager is not None, "env has no action_manager"
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            if isinstance(term, DifferentialInverseKinematicsAction):
                offset = None
                if term.cfg.body_offset is not None:
                    offset = Pose(
                        position=self.world.to_device(list(term.cfg.body_offset.pos)),
                        quaternion=self.world.to_device(list(term.cfg.body_offset.rot)),
                    )
                return term.cfg.body_name, offset
        raise NotImplementedError("schedulestream requires a DifferentialInverseKinematicsAction (IK env)")


    def _raw_eef_pose(self) -> Any:
        """World-frame EEF target as cuRobo sees it: ``from_reference(node_pose(body)) * body_offset``.

        Uses the cached IK control link/offset (``self.body_name``/``self.body_offset``). This is in
        cuRobo's ``panda_hand`` frame convention; :meth:`_eef_frame_correction` maps it into the
        env's ee_frame convention.
        """
        import torch

        link_pose = self.from_reference(self.world.get_node_pose(self.body_name))
        if self.body_offset is not None:
            link_pose = link_pose.multiply(self.body_offset)
        return link_pose.get_matrix().squeeze(0).to(device=self.datastream.device, dtype=torch.float32)

    def _eef_frame_correction(self, env_id: int) -> Any:
        """Constant body-frame transform mapping cuRobo's EEF frame to the env's ee_frame.

        Measured once with the world at the current config: ``inv(raw_eef) @ obs_eef``.

        Root cause (IsaacLab-version asset difference, NOT a quaternion-convention bug here):
        cuStream2 builds the World from its own Franka URDF (``custream2.franka.load_franka_config``),
        whose ``panda_hand`` link frame is defined ~180 deg about z relative to the ``panda_hand``
        frame in the Franka USD shipped with *this* IsaacLab version. The env's ee_frame observation
        (FrameTransformer on ``panda_hand`` + pos [0,0,0.1034], identity rot) therefore differs from
        cuRobo's ``panda_hand``-derived target by that constant 180-deg-z link-frame redefinition
        (plus the 0.107-vs-0.1034 z offset). The standalone ``applications/isaaclab`` agent does not
        need this correction only because it runs against an IsaacLab whose Franka USD ``panda_hand``
        happens to match custream2's URDF.

        Because the mismatch is a constant BODY-frame (link-local) transform, ``inv(raw0) @ obs0``
        measured at one config reproduces the true EEF pose at EVERY config (obs(q) = raw(q) @ C), so
        this is correct for full trajectories, not just a static hold. (The base ``reference_pose``
        is computed correctly: ``_pose`` converts the xyzw warp ``root_pose_w`` to wxyz for cuRobo, so
        there is no world-frame error that a body-frame correction would fail to cancel.) The
        ``[schedulestream] EEF frame check`` print validates the result reads ~0 after correction.
        """
        import torch

        eef_name = self.datastream.get_eef_names()[0]
        obs_eef = self.datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].to(
            device=self.datastream.device, dtype=torch.float32
        )
        raw_eef = self._raw_eef_pose()
        return torch.linalg.inv(raw_eef) @ obs_eef

    def _commands_to_waypoints(self, state: Any, commands: Any, env_id: int = 0) -> list[Waypoint]:
        """Convert a solved cuStream2 ``Commands`` sequence into per-step EEF-target waypoints.

        Mirrors ``isaaclab/controller.create_controller`` adapted to the cuStream2 execution model:
        each leaf command's ``execute()`` mutates the world one step at a time (``Trajectory`` sets
        arm joints per waypoint; ``Open``/``Close`` drive the gripper; ``Attach``/``Detach`` toggle
        the carried object). After every step we read the IK body-link world pose — shifted by the
        action's body offset into the EEF control frame the embodiment adapter expects — and emit a
        :class:`Waypoint` carrying the current binary gripper command.
        """
        import torch

        from schedulestream.applications.custream2.command import Attach, Close, Detach, Open

        from isaac_autodata_core.waypoint import Waypoint

        device = self.datastream.device

        # Establish the solved initial placements on the world before stepping.
        state.set()

        # Calibrate a constant frame correction. cuRobo's body link (panda_hand) differs from the
        # env's ee_frame observation by a fixed body-frame transform (measured ~180 deg about z for
        # the Franka, plus the 0.107-vs-0.1034 z offset). The world is at the current config here,
        # so `correction = inv(raw_eef) @ obs_eef` makes raw_eef @ correction == obs_eef, and since
        # the mismatch is constant it holds for every waypoint. Pure matrix math (no axis-angle
        # noise), and self-correcting if the robot model changes.
        correction = self._eef_frame_correction(env_id)

        gripper_value = OPEN_ACTION
        waypoints: list[Waypoint] = []
        for command in _leaf_commands(commands):
            # Open/Close are the physical gripper; Attach/Detach are the symbolic grasp toggles.
            # Either signals the binary gripper command for the steps that follow.
            if isinstance(command, (Close, Attach)):
                gripper_value = CLOSE_ACTION
            elif isinstance(command, (Open, Detach)):
                gripper_value = OPEN_ACTION

            for _ in command.execute():
                pose_matrix = self._raw_eef_pose() @ correction
                gripper_action = torch.tensor([gripper_value], dtype=torch.float32, device=device)
                # noise must be a float (not the Waypoint default None): MultiWaypoint.execute
                # passes it straight into the embodiment adapter, which does `noise > 0.0`. We
                # replay a planned trajectory, so no action noise.
                waypoints.append(Waypoint(pose=pose_matrix, gripper_action=gripper_action, noise=0.0))

        return waypoints

    def hold_plan(self, env_id: int, steps: int) -> list[Waypoint]:
        """Null plan: hold the current configuration for ``steps`` steps.

        Mirrors the isaaclab planner's ``hold`` — builds ``Commands`` of ``steps`` copies of the
        current :class:`Configuration` and converts them to waypoints. Useful for exercising the
        execution/recording pipeline without running TAMP.
        """
        from schedulestream.applications.custream2.command import Commands

        # No autograd context needed: this neither builds the World nor runs solve_tamp; it only
        # replays a held configuration on the (already-normal) world tensors.
        self.set_world_state(env_id)
        state = self.world.state()
        command = self.world.configuration()
        commands = Commands(self.world, steps * [command])
        self._animate(state, commands)
        return self._commands_to_waypoints(state, commands, env_id=env_id)

    def plan(self, env_id: int) -> list[Waypoint]:
        """Sync state, solve TAMP for the goal, and convert commands to executable waypoints.

        When ``self.hold`` is not ``None``, skip TAMP and return a null plan holding the current
        configuration for that many steps.

        Otherwise raises on failure (no plan / solver error) rather than returning ``None`` — see
        the note in ``ScheduleStream.plan_subtask_trajectory`` on why ``None`` livelocks the
        generation loop. The original solver exception is preserved so the true cause is visible.
        """
        if self.hold is not None:
            return self.hold_plan(env_id, self.hold)

        from schedulestream.applications.custream2.example import solve_tamp
        from schedulestream.common.utils import timeout_context

        self.set_world_state(env_id)
        state = self.world.state()
        try:
            if self.profile:
                print(f"{'=' * 30} PROFILE: solve_tamp {'=' * 30}")
            # Only solve_tamp needs autograd (it runs cuRobo IK/trajopt backward); the surrounding
            # state sync, animation, and command replay operate on the World's already-normal
            # tensors. See _create_world for why the World must be built with inference mode off.
            with _autograd_enabled(), timeout_context(timeout=2 * self.max_time):
                # solve_tamp wraps its own work in profiler(...) when profile=True.
                commands = solve_tamp(
                    state,
                    self.goal,
                    collisions=self.collisions,
                    max_time=self.max_time,
                    profile=self.profile,
                )
        except Exception:
            # Show the initial state, then re-raise the real solver error (grasp/IK/etc.) so the
            # run terminates with the true traceback instead of livelocking on a swallowed None.
            self._animate(state, None)
            raise

        # Animate the plan before executing (commands=None shows the initial state).
        self._animate(state, commands)
        if commands is None:
            raise RuntimeError(f"solve_tamp found no plan for env {env_id} (returned None).")
        return self._commands_to_waypoints(state, commands, env_id=env_id)
