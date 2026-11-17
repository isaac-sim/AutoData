# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerun views of planned paths, collision spheres and scene geometry."""

from __future__ import annotations

import atexit
import numpy as np
import os
import signal
import threading
import time
import torch
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Check if rerun is installed
try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    raise ImportError("Rerun is not installed!")

_RR_HAS_TRANSFORM_AXES = hasattr(rr, "TransformAxes3D")

import isaaclab.utils.math as PoseUtils
from curobo.types.state import JointState

# Import psutil for process management
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Process monitoring will be limited.")

if TYPE_CHECKING:  # For type hints only
    import trimesh


# Global registry to track all PlanVisualizer instances for cleanup
_GLOBAL_PLAN_VISUALIZERS: list[PlanVisualizer] = []


def _kill_owned_rerun_processes(owner_pid: int) -> int:
    """Kill Rerun viewer processes that are descendants of ``owner_pid``.

    Matching is restricted to the owner's process tree so that Rerun sessions
    belonging to other processes on the machine are never touched.

    Args:
        owner_pid: PID whose descendant Rerun processes are killed.

    Returns:
        Number of processes killed.
    """
    if not PSUTIL_AVAILABLE:
        return 0
    try:
        descendants = psutil.Process(owner_pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    killed_count = 0
    for proc in descendants:
        try:
            if "rerun" in proc.name().lower() or any("rerun" in str(arg).lower() for arg in proc.cmdline()):
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed_count


def _cleanup_all_plan_visualizers() -> None:
    """Close all registered visualizers and kill Rerun viewers spawned by this process."""
    # Close instances first so recordings are flushed and tracked viewer handles terminate.
    for visualizer in _GLOBAL_PLAN_VISUALIZERS[:]:
        if not visualizer._closed:
            visualizer.close()
    _GLOBAL_PLAN_VISUALIZERS.clear()

    # Kill any stray viewers spawned by this process that outlived their handle.
    killed_count = _kill_owned_rerun_processes(os.getpid())
    if killed_count:
        print(f"Killed {killed_count} leftover Rerun viewer processes on script exit")


# Register global cleanup on module import
atexit.register(_cleanup_all_plan_visualizers)


class PlanVisualizer:
    """Visualizes motion plans using Rerun.

    This class provides methods to visualize:
    1. Robot poses along a planned trajectory
    2. Attached objects and their collision spheres
    3. Robot collision spheres
    4. Target poses and waypoints
    """

    def __init__(
        self,
        robot_name: str = "panda",
        recording_id: str | None = None,
        debug: bool = False,
        save_path: str | None = None,
        base_pose_in_env: np.ndarray | None = None,
    ):
        """Initialize the plan visualizer.

        Args:
            robot_name: Name of the robot for visualization
            recording_id: Optional ID for the Rerun recording
            debug: Whether to print debug information
            save_path: Optional path to save the recording
            base_pose_in_env: Planner-base pose in the environment frame.
        """
        self.robot_name = robot_name
        self.debug = debug
        self.recording_id = recording_id or f"motion_plan_{robot_name}"
        self.save_path = save_path
        self._closed = False
        self.set_base_pose_in_env(np.eye(4) if base_pose_in_env is None else base_pose_in_env)

        # Enhanced process management
        self._parent_pid = os.getpid()
        self._monitor_thread = None
        self._monitor_active = False

        # Motion generator reference for sphere animation (set by CuroboPlanner)
        self._motion_gen_ref = None

        # Register this instance globally for cleanup
        _GLOBAL_PLAN_VISUALIZERS.append(self)

        # Initialize Rerun
        rr.init(self.recording_id, spawn=False)

        self._rerun_process = None
        if self.save_path is not None:
            output_path = Path(self.save_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path = str(output_path)
            rr.save(self.save_path)
            self._rerun_process = None
        else:
            # Spawn viewer and keep handle if provided so we can terminate it later.
            try:
                self._rerun_process = rr.spawn()
            except Exception as exc:
                raise RuntimeError("Failed to spawn Rerun; check that its executable is on PATH.") from exc

        # Set up coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
        self._send_blueprint("planning")

        # Store visualization state
        self._current_frame = 0
        self._plan_frame = 0
        self._active_plan_root = "world/plans"
        self._world_scene = None
        self._sphere_entities: dict[str, list[str]] = {"robot": [], "attached": [], "target": []}

        # Close orphaned viewers when their parent process exits.
        self._start_parent_process_monitoring()

        # Use weakref.finalize for cleanup when object is garbage collected
        self._finalizer = weakref.finalize(self, self._cleanup_class_resources, self.recording_id, debug)

        # Also register atexit handler as backup for normal script termination
        # Store values locally to avoid capturing self in the closure
        recording_id = self.recording_id
        debug_flag = debug
        atexit.register(self._cleanup_class_resources, recording_id, debug_flag)

        # Store original signal handlers so we can restore them after cleanup
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        # Handle Ctrl+C (SIGINT) and termination (SIGTERM) signals
        def signal_handler(signum, frame):
            if self.debug:
                print(f"Received signal {signum}, closing Rerun viewer...")
            self._cleanup_on_exit()

            # Restore original signal handler and re-raise the signal
            if signum == signal.SIGINT:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            elif signum == signal.SIGTERM:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.debug:
            print(f"Initialized Rerun visualization with recording ID: {self.recording_id}")
            if not np.allclose(self._base_pose_in_env, np.eye(4)):
                print(f"Applying planner-base pose:\n{self._base_pose_in_env}")
            if PSUTIL_AVAILABLE:
                print("Enhanced process monitoring enabled")

    def set_base_pose_in_env(self, base_pose_in_env: np.ndarray | torch.Tensor) -> None:
        """Set the planner-base pose in the environment frame.

        Args:
            base_pose_in_env: Homogeneous 4x4 transform from planner base to environment.
        """

        if torch.is_tensor(base_pose_in_env):
            base_pose_in_env = base_pose_in_env.detach().cpu().numpy()
        pose = np.asarray(base_pose_in_env, dtype=float)
        assert pose.shape == (4, 4), f"base_pose_in_env must have shape (4, 4), got {pose.shape}"
        assert np.isfinite(pose).all(), "base_pose_in_env must contain finite values"
        assert np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0]), "base_pose_in_env must be homogeneous"
        rotation = pose[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5), "base rotation must be orthonormal"
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5), "base rotation must be right-handed"
        self._base_pose_in_env = pose.copy()

    def world_scene_set(self, world_scene: trimesh.Scene | None) -> None:
        """Replace the world scene used by subsequent plan and schedule timelines."""

        self._world_scene = world_scene

    def _points_base_to_env(self, points: np.ndarray) -> np.ndarray:
        """Transform planner-base points into the environment frame."""

        points = np.asarray(points, dtype=float)
        assert points.shape[-1] == 3, f"points must end in dimension 3, got {points.shape}"
        return points @ self._base_pose_in_env[:3, :3].T + self._base_pose_in_env[:3, 3]

    def _send_blueprint(self, timeline: str | None = None) -> None:
        """Separate arm plans from complete-robot checks using the same scene camera."""

        base_position = self._base_pose_in_env[:3, 3]
        look_target = base_position + np.array([0.0, 0.35, 0.2])
        camera_position = look_target + np.array([1.0, 0.85, 0.6])
        views = [
            rrb.Spatial3DView(
                name=name,
                origin="world",
                contents=["world/scene/**", f"world/{root}/**"],
                eye_controls=rrb.EyeControls3D(
                    position=camera_position,
                    look_target=look_target,
                ),
            )
            for name, root in (("Arm plans", "plans"), ("Scheduled robot", "schedules"))
        ]
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Tabs(*views, active_tab=0),
                rrb.TimePanel(
                    state="expanded",
                    timeline=timeline,
                    fps=20.0,
                    playback_speed=1.0,
                    play_state="playing",
                    loop_mode="all",
                ),
            )
        )

    def _start_parent_process_monitoring(self) -> None:
        """Start monitoring the parent process and cleanup when it dies."""
        if not PSUTIL_AVAILABLE:
            if self.debug:
                print("psutil not available, skipping parent process monitoring")
            return

        self._monitor_active = True

        def monitor_parent_process() -> None:
            """Monitor thread function that watches the parent process."""
            if self.debug:
                print(f"Starting parent process monitor for PID {self._parent_pid}")

            # Get parent process handle
            parent_process = psutil.Process(self._parent_pid)

            # Monitor parent process
            while self._monitor_active:
                try:
                    if not parent_process.is_running():
                        if self.debug:
                            print(f"Parent process {self._parent_pid} died, cleaning up Rerun...")
                        self._kill_rerun_processes()
                        break

                    # Check every 2 seconds
                    time.sleep(2)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    if self.debug:
                        print(f"Parent process {self._parent_pid} no longer accessible, cleaning up...")
                    self._kill_rerun_processes()
                    break
                except Exception as e:
                    if self.debug:
                        print(f"Error in parent process monitor: {e}")
                    break

        # Start monitor thread
        self._monitor_thread = threading.Thread(target=monitor_parent_process, daemon=True)
        self._monitor_thread.start()

    def _kill_rerun_processes(self) -> None:
        """Kill Rerun viewer processes spawned by the owning process."""
        killed_count = _kill_owned_rerun_processes(self._parent_pid)
        if self.debug and killed_count:
            print(f"Killed {killed_count} Rerun viewer processes owned by PID {self._parent_pid}")

    @staticmethod
    def _cleanup_class_resources(recording_id: str, debug: bool) -> None:
        """Disconnect Rerun and stop viewers owned by this process.

        Args:
            recording_id: Recording identifier included in debug messages.
            debug: Whether to print cleanup progress.
        """
        if debug:
            print(f"Cleaning up Rerun visualization for {recording_id}")

        # Disconnect from Rerun
        rr.disconnect()

        # Kill only viewers spawned by this process; other Rerun sessions are left alone.
        killed_count = _kill_owned_rerun_processes(os.getpid())
        if debug and killed_count:
            print(f"Killed {killed_count} Rerun processes during cleanup")

        if debug:
            print("Cleanup completed")

    def _cleanup_on_exit(self) -> None:
        """Manual cleanup method for signal handlers."""
        if not self._closed:
            # Stop monitoring thread
            self._monitor_active = False

            self.close()
            self._kill_rerun_processes()

    def close(self) -> None:
        """Disconnect the recording and release its owned viewer resources."""
        if self._closed:
            return

        # Stop parent process monitoring
        self._monitor_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            # Give the thread a moment to stop gracefully
            time.sleep(0.1)

        # Disconnect from Rerun (closes connections, servers, and files)
        rr.disconnect()

        self._closed = True

        # Terminate viewer process if we have a handle
        try:
            process = getattr(self, "_rerun_process", None)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        except Exception:
            pass

        # Remove from global registry
        if self in _GLOBAL_PLAN_VISUALIZERS:
            _GLOBAL_PLAN_VISUALIZERS.remove(self)

        # Sweep for strays only when no other live instance remains, so closing one
        # visualizer cannot kill a sibling instance's viewer in the same process.
        if not any(not visualizer._closed for visualizer in _GLOBAL_PLAN_VISUALIZERS):
            self._kill_rerun_processes()

        if self.debug:
            print("Closed Rerun visualization with enhanced cleanup")

    def visualize_plan(
        self,
        plan: JointState,
        target_pose: torch.Tensor,
        robot_spheres: list[Any] | None = None,
        attached_spheres: list[Any] | None = None,
        ee_positions: np.ndarray | None = None,
        world_scene: trimesh.Scene | None = None,
    ) -> int:
        """Visualize a complete motion plan with all components.

        Args:
            plan: Joint state trajectory to visualize
            target_pose: Target end-effector pose
            robot_spheres: Optional list of robot collision spheres
            attached_spheres: Optional list of attached object spheres
            ee_positions: Optional end-effector positions
            world_scene: Optional world scene to visualize

        Returns:
            Monotonic plan index used to name the plan's entities.
        """
        if self.debug:
            robot_count = len(robot_spheres) if robot_spheres else 0
            attached_count = len(attached_spheres) if attached_spheres else 0
            print(
                f"Visualizing plan: {len(plan.position)} waypoints, {robot_count} robot spheres,"
                f" {attached_count} attached spheres"
            )

        plan_index = self._current_frame
        self._clear_visualization()
        self._active_plan_root = f"world/plans/{plan_index:04d}_{self.robot_name}"
        rr.reset_time()
        rr.set_time("planning", sequence=self._plan_frame)
        self._current_frame += 1

        # If a scene is supplied and not yet logged, draw it once
        if world_scene is not None:
            self._world_scene = world_scene
            self._visualize_world_scene(world_scene)

        # Visualize target pose
        self._visualize_target_pose(target_pose)

        # Visualize trajectory (end-effector positions if provided)
        self._visualize_trajectory(ee_positions)

        # Visualize spheres if provided
        if robot_spheres:
            self._visualize_robot_spheres(robot_spheres)
        if attached_spheres:
            self._visualize_attached_spheres(attached_spheres)
        else:
            # Clear any existing attached sphere visualization when no objects are attached
            self._clear_attached_spheres()
        rr.reset_time()
        return plan_index

    def visualize_failure(
        self,
        target_pose: torch.Tensor,
        robot_spheres: list[Any],
        attached_spheres: list[Any],
        world_scene: trimesh.Scene | None = None,
    ) -> None:
        """Retain the target and start-state spheres for a failed planning attempt.

        Args:
            target_pose: Failed end-effector target pose.
            robot_spheres: Robot collision spheres at the planning start state.
            attached_spheres: Attached-object spheres at the planning start state.
            world_scene: Optional cuRobo collision scene.
        """

        plan_index = self._current_frame
        self._clear_visualization()
        self._active_plan_root = f"world/plans/{plan_index:04d}_{self.robot_name}_failed"
        self._current_frame += 1
        rr.reset_time()
        rr.set_time("planning", sequence=self._plan_frame)
        if world_scene is not None:
            self._world_scene = world_scene
            self._visualize_world_scene(world_scene)
        self._visualize_target_pose(target_pose)
        self._visualize_robot_spheres(robot_spheres)
        if attached_spheres:
            self._visualize_attached_spheres(attached_spheres)
        self._plan_frame += 1
        rr.reset_time()

    def _clear_visualization(self) -> None:
        """End the current plan's visibility without deleting earlier playback frames."""
        rr.reset_time()
        rr.set_time("planning", sequence=self._plan_frame)
        rr.log(self._active_plan_root, rr.Clear(recursive=True))
        self._sphere_entities = {"robot": [], "attached": [], "target": []}
        rr.reset_time()

    def clear_visualization(self) -> None:
        """Public method to clear the visualization."""

        self._clear_visualization()

    def _visualize_target_pose(self, target_pose: torch.Tensor) -> None:
        """Visualize the target end-effector pose.

        Args:
            target_pose: Target pose as 4x4 transformation matrix
        """
        pos, rot = PoseUtils.unmake_pose(target_pose)

        # Convert to numpy arrays
        pos_np = pos.detach().cpu().numpy() if torch.is_tensor(pos) else np.array(pos)
        rot_np = rot.detach().cpu().numpy() if torch.is_tensor(rot) else np.array(rot)

        # Ensure arrays are the right shape
        pos_np = pos_np.reshape(-1)
        rot_np = rot_np.reshape(3, 3)

        # Log target position
        rr.log(
            f"{self._active_plan_root}/target/position",
            rr.Points3D(
                positions=np.array([pos_np]),
                colors=[[255, 0, 0]],  # Red
                radii=[0.02],
            ),
        )

        # Log target orientation as coordinate frame
        rr.log(
            f"{self._active_plan_root}/target/frame",
            rr.Transform3D(
                translation=pos_np,
                mat3x3=rot_np,
            ),
        )

    def _visualize_trajectory(
        self,
        ee_positions: np.ndarray | None = None,
    ) -> None:
        """Visualize the robot trajectory.

        Args:
            ee_positions: End-effector positions in the planner base frame [m].
        """
        if ee_positions is None:
            return
        positions = self._points_base_to_env(ee_positions)

        # Log trajectory points
        rr.log(
            f"{self._active_plan_root}/trajectory",
            rr.LineStrips3D(
                [positions],  # single strip consisting of all waypoints
                colors=[[0, 100, 255]],  # Blue
                radii=[0.005],
            ),
        )

        # Log keyframes
        for i, pos in enumerate(positions):
            rr.log(
                f"{self._active_plan_root}/trajectory/keyframe_{i}",
                rr.Points3D(
                    positions=np.array([pos]),
                    colors=[[0, 100, 255]],  # Blue
                    radii=[0.01],
                ),
            )

    def _visualize_robot_spheres(self, spheres: list[Any]) -> None:
        """Visualize robot collision spheres.

        Args:
            spheres: List of robot collision spheres
        """
        self._log_spheres(
            spheres=spheres,
            entity_type="robot",
            color=[0, 255, 100, 128],  # Semi-transparent green
        )

    def _visualize_attached_spheres(self, spheres: list[Any]) -> None:
        """Visualize attached object collision spheres.

        Args:
            spheres: List of attached object spheres
        """
        self._log_spheres(
            spheres=spheres,
            entity_type="attached",
            color=[255, 0, 0, 128],  # Semi-transparent red
        )

    def _clear_attached_spheres(self) -> None:
        """Clear all attached object spheres."""
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"{self._active_plan_root}/attached/{entity_id}", rr.Clear(recursive=True))
        self._sphere_entities["attached"] = []

    # ---------------------------------------------------------------------
    # PRIVATE UTILITIES
    # ---------------------------------------------------------------------

    def _log_spheres(
        self,
        spheres: list[Any],
        entity_type: str,
        color: list[int],
    ) -> None:
        """Generic helper for sphere visualization.

        Args:
            spheres: List of CuRobo ``Sphere`` objects.
            entity_type: Log path prefix (``robot`` or ``attached``).
            color: RGBA color for the spheres.
        """
        for i, sphere in enumerate(spheres):
            entity_id = f"sphere_{i}"
            # Track entities so we can clear them later
            self._sphere_entities.setdefault(entity_type, []).append(entity_id)

            # cuRobo sphere positions are expressed in the planner base frame.
            pos = (
                sphere.position.detach().cpu().numpy()
                if torch.is_tensor(sphere.position)
                else np.array(sphere.position)
            )
            pos = self._points_base_to_env(pos.reshape(1, 3))[0]

            # Log sphere via Rerun
            rr.log(
                f"{self._active_plan_root}/{entity_type}/{entity_id}",
                rr.Points3D(positions=np.array([pos]), colors=[color], radii=[float(sphere.radius)]),
            )

    def _visualize_world_scene(self, scene: trimesh.Scene) -> None:
        """Log world geometry and dynamic transforms each call.

        Geometry is sent once (cached), but transforms are updated every invocation
        so objects that move (cubes after randomization) appear at the correct
        pose per episode/frame.
        """
        import trimesh  # local import to avoid hard dependency at top

        def _to_rerun_mesh(mesh: trimesh.Trimesh) -> rr.Mesh3D:
            """Convert triangle geometry to a Rerun mesh without material properties."""
            return rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
            )

        if not hasattr(self, "_logged_geometry"):
            self._logged_geometry = set()

        for node in scene.graph.nodes_geometry:
            tform, geom_key = scene.graph.get(node)
            mesh = scene.geometry.get(geom_key)
            if mesh is None:
                continue
            rr_path = f"world/scene/{node.replace('/', '_')}"

            # cuRobo's collision scene is expressed in the planner base frame.
            tform_env = self._base_pose_in_env @ tform
            if _RR_HAS_TRANSFORM_AXES:
                # rerun >= 0.28: axis_length moved to TransformAxes3D
                rr.log(
                    rr_path,
                    rr.Transform3D(translation=tform_env[:3, 3], mat3x3=tform_env[:3, :3]),
                    static=False,
                )
                rr.log(rr_path, rr.TransformAxes3D(axis_length=0.25), static=False)
            else:
                # rerun <= 0.27: axis_length on Transform3D directly
                rr.log(
                    rr_path,
                    rr.Transform3D(
                        translation=tform_env[:3, 3],
                        mat3x3=tform_env[:3, :3],
                        axis_length=0.25,
                    ),
                    static=False,
                )

            # Geometry: send only once per node to avoid duplicates
            if rr_path not in self._logged_geometry:
                if isinstance(mesh, trimesh.Trimesh):
                    rr_mesh = _to_rerun_mesh(mesh)
                elif isinstance(mesh, trimesh.PointCloud):
                    rr_mesh = rr.Points3D(positions=mesh.vertices, colors=mesh.colors)
                else:
                    continue

                rr.log(rr_path, rr_mesh, static=True)
                self._logged_geometry.add(rr_path)

        if self.debug:
            print(f"Logged/updated {len(scene.graph.nodes_geometry)} world nodes in Rerun")

    def animate_plan(
        self,
        ee_positions: np.ndarray,
        object_positions: dict[str, np.ndarray] | None = None,
        timeline: str = "planning",
        point_radius: float = 0.01,
    ) -> None:
        """Animate robot end-effector and (optionally) attached object positions over time using Rerun.

        This helper logs a single 3-D point per timestep so that Rerun can play back the
        trajectory on the provided ``timeline``.  It is intentionally lightweight and does
        not attempt to render the full robot geometry—only key points—keeping the data
        transfer to the viewer minimal.

        Args:
            ee_positions: Array of shape (T, 3) with planner-base end-effector positions.
            object_positions: Mapping from object name to an array (T, 3) with that
                object's environment-frame positions. Each trajectory must be at least as long as
                ``ee_positions``; extra entries are ignored.
            timeline: Name of the Rerun timeline used for the animation frames.
            point_radius: Visual radius (in metres) of the rendered points.
        """
        if ee_positions is None or len(ee_positions) == 0:
            return

        rr.reset_time()
        rr.set_time(timeline, sequence=self._plan_frame)
        if self._world_scene is not None:
            self._visualize_world_scene(self._world_scene)
        # Iterate over timesteps and log a frame on the chosen timeline
        for idx, pos in enumerate(ee_positions):
            # Set time on the chosen timeline (creates it if needed)
            rr.set_time(timeline, sequence=self._plan_frame + idx)

            # End-effector positions from cuRobo FK use the planner base frame.
            rr.log(
                f"{self._active_plan_root}/anim/ee",
                rr.Points3D(
                    positions=self._points_base_to_env(np.asarray(pos).reshape(1, 3)),
                    colors=[[0, 100, 255]],  # Blue
                    radii=[point_radius],
                ),
            )

            # Optionally log attached object markers
            # NOTE: Object positions are already in world coordinates, no offset needed
            if object_positions is not None:
                for name, traj in object_positions.items():
                    if idx >= len(traj):
                        continue
                    rr.log(
                        f"{self._active_plan_root}/anim/{name}",
                        rr.Points3D(
                            positions=np.array([traj[idx]]),
                            colors=[[255, 128, 0]],  # Orange
                            radii=[point_radius],
                        ),
                    )
        rr.reset_time()

    def animate_collision_spheres(
        self,
        spheres: torch.Tensor,
        *,
        timeline_prefix: str = "full_state_schedule",
    ) -> int:
        """Animate a complete scheduled robot sphere tensor.

        Args:
            spheres: Planner-root sphere states shaped ``(T, S, 4)`` with radius last [m].
            timeline_prefix: Prefix for the isolated Rerun sequence timeline.

        Returns:
            Monotonic visualization index used in the timeline and entity path.
        """

        assert spheres.ndim == 3 and spheres.shape[-1] == 4 and len(spheres) > 0
        values = spheres.detach().cpu().numpy()
        index = self._current_frame
        self._current_frame += 1
        timeline = f"{timeline_prefix}_{index}"
        entity_path = f"world/schedules/{index:04d}/robot_spheres"

        rr.reset_time()
        rr.set_time(timeline, sequence=0)
        if self._world_scene is not None:
            self._visualize_world_scene(self._world_scene)
        for tick, frame in enumerate(values):
            valid = np.isfinite(frame).all(axis=1) & (frame[:, 3] > 0.0)
            rr.set_time(timeline, sequence=tick)
            rr.log(
                entity_path,
                rr.Points3D(
                    positions=self._points_base_to_env(frame[valid, :3]),
                    radii=frame[valid, 3],
                    colors=[0, 255, 100, 180],
                ),
            )
        rr.reset_time()
        return index

    def animate_object_collision_spheres(self, spheres: torch.Tensor, *, schedule_index: int) -> None:
        """Overlay environment-frame object spheres on a full-state schedule timeline.

        Args:
            spheres: Environment-frame sphere states shaped ``(T, S, 4)`` with radius last [m].
            schedule_index: Index returned by :meth:`animate_collision_spheres`.
        """

        assert spheres.ndim == 3 and spheres.shape[-1] == 4 and len(spheres) > 0
        assert type(schedule_index) is int and schedule_index >= 0
        values = spheres.detach().cpu().numpy()
        timeline = f"full_state_schedule_{schedule_index}"
        entity_path = f"world/schedules/{schedule_index:04d}/object_spheres"
        for tick, frame in enumerate(values):
            valid = np.isfinite(frame).all(axis=1) & (frame[:, 3] > 0.0)
            rr.set_time(timeline, sequence=tick)
            rr.log(
                entity_path,
                rr.Points3D(
                    positions=frame[valid, :3],
                    radii=frame[valid, 3],
                    colors=[255, 160, 0, 180],
                ),
            )
        rr.reset_time()

    def animate_spheres_along_path(
        self,
        plan: JointState,
        robot_spheres_at_start: list[Any] | None = None,
        attached_spheres_at_start: list[Any] | None = None,
        timeline: str = "planning",
        interpolation_steps: int = 10,
    ) -> None:
        """Animate robot and attached object spheres along the planned trajectory with smooth interpolation.

        This method creates a dense, interpolated trajectory and computes sphere positions
        at many intermediate points to create smooth animation of the robot moving along the path.

        Args:
            plan: Joint state trajectory to animate spheres along
            robot_spheres_at_start: Initial robot collision spheres (for reference)
            attached_spheres_at_start: Initial attached object spheres (for reference)
            timeline: Name of the Rerun timeline for the animation
            interpolation_steps: Number of interpolated steps between each waypoint pair
        """
        if plan is None or len(plan.position) == 0:
            if self.debug:
                print("No plan available for sphere animation")
            return

        if self.debug:
            robot_count = len(robot_spheres_at_start) if robot_spheres_at_start else 0
            attached_count = len(attached_spheres_at_start) if attached_spheres_at_start else 0
            print(f"Creating smooth animation for {robot_count} robot spheres and {attached_count} attached spheres")
            print(
                f"Original plan: {len(plan.position)} waypoints, interpolating with {interpolation_steps} steps between"
                " waypoints"
            )

        # We need access to the motion generator to compute spheres at each waypoint
        if not hasattr(self, "_motion_gen_ref") or self._motion_gen_ref is None:
            if self.debug:
                print("Motion generator reference not available for sphere animation")
            return

        motion_gen = self._motion_gen_ref

        # Validate motion generator has required attributes
        if not hasattr(motion_gen, "kinematics") or motion_gen.kinematics is None:
            if self.debug:
                print("Motion generator kinematics not available for sphere animation")
            return

        # Count robot link spheres (excluding attached objects) for consistent splitting
        robot_link_count = 0
        if robot_spheres_at_start:
            robot_link_count = len(robot_spheres_at_start)

        # Create interpolated trajectory for smooth animation
        interpolated_positions = self._create_interpolated_trajectory(plan, interpolation_steps)

        if self.debug:
            print(f"Created interpolated trajectory with {len(interpolated_positions)} total frames")

        rr.reset_time()
        rr.set_time(timeline, sequence=self._plan_frame)
        self._hide_static_spheres_for_animation()
        if self._world_scene is not None:
            self._visualize_world_scene(self._world_scene)
        # Animate spheres along the interpolated trajectory
        for frame_idx, joint_positions in enumerate(interpolated_positions):
            # Set time on the animation timeline with consistent timing
            rr.set_time(timeline, sequence=self._plan_frame + frame_idx)

            # Create joint state for this interpolated position
            if isinstance(joint_positions, torch.Tensor):
                sphere_position = joint_positions
            else:
                sphere_position = torch.tensor(joint_positions)

            # Ensure tensor is on the right device for CuRobo
            if hasattr(motion_gen, "tensor_args") and motion_gen.tensor_args is not None:
                sphere_position = motion_gen.tensor_args.to_device(sphere_position)

            # Get spheres at this configuration
            try:
                sphere_list = motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]
            except Exception as e:
                if self.debug:
                    print(f"Failed to compute spheres for frame {frame_idx}: {e}")
                continue

            # Handle sphere_list as either a list or single Sphere object
            if hasattr(sphere_list, "__iter__") and not hasattr(sphere_list, "position"):
                sphere_items = list(sphere_list)
            else:
                sphere_items = [sphere_list]

            # Separate robot and attached object spheres with different colors
            robot_sphere_positions = []
            robot_sphere_radii = []
            attached_sphere_positions = []
            attached_sphere_radii = []

            for i, sphere in enumerate(sphere_items):
                # Convert position to numpy
                pos = (
                    sphere.position.detach().cpu().numpy()
                    if torch.is_tensor(sphere.position)
                    else np.array(sphere.position)
                )
                pos = pos.reshape(-1)
                radius = float(sphere.radius)

                if i < robot_link_count:
                    robot_sphere_positions.append(self._points_base_to_env(pos.reshape(1, 3))[0])
                    robot_sphere_radii.append(radius)
                else:
                    attached_sphere_positions.append(self._points_base_to_env(pos.reshape(1, 3))[0])
                    attached_sphere_radii.append(radius)

            # Log robot spheres with green color
            if robot_sphere_positions:
                rr.log(
                    f"{self._active_plan_root}/robot_animation",
                    rr.Points3D(
                        positions=np.array(robot_sphere_positions),
                        colors=[[0, 255, 100, 220]] * len(robot_sphere_positions),  # Bright green
                        radii=robot_sphere_radii,
                    ),
                )

            # Log attached object spheres with orange color (or clear if no attached objects)
            if attached_sphere_positions:
                rr.log(
                    f"{self._active_plan_root}/attached_animation",
                    rr.Points3D(
                        positions=np.array(attached_sphere_positions),
                        colors=[[255, 150, 0, 220]] * len(attached_sphere_positions),  # Bright orange
                        radii=attached_sphere_radii,
                    ),
                )
            else:
                # Clear attached object spheres when no objects are attached
                rr.log(f"{self._active_plan_root}/attached_animation", rr.Clear(recursive=True))

        self._plan_frame += len(interpolated_positions)
        rr.reset_time()

        if self.debug:
            print(
                f"Completed smooth sphere animation with {len(interpolated_positions)} frames on timeline '{timeline}'"
            )

    def _hide_static_spheres_for_animation(self) -> None:
        """Hide static sphere visualization during animation to reduce visual clutter."""
        # Clear static robot spheres
        for entity_id in self._sphere_entities.get("robot", []):
            rr.log(f"{self._active_plan_root}/robot/{entity_id}", rr.Clear(recursive=True))

        # Clear static attached spheres
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"{self._active_plan_root}/attached/{entity_id}", rr.Clear(recursive=True))

        if self.debug:
            print("Hidden static spheres for cleaner animation view")

    def _create_interpolated_trajectory(self, plan: JointState, interpolation_steps: int) -> list[torch.Tensor]:
        """Create a smooth interpolated trajectory between waypoints.

        Args:
            plan: Original joint state trajectory
            interpolation_steps: Number of interpolation steps between each waypoint pair

        Returns:
            List of interpolated joint positions
        """
        if len(plan.position) < 2:
            # If only one waypoint, just return it
            return [plan.position[0] if isinstance(plan.position[0], torch.Tensor) else torch.tensor(plan.position[0])]  # type: ignore

        interpolated_positions = []

        # Convert plan positions to tensors if needed
        waypoints = []
        for i in range(len(plan.position)):
            pos = plan.position[i]
            if isinstance(pos, torch.Tensor):
                waypoints.append(pos)
            else:
                waypoints.append(torch.tensor(pos))

        # Interpolate between each pair of consecutive waypoints
        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]

            # Create interpolation steps between start and end
            for step in range(interpolation_steps):
                alpha = step / interpolation_steps
                interpolated_pos = start_pos * (1 - alpha) + end_pos * alpha
                interpolated_positions.append(interpolated_pos)

        # Add the final waypoint
        interpolated_positions.append(waypoints[-1])

        return interpolated_positions

    def set_motion_generator_reference(self, motion_gen: Any) -> None:
        """Set the motion generator reference for sphere animation.

        Args:
            motion_gen: CuRobo motion generator instance
        """
        self._motion_gen_ref = motion_gen

    def mark_idle(self) -> None:
        """Clear the active arm preview between episodes, retaining its recorded history."""
        self._clear_visualization()
        self._plan_frame += 1
