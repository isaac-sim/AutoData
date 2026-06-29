# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun visualizer for the cuRobo v2 motion-planner backend.

Mirrors the v1 :class:`isaac_autodata_interfaces.motion_planners.curobo.plan_visualizer.PlanVisualizer`
so the two backends visualize the same things (planned end-effector trajectory, target pose, robot
collision spheres, attached-object collision spheres, and the world obstacles) with the same Rerun
entity layout and the same process-lifecycle management.

The only backend-specific difference is how collision spheres are computed: v1 calls
``motion_gen.kinematics.get_robot_as_spheres`` on a cuRobo v1 ``MotionGen``; here we call the same
method on the cuRobo v2 ``MotionPlanner.kinematics``. The planner supplies already-split robot /
attached sphere lists (see :meth:`CuroboV2Planner._visualize_plan`); everything the planner hands us
is expressed in the robot-base frame, so no per-entity translation offset is needed.

Rerun is an optional dependency: this module is imported lazily by the planner only when
``CuroboV2PlannerCfg.visualize_plan`` is set, so non-visualized runs never require it.
"""

from __future__ import annotations

import atexit
import numpy as np
import os
import signal
import subprocess
import threading
import time
import torch
import weakref
from typing import TYPE_CHECKING, Any


def _import_rerun_sdk():
    """Import the Rerun robotics SDK, even when the deprecated ``rerun`` file-watcher package
    shadows ``rerun-sdk``.

    Both packages claim the ``rerun`` import name. When the tartley ``rerun`` file-watcher is also
    installed it can win the import (it lives directly in ``site-packages/rerun`` while ``rerun-sdk``
    ships under a ``rerun_sdk/`` path entry), yielding a module with no ``init``/``log``. We
    proactively put the ``rerun_sdk`` directory first on ``sys.path`` so the real SDK wins, and fall
    back to force-reloading it if the wrong module was already imported.
    """
    import importlib
    import os
    import sys

    # Prefer the rerun-sdk package directory so the real SDK wins over the deprecated tartley
    # ``rerun`` file-watcher. This is effective because this module is imported lazily and nothing
    # else in the planning pipeline imports ``rerun`` first — so ``rerun`` is not yet in
    # ``sys.modules`` and the inserted path decides the winner. (We do not force-reimport a wrong,
    # already-loaded ``rerun``: rerun-sdk is a native extension and re-importing it after the
    # tartley package is cached crashes the interpreter.)
    if "rerun" not in sys.modules:
        for entry in list(sys.path):
            candidate = os.path.join(entry, "rerun_sdk")
            if os.path.isfile(os.path.join(candidate, "rerun", "__init__.py")):
                sys.path.insert(0, candidate)
                break

    module = importlib.import_module("rerun")
    if hasattr(module, "init"):
        return module

    raise ImportError(
        "cuRobo v2 plan visualization needs the Rerun robotics SDK, but `import rerun` resolved to "
        f"the deprecated 'rerun' file-watcher package ({getattr(module, '__file__', '?')}) — it was "
        "imported before this module could prefer rerun-sdk. Fix the env with `pip uninstall rerun` "
        "(this keeps rerun-sdk), or disable visualize_plan."
    )


rr = _import_rerun_sdk()

_RR_HAS_TRANSFORM_AXES = hasattr(rr, "TransformAxes3D")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Rerun process monitoring will be limited.")

if TYPE_CHECKING:
    import trimesh


# Global registry to track all visualizer instances for cleanup.
_GLOBAL_PLAN_VISUALIZERS: list[PlanVisualizer] = []


def _cleanup_all_plan_visualizers() -> None:
    """Kill any lingering Rerun viewer processes and close tracked visualizers on exit."""
    if PSUTIL_AVAILABLE:
        killed = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if (proc.info["name"] and "rerun" in proc.info["name"].lower()) or (
                proc.info["cmdline"] and any("rerun" in str(arg).lower() for arg in proc.info["cmdline"])
            ):
                proc.kill()
                killed += 1
        print(f"Killed {killed} Rerun viewer processes on script exit")
    else:
        subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)

    for visualizer in _GLOBAL_PLAN_VISUALIZERS[:]:
        if not visualizer._closed:
            visualizer.close()
    _GLOBAL_PLAN_VISUALIZERS.clear()


atexit.register(_cleanup_all_plan_visualizers)


class PlanVisualizer:
    """Visualizes cuRobo v2 motion plans, collision spheres, and obstacles via Rerun.

    Args:
        robot_name: Robot identifier used in the recording id.
        recording_id: Optional Rerun recording id; defaults to ``motion_plan_<robot_name>``.
        debug: Whether to print debug information.
        save_path: Optional path to save the Rerun recording on close.
        base_translation: Optional translation added to every visualized entity. Defaults to zero
            because the v2 planner works entirely in the robot-base frame.
    """

    def __init__(
        self,
        robot_name: str = "franka",
        recording_id: str | None = None,
        debug: bool = False,
        save_path: str | None = None,
        base_translation: np.ndarray | None = None,
    ) -> None:
        self.robot_name = robot_name
        self.debug = debug
        self.recording_id = recording_id or f"motion_plan_{robot_name}"
        self.save_path = save_path
        self._closed = False
        self._base_translation = (
            np.array(base_translation, dtype=float) if base_translation is not None else np.zeros(3)
        )

        self._parent_pid = os.getpid()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_active = False

        # cuRobo v2 MotionPlanner reference, set by the planner for sphere animation.
        self._motion_planner_ref: Any = None

        global _GLOBAL_PLAN_VISUALIZERS
        _GLOBAL_PLAN_VISUALIZERS.append(self)

        rr.init(self.recording_id, spawn=False)
        self._rerun_process = None
        self._sink = self._connect_sink()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

        self._current_frame = 0
        self._sphere_entities: dict[str, list[str]] = {"robot": [], "attached": [], "target": []}

        self._start_parent_process_monitoring()

        self._finalizer = weakref.finalize(
            self, self._cleanup_class_resources, self.recording_id, self.save_path, debug
        )
        recording_id_local, save_path_local, debug_local = self.recording_id, self.save_path, debug
        atexit.register(self._cleanup_class_resources, recording_id_local, save_path_local, debug_local)

        self._original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        def signal_handler(signum, frame):
            if self.debug:
                print(f"Received signal {signum}, closing Rerun viewer...")
            self._cleanup_on_exit()
            if signum == signal.SIGINT:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            elif signum == signal.SIGTERM:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.debug:
            print(f"Initialized cuRobo v2 Rerun visualization (recording id: {self.recording_id})")

    # ------------------------------------------------------------------
    # Process lifecycle (backend-agnostic; mirrors the v1 visualizer)
    # ------------------------------------------------------------------

    def _connect_sink(self) -> str:
        """Attach a Rerun sink: a live viewer if one can be launched, else an ``.rrd`` file.

        ``rr.spawn()`` needs the ``rerun`` viewer executable. The PATH ``rerun`` script can be
        absent (e.g. removed when an unrelated ``rerun`` package is uninstalled — both ship a
        ``bin/rerun``), so we point ``spawn`` at rerun-sdk's bundled viewer binary
        (``rerun_cli/rerun``) when present. If no viewer can be launched, we fall back to writing
        an ``.rrd`` recording the user can open later. Always loud about which path was taken — the
        previous silent-failure mode is exactly what made this hard to diagnose.
        """
        exe = self._bundled_viewer_path()
        try:
            if exe is not None:
                rr.spawn(executable_path=exe)
            else:
                rr.spawn()
            print(
                f"[PlanVisualizer] live Rerun viewer launched (recording_id={self.recording_id!r}, "
                f"viewer={exe or 'rerun on PATH'}).",
                flush=True,
            )
            return "viewer"
        except Exception as exc:  # noqa: BLE001
            if self.save_path:
                try:
                    rr.save(self.save_path)
                    print(
                        f"[PlanVisualizer] no live viewer ({exc}); recording to {self.save_path!r}. "
                        f"Open it with:  rerun {self.save_path}",
                        flush=True,
                    )
                    return "file"
                except Exception as save_exc:  # noqa: BLE001
                    print(f"[PlanVisualizer] viewer spawn AND file save failed: {save_exc!r}", flush=True)
            else:
                print(f"[PlanVisualizer] could not launch a live viewer ({exc}); no save_path set.", flush=True)
            return "none"

    @staticmethod
    def _bundled_viewer_path() -> str | None:
        """Path to rerun-sdk's bundled viewer binary (``rerun_cli/rerun``), or ``None`` if absent."""
        import os

        try:
            pkg_dir = os.path.dirname(rr.__file__)  # .../rerun_sdk/rerun
            candidate = os.path.abspath(os.path.join(pkg_dir, os.pardir, "rerun_cli", "rerun"))
            return candidate if os.path.isfile(candidate) else None
        except Exception:  # noqa: BLE001
            return None

    def _start_parent_process_monitoring(self) -> None:
        if not PSUTIL_AVAILABLE:
            return
        self._monitor_active = True

        def monitor_parent_process() -> None:
            parent_process = psutil.Process(self._parent_pid)
            while self._monitor_active:
                try:
                    if not parent_process.is_running():
                        self._kill_rerun_processes()
                        break
                    time.sleep(2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._kill_rerun_processes()
                    break
                except Exception:
                    break

        self._monitor_thread = threading.Thread(target=monitor_parent_process, daemon=True)
        self._monitor_thread.start()

    def _kill_rerun_processes(self) -> None:
        try:
            if PSUTIL_AVAILABLE:
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        is_rerun = bool(proc.info["name"] and "rerun" in proc.info["name"].lower()) or bool(
                            proc.info["cmdline"] and any("rerun" in str(a).lower() for a in proc.info["cmdline"])
                        )
                        if is_rerun:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            else:
                subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)
        except Exception as exc:  # pragma: no cover
            if self.debug:
                print(f"Error killing rerun processes: {exc}")

    @staticmethod
    def _cleanup_class_resources(recording_id: str, save_path: str | None, debug: bool) -> None:
        rr.disconnect()
        if save_path is not None:
            rr.save(save_path)
        if PSUTIL_AVAILABLE:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if (proc.info["name"] and "rerun" in proc.info["name"].lower()) or (
                    proc.info["cmdline"] and any("rerun" in str(a).lower() for a in proc.info["cmdline"])
                ):
                    proc.kill()
        else:
            subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)

    def _cleanup_on_exit(self) -> None:
        if not self._closed:
            self._monitor_active = False
            self.close()
            self._kill_rerun_processes()

    def close(self) -> None:
        """Close the Rerun connection, terminate the viewer, and deregister."""
        if self._closed:
            return
        self._monitor_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            time.sleep(0.1)
        rr.disconnect()
        if self.save_path is not None:
            rr.save(self.save_path)
        self._closed = True
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
        self._kill_rerun_processes()
        global _GLOBAL_PLAN_VISUALIZERS
        if self in _GLOBAL_PLAN_VISUALIZERS:
            _GLOBAL_PLAN_VISUALIZERS.remove(self)

    def set_motion_planner_reference(self, motion_planner: Any) -> None:
        """Provide the cuRobo v2 ``MotionPlanner`` used to compute spheres during animation."""
        self._motion_planner_ref = motion_planner

    # ------------------------------------------------------------------
    # Static plan visualization
    # ------------------------------------------------------------------

    def visualize_plan(
        self,
        plan: Any,
        target_pose: torch.Tensor,
        robot_spheres: list[Any] | None = None,
        attached_spheres: list[Any] | None = None,
        ee_positions: np.ndarray | None = None,
        world_scene: trimesh.Scene | None = None,
    ) -> None:
        """Log one static snapshot of a plan: obstacles, target, EE trajectory, and spheres.

        Args:
            plan: Active-DoF joint trajectory (only ``plan.position`` is read, for the EE path
                fallback). The planner normally supplies ``ee_positions`` directly.
            target_pose: Target end-effector pose as a 4x4 matrix in the robot-base frame.
            robot_spheres: Robot collision spheres (cuRobo ``Sphere`` objects).
            attached_spheres: Attached-object collision spheres.
            ee_positions: ``[T, 3]`` end-effector positions in the robot-base frame.
            world_scene: Optional ``trimesh.Scene`` of the world obstacles.
        """
        rr.set_time("static_plan", sequence=self._current_frame)
        self._current_frame += 1

        self._clear_visualization()

        if world_scene is not None:
            self._visualize_world_scene(world_scene)
        self._visualize_target_pose(target_pose)
        self._visualize_trajectory(plan, ee_positions)

        if robot_spheres:
            self._log_spheres(robot_spheres, "robot", [0, 255, 100, 128])
        if attached_spheres:
            self._log_spheres(attached_spheres, "attached", [255, 0, 0, 128])
        else:
            self._clear_attached_spheres()

        n_ee = 0 if ee_positions is None else len(ee_positions)
        print(
            f"[PlanVisualizer] logged plan -> sink={self._sink}: {n_ee} EE waypoints, "
            f"{len(robot_spheres or [])} robot spheres, {len(attached_spheres or [])} attached spheres.",
            flush=True,
        )

    def _clear_visualization(self) -> None:
        for path in ("trajectory", "target", "anim"):
            rr.log(f"world/{path}", rr.Clear(recursive=True))
        for entity_type, entities in self._sphere_entities.items():
            for entity in entities:
                rr.log(f"world/{entity_type}/{entity}", rr.Clear(recursive=True))
            self._sphere_entities[entity_type] = []
        self._current_frame = 0

    def clear_visualization(self) -> None:
        """Public wrapper around :meth:`_clear_visualization`."""
        self._clear_visualization()

    def _visualize_target_pose(self, target_pose: torch.Tensor) -> None:
        mat = target_pose.detach().cpu().numpy() if torch.is_tensor(target_pose) else np.asarray(target_pose)
        mat = mat.reshape(4, 4)
        pos = mat[:3, 3] + self._base_translation
        rot = mat[:3, :3]
        rr.log("world/target/position", rr.Points3D(positions=np.array([pos]), colors=[[255, 0, 0]], radii=[0.02]))
        rr.log("world/target/frame", rr.Transform3D(translation=pos, mat3x3=rot))

    def _visualize_trajectory(self, plan: Any, ee_positions: np.ndarray | None) -> None:
        if ee_positions is None:
            raw = plan.position.detach().cpu().numpy() if torch.is_tensor(plan.position) else np.array(plan.position)
            if raw.ndim != 2 or raw.shape[1] < 3:
                return  # nothing sensible to draw without explicit EE positions
            positions = raw[:, :3]
        else:
            positions = np.asarray(ee_positions)
        if positions.size == 0:
            return
        positions = positions + self._base_translation
        rr.log("world/trajectory", rr.LineStrips3D([positions], colors=[[0, 100, 255]], radii=[0.005]), static=True)
        for i, pos in enumerate(positions):
            rr.log(
                f"world/trajectory/keyframe_{i}",
                rr.Points3D(positions=np.array([pos]), colors=[[0, 100, 255]], radii=[0.01]),
                static=True,
            )

    def _log_spheres(self, spheres: list[Any], entity_type: str, color: list[int]) -> None:
        for i, sphere in enumerate(spheres):
            entity_id = f"sphere_{i}"
            self._sphere_entities.setdefault(entity_type, []).append(entity_id)
            pos = (
                sphere.position.detach().cpu().numpy()
                if torch.is_tensor(sphere.position)
                else np.array(sphere.position)
            ).reshape(-1)
            pos = pos + self._base_translation
            rr.log(
                f"world/{entity_type}/{entity_id}",
                rr.Points3D(positions=np.array([pos]), colors=[color], radii=[float(sphere.radius)]),
            )

    def _clear_attached_spheres(self) -> None:
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"world/attached/{entity_id}", rr.Clear(recursive=True))
        self._sphere_entities["attached"] = []

    def _visualize_world_scene(self, scene: trimesh.Scene) -> None:
        import trimesh

        if not hasattr(self, "_logged_geometry"):
            self._logged_geometry: set[str] = set()

        for node in scene.graph.nodes_geometry:
            tform, geom_key = scene.graph.get(node)
            mesh = scene.geometry.get(geom_key)
            if mesh is None:
                continue
            rr_path = f"world/scene/{node.replace('/', '_')}"
            if _RR_HAS_TRANSFORM_AXES:
                rr.log(rr_path, rr.Transform3D(translation=tform[:3, 3], mat3x3=tform[:3, :3]), static=False)
            else:
                rr.log(
                    rr_path,
                    rr.Transform3D(translation=tform[:3, 3], mat3x3=tform[:3, :3], axis_length=0.0),
                    static=False,
                )
            if rr_path not in self._logged_geometry:
                if isinstance(mesh, trimesh.Trimesh):
                    rr.log(
                        rr_path,
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,
                            triangle_indices=mesh.faces,
                            vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
                        ),
                        static=True,
                    )
                    self._logged_geometry.add(rr_path)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def animate_plan(self, ee_positions: np.ndarray, timeline: str = "plan", point_radius: float = 0.01) -> None:
        """Play back the end-effector marker along ``ee_positions`` on ``timeline``."""
        if ee_positions is None or len(ee_positions) == 0:
            return
        for idx, pos in enumerate(ee_positions):
            rr.set_time(timeline, sequence=idx)
            rr.log(
                "world/anim/ee",
                rr.Points3D(
                    positions=np.array([pos + self._base_translation]), colors=[[0, 100, 255]], radii=[point_radius]
                ),
            )

    def animate_spheres_along_path(
        self,
        plan: Any,
        robot_sphere_count: int,
        timeline: str = "sphere_animation",
        interpolation_steps: int = 10,
    ) -> None:
        """Animate robot (green) and attached (orange) spheres along the planned trajectory.

        Recomputes collision spheres at densely interpolated configurations via the v2
        ``MotionPlanner.kinematics.get_robot_as_spheres``. ``plan`` must carry active-DoF
        positions (``[T, active_dof]``); ``robot_sphere_count`` is the number of robot self
        spheres (the remainder of each frame's active spheres are the attached object's).
        """
        motion_planner = self._motion_planner_ref
        if motion_planner is None or plan is None or len(plan.position) == 0:
            return
        device = motion_planner.device_cfg.device

        self._hide_static_spheres_for_animation()
        interpolated = self._create_interpolated_trajectory(plan, interpolation_steps)

        for frame_idx, joint_positions in enumerate(interpolated):
            rr.set_time(timeline, sequence=frame_idx)
            q = joint_positions if isinstance(joint_positions, torch.Tensor) else torch.tensor(joint_positions)
            q = q.to(device=device, dtype=torch.float32)
            if q.ndim == 1:
                q = q.unsqueeze(0)  # [active_dof] -> [1, active_dof]
            try:
                with torch.inference_mode(False), torch.enable_grad():
                    sphere_list = motion_planner.kinematics.get_robot_as_spheres(q)[0]
            except Exception as exc:
                if self.debug:
                    print(f"Failed to compute spheres for frame {frame_idx}: {exc}")
                continue

            robot_pos, robot_rad, att_pos, att_rad = [], [], [], []
            for i, sphere in enumerate(sphere_list):
                pos = (
                    sphere.position.detach().cpu().numpy()
                    if torch.is_tensor(sphere.position)
                    else np.array(sphere.position)
                ).reshape(-1) + self._base_translation
                if i < robot_sphere_count:
                    robot_pos.append(pos)
                    robot_rad.append(float(sphere.radius))
                else:
                    att_pos.append(pos)
                    att_rad.append(float(sphere.radius))

            if robot_pos:
                rr.log(
                    "world/robot_animation",
                    rr.Points3D(
                        positions=np.array(robot_pos), colors=[[0, 255, 100, 220]] * len(robot_pos), radii=robot_rad
                    ),
                )
            if att_pos:
                rr.log(
                    "world/attached_animation",
                    rr.Points3D(positions=np.array(att_pos), colors=[[255, 150, 0, 220]] * len(att_pos), radii=att_rad),
                )
            else:
                rr.log("world/attached_animation", rr.Clear(recursive=True))

    def _hide_static_spheres_for_animation(self) -> None:
        for entity_id in self._sphere_entities.get("robot", []):
            rr.log(f"world/robot/{entity_id}", rr.Clear(recursive=True))
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"world/attached/{entity_id}", rr.Clear(recursive=True))

    @staticmethod
    def _create_interpolated_trajectory(plan: Any, interpolation_steps: int) -> list[torch.Tensor]:
        positions = plan.position
        if len(positions) < 2:
            p0 = positions[0]
            return [p0 if isinstance(p0, torch.Tensor) else torch.tensor(p0)]
        waypoints = [p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in positions]
        out: list[torch.Tensor] = []
        for i in range(len(waypoints) - 1):
            start, end = waypoints[i], waypoints[i + 1]
            for step in range(interpolation_steps):
                alpha = step / interpolation_steps
                out.append(start * (1.0 - alpha) + end * alpha)
        out.append(waypoints[-1])
        return out

    def mark_idle(self) -> None:
        """Emit empty animation frames so stale spheres/markers don't linger between plans."""
        empty = np.empty((0, 3), dtype=float)
        rr.set_time("plan", sequence=self._current_frame)
        self._current_frame += 1
        rr.log("world/anim/ee", rr.Points3D(positions=empty))
        rr.set_time("sphere_animation", sequence=self._current_frame)
        rr.log("world/robot_animation", rr.Points3D(positions=empty))
        rr.log("world/attached_animation", rr.Points3D(positions=empty))
