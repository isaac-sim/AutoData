# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym rendering backed by live Isaac viewport and RGB observation frames."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Mapping
from typing import Any


def observation_video_path_resolve(
    configured_path: str | None,
    *,
    default_image_observation_names: tuple[str, ...],
    default_group_name: str = "policy",
) -> tuple[str, ...]:
    """Resolve the nested observation path used for simulator video.

    An explicit one-component path is interpreted relative to ``default_group_name``. When no
    path is configured, exactly one task-declared image observation is required; ambiguity fails
    before simulator construction instead of selecting a camera by iteration order.

    Args:
        configured_path: Optional dot-separated observation path.
        default_image_observation_names: Task-declared image terms in the default group.
        default_group_name: Observation group used for a one-component term.

    Returns:
        Canonical nested mapping keys from observation root to RGB tensor.
    """

    assert (
        isinstance(default_group_name, str) and default_group_name and default_group_name == default_group_name.strip()
    )
    assert isinstance(default_image_observation_names, tuple), "default image observation names must be a tuple"
    assert all(
        isinstance(name, str) and name and name == name.strip() and "." not in name
        for name in default_image_observation_names
    ), "default image observation names must be non-empty, trimmed local keys"
    assert len(set(default_image_observation_names)) == len(
        default_image_observation_names
    ), "default image observation names must be unique"

    if configured_path is None:
        assert len(default_image_observation_names) == 1, (
            "video recording requires --video_observation_path when the environment does not "
            "declare exactly one image_obs_list entry"
        )
        return default_group_name, default_image_observation_names[0]

    assert (
        isinstance(configured_path, str) and configured_path == configured_path.strip() and configured_path
    ), "video observation path must be a non-empty, trimmed string"
    components = tuple(configured_path.split("."))
    assert all(
        component and component == component.strip() for component in components
    ), "video observation path must contain non-empty dot-separated keys"
    return (default_group_name, components[0]) if len(components) == 1 else components


class ObservationRenderWrapper(gym.Wrapper):
    """Expose one live RGB observation through Gym's ``rgb_array`` render contract.

    Isaac Lab camera observations are refreshed as part of ``reset`` and ``step``. Caching the
    returned observation makes Gymnasium's :class:`~gymnasium.wrappers.RecordVideo` consume the
    exact frame aligned with each policy transition, without a fake physics step or a second
    viewport render product.

    Args:
        env: Environment whose observations contain the video frame.
        observation_path: Nested mapping keys leading to a batched or unbatched RGB image.
        environment_index: Batch row to record when the observation contains multiple envs.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        observation_path: tuple[str, ...],
        environment_index: int = 0,
    ) -> None:
        """Select the RGB observation and environment row used by video recording."""
        super().__init__(env)
        assert isinstance(observation_path, tuple) and observation_path, "observation_path must be non-empty"
        assert all(
            isinstance(component, str) and component and component == component.strip()
            for component in observation_path
        ), "observation_path must contain non-empty, trimmed keys"
        assert type(environment_index) is int and environment_index >= 0, "environment_index must be non-negative"
        self._observation_path = observation_path
        self._environment_index = environment_index
        self._frame: np.ndarray | None = None
        self._frame_shape: tuple[int, int, int] | None = None

    @property
    def render_mode(self) -> str:
        """Gym render mode supplied by this observation adapter."""

        return "rgb_array"

    @property
    def observation_path(self) -> tuple[str, ...]:
        """Nested observation keys backing the rendered frame."""

        return self._observation_path

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and retain its aligned RGB observation."""

        observation, info = self.env.reset(seed=seed, options=options)
        self._frame = self._frame_extract(observation)
        return observation, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Step the environment and retain its aligned RGB observation."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frame = self._frame_extract(observation)
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return an immutable snapshot of the latest RGB observation."""

        assert self._frame is not None, "video frame is unavailable before reset"
        return self._frame.copy()

    def _frame_extract(self, observation: Any) -> np.ndarray:
        """Return a copied RGB frame from the configured observation path."""
        value = observation
        traversed: list[str] = []
        for component in self._observation_path:
            assert isinstance(
                value, Mapping
            ), f"video observation {'.'.join(traversed) or '<root>'} must be a mapping, got {type(value).__name__}"
            assert (
                component in value
            ), f"video observation path {'.'.join(self._observation_path)!r} is missing key {component!r}"
            value = value[component]
            traversed.append(component)

        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.uint8, "video observation tensor must use uint8 pixels"
            frame = value.detach().cpu().numpy()
        else:
            assert isinstance(value, np.ndarray), "video observation must be a torch.Tensor or numpy.ndarray"
            assert value.dtype == np.uint8, "video observation array must use uint8 pixels"
            frame = value

        if frame.ndim == 4:
            assert (
                self._environment_index < frame.shape[0]
            ), f"video observation has {frame.shape[0]} environment rows, missing row {self._environment_index}"
            frame = frame[self._environment_index]
        else:
            assert self._environment_index == 0, "an unbatched video observation supports only environment_index=0"
        assert (
            frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0
        ), "video observation must have non-empty HWC image shape"
        assert frame.shape[2] in (3, 4), "video observation must have RGB or RGBA channels in the final axis"
        frame = np.ascontiguousarray(frame[:, :, :3]).copy()
        if self._frame_shape is None:
            self._frame_shape = frame.shape
        assert frame.shape == self._frame_shape, (
            f"video observation shape changed from {self._frame_shape} to {frame.shape}; "
            "one recording requires a fixed camera resolution"
        )
        return frame


class ViewportRenderWrapper(gym.Wrapper):
    """Expose the native Kit viewport through Gym's RGB recording interface.

    Captures the configured viewport without changing scene observation cameras or advancing
    physics. The simulator owns the visualizer and releases its render resources at close.
    """

    def __init__(self, env: gym.Env) -> None:
        """Select the existing Kit viewport without changing observation cameras."""
        super().__init__(env)
        visualizers = [
            visualizer for visualizer in env.unwrapped.sim.visualizers if visualizer.cfg.visualizer_type == "kit"
        ]
        assert len(visualizers) == 1, "Viewport recording requires exactly one Kit visualizer."
        self._visualizer = visualizers[0]

    @property
    def render_mode(self) -> str:
        """Gym render mode supplied by the viewport adapter."""
        return "rgb_array"

    def render(self) -> np.ndarray:
        """Return the current native viewport frame."""
        frame = self._visualizer.render_rgb_array()
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] in (3, 4), "Viewport capture must return RGB or RGBA."
        return np.ascontiguousarray(frame[:, :, :3])


__all__ = [
    "ObservationRenderWrapper",
    "ViewportRenderWrapper",
    "observation_video_path_resolve",
]
