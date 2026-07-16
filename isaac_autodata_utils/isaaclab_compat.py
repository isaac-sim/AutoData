# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Temporary compatibility patches for pinned Isaac Lab dependencies."""

import functools
import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

_ISAACSIM_MODULE = "isaacsim"
_FRANKA_MODULE = "isaaclab_assets.robots.franka"
_OLD_PATH_SUFFIX = "/Robots/FrankaEmika/panda_instanceable.usd"
_NEW_PATH_SUFFIX = "/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
_SIMULATION_APP_PATCH_MARKER = "_isaac_autodata_franka_asset_path_patch_installed"


def apply_franka_asset_path_patch() -> None:
    """Patch the relocated Franka asset after Isaac Sim has initialized."""
    # TODO: Remove after Isaac Lab includes 78ddcf9e331076c1eab5d817fd05dfe5c03601f7.
    try:
        module = importlib.import_module(_FRANKA_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {"isaaclab_assets", _FRANKA_MODULE}:
            return
        raise

    for config_name in ("FRANKA_PANDA_CFG", "FRANKA_PANDA_HIGH_PD_CFG"):
        config = getattr(module, config_name, None)
        usd_path = getattr(getattr(config, "spawn", None), "usd_path", None)
        if isinstance(usd_path, str) and usd_path.endswith(_OLD_PATH_SUFFIX):
            config.spawn.usd_path = usd_path.removesuffix(_OLD_PATH_SUFFIX) + _NEW_PATH_SUFFIX


def _wrap_simulation_app(module: ModuleType) -> None:
    """Apply the Franka patch after each Isaac Sim application initialization."""
    simulation_app = getattr(module, "SimulationApp", None)
    if simulation_app is None or getattr(simulation_app, _SIMULATION_APP_PATCH_MARKER, False):
        return

    original_init = simulation_app.__init__

    @functools.wraps(original_init)
    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        apply_franka_asset_path_patch()

    simulation_app.__init__ = _patched_init
    setattr(simulation_app, _SIMULATION_APP_PATCH_MARKER, True)


class _IsaacSimPatchLoader(importlib.abc.Loader):
    """Wrap the normal Isaac Sim loader and patch its ``SimulationApp`` class."""

    def __init__(self, wrapped_loader: importlib.abc.Loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._wrapped_loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self._wrapped_loader, "exec_module", None)
        assert exec_module is not None, "Isaac Sim module loader must support exec_module()."
        exec_module(module)
        _wrap_simulation_app(module)


class _IsaacSimPatchFinder(importlib.abc.MetaPathFinder):
    """Intercept only the top-level ``isaacsim`` import without importing Isaac Lab."""

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _ISAACSIM_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None and not isinstance(spec.loader, _IsaacSimPatchLoader):
            spec.loader = _IsaacSimPatchLoader(spec.loader)
        return spec


def install_franka_asset_path_patch() -> None:
    """Install the Franka path patch without importing Isaac Lab before startup."""
    isaacsim_module = sys.modules.get(_ISAACSIM_MODULE)
    if isaacsim_module is not None:
        _wrap_simulation_app(isaacsim_module)
        return

    if not any(isinstance(finder, _IsaacSimPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _IsaacSimPatchFinder())
