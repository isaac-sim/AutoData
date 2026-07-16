# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Temporary compatibility patches for pinned Isaac Lab dependencies."""

import importlib
import importlib.util

_OLD_PATH_SUFFIX = "/Robots/FrankaEmika/panda_instanceable.usd"
_NEW_PATH_SUFFIX = "/Robots/FrankaEmika/Legacy/panda_instanceable.usd"


def apply_franka_asset_path_patch() -> None:
    """Patch the relocated Franka asset before Isaac Sim starts its extension loader."""
    # TODO: Remove after Isaac Lab includes 78ddcf9e331076c1eab5d817fd05dfe5c03601f7.
    if importlib.util.find_spec("isaaclab_assets") is None:
        return

    module = importlib.import_module("isaaclab_assets.robots.franka")
    for config_name in ("FRANKA_PANDA_CFG", "FRANKA_PANDA_HIGH_PD_CFG"):
        config = getattr(module, config_name, None)
        usd_path = getattr(getattr(config, "spawn", None), "usd_path", None)
        if isinstance(usd_path, str) and usd_path.endswith(_OLD_PATH_SUFFIX):
            config.spawn.usd_path = usd_path.removesuffix(_OLD_PATH_SUFFIX) + _NEW_PATH_SUFFIX
