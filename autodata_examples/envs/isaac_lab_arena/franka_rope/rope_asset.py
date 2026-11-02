# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path

from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_arena.assets.asset import Asset

_ROPE_USD_PATH = Path(__file__).resolve().parent / "assets" / "Rope.usd"


class FrankaRopeAsset(Asset):
    """Deformable rope asset that can participate in an Arena :class:`Scene`."""

    def __init__(self, name: str = "object") -> None:
        super().__init__(name=name, tags=["deformable"])
        self.object_cfg = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.02),
                rot=(0.0, 0.0, 0.707, 0.707),
            ),
            spawn=UsdFileCfg(usd_path=str(_ROPE_USD_PATH)),
            debug_vis=False,
        )

    def get_object_cfg(self) -> tuple[str, DeformableObjectCfg]:
        """Return the scene key and Isaac Lab deformable-object configuration."""

        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, None]:
        """Return no asset-level event; the Arena task owns nodal reset semantics."""

        return self.name, None
