# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path

from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_arena.assets.asset import Asset

_ROPE_USD_PATH = Path(__file__).resolve().parent / "assets" / "Rope.usd"


def rope_newton_spawn_cfg() -> UsdFileCfg:
    """Load the shared rope topology with Newton material properties."""
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

    # VBD needs a compliant volume material for the gripper to bend this slender
    # mesh. Keep the shared topology and density, with backend-specific elasticity.
    youngs_modulus, poissons_ratio = 2.0e5, 0.45
    return UsdFileCfg(
        usd_path=str(_ROPE_USD_PATH.with_name("RopeNewton.usda")),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material_path="NewtonPhysicsMaterial",
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=1000.0,
            particle_radius=0.002,
            k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
            k_lambda=youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)),
            k_damp=1000.0,
        ),
    )


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
