# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the AVP override against the relocated Isaac Lab G1 task."""

import importlib

from autodata_utils import g1_avp_teleop


def test_avp_override_reaches_g1_task(monkeypatch):
    module = importlib.import_module("isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg")
    # Restore the task's original builder and marker after the test.
    monkeypatch.setattr(module, "_build_g1_locomanipulation_pipeline", module._build_g1_locomanipulation_pipeline)
    monkeypatch.setattr(module, "_autodata_g1_avp_teleop_patch_installed", False, raising=False)

    g1_avp_teleop.install_g1_avp_teleop_patch_if_requested(
        ["--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--cloudxr_env", "avp"]
    )

    cfg = module.LocomanipulationG1EnvCfg()
    assert cfg.isaac_teleop.pipeline_builder is g1_avp_teleop._build_g1_avp_locomanipulation_pipeline
