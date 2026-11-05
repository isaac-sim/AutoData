# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI test to catch when we can remove the staging bucket override."""

import runpy

import pytest
from isaaclab_arena.assets import nucleus as arena_assets


@pytest.mark.parametrize("root_name", ["ARENA_NUCLEUS_DIR", "ISAAC_STAGING_NUCLEUS_DIR"])
def test_staging_asset_override_is_still_needed(root_name: str) -> None:
    upstream_url = runpy.run_path(arena_assets.__file__)[root_name]
    assert "/Isaac/6.2/" not in upstream_url, (
        f"Upstream {root_name} already points to 6.2: {upstream_url}. "
        "Remove the manual staging override in autodata_examples/envs/__init__.py and this test."
    )
