#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep Sim's transitive dependencies locked, but use the simulator supplied by the base image.
sim_packages=$("${UV_PYTHON}" - <<'PY'
import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text())
version = Path("/isaac-sim/VERSION").read_text().strip().split("-")[0]
packages = [p for p in lock["package"] if p["name"] == "isaacsim" or p["name"].startswith("isaacsim-")]
assert packages, "uv.lock must include Isaac Sim"
for package in packages:
    assert package["version"] == f"{version}.0", (
        f"Base Isaac Sim {version} does not match {package['name']}=={package['version']} in uv.lock"
    )
    print(package["name"])
PY
)

args=(--locked)
while read -r package; do
    args+=(--no-install-package "$package")
done <<< "$sim_packages"
uv sync "${args[@]}" "$@"
