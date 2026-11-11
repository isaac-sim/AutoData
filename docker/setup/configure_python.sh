#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

source /isaac-sim/setup_python_env.sh
"$VIRTUAL_ENV/bin/python" -I - <<'PY'
import os
from pathlib import Path
import sysconfig

# Scope Sim's modules to this environment, after its stdlib and locked packages.
paths = [path for path in os.environ["PYTHONPATH"].split(os.pathsep) if path.startswith("/isaac-sim/")]
Path(sysconfig.get_path("purelib"), "autodata_sim.pth").write_text("\n".join(paths) + "\n")
# Direct Python and docker exec need the same libraries as the interactive shell.
libraries = [path for path in os.environ["LD_LIBRARY_PATH"].split(os.pathsep) if path.startswith("/isaac-sim/")]
Path("/etc/ld.so.conf.d/autodata-sim.conf").write_text("\n".join(libraries) + "\n")
PY
ldconfig
# Kit keeps per-application state here even when Python runs from a separate venv.
install -d -g isaac-sim -m 2775 /isaac-sim/kit/{cache,data,logs}
