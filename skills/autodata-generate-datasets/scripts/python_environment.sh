# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  python_cmd=("$VIRTUAL_ENV/bin/python")
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  python_cmd=("$CONDA_PREFIX/bin/python")
elif [[ -x /isaac-sim/python.sh ]]; then
  python_cmd=(/isaac-sim/python.sh)
elif [[ -d .venv ]]; then
  python_cmd=(.venv/bin/python)
else
  python_cmd=(python)
fi

if ! "${python_cmd[@]}" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("isaaclab_arena") is None)' >/dev/null 2>&1; then
  echo "ERROR: Isaac Lab-Arena was not found using ${python_cmd[*]}." >&2
  echo "       Start ./docker/run_docker.sh, or activate a complete environment created with uv sync --locked." >&2
  exit 1
fi
