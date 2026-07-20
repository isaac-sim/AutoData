# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU/App import smoke for the pinned ScheduleStream cuRobo-v1 development image."""

from __future__ import annotations

import json
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher

_EXPECTED_SCHEDULESTREAM_COMMIT = "f6351b8db8d7da9cb6ddd6854dbfc3123ab048f5"
_EXPECTED_SCHEDULESTREAM_VERSION = "0.0.0.dev0+f6351b8"


def main() -> None:
    """Launch headless Isaac, import the v1 integration, and print machine-readable capabilities."""

    launcher = AppLauncher({"headless": True})
    try:
        from schedulestream.applications.isaaclab.controller import PathController
        from schedulestream.applications.isaaclab.planner import Planner

        from isaac_autodata_interfaces.motion_planners.curobo.compat import (
            detect_curobo_runtime,
            select_schedulestream_backend,
        )

        capabilities = detect_curobo_runtime()
        selection = select_schedulestream_backend("curobo_v1", capabilities)
        source_commit = os.environ.get("SCHEDULESTREAM_SOURCE_COMMIT")
        expected_version = os.environ.get("SCHEDULESTREAM_VERSION")
        installed_version = version("schedulestream")
        if source_commit != _EXPECTED_SCHEDULESTREAM_COMMIT:
            raise RuntimeError(
                f"ScheduleStream source commit {source_commit!r} does not match reviewed "
                f"commit {_EXPECTED_SCHEDULESTREAM_COMMIT!r}"
            )
        if expected_version != _EXPECTED_SCHEDULESTREAM_VERSION or installed_version != expected_version:
            raise RuntimeError(
                "ScheduleStream version identity mismatch "
                f"(image={expected_version!r}, installed={installed_version!r}, "
                f"reviewed={_EXPECTED_SCHEDULESTREAM_VERSION!r})"
            )
        print(
            json.dumps(
                {
                    "controller_class": PathController.__name__,
                    "curobo": capabilities.to_dict(),
                    "planner_class": Planner.__name__,
                    "schedulestream_source_commit": source_commit,
                    "schedulestream_version": installed_version,
                    "selection": {
                        "application": selection.schedulestream_application,
                        "motion_backend": selection.motion_backend,
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
