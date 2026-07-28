# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy native symbols for experimental ScheduleStream v1/v2 command lowering."""

from __future__ import annotations

import importlib
from typing import Any

from isaac_autodata_interfaces.autonomous.schedulestream.command_types import (
    ScheduleStreamCommandSymbols,
    ScheduleStreamImportError,
)


def load_schedulestream_command_symbols(application: str) -> ScheduleStreamCommandSymbols:
    """Import only the selected ScheduleStream application and return its command surface.

    Args:
        application: ``custream`` for cuRobo v1 or ``custream2`` for cuRobo v2.
    """

    if application not in ("custream", "custream2"):
        raise ScheduleStreamImportError(
            f"unknown ScheduleStream application {application!r}; expected 'custream' or 'custream2'"
        )
    command_module_name = f"schedulestream.applications.{application}.command"
    state_module_name = f"schedulestream.applications.{application}.state"
    utils_module_name = f"schedulestream.applications.{application}.utils"
    try:
        command = importlib.import_module(command_module_name)
        state = importlib.import_module(state_module_name)
        utils = importlib.import_module(utils_module_name)
        link_path_type = command.ArmPath if application == "custream" else command.LinkPath
        return ScheduleStreamCommandSymbols(
            application=application,
            commands_type=command.Commands,
            composite_type=command.Composite,
            configuration_type=state.Configuration,
            trajectory_type=command.Trajectory,
            link_path_type=link_path_type,
            open_type=command.Open,
            close_type=command.Close,
            attach_type=command.Attach,
            detach_type=command.Detach,
            pose_to_matrix=utils.matrix_from_pose,
        )
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:500]
        raise ScheduleStreamImportError(
            f"failed to load reviewed {application} command API ({type(exc).__name__}: {message})"
        ) from exc


def native_type_name(value: Any) -> str:
    """Return a bounded native type name without invoking an object's representation."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"[:512]
