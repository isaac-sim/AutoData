# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision-aware waypoint and block scheduling without simulator dependencies."""

from autodata_core.scheduling.retiming import schedule_paths
from autodata_core.scheduling.types import Schedule, ScheduleFailure, SchedulePath, SegmentWaypoint, WaypointConflict

__all__ = [
    "Schedule",
    "ScheduleFailure",
    "SchedulePath",
    "SegmentWaypoint",
    "WaypointConflict",
    "schedule_paths",
]
