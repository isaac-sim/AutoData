# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The :class:`Datastream` facade.

A single object the data generator reads the world through, composing the task descriptor
(subtask semantics), the embodiment adapter (kinematics), and the live env (scene), and owning the
source-demo pool.
"""

from isaac_autodata_interfaces.datastream.datastream import Datastream

__all__ = [
    "Datastream",
]
