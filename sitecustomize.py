# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the G1 Apple Vision Pro teleoperation override when requested."""

from importlib.util import find_spec

# An older editable installation may expose this module without exposing the
# renamed AutoData packages. Unrelated Python commands (including venv and docs
# builds) must still start normally in that environment.
if find_spec("autodata_utils") is not None:
    from autodata_utils.g1_avp_teleop import install_g1_avp_teleop_patch_if_requested

    install_g1_avp_teleop_patch_if_requested()
