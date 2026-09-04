# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install temporary compatibility patches for Autodata Python workflows."""

from autodata_utils.g1_avp_teleop import install_g1_avp_teleop_patch_if_requested
from autodata_utils.isaaclab_compat import install_franka_asset_path_patch

install_franka_asset_path_patch()
install_g1_avp_teleop_patch_if_requested()
