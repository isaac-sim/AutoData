# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Install temporary compatibility patches for AutoData Python workflows."""

from isaac_autodata_utils.isaaclab_compat import install_franka_asset_path_patch

install_franka_asset_path_patch()
