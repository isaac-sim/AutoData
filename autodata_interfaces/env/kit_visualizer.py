# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kit viewport capture with camera-driven rendering in headless mode."""

from isaaclab_visualizers.kit import KitVisualizer


class KitVisualizerHeadless(KitVisualizer):
    """Keep observation cameras active when the viewport renders only on demand."""

    def pumps_app_update(self) -> bool:
        """Report whether simulation steps update the Kit app before cameras are read."""

        return False if self.cfg.headless else super().pumps_app_update()
