# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


class _TestPaths:
    def __init__(self):
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        self.tests_dir = os.path.dirname(utils_dir)
        self.repo_root = os.path.dirname(self.tests_dir)

        # Test data directory
        self.test_data_dir = os.path.join(self.tests_dir, "test_data")

        # Config directories
        self.examples_dir = os.path.join(self.repo_root, "isaac_autodata_examples")
        self.scripts_dir = os.path.join(self.repo_root, "scripts")
        self.generate_dataset_script = os.path.join(self.scripts_dir, "generate_dataset.py")
        self.tasks_dir = os.path.join(self.examples_dir, "tasks")
        self.embodiments_dir = os.path.join(self.examples_dir, "embodiments")

        self.python_path = sys.executable


TestPaths = _TestPaths()
