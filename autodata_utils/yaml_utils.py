# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML loading helpers."""

import yaml


class UniqueKeySafeLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that rejects duplicate mapping keys instead of silently keeping the last one.

    ``yaml.safe_load`` is last-key-wins, so a repeated key would quietly override an earlier value before
    any schema validation runs. Checking in ``construct_mapping`` covers every mapping node, including
    nested ones.

    Example:
        ``yaml.load(f, Loader=UniqueKeySafeLoader)``
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in seen, f"duplicate key {key!r} in YAML mapping at {key_node.start_mark}."
            seen.add(key)
        return super().construct_mapping(node, deep)
