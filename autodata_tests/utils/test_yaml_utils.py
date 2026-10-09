# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the duplicate-key-rejecting YAML loader."""

import yaml

import pytest

from autodata_utils.yaml_utils import UniqueKeySafeLoader


def test_unique_key_loader_loads_valid_yaml():
    assert yaml.load("a: 1\nb: {c: 2}\n", Loader=UniqueKeySafeLoader) == {"a": 1, "b": {"c": 2}}


@pytest.mark.parametrize(
    "text",
    [
        "a: 1\na: 2\n",
        "outer:\n  inner: 1\n  inner: 2\n",
        "items:\n  - x: 1\n    x: 2\n",
    ],
)
def test_unique_key_loader_rejects_duplicate_keys(text):
    with pytest.raises(AssertionError, match="duplicate key"):
        yaml.load(text, Loader=UniqueKeySafeLoader)
