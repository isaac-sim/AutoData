# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration callbacks consume provider options without seeing AutoData descriptors."""

import argparse
import sys
from types import SimpleNamespace

import pytest

from autodata_interfaces.env.registration import register_external_environment


def test_callback_receives_only_environment_options(monkeypatch):
    original = ["annotate_demos.py", "--embodiment", "robot.yaml", "--task_descriptor", "task.yaml"]
    monkeypatch.setattr(sys, "argv", original)
    captured = []

    def register():
        captured.extend(sys.argv[1:])
        return []

    monkeypatch.setitem(sys.modules, "external_provider", SimpleNamespace(register=register))
    args = argparse.Namespace(
        external_callback="external_provider.register",
        env_name="external-task",
        device="cpu",
        record_images=True,
        num_envs=2,
    )
    register_external_environment(args, ["--object", "box", "--", "--embodiment", "provider_robot"])
    assert captured == [
        "--task",
        "external-task",
        "--device",
        "cpu",
        "--num_envs",
        "2",
        "--enable_cameras",
        "--object",
        "box",
        "--embodiment",
        "provider_robot",
    ]
    assert sys.argv is original


def test_callback_failure_restores_process_arguments(monkeypatch):
    original = sys.argv

    def register():
        raise RuntimeError("registration failed")

    monkeypatch.setitem(sys.modules, "external_provider", SimpleNamespace(register=register))
    args = argparse.Namespace(
        external_callback="external_provider:register",
        env_name="external-task",
        device="cpu",
        record_images=False,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        register_external_environment(args, [])
    assert sys.argv is original
