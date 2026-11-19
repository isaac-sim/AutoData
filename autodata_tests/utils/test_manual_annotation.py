# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recreate bundled annotation boundaries through manual marking callbacks without a viewer."""

import ast
import h5py
import torch
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.tasks.segment_spec import SegmentEventPoint
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from autodata_interfaces.tasks.task_descriptor_utils import resolve_task_algorithm
from autodata_tests.utils.constants import TestPaths


@pytest.mark.parametrize(
    "dataset,descriptor",
    [
        ("annotated_dataset_franka_stack_mimicgen.hdf5", "franka_cube_stack.yaml"),
        ("annotated_dataset_franka_stack_skillgen.hdf5", "franka_cube_stack_skillgen.yaml"),
        ("annotated_dataset_gr1_pick_place_dexmimicgen.hdf5", "gr1_pick_place.yaml"),
        ("annotated_dataset_g1_pick_place_dexmimicgen.hdf5", "g1_pick_place.yaml"),
        ("annotated_dataset_franka_rope_softmimicgen.hdf5", "franka_rope_softmimicgen.yaml"),
        ("annotated_dataset_franka_rope_softmimicgen_newton.hdf5", "franka_rope_softmimicgen.yaml"),
        ("dataset_gr1t2_nut_pour_schedulegen_annotated.hdf5", "gr1t2_nut_pour_schedulegen.yaml"),
    ],
)
def test_manual_callbacks_recreate_bundled_annotations(dataset, descriptor):
    # Load only the callback functions: importing the entry point would start Isaac Sim.
    script = Path(TestPaths.scripts_dir) / "annotate_demos.py"
    tree = ast.parse(script.read_text())
    names = {"_build_signal_names", "mark_subtask_cb", "annotate_episode_in_manual_mode"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    namespace = {"torch": torch, "SegmentEventPoint": SegmentEventPoint}
    exec(compile(ast.fix_missing_locations(module), str(script), "exec"), namespace)
    task_path = str(Path(TestPaths.tasks_dir) / descriptor)
    task = TaskDescriptor.from_yaml(task_path)
    algorithm = resolve_task_algorithm(None, task_path)
    signals = namespace["_build_signal_names"](task, algorithm in ("skillgen", "schedulegen"))
    source = Path(TestPaths.test_data_dir) / dataset
    if descriptor == "gr1t2_nut_pour_schedulegen.yaml":
        source = Path(TestPaths.repo_root) / "datasets/annotated_datasets" / dataset

    with h5py.File(source) as data:
        for demo in data["data"].values():
            count = len(demo["actions"])
            expected = demo["obs/datagen_info"]
            marks = iter([
                [
                    DataGenInfoPool._find_first_rising_edge(expected[group][name][:], signal_name=name)
                    for group, name in arm_signals
                ]
                for arm_signals in signals.values()
                if arm_signals
            ])

            def replay(env, episode, success_term):
                for index in next(marks):
                    namespace["current_action_index"] = index
                    namespace["mark_subtask_cb"]()
                return True

            namespace["replay_episode"] = replay
            annotated = SimpleNamespace(data={"obs": {"datagen_info": {}}})
            env = SimpleNamespace(recorder_manager=SimpleNamespace(get_episode=lambda index: annotated))
            episode = SimpleNamespace(data={"actions": torch.empty(count, 0)})
            assert namespace["annotate_episode_in_manual_mode"](env, episode, None, signals)
            for arm_signals in signals.values():
                for group, name in arm_signals:
                    actual = annotated.data["obs"]["datagen_info"][group][name]
                    # Automatic task flags may toggle; manual marks stay true. Both must locate the same boundary.
                    assert actual.shape == (count,) and actual.dtype == torch.bool
                    assert DataGenInfoPool._find_first_rising_edge(actual, signal_name=name) == (
                        DataGenInfoPool._find_first_rising_edge(expected[group][name][:], signal_name=name)
                    )
