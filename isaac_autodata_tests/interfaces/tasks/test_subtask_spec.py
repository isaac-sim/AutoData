# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaac_autodata_interfaces.tasks.subtask_spec`."""

import pytest

from isaac_autodata_interfaces.tasks.subtask_spec import (
    ALGO_PARAMS_REGISTRY,
    DexMimicGenSubtaskAlgoParams,
    MimicGenSubtaskAlgoParams,
    SkillGenSubtaskAlgoParams,
    Subtask,
    SubtaskAlgoParams,
)


def test_subtask_defaults():
    st = Subtask(algo_params=MimicGenSubtaskAlgoParams())
    assert st.object_ref == ""
    assert st.description == ""
    assert st.subtask_start_signal == ""
    assert st.subtask_term_signal == ""
    assert st.selection_strategy == "random"
    assert st.selection_strategy_kwargs == {}
    assert st.first_subtask_start_offset_range == (0, 0)
    assert st.subtask_term_offset_range == (0, 0)
    assert st.action_noise == 0.0
    assert st.num_interpolation_steps == 0
    assert st.num_fixed_steps == 0
    assert st.apply_noise_during_interpolation is False
    assert isinstance(st.algo_params, MimicGenSubtaskAlgoParams)


def test_subtask_requires_algo_params():
    # algo_params is declared with a MISSING default, so it has no usable default.
    with pytest.raises(TypeError):
        Subtask()


def test_subtask_is_keyword_only():
    # kw_only=True: positional construction is rejected.
    with pytest.raises(TypeError):
        Subtask("cube_1", algo_params=MimicGenSubtaskAlgoParams())  # type: ignore[misc]


def test_subtask_kwargs_default_factory_is_isolated():
    a = Subtask(algo_params=MimicGenSubtaskAlgoParams())
    b = Subtask(algo_params=MimicGenSubtaskAlgoParams())
    a.selection_strategy_kwargs["nn_k"] = 3
    assert b.selection_strategy_kwargs == {}, "default_factory dict must not be shared across instances"


def test_subtask_field_values_round_trip():
    algo = SkillGenSubtaskAlgoParams(subtask_start_offset_range=(1, 2))
    st = Subtask(
        object_ref="cube_2",
        description="grasp the red cube",
        subtask_start_signal="start_1",
        subtask_term_signal="grasp_1",
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        first_subtask_start_offset_range=(0, 5),
        subtask_term_offset_range=(-2, 2),
        action_noise=0.03,
        num_interpolation_steps=4,
        num_fixed_steps=1,
        apply_noise_during_interpolation=True,
        algo_params=algo,
    )
    assert st.object_ref == "cube_2"
    assert st.selection_strategy_kwargs == {"nn_k": 3}
    assert st.subtask_term_offset_range == (-2, 2)
    assert st.apply_noise_during_interpolation is True
    assert st.algo_params is algo
    assert algo.subtask_start_offset_range == (1, 2)


def test_algo_params_registry_contents():
    assert ALGO_PARAMS_REGISTRY == {
        "mimicgen": MimicGenSubtaskAlgoParams,
        "dexmimicgen": DexMimicGenSubtaskAlgoParams,
        "skillgen": SkillGenSubtaskAlgoParams,
    }
    for cls in ALGO_PARAMS_REGISTRY.values():
        assert issubclass(cls, SubtaskAlgoParams)


def test_skillgen_algo_params_default_and_override():
    assert SkillGenSubtaskAlgoParams().subtask_start_offset_range == (0, 0)
    assert SkillGenSubtaskAlgoParams(subtask_start_offset_range=(1, 3)).subtask_start_offset_range == (1, 3)


@pytest.mark.parametrize("cls", [MimicGenSubtaskAlgoParams, DexMimicGenSubtaskAlgoParams])
def test_plain_algo_params_have_no_extra_fields(cls):
    # MimicGen / DexMimicGen carry no params beyond the shared ones on Subtask.
    from dataclasses import fields

    assert fields(cls) == ()
