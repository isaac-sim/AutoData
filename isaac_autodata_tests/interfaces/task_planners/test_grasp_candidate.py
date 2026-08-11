# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaac_autodata_core.transform import IDENTITY_MATRIX4
from isaac_autodata_interfaces.task_planners import GraspCandidateSet


def _translated(x: float):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_grasp_candidate_set_normalizes_finite_candidates_and_scores() -> None:
    candidates = GraspCandidateSet(
        method="graspgen",
        object_id="mug",
        link_from_object=(IDENTITY_MATRIX4, _translated(0.1)),
        scores=(0.2, 0.8),
    )

    assert len(candidates) == 2
    assert candidates.scores == (0.2, 0.8)
    assert candidates.to_dict()["object_id"] == "mug"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "auto"}, "method"),
        ({"object_id": ""}, "object_id"),
        ({"link_from_object": ()}, "must not be empty"),
        ({"link_from_object": (IDENTITY_MATRIX4, IDENTITY_MATRIX4)}, "unique"),
        ({"scores": (0.1, 0.2)}, "one-to-one"),
        ({"scores": (True,)}, "numbers"),
        ({"scores": (float("nan"),)}, "finite"),
    ],
)
def test_grasp_candidate_set_rejects_invalid_values(kwargs, message: str) -> None:
    values = {
        "method": "analytical",
        "object_id": "cube",
        "link_from_object": (IDENTITY_MATRIX4,),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        GraspCandidateSet(**values)
