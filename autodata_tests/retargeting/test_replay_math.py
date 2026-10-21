# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the retargeting replay math: subtask spans, offsets, and EEF-offset composition.

These exercise the tensor math directly (no env, no GPU) so they can run pre-merge instead of only in the
GPU nightly -- the risky logic here (offset timelines, carry-segment offset propagation, span boundary
resolution) is pure function of its inputs and does not need a live simulation.
"""

import torch

from autodata_retargeting.config import parse_subtask
from autodata_retargeting.eef_offset import build_eef_offsets, compose_retarget_eef_offsets
from autodata_retargeting.object_tracking import iter_subtask_spans
from autodata_retargeting.parallel_replay import _DrainBarrier
from autodata_retargeting.replay import apply_subtask_offsets
from autodata_retargeting.trajectory import _link_state_quat_order


def _eye_poses(num_steps: int) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).repeat(num_steps, 1, 1)


# ---------------------------------------------------------------------------
# iter_subtask_spans
# ---------------------------------------------------------------------------


def test_iter_subtask_spans_fixed_length_and_last_subtask_runs_to_end():
    subtasks = {
        "right": [
            parse_subtask({"subtask_end": {"method": "fixed_length", "length": 5}}),
            parse_subtask({}),
        ]
    }
    spans = list(iter_subtask_spans(subtasks, gripper_closed={}, signals={}, num_steps=20))
    assert [(s[4], s[5]) for s in spans] == [(0, 5), (5, 19)]
    assert spans[0][6] == 5  # resolved boundary
    assert spans[1][6] is None  # last subtask has no boundary


def test_iter_subtask_spans_gripper_close_resolves_crossing_step():
    frac = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    subtasks = {"right": [parse_subtask({"subtask_end": "gripper_close"}), parse_subtask({})]}
    spans = list(iter_subtask_spans(subtasks, gripper_closed={"right": frac}, signals={}, num_steps=6))
    assert spans[0][4:6] == (0, 3)  # ends where closedness first crosses 0.5


# ---------------------------------------------------------------------------
# apply_subtask_offsets
# ---------------------------------------------------------------------------


def test_apply_subtask_offsets_is_identity_without_offsets():
    subtasks = {"right": [parse_subtask({})]}
    poses = {"right": _eye_poses(4)}
    out_poses, out_objects = apply_subtask_offsets(poses, subtasks, {}, {}, 4, None, 0.5, {})
    assert torch.allclose(out_poses["right"], poses["right"])
    assert out_objects == {}


def test_apply_subtask_offsets_world_frame_translates_eef_path():
    subtasks = {"right": [parse_subtask({"offset": {"translation": [0.1, 0.0, 0.0]}})]}
    poses = {"right": _eye_poses(4)}
    out_poses, _ = apply_subtask_offsets(poses, subtasks, {}, {}, 4, None, 0.5, {})
    assert torch.allclose(out_poses["right"][:, :3, 3], torch.tensor([0.1, 0.0, 0.0]).expand(4, 3))
    # The input dict is untouched (copy-on-write).
    assert torch.allclose(poses["right"], _eye_poses(4))


def test_apply_subtask_offsets_propagates_to_tracked_object():
    """Regression test: a subtask offset on a tracked (carried) object must reach the object's own
    trajectory, not just the (carry-overridden) EEF path -- see replay.apply_subtask_offsets docstring."""
    subtasks = {
        "right": [
            parse_subtask({
                "object_tracking": "cube",
                "offset": {"translation": [0.0, 0.0, 0.02], "frame": "world"},
            })
        ]
    }
    poses = {"right": _eye_poses(4)}
    source_objects = {"cube": _eye_poses(4)}
    out_poses, out_objects = apply_subtask_offsets(poses, subtasks, {}, {}, 4, None, 0.5, source_objects)
    assert torch.allclose(out_objects["cube"][:, :3, 3], torch.tensor([0.0, 0.0, 0.02]).expand(4, 3))
    # The EEF path is still carried through too (the override later replaces it during the carry).
    assert torch.allclose(out_poses["right"][:, :3, 3], torch.tensor([0.0, 0.0, 0.02]).expand(4, 3))
    # The input object dict is untouched (copy-on-write).
    assert torch.allclose(source_objects["cube"], _eye_poses(4))


def test_apply_subtask_offsets_leaves_untracked_objects_alone():
    subtasks = {"right": [parse_subtask({"object_tracking": "cube", "offset": {"translation": [0, 0, 0.02]}})]}
    poses = {"right": _eye_poses(4)}
    source_objects = {"cube": _eye_poses(4), "other": _eye_poses(4)}
    _, out_objects = apply_subtask_offsets(poses, subtasks, {}, {}, 4, None, 0.5, source_objects)
    assert torch.allclose(out_objects["other"], _eye_poses(4))
    assert out_objects["other"] is source_objects["other"]  # untouched entries are not copied


# ---------------------------------------------------------------------------
# compose_retarget_eef_offsets / build_eef_offsets
# ---------------------------------------------------------------------------


def test_build_eef_offsets_translates_in_rotated_frame():
    import math

    offsets = build_eef_offsets(
        {"right": {"axis_angle": [0.0, 0.0, math.pi / 2], "translation": [1.0, 0.0, 0.0]}},
        ["right"],
        torch.device("cpu"),
    )
    # A +90deg yaw rotates the canonical +x translation axis onto world +y.
    assert torch.allclose(offsets["right"][:3, 3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_compose_retarget_eef_offsets_none_without_declared_offsets(tmp_path):
    src = tmp_path / "src.yaml"
    tgt = tmp_path / "tgt.yaml"
    src.write_text("name: src\n")
    tgt.write_text("name: tgt\n")
    assert compose_retarget_eef_offsets(str(src), str(tgt), ["right"], torch.device("cpu")) is None


def test_compose_retarget_eef_offsets_composes_single_arm_schemas(tmp_path):
    src = tmp_path / "src.yaml"
    tgt = tmp_path / "tgt.yaml"
    src.write_text("eef_offset: [0.1, 0.0, 0.0]\n")
    tgt.write_text("eef_offset: [0.0, 0.0, 0.0]\n")
    composed = compose_retarget_eef_offsets(str(src), str(tgt), ["right"], torch.device("cpu"))
    assert composed is not None
    assert torch.allclose(composed["right"][:3, 3], torch.tensor([0.1, 0.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# trajectory._link_state_quat_order
# ---------------------------------------------------------------------------


def test_link_state_quat_order_selects_legacy_wxyz_to_xyzw_permutation():
    assert _link_state_quat_order(is_legacy_quaternion_format=True) == (1, 2, 3, 0)


def test_link_state_quat_order_is_identity_for_current_xyzw_format():
    assert _link_state_quat_order(is_legacy_quaternion_format=False) == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# parallel_replay._DrainBarrier
# ---------------------------------------------------------------------------


def test_drain_barrier_all_idle_only_once_every_worker_has_gone_idle():
    barrier = _DrainBarrier(num_envs=3)
    assert not barrier.all_idle()
    barrier.mark_idle()
    barrier.mark_idle()
    assert not barrier.all_idle()
    barrier.mark_idle()
    assert barrier.all_idle()
