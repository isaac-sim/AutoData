# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level orchestration: resolve config, build the env, replay every episode."""

import gymnasium as gym
import numpy as np
import random
import torch
from copy import deepcopy

from isaaclab.utils.datasets import HDF5DatasetFileHandler

from autodata_interfaces.embodiments import embodiment_adapter_from_yaml
from autodata_interfaces.env import setup_env_config, setup_output_paths
from autodata_interfaces.tasks.generation_policy_spec import GenerationPolicy

from .config import RetargetConfig
from .eef_offset import build_eef_offsets, compose_retarget_eef_offsets
from .gripper_retargeting import build_passthrough_remapper, load_hand_postures
from .provider import DatasetReplayProvider
from .replay import resolve_eef_reference_links, validate_eef_agreement
from .replayer import Replayer


def _resolve_retarget_params(args) -> RetargetConfig:
    """Resolve the task/pair retargeting parameters from ``--retarget_config`` or the individual flags."""
    if args.retarget_config:
        return RetargetConfig.from_yaml(args.retarget_config)
    missing = [
        name for name in ("source_embodiment", "target_embodiment", "target_env_name") if getattr(args, name) is None
    ]
    assert not missing, f"Without --retarget_config, these flags are required: {missing}"
    # The retargeting knobs (hand policy, replay_speed, frame reconciliation, reference_pose,
    # eef_reference_link, subtasks/object tracking, write_datagen_info, ...) are descriptor-only; a flag-only
    # run takes their RetargetConfig defaults (empty subtasks -> no segmentation / object tracking).
    return RetargetConfig(
        source_embodiment=args.source_embodiment,
        target_embodiment=args.target_embodiment,
        target_env_name=args.target_env_name,
    )


def run(args, simulation_app) -> int:
    """Retarget every source episode onto the target embodiment and export the results."""
    output_dir, output_file_name = setup_output_paths(args.output_file)
    config = _resolve_retarget_params(args)

    # Generation policy is now built straight from the (self-contained) retarget config -- the retargeting
    # path only reads seed and keep_failed (success_settle_steps is read directly from the config);
    # num_trials/guarantee_success are superseded by the provider's stop condition
    # (--target_successes / --target_runs).
    generation_policy = GenerationPolicy(
        name=config.name,
        seed=config.seed,
        keep_failed=args.keep_failed,
    )

    # Adapters. The source adapter only decodes recorded actions (no env needed); the target adapter
    # encodes actions against the live target env and is bound after the env is created.
    source_adapter = embodiment_adapter_from_yaml(config.source_embodiment)
    target_adapter = embodiment_adapter_from_yaml(config.target_embodiment)
    validate_eef_agreement(source_adapter, target_adapter, config.eef_name_map)
    eef_names = list(target_adapter.get_eef_names())
    # Source hand_open/hand_close postures (keyed by the target EEF names), reused for both the hand
    # policy and, under object_tracking="auto", the gripper-closed gate in carry detection.
    source_hand_postures = load_hand_postures(config.source_embodiment, eef_names)
    remap_passthrough = build_passthrough_remapper(
        source_adapter,
        target_adapter,
        hand_policy=config.hand_policy,
        hand_interp_norm=config.hand_interp_norm,
        hand_binary_close_threshold=config.hand_binary_close_threshold,
        hand_interp_band=config.hand_interp_band,
        joint_mapping=config.joint_mapping,
        source_hand_postures=source_hand_postures,
        target_hand_postures=load_hand_postures(config.target_embodiment, eef_names),
        target_channel_defaults=config.target_channel_defaults,
        eef_name_map=config.eef_name_map,
    )
    device = torch.device(args.device)
    # EEF frame reconciliation: the descriptor's inline ``eef_offsets`` if given, else composed from each
    # embodiment's own eef_offset (the general per-robot design).
    if config.eef_offsets:
        eef_offsets = build_eef_offsets(config.eef_offsets, eef_names, device)
    else:
        eef_offsets = compose_retarget_eef_offsets(
            config.source_embodiment, config.target_embodiment, eef_names, device
        )

    num_envs = getattr(args, "num_envs", 1)
    env_cfg, success_term = setup_env_config(
        env_name=config.target_env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args.device,
        generation_policy_params=generation_policy,
    )

    # Compensate the target env's (auto-detected) arm-action scale *in the encoded action* rather than
    # changing the env: the adapter emits (delta / scale) so the env re-applying its own (unchanged) scale
    # realizes the full commanded delta each step. The source per-step delta was physically achieved, so it
    # is reachable in one step; without this the Franka IK-Rel scale=0.5 halves it and the robot lags. The
    # compensation lives in the action, so the recorded action stays correct for the standard env. Only
    # single-arm delta-pose adapters carry the ``action_scale`` field (bimanual/whole-body left untouched).
    arm_action = getattr(env_cfg.actions, "arm_action", None)
    action_scale = getattr(arm_action, "scale", None)
    if isinstance(action_scale, (int, float)) and hasattr(target_adapter, "action_scale"):
        target_adapter.action_scale = float(action_scale)
        print(f"[retarget] compensating target arm-action scale={float(action_scale)} in the encoded action")

    env = gym.make(config.target_env_name, cfg=env_cfg).unwrapped

    random.seed(generation_policy.seed)
    np.random.seed(generation_policy.seed)
    torch.manual_seed(generation_policy.seed)

    try:
        target_adapter.bind_env(env)
        env.reset()
        # Snapshot the target's home scene state once; each replay restores its robot from here
        # (deepcopy so later sim steps cannot mutate the cached tensors).
        target_default_state = deepcopy(env.scene.get_state(is_relative=True))

        # Resolve the controlled-link reference reconstruction once (needs the env's action term to
        # infer links when set to "controlled"); None (default) leaves the observed eef_pose in place.
        eef_reference_link = resolve_eef_reference_links(target_adapter, config.eef_reference_link)
        if eef_reference_link is not None:
            where = "matched target_eef_pose per episode" if isinstance(eef_reference_link, str) else eef_reference_link
            print(
                f"[retarget] reconstructing eef_pose reference at controlled links ({where}); "
                "tracking error is measured there too."
            )

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args.input_file)
        episode_names = list(dataset_file_handler.get_episode_names())
        if len(episode_names) == 0:
            print("No episodes found in the source dataset.")
            return 0

        # Optionally retarget only a chosen subset of source episodes (by index into the dataset).
        select_episodes = getattr(args, "select_episodes", None) or []
        if select_episodes:
            out_of_range = [i for i in select_episodes if not 0 <= i < len(episode_names)]
            assert (
                not out_of_range
            ), f"--select_episodes {out_of_range} out of range; the source dataset has {len(episode_names)} episodes."
            episode_names = [episode_names[i] for i in select_episodes]
            print(f"[retarget] replaying {len(episode_names)} selected episode(s): {select_episodes}")

        # Provider: hands out examples and owns the stop condition (target successes / runs / exhaust).
        # Copy retargeting replays each recorded source demo once on the target robot.
        provider = DatasetReplayProvider(
            dataset_file_handler,
            episode_names,
            env.device,
            target_successes=getattr(args, "target_successes", None),
            target_runs=getattr(args, "target_runs", None),
        )
        # Replayer: executes each plan on the target robot (single-env or parallel).
        replayer = Replayer(
            env=env,
            source_adapter=source_adapter,
            target_adapter=target_adapter,
            remap_passthrough=remap_passthrough,
            config=config,
            generation_policy=generation_policy,
            success_term=success_term,
            robot_asset_name=getattr(target_adapter, "robot_asset_name", "robot"),
            target_default_state=target_default_state,
            eef_offsets=eef_offsets,
            eef_reference_link=eef_reference_link,
            source_hand_postures=source_hand_postures,
            output_file=args.output_file,
        )
        return replayer.run(provider, num_envs, simulation_app)
    finally:
        env.close()
