# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Replayer: executes retargeting :class:`~.provider.Plan` objects on the target robot.

Holds the execution context (target env + adapters + resolved retargeting artifacts) and pulls plans
from a :class:`~.provider.PlanProvider` until it is done, driving each on the target robot's IK and
recording the rollout. Knows nothing about *where* plans come from -- full-replay of a dataset today, a
scene generator tomorrow -- so the two concerns stay separate. Supports single-env (with the per-step
tracking-error report) and parallel multi-env replay.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

from .parallel_replay import run_parallel_replay
from .provider import PlanProvider, ReplayResult
from .replay import ReplayContext, format_eef_errors, replay_episode_on_target


@dataclass
class Replayer:
    """Executes plans on the target embodiment. Built once per run from the resolved run context."""

    env: Any
    ctx: ReplayContext
    generation_policy: Any
    output_file: str

    def _ref_label(self) -> str:
        return (
            "source executed path (eef_pose)"
            if self.ctx.config.reference_pose == "eef_pose"
            else "commanded ideal (target_eef_pose)"
        )

    def replay_plan(self, plan) -> ReplayResult:
        """Replay one plan on the target env (single-env), record it, and return its outcome."""
        succeeded, eef_errors = replay_episode_on_target(self.ctx, self.env, plan.episode)
        # The recorder's export mode (from the generation policy's keep_failed flag) decides whether a
        # failed replay is written; always report the outcome to it.
        env_ids = torch.tensor([0], device=self.env.device)
        self.env.recorder_manager.set_success_to_episodes(
            env_ids, torch.tensor([[succeeded]], dtype=torch.bool, device=self.env.device)
        )
        self.env.recorder_manager.export_episodes(env_ids)
        return ReplayResult(success=succeeded, eef_errors=eef_errors)

    def _run_single(self, provider: PlanProvider, simulation_app: Any) -> int:
        """Pull plans from ``provider`` and replay them one at a time until it is done."""
        ref_label = self._ref_label()
        num_success = 0
        num_processed = 0
        all_eef_errors: list[dict[str, dict[str, torch.Tensor]]] = []
        with torch.inference_mode():
            while True:
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break
                plan = provider.next()
                if plan is None:
                    break
                print(f"\nRetargeting example #{plan.index} ({plan.name})")
                result = self.replay_plan(plan)
                provider.observe(plan, result)
                num_processed += 1
                num_success += int(result.success)
                all_eef_errors.append(result.eef_errors)
                print(
                    f"\t{'Succeeded' if result.success else 'Failed'}. Tracking error "
                    f"(object while carried, else EEF vs {ref_label}):"
                )
                print(format_eef_errors(result.eef_errors))

        rate = 100 * num_success / num_processed if num_processed > 0 else 0.0
        print(
            f"\nRetargeted {num_processed} example{'s' if num_processed != 1 else ''} "
            f"({num_success} successful, {rate:.1f}%) to {self.output_file}."
        )
        if all_eef_errors:
            eef_names = list(all_eef_errors[0].keys())
            overall = {
                eef_name: {key: torch.cat([demo[eef_name][key] for demo in all_eef_errors]) for key in ("pos", "rot")}
                for eef_name in eef_names
            }
            print(f"Overall tracking error (object while carried, else EEF vs {ref_label}, all examples):")
            print(format_eef_errors(overall))
        return num_success

    def _run_parallel(self, provider: PlanProvider, num_envs: int) -> int:
        """Replay plans across ``num_envs`` workers pulling from ``provider``."""
        results = run_parallel_replay(self.ctx, self.env, provider, num_envs, self.generation_policy)
        n_success = sum(1 for _, ok in results if ok)
        rate = 100 * n_success / len(results) if results else 0.0
        print(
            f"\nRetargeted {len(results)} example{'s' if len(results) != 1 else ''} "
            f"({n_success} successful, {rate:.1f}%) to {self.output_file} across {num_envs} envs."
        )
        return n_success

    def run(self, provider: PlanProvider, num_envs: int, simulation_app: Any) -> int:
        """Drive ``provider`` to completion on the target env; return the number of successful replays."""
        if num_envs > 1:
            return self._run_parallel(provider, num_envs)
        return self._run_single(provider, simulation_app)
