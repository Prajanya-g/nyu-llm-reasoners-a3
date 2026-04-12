"""GRPO-style helpers (group rewards, policy objectives)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, Any]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Score rollouts; mean-center each group; optionally scale by group std + eps.

    Groups are contiguous ``group_size`` blocks along the rollout batch.

    Args:
        reward_fn: ``(response, ground_truth)`` -> dict with ``"reward"``.
        rollout_responses: Length ``rollout_batch_size``.
        repeated_ground_truths: Same length (GT repeated per group member).
        group_size: Rollouts per prompt.
        advantage_eps: Added to std when ``normalize_by_std``.
        normalize_by_std: Unbiased within-group std vs mean-only centering.

    Returns:
        Advantages, raw rewards, metadata (each rollout is one element).
    """
    n = len(rollout_responses)
    if len(repeated_ground_truths) != n:
        raise ValueError("repeated_ground_truths must match rollout_responses length")
    if group_size <= 0 or n % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")

    raw_list = [
        float(reward_fn(resp, gt)["reward"])
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw = torch.tensor(raw_list, dtype=torch.float32)
    groups = raw.view(-1, group_size)

    mean = groups.mean(dim=1, keepdim=True)
    centered = groups - mean

    if normalize_by_std:
        std = groups.std(dim=1, unbiased=True, keepdim=True)
        advantages_g = centered / (std + float(advantage_eps))
    else:
        advantages_g = centered

    advantages = advantages_g.reshape(-1)

    metadata: dict[str, float] = {
        "raw_mean": float(raw.mean().item()),
        "raw_std": float(raw.std(unbiased=False).item()),
        "raw_min": float(raw.min().item()),
        "raw_max": float(raw.max().item()),
        "n_groups": float(n // group_size),
        "group_size": float(group_size),
    }

    return advantages, raw, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clipped surrogate per token: ``-min(r*A, clip(r)*A)`` with ``A`` broadcast on seq.

    Args:
        advantages: ``(batch_size, 1)``.
        policy_log_probs / old_log_probs: ``(batch_size, sequence_length)``.
        cliprange: Clips ``ratio = exp(new - old)`` to ``[1-ε, 1+ε]``.

    Returns:
        Per-token loss and metadata (``clipped`` mask: clipped surrogate strictly lower).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    adv = advantages.to(dtype=policy_log_probs.dtype)
    surr_unclipped = ratio * adv
    ratio_clipped = torch.clamp(
        ratio,
        min=1.0 - float(cliprange),
        max=1.0 + float(cliprange),
    )
    surr_clipped = ratio_clipped * adv
    loss = -torch.min(surr_unclipped, surr_clipped)
    clipped = (surr_clipped < surr_unclipped).to(dtype=policy_log_probs.dtype)
    metadata: dict[str, Tensor] = {
        "clipped": clipped,
        "ratio": ratio.detach(),
    }
    return loss, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """Per-token policy-gradient loss: negate broadcast product of scalar reward and log-prob.

    Minimizing the mean of this tensor implements the usual REINFORCE objective
    ``-A * log pi`` at each sequence position (``A`` is constant over ``sequence_length``).

    Args:
        raw_rewards_or_advantages: Shape ``(batch_size, 1)`` reward or advantage per rollout.
        policy_log_probs: Shape ``(batch_size, sequence_length)``.

    Returns:
        Shape ``(batch_size, sequence_length)``.
    """
    return -(raw_rewards_or_advantages.to(policy_log_probs.dtype) * policy_log_probs)
