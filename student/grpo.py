"""GRPO-style helpers (group rewards, policy objectives)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatch to naive PG (reward or advantage) or GRPO-Clip; return per-token loss + metadata.

    Args:
        policy_log_probs: ``(batch_size, sequence_length)``.
        loss_type: Which objective to use.
        raw_rewards: ``(batch_size, 1)``, required for ``no_baseline``.
        advantages: ``(batch_size, 1)``, required for ``reinforce_with_baseline`` and ``grpo_clip``.
        old_log_probs: ``(batch_size, sequence_length)``, required for ``grpo_clip``.
        cliprange: PPO/GRPO ratio clip ε, required for ``grpo_clip``.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required when loss_type is 'no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages is required when loss_type is 'reinforce_with_baseline'"
            )
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    if loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required when loss_type is 'grpo_clip'")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when loss_type is 'grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required when loss_type is 'grpo_clip'")
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    raise ValueError(f"unknown loss_type: {loss_type!r}")


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked sum of per-token PG loss with grad-accum scaling (matches course snapshots).

    Let :math:`S = \\sum(\\text{mask} \\cdot \\ell)` and :math:`g` =
    ``gradient_accumulation_steps``. We call ``backward`` on :math:`S / (g^3 + g^4)`
    and return :math:`S / g^4` for logging (mirrors the SFT microbatch split used in
    this assignment’s snapshots). Gradients accumulate across repeated calls unless
    the caller zeros ``policy_log_probs.grad``.

    Args:
        policy_log_probs: ``(B, T)`` trainable log-probs.
        response_mask: ``(B, T)``; 1 on response tokens to optimize.
        gradient_accumulation_steps: Microbatches per optimizer step (squared in divisor).
        loss_type: PG variant (see :func:`compute_policy_gradient_loss`).
        raw_rewards / advantages / old_log_probs / cliprange: Passed through when required.

    Returns:
        Scalar ``loss`` (same tensor used for ``backward``) and ``metadata``.
    """
    token_loss, pg_meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    gas = float(gradient_accumulation_steps)
    sum_pg = (token_loss * mask).sum()
    denom_backward = gas**3 + gas**4
    denom_log = gas**4
    loss_backward = sum_pg / denom_backward
    loss_log = sum_pg / denom_log

    masked_pg_sum = sum_pg.detach()
    metadata: dict[str, Tensor] = {
        "masked_pg_sum": masked_pg_sum.detach(),
        "num_response_tokens": mask.sum().detach(),
    }
    for key, value in pg_meta.items():
        metadata[key] = value.detach() if isinstance(value, Tensor) else value

    loss_backward.backward()
    return loss_log, metadata


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
