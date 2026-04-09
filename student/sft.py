"""Supervised fine-tuning utilities (prompt/response tokenization for causal LM)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompts and outputs separately, concatenate, build ``response_mask``.

    Prompt uses ``add_special_tokens=True`` (matches course snapshot / HF SFT). Output uses
    ``add_special_tokens=False``. If concat produces two consecutive EOS ids, drop one so
    labels do not repeat EOS where the snapshot expects a pad slot.

    Args:
        prompt_strs: Batch of prompts.
        output_strs: Batch of responses (same length as ``prompt_strs``).
        tokenizer: HF tokenizer (e.g. Qwen2.5-Math-1.5B).

    Returns:
        ``input_ids``, ``labels``, ``response_mask`` per assignment spec.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    full_sequences: list[list[int]] = []
    prompt_token_lens: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
        full = prompt_ids + output_ids
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and len(full) >= 2 and full[-1] == eos_id and full[-2] == eos_id:
            full = full[:-1]
        full_sequences.append(full)
        prompt_token_lens.append(len(prompt_ids))

    max_len = max(len(f) for f in full_sequences)
    width = max_len - 1

    batch = len(full_sequences)
    input_ids = torch.empty((batch, width), dtype=torch.long)
    labels = torch.empty((batch, width), dtype=torch.long)
    response_mask = torch.zeros((batch, width), dtype=torch.long)

    for row, full, p_len in zip(range(batch), full_sequences, prompt_token_lens):
        padded = full + [pad_token_id] * (max_len - len(full))
        inp = padded[:-1]
        lab = padded[1:]
        input_ids[row] = torch.tensor(inp, dtype=torch.long)
        labels[row] = torch.tensor(lab, dtype=torch.long)
        if len(full) >= 2:
            start = max(p_len - 1, 0)
            response_mask[row, start : len(full) - 1] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Per-position Shannon entropy of next-token distribution (over vocab), in nats.

    Uses ``logsumexp`` for stable normalization and :func:`torch.special.entr` so ``0 log 0``
    terms do not produce NaNs.

    Args:
        logits: Unnormalized log-probabilities, shape ``(batch, seq, vocab)``.

    Returns:
        Entropy ``-(p * log(p)).sum()`` over vocab at each position, shape ``(batch, seq)``.
    """
    log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(logits - log_norm)
    return torch.special.entr(probs).sum(dim=-1)


def get_response_log_probs(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Per-token log p(label_t | x_{≤t}) from causal LM logits, optional entropy per step.

    Uses ``model(input_ids).logits`` (shape ``batch × seq × vocab``). At time step ``t`` the
    row ``logits[:, t]`` is the next-token distribution after consuming ``input_ids[:, : t+1]``;
    the assignment ``labels`` tensor supplies the target id at each position (same length as
    ``input_ids``). Negative labels (e.g. padding ignored in loss) are clamped for ``gather``.

    Args:
        model: HF causal LM (already on the intended device).
        input_ids: Context tokens, shape ``(batch, seq)``.
        labels: Target token ids aligned with each step, shape ``(batch, seq)``.
        return_token_entropy: If True, include ``token_entropy`` via :func:`compute_entropy`.

    Returns:
        ``log_probs`` shape ``(batch, seq)``. If ``return_token_entropy``, also ``token_entropy``
        with the same shape.
    """
    device = next(model.parameters()).device
    x = input_ids.to(device)
    y = labels.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(x).logits
    finally:
        if was_training:
            model.train()

    log_probs_v = torch.log_softmax(logits, dim=-1)
    vocab = logits.size(-1)
    idx = y.long().clamp(min=0, max=vocab - 1)
    token_log_probs = log_probs_v.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    out: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum selected elements (``mask == 1``) and divide by ``normalize_constant``.

    Args:
        tensor: Values to sum.
        mask: Same shape as ``tensor``; ones select entries, zeros are ignored.
        normalize_constant: Divisor applied after summing.
        dim: Dimension to sum along, or ``None`` to sum every element.

    Returns:
        Normalized sum tensor with summed dimensions removed when ``dim`` is set.
    """
    weighted = tensor * mask.to(dtype=tensor.dtype)
    if dim is None:
        return weighted.sum() / normalize_constant
    return weighted.sum(dim=dim) / normalize_constant
