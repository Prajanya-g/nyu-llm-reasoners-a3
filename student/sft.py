"""Supervised fine-tuning utilities (prompt/response tokenization for causal LM)."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompts and outputs separately, concatenate, build ``response_mask``.

    At LM position ``j`` the target is ``full[j + 1]``. The mask is 1 when that token is in
    the output span (SFT loss on responses only); 0 for prompt targets and padding.

    Args:
        prompt_strs: Batch of prompts.
        output_strs: Batch of responses (same length as ``prompt_strs``).
        tokenizer: HF tokenizer (e.g. ``Qwen/Qwen2.5-Math-1.5B`` per §4.1).

    Returns:
        ``input_ids``: (batch, max(prompt_and_output_lens) - 1), ``full[:-1]``, padded.
        ``labels``: same shape, ``full[1:]``, pad value ``-100``.
        ``response_mask``: same shape, 1 on response label positions, else 0.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must set pad_token_id or eos_token_id")

    full_sequences: list[list[int]] = []
    prompt_token_lens: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
        full_sequences.append(prompt_ids + output_ids)
        prompt_token_lens.append(len(prompt_ids))

    lengths = [len(s) for s in full_sequences]
    max_full = max(lengths)
    width = max_full - 1

    batch = len(full_sequences)
    input_ids = torch.full((batch, width), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, width), fill_value=-100, dtype=torch.long)
    response_mask = torch.zeros((batch, width), dtype=torch.long)

    for row, full, p_tokens in zip(range(batch), full_sequences, prompt_token_lens):
        seq_len = len(full)
        if seq_len < 2:
            continue
        content_len = seq_len - 1
        inp = full[:-1]
        lab = full[1:]
        input_ids[row, :content_len] = torch.tensor(inp, dtype=torch.long)
        labels[row, :content_len] = torch.tensor(lab, dtype=torch.long)
        start = max(p_tokens - 1, 0)
        response_mask[row, start:content_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
