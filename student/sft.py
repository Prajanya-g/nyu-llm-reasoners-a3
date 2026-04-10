"""Supervised fine-tuning utilities (prompt/response tokenization for causal LM)."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import typer
from datasets import Dataset, load_from_disk
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


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
    *,
    for_training: bool = False,
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
        for_training: If True, run a differentiable forward (no ``no_grad``); leave the model in
            training mode. If False (default), match unit-test behavior (eval + no grad).

    Returns:
        ``log_probs`` shape ``(batch, seq)``. If ``return_token_entropy``, also ``token_entropy``
        with the same shape.
    """
    device = next(model.parameters()).device
    x = input_ids.to(device)
    y = labels.to(device)

    if for_training:
        logits = model(x).logits
    else:
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


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked NLL over the response mask, ``backward`` on the same scalar that is returned.

    Loss / backward graph: :math:`\\sum (\\text{mask} \\cdot -\\log p) / (\\text{normalize_constant}
    \\cdot \\text{gradient_accumulation_steps}^2)`. Gradients accumulate across repeated calls
    unless the caller zeros ``policy_log_probs.grad`` (the 10-step test relies on accumulation
    and repeated references to the same ``.grad`` tensor).

    Args:
        policy_log_probs: Per-step log-probabilities of the labeled token, shape ``(B, T)``.
        response_mask: Same shape; 1 for supervised response positions.
        gradient_accumulation_steps: Microbatches per optimizer step (squared in the divisor).
        normalize_constant: Divisor on the masked sum (before accumulation scaling).

    Returns:
        Scalar ``loss`` (for logging) and ``metadata`` tensors.
    """
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    token_nll = -policy_log_probs
    gas = float(gradient_accumulation_steps)
    denom = normalize_constant * gas * gas
    loss = masked_normalize(token_nll, mask, denom, dim=None)

    masked_nll_sum = (token_nll * mask).sum()
    metadata: dict[str, Tensor] = {
        "masked_nll_sum": masked_nll_sum.detach(),
        "num_response_tokens": mask.sum().detach(),
    }

    loss.backward()
    return loss, metadata


# --- Intellect SFT dataset & batching -------------------------------------------------


def _messages_to_prompt_and_output(messages: list[dict[str, str]]) -> tuple[str, str]:
    sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
    return prompt, assistant_msg


def load_prime_intellect(
    train_path: str | Path,
    max_examples: int | None,
) -> Dataset:
    """Load Prime Intellect-style SFT data (``messages`` + ``ground_truth``) from disk."""
    path = Path(train_path)
    ds: Dataset = load_from_disk(str(path))
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    return ds


def sample_batch(dataset: Dataset, batch_size: int, *, generator: torch.Generator | None = None) -> dict[str, list[str]]:
    """Sample a batch of (prompt, assistant output) string pairs."""
    n = len(dataset)
    if n == 0:
        raise ValueError("dataset is empty")
    idx = torch.randint(0, n, (batch_size,), generator=generator)
    prompts: list[str] = []
    outputs: list[str] = []
    for i in idx.tolist():
        row = dataset[int(i)]
        p, o = _messages_to_prompt_and_output(row["messages"])
        prompts.append(p)
        outputs.append(o)
    return {"prompts": prompts, "outputs": outputs}


def get_microbatches(
    batch: dict[str, list[str]],
    gradient_accumulation_steps: int,
) -> Iterator[dict[str, list[str]]]:
    """Split a batch into up to ``gradient_accumulation_steps`` contiguous microbatches."""
    prompts = batch["prompts"]
    outputs = batch["outputs"]
    n = len(prompts)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    n_mb = min(gradient_accumulation_steps, max(n, 1))
    chunk = (n + n_mb - 1) // n_mb
    start = 0
    while start < n:
        end = min(start + chunk, n)
        yield {"prompts": prompts[start:end], "outputs": outputs[start:end]}
        start = end


# --- vLLM eval ------------------------------------------------------------------------


def init_vllm(model_id: str, device: str | None = None, seed: int = 42, **kwargs: Any) -> Any:
    """Construct a vLLM ``LLM`` for periodic eval. Extra kwargs are forwarded to ``LLM(...)``.

    Note:
        Pinning vLLM to a specific GPU (e.g. ``cuda:1``) while training on ``cuda:0`` is
        environment-dependent; set ``CUDA_VISIBLE_DEVICES`` appropriately or pass vLLM-supported
        engine args via ``kwargs``.
    """
    from vllm import LLM

    if device is not None:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    llm_kw: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": seed,
        "gpu_memory_utilization": kwargs.pop("gpu_memory_utilization", 0.9),
        **kwargs,
    }
    llm = LLM(model=model_id, **llm_kw)
    reload_kw = {k: v for k, v in llm_kw.items()}
    setattr(llm, "_sft_vllm_reload_kw", reload_kw)
    return llm


def load_policy_into_vllm_instance(
    policy: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    llm: Any,
) -> Any:
    """Save the HF policy to a temp dir and rebuild vLLM from those weights.

    Returns the new ``LLM`` instance; callers must reassign their handle, e.g.
    ``llm = load_policy_into_vllm_instance(policy, tokenizer, llm)``.
    """
    from vllm import LLM

    reload_kw: dict[str, Any] = getattr(llm, "_sft_vllm_reload_kw", {"trust_remote_code": True, "dtype": "bfloat16"})
    tmp = Path(tempfile.mkdtemp(prefix="sft_vllm_ckpt_"))
    policy.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    new_llm = LLM(model=str(tmp), **reload_kw)
    setattr(new_llm, "_sft_vllm_reload_kw", reload_kw)
    return new_llm


def evaluate_on_math_val(
    llm: Any,
    tokenizer: PreTrainedTokenizerBase,
    val_path: str | Path,
    max_examples: int | None = 500,
) -> float:
    """Intellect-style validation accuracy using :func:`question_only_reward_fn`."""
    from vllm import SamplingParams

    from .drgrpo_grader import question_only_reward_fn

    _ = tokenizer
    ds = load_from_disk(str(val_path))
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    prompts: list[str] = []
    gts: list[str] = []
    for ex in ds:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)
    correct = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        correct += int(question_only_reward_fn(text, gts[i])["reward"])
    return correct / max(len(outputs), 1)


# --- Full training loop & CLI ---------------------------------------------------------


def run_sft_training_run(
    *,
    model_id: str,
    train_path: Path,
    val_path: Path,
    output_root: Path,
    max_train_examples: int | None,
    n_sft_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    eval_every: int,
    max_eval_examples: int | None,
    policy_device: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    seed: int,
) -> None:
    """One training run for a fixed training subset size (``max_train_examples``)."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    device = torch.device(policy_device)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    torch.ones(1).to(policy_device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("GPU warmed up", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_prime_intellect(train_path, max_train_examples)
    n_tag = "full" if max_train_examples is None else str(max_train_examples)
    save_dir = output_root / f"sft_n{n_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    llm = init_vllm(model_id, seed=seed)

    wb: Any = None
    if use_wandb:
        import wandb as wb_mod

        wb = wb_mod
        wb.init(
            project=wandb_project,
            name=wandb_run_name or f"sft_n{n_tag}",
            config={
                "model_id": model_id,
                "max_train_examples": max_train_examples,
                "n_sft_steps": n_sft_steps,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lr": lr,
                "eval_every": eval_every,
            },
            reinit=True,
        )
        wb.define_metric("train_step")
        wb.define_metric("eval_step")
        wb.define_metric("train/*", step_metric="train_step")
        wb.define_metric("eval/*", step_metric="eval_step")

    train_step = 0
    policy.train()
    use_amp = device.type == "cuda"

    n_examples = "full" if max_train_examples is None else max_train_examples
    print(f"Starting SFT run: n_examples={n_examples}, lr={lr}", flush=True)

    try:
        for _step in range(n_sft_steps):
            optimizer.zero_grad(set_to_none=True)
            batch = sample_batch(dataset, batch_size, generator=gen)

            for micro in get_microbatches(batch, gradient_accumulation_steps):
                tokens = tokenize_prompt_and_output(
                    micro["prompts"], micro["outputs"], tokenizer
                )
                input_ids = tokens["input_ids"].to(device)
                labels = tokens["labels"].to(device)
                mask = tokens["response_mask"].to(device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    log_probs = get_response_log_probs(
                        policy,
                        input_ids,
                        labels,
                        for_training=True,
                    )["log_probs"]
                loss, _meta = sft_microbatch_train_step(
                    log_probs,
                    mask,
                    gradient_accumulation_steps,
                )
                if wb is not None and wb.run is not None:
                    wb.log({"train/loss": float(loss.detach()), "train_step": train_step})

            print(f"[step {train_step}] loss={float(loss.detach()):.4f}", flush=True)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            train_step += 1

            if train_step % eval_every == 0:
                policy.eval()
                llm = load_policy_into_vllm_instance(policy, tokenizer, llm)
                acc = evaluate_on_math_val(
                    llm, tokenizer, val_path, max_examples=max_eval_examples
                )
                print(f"[eval step {train_step}] accuracy={acc:.4f}", flush=True)
                if wb is not None and wb.run is not None:
                    wb.log({"eval/accuracy": acc, "eval_step": train_step})
                policy.train()
    except Exception as e:
        print(f"CRASH: {e}", flush=True)
        raise

    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if wb is not None and wb.run is not None:
        wb.finish()


app = typer.Typer(add_completion=False, help="Intellect SFT training (5 dataset sizes + wandb).")


@app.command()
def train(
    model_id: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", help="HF model id or local path"),
    train_path: Path = typer.Option(
        Path("data-distrib/intellect_math/train"),
        help="HuggingFace ``datasets`` disk path (train)",
    ),
    val_path: Path = typer.Option(
        Path("data-distrib/intellect_math/dev"),
        help="Validation set path (Intellect dev)",
    ),
    output_root: Path = typer.Option(
        Path(os.environ.get("SFT_OUTPUT_ROOT", "./sft_checkpoints")),
        help="Directory for per-run ``sft_n*`` saves",
    ),
    n_sft_steps: int = typer.Option(100, help="Optimizer steps per dataset-size experiment"),
    batch_size: int = typer.Option(4, help="Samples per optimizer step (before microbatch split)"),
    gradient_accumulation_steps: int = typer.Option(2, help="Microbatches per optimizer step"),
    lr: float = typer.Option(1e-5, help="AdamW learning rate"),
    eval_every: int = typer.Option(10, help="Run vLLM eval every N train steps"),
    policy_device: str = typer.Option("cuda:0", help="Device for HF policy training"),
    max_eval_examples: int = typer.Option(500, help="Cap validation examples per eval"),
    seed: int = typer.Option(42),
    wandb_project: str = typer.Option("intellect-sft", help="Weights & Biases project"),
    no_wandb: bool = typer.Option(False, help="Disable wandb logging"),
) -> None:
    """Run five experiments with N ∈ {128, 256, 512, 1024, full train set}."""
    dataset_sizes: tuple[int | None, ...] = (128, 256, 512, 1024, None)
    for n in dataset_sizes:
        run_sft_training_run(
            model_id=model_id,
            train_path=train_path,
            val_path=val_path,
            output_root=output_root,
            max_train_examples=n,
            n_sft_steps=n_sft_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=lr,
            eval_every=eval_every,
            max_eval_examples=max_eval_examples,
            policy_device=policy_device,
            use_wandb=not no_wandb,
            wandb_project=wandb_project,
            wandb_run_name=None,
            seed=seed,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
