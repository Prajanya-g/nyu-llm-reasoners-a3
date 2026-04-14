"""Supervised fine-tuning utilities (prompt/response tokenization for causal LM)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from datasets import Dataset, load_from_disk
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)


def _configure_run_logging(level: str) -> None:
    """Console logging for training CLI (idempotent if handlers already exist)."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    if not logging.root.handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logging.getLogger().setLevel(lvl)
    log.setLevel(lvl)


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
    # Never materializes full (B, T, V) probs tensor
    return torch.distributions.Categorical(logits=logits).entropy()


def get_response_log_probs(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
    *,
    for_training: bool = False,
) -> dict[str, Tensor]:
    """Per-token log p(label_t | x_{≤t}) from causal LM logits, optional entropy per step.

    Uses ``model(input_ids).logits`` (shape ``batch × seq × vocab``). Log-probs for the labeled
    token use fused :func:`torch.nn.functional.cross_entropy` on a flattened batch so PyTorch
    never materializes a full ``(batch, seq, vocab)`` ``log_softmax`` tensor (saves VRAM).

    Args:
        model: HF causal LM (already on the intended device).
        input_ids: Context tokens, shape ``(batch, seq)``.
        labels: Target token ids per step, shape ``(batch, seq)``; clamped to
            ``[0, vocab_size - 1]`` where invalid (same as the old gather path).
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

    vocab = logits.size(-1)
    y_flat = y.long().clamp(min=0, max=vocab - 1).view(-1)
    token_log_probs = (
        -F.cross_entropy(
            logits.view(-1, vocab),
            y_flat,
            reduction="none",
        )
    ).view(y.shape)

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


def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """Average selected elements (``mask`` true); reduction follows :meth:`torch.Tensor.mean`.

    Args:
        tensor: Values to average.
        mask: Same shape as ``tensor``; true (or 1) includes entries in the mean.
        dim: Axis to reduce, or ``None`` for one scalar mean over all masked entries.

    Returns:
        ``sum(tensor * mask) / sum(mask)`` with the same reduced shape as ``tensor.mean(dim)``.
    """
    m = mask.to(dtype=tensor.dtype)
    weighted = tensor * m
    if dim is None:
        return weighted.sum() / m.sum()
    return weighted.sum(dim=dim) / m.sum(dim=dim)


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    use_masked_normalize: bool = True,
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
        use_masked_normalize: If ``True``, use :func:`masked_normalize`; else use
            :func:`masked_mean` for the pre-accumulation reduction.

    Returns:
        Scalar ``loss`` (for logging) and ``metadata`` tensors.
    """
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    token_nll = -policy_log_probs
    gas = float(gradient_accumulation_steps)
    if use_masked_normalize:
        denom = normalize_constant * gas * gas
        loss = masked_normalize(token_nll, mask, denom, dim=None)
    else:
        loss = masked_mean(token_nll, mask, dim=None) / (gas * gas)

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


def _vllm_llm_construct(model: str, llm_kw: dict[str, Any]) -> Any:
    """Instantiate vLLM ``LLM`` with patches that avoid profiling / distributed assumptions."""
    from vllm import LLM

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(model=model, **llm_kw)


def init_vllm(
    model_id: str,
    device: str = "cuda:1",
    seed: int = 42,
    gpu_memory_utilization: float = 0.6,
    **kwargs: Any,
) -> Any:
    """Construct a vLLM ``LLM`` for periodic eval on ``device`` (default ``cuda:1``).

    vLLM 0.7.x forwards ``device`` to ``EngineArgs`` so inference can use a different GPU than
    the HF policy (typically policy on ``cuda:0``, vLLM on ``cuda:1``). The ``LLM(...)`` call is
    wrapped in mocks so ``_assert_memory_footprint_increased_during_profiling`` is skipped when
    another process holds most of ``cuda:0``. Extra ``kwargs`` are forwarded to ``LLM(...)``.
    """
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    llm_kw: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": seed,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        **kwargs,
        "device": device,
    }
    llm = _vllm_llm_construct(model_id, llm_kw)
    reload_kw = {k: v for k, v in llm_kw.items()}
    setattr(llm, "_sft_vllm_reload_kw", reload_kw)
    return llm


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: Any) -> Any:
    """Copy HF ``policy`` weights into the existing vLLM engine (no second ``LLM`` allocation).

    Mutates the runner model in place so eval uses updated weights without peak VRAM from
    two concurrent vLLM instances on the same GPU.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    return llm


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
    vllm_device: str,
    vllm_gpu_memory_utilization: float = 0.6,
    use_masked_normalize: bool = True,
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
    policy.gradient_checkpointing_enable()
    log.info("Gradient checkpointing enabled (trades compute for activation memory)")
    torch.ones(1).to(policy_device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log.info("GPU warmed up (policy_device=%s)", policy_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_prime_intellect(train_path, max_train_examples)
    n_tag = "full" if max_train_examples is None else str(max_train_examples)
    save_dir = output_root / f"sft_n{n_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Run config | model_id=%s policy_device=%s vllm_device=%s vllm_gpu_mem=%.3f "
        "train_path=%s val_path=%s n_train=%d n_sft_steps=%d batch_size=%d grad_acc=%d lr=%g "
        "eval_every=%d max_eval_examples=%s save_dir=%s wandb=%s",
        model_id,
        policy_device,
        vllm_device,
        vllm_gpu_memory_utilization,
        train_path,
        val_path,
        len(dataset),
        n_sft_steps,
        batch_size,
        gradient_accumulation_steps,
        lr,
        eval_every,
        max_eval_examples,
        save_dir,
        use_wandb,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        dev_idx = device.index if device.index is not None else 0
        print(
            f"Model memory: {torch.cuda.memory_allocated(dev_idx) / 1e9:.1f}GB",
            flush=True,
        )
    log.info(
        "Initializing vLLM for periodic eval (model_id=%s device=%s)",
        model_id,
        vllm_device,
    )
    t_vllm = time.perf_counter()
    llm = init_vllm(
        model_id,
        device=vllm_device,
        seed=seed,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    log.info("vLLM ready (%.1fs)", time.perf_counter() - t_vllm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 2:
            print(
                f"After vLLM init - cuda:0: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB, "
                f"cuda:1: {torch.cuda.memory_allocated(1) / 1e9:.1f}GB",
                flush=True,
            )
        else:
            print(
                f"After vLLM init - cuda:0: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB "
                f"(only {n_gpu} GPU(s) visible; use 2 GPUs to split policy vs vLLM)",
                flush=True,
            )

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
                "vllm_device": vllm_device,
                "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            },
            reinit=True,
        )
        wb.define_metric("train_step")
        wb.define_metric("eval_step")
        wb.define_metric("train/*", step_metric="train_step")
        wb.define_metric("eval/*", step_metric="eval_step")
        log.info("wandb initialized project=%s run=%s", wandb_project, wandb_run_name or f"sft_n{n_tag}")

    train_step = 0
    policy.train()
    use_amp = device.type == "cuda"

    n_examples = "full" if max_train_examples is None else max_train_examples
    log.info("Starting training loop | n_examples=%s lr=%g total_optimizer_steps=%d", n_examples, lr, n_sft_steps)

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
                    use_masked_normalize=use_masked_normalize,
                )
                if wb is not None and wb.run is not None:
                    wb.log({"train/loss": float(loss.detach()), "train_step": train_step})

            grad_pre_clip = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            train_step += 1
            log.info(
                "train | step=%d/%d train_step=%d loss=%.4f grad_norm=%.4f",
                _step + 1,
                n_sft_steps,
                train_step,
                float(loss.detach()),
                float(grad_pre_clip),
            )

            if train_step % eval_every == 0:
                policy.eval()
                log.info("eval | reloading vLLM weights train_step=%d", train_step)
                t0 = time.perf_counter()
                load_policy_into_vllm_instance(policy, llm)
                log.info("eval | vLLM reload done (%.1fs)", time.perf_counter() - t0)
                t1 = time.perf_counter()
                acc = evaluate_on_math_val(
                    llm, tokenizer, val_path, max_examples=max_eval_examples
                )
                log.info(
                    "eval | train_step=%d accuracy=%.4f (%.1fs)",
                    train_step,
                    acc,
                    time.perf_counter() - t1,
                )
                if wb is not None and wb.run is not None:
                    wb.log({"eval/accuracy": acc, "eval_step": train_step})
                policy.train()
    except Exception:
        log.exception("Training crashed (n_examples=%s)", n_examples)
        raise

    log.info("Saving checkpoint to %s", save_dir)
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info("Saved policy and tokenizer | n_examples=%s", n_examples)
    if wb is not None and wb.run is not None:
        wb.finish()
        log.info("wandb run finished")


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
    batch_size: int = typer.Option(
        2,
        help="Samples per optimizer step (before microbatch split); use 2 with GC if 1.5B OOMs at 4",
    ),
    gradient_accumulation_steps: int = typer.Option(
        4,
        help="Microbatches per optimizer step (e.g. batch_size=2 × 4 ⇒ effective batch 8)",
    ),
    lr: float = typer.Option(1e-5, help="AdamW learning rate"),
    eval_every: int = typer.Option(10, help="Run vLLM eval every N train steps"),
    policy_device: str = typer.Option("cuda:0", help="Device for HF policy training"),
    vllm_device: str = typer.Option(
        "cuda:1",
        help="CUDA device string passed to vLLM EngineArgs (keep off policy GPU, e.g. cuda:1)",
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.6,
        help="vLLM gpu_memory_utilization (fraction of GPU memory for weights + KV cache)",
    ),
    max_eval_examples: int = typer.Option(500, help="Cap validation examples per eval"),
    use_masked_normalize: bool = typer.Option(
        True,
        help="Use masked_normalize for SFT loss reduction (set false to use masked_mean).",
    ),
    seed: int = typer.Option(42),
    wandb_project: str = typer.Option("intellect-sft", help="Weights & Biases project"),
    no_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    log_level: str = typer.Option(
        "INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Run five experiments with N ∈ {128, 256, 512, 1024, full train set}."""
    _configure_run_logging(log_level)
    dataset_sizes: tuple[int | None, ...] = (128, 256, 512, 1024, None)
    log.info(
        "Multi-run job | %d dataset sizes: %s | model_id=%s",
        len(dataset_sizes),
        [x if x is not None else "full" for x in dataset_sizes],
        model_id,
    )
    for i, n in enumerate(dataset_sizes, start=1):
        label = n if n is not None else "full"
        log.info("=== Experiment %d/%d | max_train_examples=%s ===", i, len(dataset_sizes), label)
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
            use_masked_normalize=use_masked_normalize,
            policy_device=policy_device,
            vllm_device=vllm_device,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            use_wandb=not no_wandb,
            wandb_project=wandb_project,
            wandb_run_name=None,
            seed=seed,
        )
    log.info("Multi-run job complete | all %d experiments finished", len(dataset_sizes))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
