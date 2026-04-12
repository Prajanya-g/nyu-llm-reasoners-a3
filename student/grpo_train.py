"""Countdown GRPO training loop (rollouts + group-normalized policy gradient)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import typer
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from student.countdown_reward import countdown_reward_fn
from student.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from student.sft import (
    _configure_run_logging,
    get_response_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
)

log = logging.getLogger(__name__)

def _normalize_completion_for_reward(text: str) -> str:
    """If generation stopped at ``</answer>``, vLLM may omit the closing tag from the string."""
    lower = text.lower()
    if "<answer" in lower and "</answer>" not in lower:
        return text + "</answer>"
    return text


def load_countdown_prompt_template() -> str:
    path = Path(__file__).resolve().parent / "prompts" / "countdown.prompt"
    return path.read_text()


def format_countdown_prompt(template: str, nums: list[int], target: int) -> str:
    question = (
        f"Using the numbers in the list {list(nums)}, create an equation that equals {target}."
    )
    return template.replace("{question}", question)


def load_countdown_split(path: str | Path, max_examples: int | None) -> Dataset:
    ds: Dataset = load_from_disk(str(path))
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    return ds


def sample_prompt_indices(
    n_prompts: int,
    dataset_len: int,
    generator: torch.Generator,
) -> list[int]:
    if n_prompts > dataset_len:
        raise ValueError("n_prompts exceeds dataset size")
    perm = torch.randperm(dataset_len, generator=generator)[:n_prompts]
    return perm.tolist()


def build_rollout_batch(
    ds: Dataset,
    indices: list[int],
    group_size: int,
    template: str,
) -> tuple[list[str], list[str]]:
    """Expand ``len(indices)`` prompts each ``group_size`` times; parallel JSON ground truths."""
    prompts: list[str] = []
    gts: list[str] = []
    for idx in indices:
        row = ds[int(idx)]
        nums = list(row["nums"])
        target = int(row["target"])
        gt = json.dumps({"numbers": nums, "target": target})
        p = format_countdown_prompt(template, nums, target)
        for _ in range(group_size):
            prompts.append(p)
            gts.append(gt)
    return prompts, gts


def evaluate_countdown_val(
    llm: Any,
    ds: Dataset,
    template: str,
    tokenizer: PreTrainedTokenizerBase,
    max_examples: int,
    *,
    max_tokens: int,
) -> dict[str, float]:
    """Mean reward / format / answer on a subset of the validation set (greedy)."""
    from vllm import SamplingParams

    _ = tokenizer
    n = min(max_examples, len(ds))
    prompts: list[str] = []
    gts: list[str] = []
    for i in range(n):
        row = ds[i]
        nums = list(row["nums"])
        target = int(row["target"])
        prompts.append(format_countdown_prompt(template, nums, target))
        gts.append(json.dumps({"numbers": nums, "target": target}))
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
    )
    outputs = llm.generate(prompts, params)
    rewards: list[float] = []
    fmt: list[float] = []
    ans: list[float] = []
    for i, out in enumerate(outputs):
        text = _normalize_completion_for_reward(out.outputs[0].text)
        r = countdown_reward_fn(text, gts[i])
        rewards.append(float(r["reward"]))
        fmt.append(float(r["format_reward"]))
        ans.append(float(r["answer_reward"]))
    m = max(len(rewards), 1)
    return {
        "val/mean_reward": sum(rewards) / m,
        "val/mean_format_reward": sum(fmt) / m,
        "val/mean_answer_reward": sum(ans) / m,
    }


def run_grpo_training(
    *,
    model_id: str,
    train_path: Path,
    val_path: Path,
    output_root: Path,
    n_grpo_steps: int,
    learning_rate: float,
    advantage_eps: float,
    rollout_batch_size: int,
    group_size: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    epochs_per_rollout_batch: int,
    sampling_temperature: float,
    sampling_max_tokens: int,
    gpu_memory_utilization: float,
    loss_type: str,
    use_std_normalization: bool,
    policy_device: str,
    vllm_device: str,
    eval_every: int,
    max_val_examples: int,
    log_rollout_every: int,
    cliprange: float | None,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    max_train_examples: int | None,
) -> None:
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    assert rollout_batch_size % micro_train_batch_size == 0, (
        "rollout_batch_size must be divisible by micro_train_batch_size "
        f"({rollout_batch_size=} {micro_train_batch_size=})"
    )
    n_microbatches = rollout_batch_size // micro_train_batch_size
    assert n_microbatches == gradient_accumulation_steps, (
        "Require rollout_batch_size // micro_train_batch_size == gradient_accumulation_steps "
        f"({n_microbatches=} vs {gradient_accumulation_steps=}); "
        "adjust train_batch_size so micro_train_batch_size * GAS == rollout_batch_size."
    )
    assert train_batch_size == rollout_batch_size, (
        "This loop assumes one rollout batch fills one optimizer step: "
        f"set train_batch_size == rollout_batch_size (got {train_batch_size=} {rollout_batch_size=})."
    )
    if loss_type not in ("no_baseline", "reinforce_with_baseline", "grpo_clip"):
        raise ValueError(f"unknown loss_type: {loss_type!r}")
    if loss_type == "grpo_clip":
        raise NotImplementedError(
            "This trainer is on-policy only: use loss_type=reinforce_with_baseline (or no_baseline). "
            "grpo_clip needs per-token old_log_probs from the behavior policy at rollout time."
        )

    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    device = torch.device(policy_device)
    template = load_countdown_prompt_template()

    train_ds = load_countdown_split(train_path, max_train_examples)
    val_ds = load_countdown_split(val_path, None)

    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    policy.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    plot_path = output_root / "grpo_val_rewards.png"
    rollout_log_path = output_root / "grpo_rollouts.jsonl"
    metrics_path = output_root / "grpo_metrics.jsonl"

    if device.type == "cuda":
        torch.cuda.synchronize()

    log.info("Initializing vLLM on %s (gpu_memory_utilization=%.2f)", vllm_device, gpu_memory_utilization)
    llm = init_vllm(
        model_id,
        device=vllm_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    load_policy_into_vllm_instance(policy, llm)

    wb: Any = None
    if use_wandb:
        import wandb as wb_mod

        wb = wb_mod
        wb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_id": model_id,
                "n_grpo_steps": n_grpo_steps,
                "rollout_batch_size": rollout_batch_size,
                "group_size": group_size,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lr": learning_rate,
                "loss_type": loss_type,
                "use_std_normalization": use_std_normalization,
            },
            reinit=True,
        )
        wb.define_metric("train_step")
        wb.define_metric("eval_step")
        wb.define_metric("train/*", step_metric="train_step")
        wb.define_metric("eval/*", step_metric="eval_step")

    from vllm import SamplingParams

    train_step = 0
    val_steps: list[int] = []
    val_rewards: list[float] = []
    policy.train()
    use_amp = device.type == "cuda"

    try:
        for outer in range(n_grpo_steps):
            load_policy_into_vllm_instance(policy, llm)

            idxs = sample_prompt_indices(
                n_prompts_per_rollout_batch,
                len(train_ds),
                gen,
            )
            prompts, repeated_gts = build_rollout_batch(train_ds, idxs, group_size, template)

            sp = SamplingParams(
                temperature=float(sampling_temperature),
                max_tokens=int(sampling_max_tokens),
                stop=["</answer>"],
            )
            t0 = time.perf_counter()
            vout = llm.generate(prompts, sp)
            completions_raw = [o.outputs[0].text for o in vout]
            completions = [_normalize_completion_for_reward(t) for t in completions_raw]
            log.info("rollout | step=%d generate_s=%.1f", outer + 1, time.perf_counter() - t0)

            advantages_t, raw_t, reward_meta = compute_group_normalized_rewards(
                countdown_reward_fn,
                completions,
                repeated_gts,
                group_size,
                advantage_eps,
                use_std_normalization,
            )
            advantages_2d = advantages_t.unsqueeze(1).to(device=device, dtype=torch.float32)
            raw_2d = raw_t.unsqueeze(1).to(device=device, dtype=torch.float32)

            train_reward_mean = float(raw_t.mean().item())
            train_fmt_mean = reward_meta["mean_format_reward"]
            train_ans_mean = reward_meta["mean_answer_reward"]

            for _epoch in range(epochs_per_rollout_batch):
                optimizer.zero_grad(set_to_none=True)
                last_loss: torch.Tensor | None = None
                last_meta: dict[str, torch.Tensor] = {}
                last_entropy: float | None = None
                grad_norm: float = 0.0

                for m in range(n_microbatches):
                    lo = m * micro_train_batch_size
                    hi = lo + micro_train_batch_size
                    mb_prompts = prompts[lo:hi]
                    mb_out = completions[lo:hi]
                    tokens = tokenize_prompt_and_output(mb_prompts, mb_out, tokenizer)
                    input_ids = tokens["input_ids"].to(device)
                    labels = tokens["labels"].to(device)
                    mask = tokens["response_mask"].to(device)

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        rp = get_response_log_probs(
                            policy,
                            input_ids,
                            labels,
                            return_token_entropy=True,
                            for_training=True,
                        )
                        log_probs = rp["log_probs"]
                        ent = rp["token_entropy"]
                    adv_mb = advantages_2d[lo:hi].to(dtype=log_probs.dtype)
                    raw_mb = raw_2d[lo:hi].to(dtype=log_probs.dtype)

                    last_loss, last_meta = grpo_microbatch_train_step(
                        log_probs,
                        mask,
                        gradient_accumulation_steps,
                        loss_type,
                        raw_rewards=raw_mb if loss_type == "no_baseline" else None,
                        advantages=adv_mb if loss_type != "no_baseline" else None,
                        old_log_probs=None,
                        cliprange=cliprange if loss_type == "grpo_clip" else None,
                    )
                    ent_f = ent.to(dtype=torch.float32)
                    m_f = mask.to(dtype=torch.float32)
                    denom = torch.clamp(m_f.sum(), min=1.0)
                    last_entropy = float((ent_f * m_f).sum().item() / denom.item())

                assert last_loss is not None
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
                )
                optimizer.step()
                train_step += 1

                log.info(
                    "train | grpo_step=%d train_step=%d loss=%.6f grad_norm=%.4f "
                    "entropy=%s reward_mean=%.4f fmt=%.4f ans=%.4f raw_mean=%.4f",
                    outer + 1,
                    train_step,
                    float(last_loss.detach()),
                    grad_norm,
                    f"{last_entropy:.4f}" if last_entropy is not None else "n/a",
                    train_reward_mean,
                    train_fmt_mean,
                    train_ans_mean,
                    reward_meta.get("raw_mean", 0.0),
                )
                if wb is not None and wb.run is not None:
                    log_payload: dict[str, Any] = {
                        "train/loss": float(last_loss.detach()),
                        "train/grad_norm": grad_norm,
                        "train/mean_reward": train_reward_mean,
                        "train/mean_format_reward": train_fmt_mean,
                        "train/mean_answer_reward": train_ans_mean,
                        "train_step": train_step,
                    }
                    if last_entropy is not None:
                        log_payload["train/token_entropy"] = last_entropy
                    if "clipped" in last_meta and "ratio" in last_meta:
                        msum = mask.sum().clamp(min=1)
                        cf = (last_meta["clipped"] * mask.to(last_meta["clipped"].dtype)).sum() / msum
                        log_payload["train/clip_fraction"] = float(cf.detach())
                    wb.log(log_payload)

                with metrics_path.open("a", encoding="utf-8") as mf:
                    mf.write(
                        json.dumps(
                            {
                                "train_step": train_step,
                                "grpo_step": outer + 1,
                                "loss": float(last_loss.detach()),
                                "grad_norm": grad_norm,
                                "mean_reward": train_reward_mean,
                                "reward_meta": {k: float(v) for k, v in reward_meta.items()},
                            }
                        )
                        + "\n"
                    )

            if log_rollout_every > 0 and (outer + 1) % log_rollout_every == 0:
                with rollout_log_path.open("a", encoding="utf-8") as rf:
                    for j in range(min(3, len(completions))):
                        rf.write(
                            json.dumps(
                                {
                                    "train_step": train_step,
                                    "prompt_tail": prompts[j][-200:],
                                    "completion": completions[j][:2000],
                                    "ground_truth": repeated_gts[j],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            if eval_every > 0 and train_step % eval_every == 0:
                policy.eval()
                load_policy_into_vllm_instance(policy, llm)
                ev = evaluate_countdown_val(
                    llm,
                    val_ds,
                    template,
                    tokenizer,
                    max_val_examples,
                    max_tokens=sampling_max_tokens,
                )
                log.info(
                    "eval | train_step=%d mean_reward=%.4f format=%.4f answer=%.4f",
                    train_step,
                    ev["val/mean_reward"],
                    ev["val/mean_format_reward"],
                    ev["val/mean_answer_reward"],
                )
                val_steps.append(train_step)
                val_rewards.append(ev["val/mean_reward"])
                if wb is not None and wb.run is not None:
                    wb.log({**ev, "eval_step": train_step})
                policy.train()

    except Exception:
        log.exception("GRPO training crashed")
        raise

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if val_steps:
            plt.figure(figsize=(8, 4))
            plt.plot(val_steps, val_rewards, marker="o")
            plt.xlabel("train_step")
            plt.ylabel("validation mean reward")
            plt.title("Countdown GRPO — validation reward")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            log.info("Saved validation reward plot to %s", plot_path)
        else:
            log.warning("No eval points recorded; skip plot (eval_every=%s)", eval_every)
    except Exception:
        log.exception("Could not save matplotlib plot")

    save_dir = output_root / "grpo_final"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info("Saved policy and tokenizer to %s", save_dir)
    if wb is not None and wb.run is not None:
        wb.finish()


app = typer.Typer(add_completion=False, help="Countdown GRPO training.")


@app.command()
def train(
    model_id: str = typer.Option("Qwen/Qwen2.5-Math-1.5B"),
    train_path: Path = typer.Option(Path("data-distrib/countdown/dataset/train")),
    val_path: Path = typer.Option(Path("data-distrib/countdown/dataset/dev")),
    output_root: Path = typer.Option(Path(os.environ.get("GRPO_OUTPUT_ROOT", "./grpo_checkpoints"))),
    n_grpo_steps: int = typer.Option(200),
    learning_rate: float = typer.Option(1e-5),
    advantage_eps: float = typer.Option(1e-6),
    rollout_batch_size: int = typer.Option(16),
    group_size: int = typer.Option(8),
    train_batch_size: int = typer.Option(16),
    gradient_accumulation_steps: int = typer.Option(4),
    epochs_per_rollout_batch: int = typer.Option(1),
    sampling_temperature: float = typer.Option(0.7),
    sampling_max_tokens: int = typer.Option(1024),
    gpu_memory_utilization: float = typer.Option(0.8),
    loss_type: str = typer.Option("reinforce_with_baseline"),
    use_std_normalization: bool = typer.Option(True),
    policy_device: str = typer.Option("cuda:0"),
    vllm_device: str = typer.Option("cuda:1"),
    eval_every: int = typer.Option(10),
    max_val_examples: int = typer.Option(256),
    log_rollout_every: int = typer.Option(10),
    cliprange: float | None = typer.Option(None),
    max_train_examples: int | None = typer.Option(None),
    seed: int = typer.Option(42),
    wandb_project: str = typer.Option("countdown-grpo"),
    wandb_run_name: str | None = typer.Option(None),
    no_wandb: bool = typer.Option(False),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Train with GRPO on Countdown (on-policy defaults: train_batch_size == rollout_batch_size).

    The handout pair ``train_batch_size=64, gradient_accumulation_steps=128`` is inconsistent
    (64 % 128 != 0). Use ``train_batch_size == rollout_batch_size`` and choose ``GAS`` so
    ``rollout_batch_size % (train_batch_size // GAS) == 0`` and
    ``rollout_batch_size // (train_batch_size // GAS) == GAS`` (default: 16 / 4 = 4 microbatches, GAS=4).
    """
    _configure_run_logging(log_level)
    run_grpo_training(
        model_id=model_id,
        train_path=train_path,
        val_path=val_path,
        output_root=output_root,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        sampling_temperature=sampling_temperature,
        sampling_max_tokens=sampling_max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        policy_device=policy_device,
        vllm_device=vllm_device,
        eval_every=eval_every,
        max_val_examples=max_val_examples,
        log_rollout_every=log_rollout_every,
        cliprange=cliprange,
        seed=seed,
        use_wandb=not no_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        max_train_examples=max_train_examples,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
