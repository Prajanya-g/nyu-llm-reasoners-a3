"""Minimal evaluation script for MATH and Intellect test sets."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths):
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)
    correct = 0
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]
        results.append(
            {
                "prompt": prompts[i][-200:],  # last 200 chars to save space
                "output": text,
                "ground_truth": ground_truths[i],
                "reward": reward["reward"],
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
            }
        )
    # Print category counts
    cat1 = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 1)
    cat2 = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 0)
    cat3 = sum(1 for r in results if r["format_reward"] == 0 and r["answer_reward"] == 0)
    print(f"  Cat1 (format=1, answer=1): {cat1}")
    print(f"  Cat2 (format=1, answer=0): {cat2}")
    print(f"  Cat3 (format=0, answer=0): {cat3}")
    return correct / len(outputs), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument(
        "--intellect-path",
        default="data-distrib/intellect_math/test",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample] {prompts[0][:200]}...")
    acc, results = evaluate(llm, prompts, gts)
    print(f"Intellect Accuracy: {acc:.4f}")
    with open("intellect_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")
    acc, results = evaluate(llm, prompts, gts)
    print(f"MATH Accuracy: {acc:.4f}")
    with open("math_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
