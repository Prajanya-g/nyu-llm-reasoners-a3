#!/bin/bash
#SBATCH --job-name=sft_tune
#SBATCH --partition=c24m170-a100-2
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:2
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --output=/scratch/pg2963/sft_runs/tune_%j.log
#
# Copy or edit paths for your cluster user. Submit: sbatch scripts/sft_tune.slurm
# Effective batch = batch_size × gradient_accumulation_steps (default 2×4 = 8).

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${LR:=1e-5}"
REPO_ROOT="${SFT_REPO_ROOT:-/scratch/pg2963/nyu-llm-reasoners-a3}"
OUT_ROOT="${SFT_OUTPUT_ROOT:-/scratch/pg2963/sft_runs}"

cd "${REPO_ROOT}" || exit 1
uv run python -m student.sft train \
  --output-root "${OUT_ROOT}" \
  --wandb-project intellect-sft \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --lr "${LR}" \
  --n-sft-steps 100 \
  --eval-every 10 \
  --batch-size 2 \
  --gradient-accumulation-steps 4
