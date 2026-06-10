#!/bin/bash
# ExCoT Step 2.3 — job 3 of 3 : CoT-DPO (7B, GPU, NO vLLM)
# ---------------------------------------------------------------------------
# DPO on the on-policy CoT pairs, ON TOP OF the CoT-SFT adapter. chosen/rejected
# are full reasoning chains -> the model learns to prefer correct reasoning.
# Plain trl DPOTrainer (no vLLM). beta 0.05, 1 epoch (anti-overfit guardrails).
# Output is a NEW dir (round 1); does not touch the CoT-SFT adapter.
# Run after build-pairs wrote bird_cot_pairs_7b.json. Expected ~1-2h on H200.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-cotdpo-train-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cotdpo_train_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cotdpo_train_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

pip install -q "trl==0.12.2" "transformers==4.46.3" 2>&1 | tail -3

BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
COT_SFT_ADAPTER=/scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter   # DPO on top of CoT-SFT
PAIRS=/scratch/phalle.y/results_cot_dpo_7b/bird_cot_pairs_7b.json
OUT=/scratch/phalle.y/bird_cot_dpo_r1_7b                                  # round-1 output

PYTHONUNBUFFERED=1 python -m src.bird.dpo_train \
    --pairs-file  "$PAIRS" \
    --sft-adapter "$COT_SFT_ADAPTER" \
    --output-dir  "$OUT" \
    --base-model  "$BASE_MODEL" \
    --cutoff-len  8192 \
    --beta        0.05 \
    --epochs      1 \
    --save-steps  50
