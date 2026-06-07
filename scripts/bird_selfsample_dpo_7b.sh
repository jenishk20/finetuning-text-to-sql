#!/bin/bash
# ExCoT Stage 1 — job 2 of 3 : DPO TRAINING (7B)
# ---------------------------------------------------------------------------
# Train DPO on the self-sampled pairs, on top of the 7B SFT adapter.
# NO vLLM here — this is plain TRL DPOTrainer, the robust part of the pipeline.
# Run ONLY after job 1 finished and bird_ss_pairs_7b.json exists & is non-empty.
#
# Guardrails from your past failures: beta 0.05 (stay near SFT base),
# 1 epoch (avoid the overfit that regressed you before).
# Output goes to a NEW dir so your 50.3% bird_dpo_adapter_7b is never touched.
# Expected: ~1-2h on H200.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-ss-dpo-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_ss_dpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_ss_dpo_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
SFT_ADAPTER=/scratch/phalle.y/bird_sft_adapter_7b/final_adapter   # same as job 1 (verified)
PAIRS=/scratch/phalle.y/results_self_sampling_7b/bird_ss_pairs_7b.json
OUT=/scratch/phalle.y/bird_dpo_adapter_7b_selfsampling   # NEW dir — does not overwrite 50.3% adapter

PYTHONUNBUFFERED=1 python -m src.bird.dpo_train \
    --pairs-file  "$PAIRS" \
    --sft-adapter "$SFT_ADAPTER" \
    --output-dir  "$OUT" \
    --base-model  "$BASE_MODEL" \
    --cutoff-len  8192 \
    --beta        0.05 \
    --epochs      1 \
    --save-steps  50
