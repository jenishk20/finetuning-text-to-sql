#!/bin/bash
# Eval a 7B GRPO adapter on BIRD dev — GREEDY, result-set accuracy.
# Uses src.bird.eval_finetuned (already on this branch): greedy decode, same prompt
# template GRPO trained on, directly comparable to the SFT/DPO baselines.
# No vLLM — runs in the grpo_env training env. Set ADAPTER_DIR, then sbatch.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # any GPU >=24GB works for 4-bit eval
#SBATCH --time=04:00:00          # ~3-4h for the full 1534-question greedy eval
#SBATCH --job-name=bird-eval-grpo-7b
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_grpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_grpo_7b_%j.err

source activate /scratch/phalle.y/grpo_env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# ── CHANGE THIS to eval the variant you want ─────────────────────────────────
#   GRPO quick run:  /scratch/phalle.y/bird_grpo_adapter_7b_quick
#   SFT baseline:    /scratch/phalle.y/bird_sft_adapter_7b      (for a clean A/B)
#   full from-SFT:   /scratch/phalle.y/bird_grpo_adapter_7b_from_sft
#   full from-DPO:   /scratch/phalle.y/bird_grpo_adapter_7b_from_dpo
ADAPTER_DIR=/scratch/phalle.y/bird_grpo_adapter_7b_quick

# pick final_adapter if present, else the highest-numbered checkpoint
LATEST_CKPT=$(ls -d $ADAPTER_DIR/final_adapter 2>/dev/null || ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Evaluating adapter: $LATEST_CKPT"

# Per-adapter output dir so A/B runs don't clobber each other
OUT_DIR=/scratch/phalle.y/results_eval_$(basename "$ADAPTER_DIR")

# Dev set lives on /scratch (NOT /home — /home is not mounted on compute nodes)
PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter    "$LATEST_CKPT" \
    --dev-json   /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
    --db-dir     /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
    --output-dir "$OUT_DIR"
