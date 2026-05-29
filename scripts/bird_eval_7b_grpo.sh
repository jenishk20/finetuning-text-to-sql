#!/bin/bash
# Eval a 7B GRPO adapter on BIRD dev (with Best-of-N self-consistency)
# Usage: edit ADAPTER_DIR to point at the GRPO output you want to eval

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-eval-grpo-7b
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_grpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_grpo_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" 2>&1 | tail -3

# ── CHANGE THIS to eval the variant you want ─────────────────────────────────
# Options:
#   /scratch/phalle.y/bird_grpo_adapter_7b_from_sft/final_adapter
#   /scratch/phalle.y/bird_grpo_adapter_7b_from_dpo/final_adapter
ADAPTER_DIR=/scratch/phalle.y/bird_grpo_adapter_7b_from_dpo
LATEST_CKPT=$(ls -d $ADAPTER_DIR/final_adapter 2>/dev/null || ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Using GRPO checkpoint: $LATEST_CKPT"

PYTHONUNBUFFERED=1 python -m src.bird.eval_best_of_n \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter     "$LATEST_CKPT" \
    --dev-json    /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir      /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir  /scratch/phalle.y/results_7b_grpo_eval \
    --k           4 \
    --temperature 0.8
