#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # H200 not strictly required for eval, any GPU ≥24GB
#SBATCH --time=07:00:00          # 14B inference is slower than 7B
#SBATCH --job-name=bird-eval-14b-sft
#SBATCH --mem=70GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_14b_sft_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_14b_sft_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Evaluate 14B SFT-only adapter on BIRD dev (baseline before DPO)
PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter    /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --dev-json   /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir     /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir /scratch/phalle.y/results_14b_sft_eval
