#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00          # ~2-3h for 1,219 pairs on 14B
#SBATCH --job-name=bird-dpo-14b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_dpo_14b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_dpo_14b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# DPO on top of 14B SFT adapter, using the same 1,219 clear-preference pairs
# that gave the 7B model its 50.3% on BIRD dev
PYTHONUNBUFFERED=1 python -m src.bird.dpo_train \
    --pairs-file  /scratch/phalle.y/results_frontier_pairs/bird_frontier_dpo_data.json \
    --sft-adapter /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --output-dir  /scratch/phalle.y/bird_dpo_adapter_14b \
    --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \
    --cutoff-len  8192 \
    --beta        0.05 \
    --epochs      1 \
    --save-steps  50
