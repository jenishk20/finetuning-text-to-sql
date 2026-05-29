#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-dpo-7b
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_dpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_dpo_7b_%j.err

# Stage 2 of the clean BIRD pipeline: offline DPO on top of the 7B SFT adapter
# (output of bird_sft_7b.sh) using strong-teacher preference pairs.
# Filter pairs to clear_preference only (one right, one wrong) — both-correct /
# judge pairs regressed accuracy in earlier runs.

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

# Redirect all caches to scratch — home dir has small quota
export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.dpo_train \
    --pairs-file  /scratch/phalle.y/results_frontier_pairs/bird_frontier_dpo_data.json \
    --sft-adapter /scratch/phalle.y/bird_sft_adapter_7b/final_adapter \
    --output-dir  /scratch/phalle.y/bird_dpo_adapter_7b \
    --cutoff-len  8192 \
    --beta        0.05 \
    --epochs      1 \
    --save-steps  50