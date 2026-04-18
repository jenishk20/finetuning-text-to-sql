#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-dpo
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_dpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_dpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

# Redirect all caches to scratch — home dir has small quota
export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.dpo_train \
    --pairs-file /scratch/phalle.y/results_frontier_pairs/bird_frontier_dpo_data.json \
    --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
    --output-dir /scratch/phalle.y/bird_frontier_dpo_adapter \
    --cutoff-len 8192 \
    --beta 0.05 \
    --epochs 1 \
    --save-steps 50
