#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # H200 not required for eval — any GPU ≥12GB works
#SBATCH --time=04:00:00          # ~3-4h for full 1534-question eval
#SBATCH --job-name=bird-eval
#SBATCH --mem=40GB               # Less than training — no optimizer states
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

# Redirect all caches to scratch — home dir has small quota
export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.eval_finetuned \
    --adapter    /scratch/phalle.y/bird_frontier_dpo_adapter_4684pairs/final_adapter \
    --dev-json   /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir     /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir /scratch/phalle.y/results_finetuned_4684pairs
