#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00          # ~1h for 500 mini-dev questions on H200
#SBATCH --job-name=bird-minidev
#SBATCH --mem=70GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_minidev_%j.out
#SBATCH --error=/scratch/phalle.y/bird_minidev_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Run BIRD DPO adapter on the 500 mini-dev questions
PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --adapter    /scratch/phalle.y/bird_frontier_dpo_adapter/final_adapter \
    --dev-json   /home/phalle.y/Jenish-DPO-GRPO/data_minidev/MINIDEV/mini_dev_sqlite.json \
    --db-dir     /home/phalle.y/Jenish-DPO-GRPO/data_minidev/MINIDEV/dev_databases \
    --output-dir /scratch/phalle.y/results_bird_minidev
