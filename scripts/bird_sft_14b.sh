#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=09:00:00          # ~5-6h for 9,428 examples × 3 epochs on 14B
#SBATCH --job-name=bird-sft-14b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_sft_14b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_sft_14b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.bird.sft_train \
    --sft-data   /scratch/phalle.y/bird_sft_data.json \
    --output-dir /scratch/phalle.y/bird_sft_adapter_14b \
    --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --epochs     3 \
    --cutoff-len 8192 \
    --save-steps 100
