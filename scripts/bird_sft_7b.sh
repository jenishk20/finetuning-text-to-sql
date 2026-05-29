#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-sft-7b
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_sft_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_sft_7b_%j.err

# Stage 1 of the clean BIRD pipeline: SFT a fresh 7B LoRA adapter on BIRD
# train gold SQL. Produces the adapter that bird_dpo_7b.sh trains DPO on top of.
# Model is pinned here (the trainer itself is model-agnostic).

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

# Redirect all caches to scratch — home dir has small quota
export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.sft_train \
    --sft-data    /scratch/phalle.y/bird_sft_data.json \
    --output-dir  /scratch/phalle.y/bird_sft_adapter_7b \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs      3 \
    --lr          2e-4 \
    --cutoff-len  8192 \
    --save-steps  100