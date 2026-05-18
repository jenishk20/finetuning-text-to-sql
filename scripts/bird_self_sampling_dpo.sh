#!/bin/bash
# Pipeline 2 — job 2 of 3
# DPO training on self-sampled preference pairs on top of 14B SFT adapter
# Run only after bird_self_sampling_generate.sh completes successfully

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-selfsample-dpo
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_selfsample_dpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_selfsample_dpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.bird.dpo_train \
    --pairs-file  /scratch/phalle.y/results_self_sampling/bird_self_sampling_pairs.json \
    --sft-adapter /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --output-dir  /scratch/phalle.y/bird_dpo_adapter_14b_selfsampling \
    --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \
    --cutoff-len  8192 \
    --beta        0.1 \
    --epochs      1 \
    --save-steps  50
