#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00          # finish the last ~12% of training
#SBATCH --job-name=bird-sft-14b-resume
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_sft_14b_resume_%j.out
#SBATCH --error=/scratch/phalle.y/bird_sft_14b_resume_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Auto-detect latest checkpoint in the output dir
ADAPTER_DIR=/scratch/phalle.y/bird_sft_adapter_14b
LATEST_CKPT=$(ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Resuming from: $LATEST_CKPT"

PYTHONUNBUFFERED=1 python -m src.bird.sft_train \
    --sft-data   /scratch/phalle.y/bird_sft_data.json \
    --output-dir $ADAPTER_DIR \
    --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --epochs     3 \
    --cutoff-len 8192 \
    --save-steps 100 \
    --resume     "$LATEST_CKPT"
