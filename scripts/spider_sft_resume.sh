#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00          # chain as many of these as needed to reach 2625 steps (3 epochs)
#SBATCH --job-name=spider-sft-7b-resume
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_sft_7b_resume_%j.out
#SBATCH --error=/scratch/phalle.y/spider_sft_7b_resume_%j.err

# Resume the Spider SFT from its latest checkpoint and continue the 3-epoch
# cosine schedule. Submit this repeatedly until training reports "complete"
# (i.e. it reaches step 2625). Same pattern used to finish the BIRD 14B SFT.

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Auto-detect the latest checkpoint in the output dir.
ADAPTER_DIR=/scratch/phalle.y/spider_sft_adapter_7b
LATEST_CKPT=$(ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Resuming from: $LATEST_CKPT"
[ -z "$LATEST_CKPT" ] && echo "No checkpoint found in $ADAPTER_DIR — run spider_sft_7b.sh first." && exit 1

PYTHONUNBUFFERED=1 python -m src.bird.sft_train \
    --sft-data   /home/phalle.y/Jenish-DPO-GRPO/sft_data.json \
    --output-dir $ADAPTER_DIR \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs     3 \
    --lr         2e-4 \
    --cutoff-len 8192 \
    --save-steps 100 \
    --resume     "$LATEST_CKPT"
