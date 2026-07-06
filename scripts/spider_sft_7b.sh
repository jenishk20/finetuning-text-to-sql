#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=spider-sft-7b
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_sft_7b_%j.out
#SBATCH --error=/scratch/phalle.y/spider_sft_7b_%j.err

# Clean Spider pipeline, stage 1: SFT a fresh 7B LoRA adapter on Spider TRAIN
# gold SQL (7,000 examples). Produces the SFT-only checkpoint the clean DPO will
# train on top of. Reuses src.bird.sft_train (model/data-agnostic: r=32, alpha=64,
# 4-bit NF4, cosine LR — identical config to the original Spider SFT).
#
# Data is already on the cluster — /home/phalle.y/Jenish-DPO-GRPO/sft_data.json
# (schema is embedded in each example → NO databases needed to train).
# If a compute node can't read /home, copy it to /scratch first and repoint --sft-data:
#   cp /home/phalle.y/Jenish-DPO-GRPO/sft_data.json /scratch/phalle.y/spider_sft_data.json

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

# Caches on /scratch — home quota is small
export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.sft_train \
    --sft-data    /home/phalle.y/Jenish-DPO-GRPO/sft_data.json \
    --output-dir  /scratch/phalle.y/spider_sft_adapter_7b \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs      3 \
    --lr          2e-4 \
    --cutoff-len  8192 \
    --save-steps  100

# Output adapter: /scratch/phalle.y/spider_sft_adapter_7b/final_adapter
# Then sanity-check it with scripts/spider_eval_quick.sh (set ADAPTER_DIR to the above).
