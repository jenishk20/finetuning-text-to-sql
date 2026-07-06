#!/bin/bash
# GRPO (execution-reward RL) for Spider 7B — starts from the SFT adapter (77%).
# Reward: binary SQL-execution match against gold. NO vLLM (HF generate) — vLLM pins
# break TRL in-env (your prior error); grpo_env has vllm OUT by design.
#
# Research-informed config: constant LR (fixes the BIRD cosine-died stall),
# low KL beta 0.01 (verifiable reward), num_gen 8. Constant LR → resume-chainable.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00          # ~2000 Q at num_gen 8; chain via --resume if it doesn't finish
#SBATCH --job-name=spider-grpo-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_grpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/spider_grpo_7b_%j.err

source activate /scratch/phalle.y/grpo_env      # trl 0.15.2, NO vllm (keep it that way)
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.spider.grpo_train \
    --train-json  /home/phalle.y/Jenish-DPO-GRPO/train_spider.json \
    --data-dir    /home/phalle.y/Jenish-DPO-GRPO \
    --sft-adapter /scratch/phalle.y/spider_sft_adapter_7b/final_adapter \
    --output-dir  /scratch/phalle.y/spider_grpo_adapter_7b \
    --limit 2000 \
    --epochs 1 \
    --num-gen 8 \
    --lr 2e-6 \
    --lr-scheduler constant \
    --beta 0.01 \
    --max-tokens 256 \
    --max-prompt-length 2048 \
    --batch-size 8 \
    --grad-accum 8 \
    --save-steps 25

# To continue if it hits the 8h wall (constant LR → clean resume):
#   LATEST=$(ls -d /scratch/phalle.y/spider_grpo_adapter_7b/checkpoint-* | sort -t- -k2 -n | tail -1)
#   ... add: --resume "$LATEST"
# Eval after: scripts/spider_eval_full.sh with ADAPTER_DIR=/scratch/phalle.y/spider_grpo_adapter_7b
