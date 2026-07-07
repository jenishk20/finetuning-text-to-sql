#!/bin/bash
# GRPO on the HELD-OUT 1,500 Spider questions (unseen by the re-SFT).
# The model won't have memorized these → reward starts ~0.80 with real variance →
# actual learning signal (unlike the flat-0.98 run on memorized train data).
# Starts from the re-SFT-on-5,500 adapter. NO vLLM (grpo_env).

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00          # ~2-3h for 1,500 Q at num_gen 8
#SBATCH --job-name=spider-grpo-holdout
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_grpo_holdout_%j.out
#SBATCH --error=/scratch/phalle.y/spider_grpo_holdout_%j.err

source activate /scratch/phalle.y/grpo_env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.spider.grpo_train \
    --train-json  /scratch/phalle.y/spider_splits/spider_grpo_holdout.json \
    --data-dir    /home/phalle.y/Jenish-DPO-GRPO \
    --sft-adapter /scratch/phalle.y/spider_sft_split_adapter_7b/final_adapter \
    --output-dir  /scratch/phalle.y/spider_grpo_holdout_adapter_7b \
    --limit 1500 \
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

# WATCH THE REWARD: it should start ~0.80 and CLIMB. If it's ~0.98 flat again,
# the holdout wasn't actually unseen — stop and tell me.
# Eval after: spider_eval_full.sh with ADAPTER_DIR=/scratch/phalle.y/spider_grpo_holdout_adapter_7b
