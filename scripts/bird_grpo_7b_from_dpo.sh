#!/bin/bash
# GRPO training for 7B BIRD — starts from DPO adapter (50.3%)
# Goal: push the existing best 7B result higher using RL with execution reward
# Runs in parallel with bird_grpo_7b_from_sft.sh (different output dir, different GPU)

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bird-grpo-7b-dpo
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_7b_dpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_7b_dpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# train.json should already exist from the SFT-from job; if not, create it
TRAIN_JSON=/scratch/phalle.y/bird_train/train/train.json
if [ ! -f "$TRAIN_JSON" ]; then
    python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('xu3kev/BIRD-SQL-data-train', split='train')
out = [{'question_id': i, 'db_id': r['db_id'], 'question': r['question'],
        'evidence': r.get('evidence', ''), 'SQL': r['SQL']} for i, r in enumerate(ds)][:2000]
json.dump(out, open('$TRAIN_JSON', 'w'), indent=2)
"
fi

# Start GRPO from the 7B DPO adapter (your current best at 50.3%)
PYTHONUNBUFFERED=1 python -m src.bird.grpo_train \
    --train-json  /scratch/phalle.y/bird_train/train/train.json \
    --db-dir      /scratch/phalle.y/bird_train/train/train_databases \
    --sft-adapter /scratch/phalle.y/bird_frontier_dpo_adapter/final_adapter \
    --output-dir  /scratch/phalle.y/bird_grpo_adapter_7b_from_dpo \
    --epochs      1 \
    --num-gen     4 \
    --lr          5e-7 \
    --max-tokens  512 \
    --batch-size  1 \
    --grad-accum  8
