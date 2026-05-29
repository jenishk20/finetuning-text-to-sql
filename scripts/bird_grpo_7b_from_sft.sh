#!/bin/bash
# GRPO training for 7B BIRD — starts from SFT adapter
# Reward: SQL execution match against gold (binary 1/0)
# Expected: ~7-8h on H200 for 2,000 train questions × 4 generations × 1 epoch

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bird-grpo-7b-sft
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_7b_sft_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_7b_sft_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# ── Step 1: Build train.json from HF dataset (one-time, ~30s) ────────────────
TRAIN_JSON=/scratch/phalle.y/bird_train/train/train.json
if [ ! -f "$TRAIN_JSON" ]; then
    echo "Building train.json from HF dataset..."
    python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('xu3kev/BIRD-SQL-data-train', split='train')
out = []
for i, row in enumerate(ds):
    out.append({
        'question_id': i,
        'db_id':       row['db_id'],
        'question':    row['question'],
        'evidence':    row.get('evidence', ''),
        'SQL':         row['SQL'],
    })
# Cap to 2000 questions to fit in 8h budget (full 9k would take ~24h)
out = out[:2000]
json.dump(out, open('$TRAIN_JSON', 'w'), indent=2)
print(f'Saved {len(out)} train examples to $TRAIN_JSON')
"
fi

# ── Step 2: GRPO training ────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python -m src.bird.grpo_train \
    --train-json  /scratch/phalle.y/bird_train/train/train.json \
    --db-dir      /scratch/phalle.y/bird_train/train/train_databases \
    --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
    --output-dir  /scratch/phalle.y/bird_grpo_adapter_7b_from_sft \
    --epochs      1 \
    --num-gen     4 \
    --lr          1e-6 \
    --max-tokens  512 \
    --batch-size  1 \
    --grad-accum  8
