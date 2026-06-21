#!/bin/bash
# QUICK GRPO validation run for 7B BIRD — ~3-4h on H200, then eval to sanity-check.
# Uses a 1,000-question subset (vs 2,000 in the full run) and saves every 50 steps,
# so even if the 4h wall is hit you have an evaluable checkpoint.
# Starts from the SFT adapter; reward = SQL execution match against gold (binary 1/0).

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-grpo-7b-quick
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_7b_quick_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_7b_quick_%j.err

source activate /scratch/phalle.y/grpo_env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# ── Step 1: build a 1,000-question subset (random, seeded) from full train.json ──
FULL_JSON=/scratch/phalle.y/bird_train/train/train.json
QUICK_JSON=/scratch/phalle.y/bird_train/train/train_quick1000.json
if [ ! -f "$QUICK_JSON" ]; then
    echo "Building 1,000-question subset..."
    python3 -c "
import json, random
random.seed(42)
d = json.load(open('$FULL_JSON'))
sub = random.sample(d, 1000)
json.dump(sub, open('$QUICK_JSON', 'w'), indent=2)
print(f'Saved {len(sub)} examples to $QUICK_JSON')
"
fi

# ── Step 2: GRPO training (1 epoch over 1,000 questions) ─────────────────────────
PYTHONUNBUFFERED=1 python -m src.bird.grpo_train \
    --train-json  "$QUICK_JSON" \
    --db-dir      /scratch/phalle.y/bird_train/train/train_databases \
    --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
    --output-dir  /scratch/phalle.y/bird_grpo_adapter_7b_quick \
    --epochs      1 \
    --num-gen     4 \
    --lr          1e-6 \
    --max-tokens  512 \
    --batch-size  4 \
    --grad-accum  8 \
    --save-steps  50
