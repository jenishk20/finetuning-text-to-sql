#!/bin/bash
# ExCoT Step 2.2-eval — CoT eval of the CoT-SFT adapter on BIRD dev
# ---------------------------------------------------------------------------
# Greedy eval with --cot (reasoning prompt + final-SQL extraction). Interleaved
# (one GPU generation per question) -> idle-safe. Compare result_accuracy to
# your 49.3% self-sampling baseline / 50.3% frontier-DPO best.
#
# Greedy stops at EOS, so the 768-token cap is rarely hit; per-question time is
# set by the model's natural CoT length (~370 tok). Full 1534 ~5h; checkpoints
# every 50 and is resumable (--resume). Default is the FULL 1534 (~5h).
#   full:   sbatch scripts/bird_eval_7b_cot_sft.sh             (all 1534, default)
#   quick:  LIMIT=500 sbatch scripts/bird_eval_7b_cot_sft.sh   (first 500, ~2h)
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bird-eval-cotsft-7b
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_cotsft_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_cotsft_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

LIMIT=${LIMIT:-0}                         # full 1534 by default; LIMIT=500 = quick checkpoint
EXTRA=""
[ "$LIMIT" -gt 0 ] && EXTRA="--limit $LIMIT"

PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --adapter    /scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter \
    --dev-json   /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
    --db-dir     /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
    --output-dir /scratch/phalle.y/results_cot_sft_7b_eval \
    --cot \
    $EXTRA
