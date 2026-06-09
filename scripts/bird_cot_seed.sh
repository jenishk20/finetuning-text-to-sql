#!/bin/bash
# ExCoT Step 2.1 — CoT SEED GENERATION (CPU, NO GPU)
# ---------------------------------------------------------------------------
# Qwen3-Coder-480B (W&B Inference) generates step-by-step CoT + SQL over BIRD
# train; we execute each and KEEP ONLY execution-correct traces -> cot_sft_data.json
# for src.bird.sft_train. API + sqlite only: no GPU (so no idle-kill), runs on a
# CPU partition (Discovery compute nodes have outbound internet for the W&B API).
#
# Prereqs:
#   - WANDB_API_KEY must be in the repo .env on the cluster (it's loaded via dotenv).
#   - verify the train paths below with: ls /scratch/phalle.y/bird_train/train/
# Expected: ~2-3h, ~$5-8 of W&B credits for the full ~9.4k train set
# (yield ~50-55% -> ~4700-5000 correct CoT seed examples).
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # ⚠️ CPU partition — verify with `sinfo -s`
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-cot-seed
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cot_seed_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cot_seed_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
# Needs openai + python-dotenv (already in env). No vLLM, no torch, no GPU.

QJSON=/scratch/phalle.y/bird_train/train/train.json
DB_DIR=/scratch/phalle.y/bird_train/train/train_databases   # same dir the self-sampling run used
OUT=/scratch/phalle.y/cot_sft_data.json
MAXQ=${MAXQ:-0}                                              # 0 = full set; MAXQ=20 sbatch ... for a cluster smoke

EXTRA=""
[ "$MAXQ" -gt 0 ] && EXTRA="--max-questions $MAXQ"

PYTHONUNBUFFERED=1 python -m src.bird.build_cot_sft_data \
    --questions-json "$QJSON" \
    --db-dir         "$DB_DIR" \
    --output-file    "$OUT" \
    --concurrency    8 \
    --max-tokens     2048 \
    $EXTRA
