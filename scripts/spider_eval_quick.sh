#!/bin/bash
# Quick Spider dev eval of a LoRA adapter (4-bit base + adapter), result-set accuracy.
# Uses src.spider.eval_finetuned (already in the repo). Set ADAPTER_DIR + SPIDER_DATA
# below, then:   sbatch scripts/spider_eval_quick.sh
#
# Data lives at /home/phalle.y/Jenish-DPO-GRPO (dev.json + database/) — readable on
# your compute nodes, so no staging needed. The SFT adapter is on /scratch.
# If a node ever can't read /home, copy the data to /scratch and repoint SPIDER_DATA.

#SBATCH --partition=gpu
#SBATCH --nodes=1No
#SBATCH --gres=gpu:h200:1        # any GPU >=12GB works for 4-bit eval
#SBATCH --time=01:00:00          # quick (--limit). Bump to 03:00:00 for the full 1034.
#SBATCH --job-name=spider-eval-sft
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_eval_sft_%j.out
#SBATCH --error=/scratch/phalle.y/spider_eval_sft_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# ── EDIT THESE ───────────────────────────────────────────────────────────────
# Spider SFT adapter dir (holds adapter_config.json, or a parent with final_adapter/checkpoint-*).
# Set this to whatever scripts/spider_find_adapters.sh turns up.
ADAPTER_DIR=/scratch/phalle.y/spider_sft_adapter_7b
# Spider data with dev.json + database/ (already on the cluster under /home).
SPIDER_DATA=/home/phalle.y/Jenish-DPO-GRPO
# Quick smoke: 100 questions (~10-15 min). Set LIMIT=0 to run the full 1034-question dev set.
LIMIT=100
# ─────────────────────────────────────────────────────────────────────────────

# Pick final_adapter if present, else the highest-numbered checkpoint, else ADAPTER_DIR itself.
CKPT=$(ls -d "$ADAPTER_DIR"/final_adapter 2>/dev/null \
       || ls -d "$ADAPTER_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
CKPT=${CKPT:-$ADAPTER_DIR}

if [ ! -f "$CKPT/adapter_config.json" ]; then
  echo "ERROR: no adapter_config.json under $CKPT"
  echo "Run scripts/spider_find_adapters.sh first and set ADAPTER_DIR correctly."
  exit 1
fi
echo "Evaluating adapter: $CKPT"
echo "Spider data:        $SPIDER_DATA"

OUT_DIR=/scratch/phalle.y/results_spider_eval_$(basename "$ADAPTER_DIR")
LIMIT_ARG=""
[ "${LIMIT:-0}" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

PYTHONUNBUFFERED=1 python -m src.spider.eval_finetuned \
    --adapter    "$CKPT" \
    --data-dir   "$SPIDER_DATA" \
    --output-dir "$OUT_DIR" \
    $LIMIT_ARG

# Tip: to also get the BASE-model (no-adapter) Spider baseline for an A/B, run again with --base-only.
