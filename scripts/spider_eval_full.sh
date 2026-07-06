#!/bin/bash
# FULL Spider dev eval (all 1,034 questions) of a LoRA adapter (4-bit base + adapter),
# result-set accuracy. Uses src.spider.eval_finetuned. Set ADAPTER_DIR below, then:
#   sbatch scripts/spider_eval_full.sh
#
# Data lives at /home/phalle.y/Jenish-DPO-GRPO (dev.json + database/) — readable on
# your compute nodes, so no staging needed. The SFT adapter is on /scratch.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # any GPU >=12GB works for 4-bit eval
#SBATCH --time=03:00:00          # full 1,034-question dev eval (~2-3h)
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

# ── EDIT IF NEEDED ───────────────────────────────────────────────────────────
ADAPTER_DIR=/scratch/phalle.y/spider_sft_adapter_7b       # SFT adapter output dir
SPIDER_DATA=/home/phalle.y/Jenish-DPO-GRPO                # dev.json + database/
# ─────────────────────────────────────────────────────────────────────────────

# Pick final_adapter if present, else the highest-numbered checkpoint, else the dir itself.
CKPT=$(ls -d "$ADAPTER_DIR"/final_adapter 2>/dev/null \
       || ls -d "$ADAPTER_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
CKPT=${CKPT:-$ADAPTER_DIR}

if [ ! -f "$CKPT/adapter_config.json" ]; then
  echo "ERROR: no adapter_config.json under $CKPT"
  exit 1
fi
echo "Evaluating adapter: $CKPT"
echo "Spider data:        $SPIDER_DATA"

OUT_DIR=/scratch/phalle.y/results_spider_eval_$(basename "$ADAPTER_DIR")

# No --limit → full 1,034-question Spider dev set.
PYTHONUNBUFFERED=1 python -m src.spider.eval_finetuned \
    --adapter    "$CKPT" \
    --data-dir   "$SPIDER_DATA" \
    --output-dir "$OUT_DIR"

# Tip: run again with --base-only (no adapter) for the raw Qwen-7B baseline;
#      SFT lift = adapter accuracy − base accuracy.
