#!/bin/bash
# ============================================================================
# CoT pass@k  —  JOB 1 of 2  (GPU: generate candidates)
# ============================================================================
# Generates K=8 chain-of-thought candidates per BIRD DEV question and saves the
# raw candidates to a JSON file. It does NOT score anything (that is job 2, CPU).
# Splitting GPU generation from CPU scoring is deliberate: scoring executes
# ~12k SQL queries on CPU for a while, and an idle-GPU watchdog would kill a job
# that held a GPU during that. Same two-phase design as your self-sampling runs.
#
# WHY THIS EXISTS
#   No saved candidate file was found on /scratch, so we regenerate the exact
#   samples the 58.5% Best-of-N number came from, then (job 2) also read pass@k
#   off the very same samples. One generation, both numbers, apples-to-apples.
#
# MODEL SOURCE
#   Uses the MERGED CoT model straight from HuggingFace
#   (jk200201/qwen2.5-coder-7b-bird-cot), so this works even though the local
#   /scratch adapter may have been wiped. No LoRA needed — it is already merged.
#
# REPRODUCIBILITY NOTE
#   K=8, temperature 0.8 match the 58.5% self-consistency recipe (model card).
#   max-new-tokens is 768 (your greedy CoT eval's cap; natural CoT length ~370
#   tok) so reasoning is never truncated. If you want to reproduce 58.5% to the
#   decimal, set this to whatever the original Best-of-N generation used.
#
# RUN
#   sbatch scripts/bird_cot_passk_generate.sh
#   # quick smoke test on 200 questions first:
#   LIMIT=200 sbatch scripts/bird_cot_passk_generate.sh
# ----------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-cot-passk-gen
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cot_passk_gen_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cot_passk_gen_%j.err

# Activate conda BEFORE any strict-mode flag. Conda's activate script
# references $PS1 (unset in a batch shell), so running it under `set -u`
# aborts the job with "PS1: unbound variable". Activate first, then turn on
# strict mode for the actual work.
source activate /scratch/phalle.y/py310env
set -eo pipefail
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" datasets 2>&1 | tail -3

# ── Config ──────────────────────────────────────────────────────────────────
MODEL=jk200201/qwen2.5-coder-7b-bird-cot                                   # merged CoT model, from HF
DEVJSON=/scratch/phalle.y/bird_dev/dev_20240627/dev.json                   # BIRD dev questions
DEVDB=/scratch/phalle.y/bird_dev/dev_20240627/dev_databases                # BIRD dev sqlite DBs
OUTDIR=/scratch/phalle.y/results_cot_passk                                 # FRESH output dir
CANDFILE=$OUTDIR/dev_cot_candidates_k8.json
LIMIT=${LIMIT:-0}                                                          # 0 = all 1534 dev questions

mkdir -p "$OUTDIR"

# ── Free-path guard: if candidates already exist, skip generation entirely ────
if [ -f "$CANDFILE" ]; then
    echo "Candidates already present at $CANDFILE — skipping generation."
    echo "Go straight to job 2: sbatch/bash scripts/bird_cot_passk_score.sh"
    exit 0
fi

# ── Data guard: BIRD dev may have been wiped from /scratch ────────────────────
if [ ! -f "$DEVJSON" ] || [ ! -d "$DEVDB" ]; then
    echo "ERROR: BIRD dev not found."
    echo "  expected questions: $DEVJSON"
    echo "  expected databases: $DEVDB"
    echo "Restore the official BIRD dev release (dev.json + dev_databases/) to those paths, then re-run."
    exit 1
fi

EXTRA=""
[ "$LIMIT" -gt 0 ] && EXTRA="--max-questions $LIMIT"

# ── Generate (GPU) and save raw candidates, then exit before any execution ────
# NOTE: --train-json just means "the questions file" here; we point it at DEV.
#       --output-file is a required arg but unused in --generate-only mode.
PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model      "$MODEL" \
    --train-json      "$DEVJSON" \
    --db-dir          "$DEVDB" \
    --output-file     "$OUTDIR/_unused_pairs.json" \
    --k               8 \
    --temperature     0.8 \
    --max-new-tokens  768 \
    --cot \
    --generate-only \
    --save-candidates "$CANDFILE" \
    $EXTRA

echo "Done. Candidates saved to $CANDFILE"
echo "Next: sbatch/bash scripts/bird_cot_passk_score.sh"
