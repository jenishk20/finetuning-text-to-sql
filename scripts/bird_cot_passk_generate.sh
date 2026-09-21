#!/bin/bash
# ============================================================================
# CoT pass@k  —  JOB 1 of 2  (GPU: generate candidates, HF transformers, NO vLLM)
# ============================================================================
# Generates K=8 chain-of-thought candidates per BIRD DEV question and saves the
# raw candidates JSON. Job 2 (CPU) then executes and scores them.
#
# This uses plain HuggingFace transformers, not vLLM: the vLLM env on /scratch
# was purged and reinstalling it pulls a brittle dependency chain. Generation
# only needs torch + transformers, which live in (or install cleanly into)
# mergeenv310. The model is loaded from its LOCAL path, so there is no HF
# download and no proxy dependency at run time.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP (do this once; needs the proxy for pip):
#   python3 -m venv /scratch/phalle.y/passk_env
#   source /scratch/phalle.y/passk_env/bin/activate
#   export HTTPS_PROXY=http://10.99.0.130:3128 HTTP_PROXY=http://10.99.0.130:3128
#   pip install --upgrade pip && pip install torch==2.4.1 transformers==4.46.3
# ---------------------------------------------------------------------------
#
# RUN (on a GPU node):
#   sbatch scripts/bird_cot_passk_generate.sh                 # full 1534
#   LIMIT=200 sbatch scripts/bird_cot_passk_generate.sh       # smoke test
#   # or directly in a GPU interactive session:
#   LIMIT=200 bash scripts/bird_cot_passk_generate.sh
# ----------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-cot-passk-gen
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cot_passk_gen_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cot_passk_gen_%j.err

# Activate the venv BEFORE strict mode (activate, then turn on error-exit).
source /scratch/phalle.y/passk_env/bin/activate
set -eo pipefail
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache

# ── Config ──────────────────────────────────────────────────────────────────
MODEL=/scratch/phalle.y/bird_cot_sft_merged_7b                              # LOCAL merged CoT model
DEVJSON=/scratch/phalle.y/bird_dev/dev_20240627/dev.json
DEVDB=/scratch/phalle.y/bird_dev/dev_20240627/dev_databases
OUTDIR=/scratch/phalle.y/results_cot_passk
CANDFILE=${OUT:-$OUTDIR/dev_cot_candidates_k8.json}                         # override with OUT=...
LIMIT=${LIMIT:-0}
BATCH=${BATCH:-4}                                                           # lower to 2 or 1 if OOM
# GREEDY=1 -> greedy reproduce check (do_sample off, 1 sample/q). Use a distinct
# OUT so it doesn't collide with the sampled candidates, e.g.:
#   GREEDY=1 LIMIT=300 BATCH=1 OUT=$OUTDIR/dev_greedy_check.json sbatch scripts/bird_cot_passk_generate.sh
GREEDY_FLAG=""
[ -n "${GREEDY:-}" ] && GREEDY_FLAG="--greedy"

mkdir -p "$OUTDIR"

# ── Preconditions ─────────────────────────────────────────────────────────────
python -c "import transformers" 2>/dev/null || {
    echo "ERROR: transformers not importable in passk_env."
    echo "Run the ONE-TIME SETUP block in this script's header, then re-run."
    exit 1
}
if [ ! -d "$MODEL" ];   then echo "ERROR: model dir missing: $MODEL"; exit 1; fi
if [ ! -f "$DEVJSON" ]; then echo "ERROR: dev.json missing: $DEVJSON"; exit 1; fi
if [ ! -d "$DEVDB" ];   then echo "ERROR: dev_databases missing: $DEVDB"; exit 1; fi
if [ -f "$CANDFILE" ]; then
    echo "Candidates already exist at $CANDFILE — skipping generation."
    echo "To force a clean re-run: rm $CANDFILE"
    echo "Otherwise go to job 2: bash scripts/bird_cot_passk_score.sh"
    exit 0
fi

EXTRA=""
[ "$LIMIT" -gt 0 ] && EXTRA="--limit $LIMIT"

# ── Generate ──────────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python -m src.bird.generate_cot_candidates_hf \
    --model          "$MODEL" \
    --dev-json       "$DEVJSON" \
    --db-dir         "$DEVDB" \
    --out            "$CANDFILE" \
    --k              8 \
    --temperature    0.8 \
    --max-new-tokens 768 \
    --batch-size     "$BATCH" \
    $GREEDY_FLAG $EXTRA

echo "Done. Candidates at $CANDFILE"
echo "Next: bash scripts/bird_cot_passk_score.sh"
