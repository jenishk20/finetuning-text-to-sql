#!/bin/bash
# ============================================================================
# CoT pass@k  —  JOB 2 of 2  (CPU: score pass@k + self-consistency)
# ============================================================================
# Reads the candidate file from job 1 and computes, from the SAME 8 samples:
#   - pass@1 / pass@2 / pass@4 / pass@8   (unbiased HumanEval estimator)
#   - self-consistency@8 (majority vote)  -> should reproduce your ~58.5%
#   - the headroom gap  pass@8 - self-consistency  (what CoT-GRPO could chase)
#   - a per-difficulty breakdown (where the headroom actually lives)
#
# CPU ONLY — no GPU, no vLLM. `src.bird.inference` imports vLLM lazily (only
# inside the engine), so importing extract_final_sql here is GPU-free.
#
# HOW TO RUN
#   Easiest: grab a short interactive CPU allocation and run it directly:
#       bash scripts/bird_cot_passk_score.sh
#   Or submit it to a CPU partition by adding the #SBATCH lines your cluster
#   uses for CPU jobs (this script intentionally requests no GPU).
# ----------------------------------------------------------------------------

set -euo pipefail

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

OUTDIR=/scratch/phalle.y/results_cot_passk
CANDFILE=$OUTDIR/dev_cot_candidates_k8.json

if [ ! -f "$CANDFILE" ]; then
    echo "ERROR: $CANDFILE not found. Run job 1 first: scripts/bird_cot_passk_generate.sh"
    echo "(Or, if you found an existing candidates file elsewhere, point CANDFILE at it.)"
    exit 1
fi

PYTHONUNBUFFERED=1 python -m src.bird.score_pass_k \
    --candidates-file "$CANDFILE" \
    --output-dir      "$OUTDIR" \
    --ks 1 2 4 8
