#!/bin/bash
# ExCoT Stage 1 — job 2 of 3 : EXECUTE + BUILD PAIRS (7B, CPU — NO GPU)
# ---------------------------------------------------------------------------
# Reads the raw candidates saved by bird_selfsample_generate_7b.sh, executes
# each candidate against its SQLite DB, and builds (correct, wrong) DPO pairs.
# This runs on a CPU partition with NO GPU requested, so the GPU-idle watchdog
# that killed the old single-job run cannot touch it. vLLM is never imported.
# Run only AFTER the generate job wrote candidates_7b.json.
# Expected: ~1.5h for 1500 questions (SQLite execution, 15s timeout each).
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # ⚠️ CPU partition — verify the name with `sinfo -s` (e.g. short / express / cpu)
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-ss-pairs-7b
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_ss_pairs_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_ss_pairs_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
# No pip install / no vLLM here — this phase only needs json + sqlite3 (stdlib).

CAND=/scratch/phalle.y/results_self_sampling_7b/candidates_7b.json
OUT=/scratch/phalle.y/results_self_sampling_7b/bird_ss_pairs_7b.json

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --candidates-file "$CAND" \
    --output-file     "$OUT"
