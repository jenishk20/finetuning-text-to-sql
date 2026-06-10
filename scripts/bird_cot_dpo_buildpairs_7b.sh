#!/bin/bash
# ExCoT Step 2.3 — job 2 of 3 : EXECUTE + BUILD CoT PAIRS (7B, CPU — NO GPU)
# ---------------------------------------------------------------------------
# Reads the raw CoT candidates, extracts each final SQL, executes it, and builds
# (correct CoT, wrong CoT) pairs — the chosen/rejected are the FULL reasoning
# chains. --cot is REQUIRED here so it extracts the final SQL + uses the CoT prompt.
# CPU partition, no GPU -> watchdog-safe. Run after the generate job.
# Expected ~3-4h for 3000 questions (sqlite execution).
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # ⚠️ CPU partition — verify with `sinfo -s`
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=bird-cotdpo-pairs-7b
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cotdpo_pairs_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cotdpo_pairs_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
# No vLLM / no torch here — json + sqlite3 only.

CAND=/scratch/phalle.y/results_cot_dpo_7b/candidates_cot_7b.json
OUT=/scratch/phalle.y/results_cot_dpo_7b/bird_cot_pairs_7b.json

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --candidates-file "$CAND" \
    --output-file     "$OUT" \
    --cot
