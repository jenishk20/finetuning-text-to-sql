#!/bin/bash
# STaR / RFT — job 2 of 3: EXECUTE + KEEP CORRECT CoT → SFT data (CPU, NO GPU)
# ---------------------------------------------------------------------------
# Execute each CoT sample's final SQL, keep the SHORTEST execution-correct chain
# per question as an SFT target (--rft). Then merge with the teacher CoT seed so
# SFT trains on BOTH the teacher's reasoning and the model's own correct reasoning
# (STaR self-improvement). CPU partition, no GPU → watchdog-safe.
# Expected ~1.5-3h (sqlite execution of 3000 x 8 samples).
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # ⚠️ CPU partition — verify with `sinfo -s`
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=bird-star-build-7b
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_star_build_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_star_build_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
# json + sqlite3 only — no vLLM / no torch here.

CAND=/scratch/phalle.y/results_star_7b/candidates_star.json
SELF=/scratch/phalle.y/results_star_7b/bird_star_self_data.json
TEACHER_SEED=/scratch/phalle.y/cot_sft_data.json                 # the 5,593 teacher CoT seed (→ CoT-SFT 52.1%)
MERGED=/scratch/phalle.y/results_star_7b/bird_star_sft_data.json

# ── Execute candidates, keep shortest correct CoT per question → SFT alpaca data ──
PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --candidates-file "$CAND" \
    --output-file     "$SELF" \
    --cot \
    --rft

# ── Merge teacher seed + self-generated correct CoTs → one SFT file ──
python3 - <<PY
import json
seed  = json.load(open("$TEACHER_SEED"))
self_ = json.load(open("$SELF"))
merged = seed + self_
json.dump(merged, open("$MERGED", "w"), indent=2)
print(f"Merged SFT data: {len(seed)} teacher-seed + {len(self_)} self-generated = {len(merged)} → $MERGED")
PY

# Next: sbatch scripts/bird_star_sft_7b.sh  (SFT base 7B on the merged data)
