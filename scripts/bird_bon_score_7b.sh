#!/bin/bash
# ExCoT Best-of-N — job 2 of 2 : self-consistency SCORE on BIRD dev (CPU — NO GPU)
# ---------------------------------------------------------------------------
# Reads the K CoT candidates, extracts each final SQL, executes, picks the result
# set shared by the MOST candidates (majority vote), scores vs gold. CPU only ->
# no idle kill. Run after the generate job. Prints result/exec accuracy, per-
# difficulty, and pick-rank distribution. ~3.5-4h for K=8 over 1534 dev.
# Compare the printed result_accuracy to your CoT-SFT greedy 52.1%.
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # ⚠️ CPU partition — verify with `sinfo -s`
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-bon-score-7b
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_bon_score_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_bon_score_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
# json + sqlite3 only — no vLLM/torch.

PYTHONUNBUFFERED=1 python -m src.bird.score_best_of_n \
    --candidates-file /scratch/phalle.y/results_bon_7b/dev_cot_candidates.json \
    --output-dir      /scratch/phalle.y/results_bon_7b_eval
