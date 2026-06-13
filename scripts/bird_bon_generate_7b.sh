#!/bin/bash
# ExCoT Best-of-N — job 1 of 2 : generate K=8 CoT candidates per BIRD DEV question
# ---------------------------------------------------------------------------
# Reuses build_self_sampling_pairs in --generate-only --cot mode, pointed at the
# DEV set, sampling K=8 with temperature 0.8 from the CoT-SFT model. Saves raw
# CoT candidates; the CPU score job does the self-consistency vote + scoring.
# Generate-only -> GPU busy throughout -> no idle kill. ~40-90 min on H200.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-bon-gen-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_bon_gen_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_bon_gen_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" "transformers==4.46.3" datasets 2>&1 | tail -3

BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
COT_SFT_ADAPTER=/scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter
DEV_JSON=/scratch/phalle.y/bird_dev/dev_20240627/dev.json
DB_DIR=/scratch/phalle.y/bird_dev/dev_20240627/dev_databases
CAND=/scratch/phalle.y/results_bon_7b/dev_cot_candidates.json
K=${K:-8}                                                # Best-of-N width; K=4 for a faster run

# --train-json works on dev.json (same db_id/question/evidence/SQL fields).
# --output-file is required by the CLI but unused in --generate-only mode.
PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model      "$BASE_MODEL" \
    --adapter         "$COT_SFT_ADAPTER" \
    --train-json      "$DEV_JSON" \
    --db-dir          "$DB_DIR" \
    --output-file     /scratch/phalle.y/results_bon_7b/_unused_pairs.json \
    --k               "$K" \
    --temperature     0.8 \
    --cot \
    --max-new-tokens  1024 \
    --generate-only \
    --save-candidates "$CAND"
