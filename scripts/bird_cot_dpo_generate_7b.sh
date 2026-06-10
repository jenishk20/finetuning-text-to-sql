#!/bin/bash
# ExCoT Step 2.3 — job 1 of 3 : ON-POLICY CoT GENERATION (7B, GPU)
# ---------------------------------------------------------------------------
# Sample K CoT (reasoning + SQL) per BIRD train question FROM THE CoT-SFT MODEL
# (bird_cot_sft_adapter_7b). Saves raw CoT candidates; the CPU job executes the
# extracted SQL and builds (correct CoT, wrong CoT) pairs. Generate-only -> the
# GPU is busy throughout -> no idle-watchdog kill.
# Expected ~20-40 min on H200 for 3000 questions x K=4 (vLLM batched).
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --job-name=bird-cotdpo-gen-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cotdpo_gen_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cotdpo_gen_7b_%j.err

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
COT_SFT_ADAPTER=/scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter   # sample from the CoT-SFT model
DB_DIR=/scratch/phalle.y/bird_train/train/train_databases
CAND=/scratch/phalle.y/results_cot_dpo_7b/candidates_cot_7b.json
OUT=/scratch/phalle.y/results_cot_dpo_7b/bird_cot_pairs_7b.json
MAXQ=${MAXQ:-3000}                                       # MAXQ=20 sbatch ... for a smoke test

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model      "$BASE_MODEL" \
    --adapter         "$COT_SFT_ADAPTER" \
    --output-file     "$OUT" \
    --db-dir          "$DB_DIR" \
    --k               4 \
    --temperature     0.8 \
    --max-questions   "$MAXQ" \
    --use-hf \
    --cot \
    --max-new-tokens  1024 \
    --generate-only \
    --save-candidates "$CAND"
