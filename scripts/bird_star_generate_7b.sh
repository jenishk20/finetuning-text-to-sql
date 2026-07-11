#!/bin/bash
# STaR / RFT — job 1 of 3: GENERATE K CoT samples (7B, GPU, vLLM)
# ---------------------------------------------------------------------------
# Sample K=8 CoT (reasoning + SQL) per BIRD train question FROM the CoT-SFT model
# (bird_cot_sft_adapter_7b, 52.1%). Generate-only → GPU busy throughout → no
# idle-watchdog kill. The CPU job (job 2) executes + keeps the correct CoTs.
# Expected ~1.5-3h for 3000 questions x K=8 (vLLM batched) on H200.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-star-gen-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_star_gen_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_star_gen_7b_%j.err

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
COT_SFT_ADAPTER=/scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter   # sample from the 52.1% CoT model
DB_DIR=/scratch/phalle.y/bird_train/train/train_databases
CAND=/scratch/phalle.y/results_star_7b/candidates_star.json
OUT=/scratch/phalle.y/results_star_7b/bird_star_self_data.json           # real output written by job 2
MAXQ=${MAXQ:-3000}                                                        # MAXQ=20 sbatch ... for a smoke test

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model      "$BASE_MODEL" \
    --adapter         "$COT_SFT_ADAPTER" \
    --output-file     "$OUT" \
    --db-dir          "$DB_DIR" \
    --k               8 \
    --temperature     0.8 \
    --max-questions   "$MAXQ" \
    --use-hf \
    --cot \
    --max-new-tokens  1024 \
    --generate-only \
    --save-candidates "$CAND"

# Next: sbatch scripts/bird_star_build_7b.sh  (CPU: execute + keep correct CoTs → SFT data)
