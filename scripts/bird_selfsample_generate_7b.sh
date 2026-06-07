#!/bin/bash
# ExCoT Stage 1 — job 1 of 3 : ON-POLICY GENERATION (7B)
# ---------------------------------------------------------------------------
# Generate K self-samples per BIRD train question from the 7B SFT adapter,
# execute each against its database, and build (correct, wrong) DPO pairs.
# This is the SAME proven code that ran on your 14B — only the paths changed.
#
# vLLM lives ONLY in this job. If it crashes, training is untouched.
# Expected: ~2-3h on H200 for 1500 questions (7B is ~2x faster than 14B).
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-ss-gen-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_ss_gen_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_ss_gen_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoids vLLM fork crashes

pip install -q "vllm==0.6.3" datasets 2>&1 | tail -3

# ── Config (verify SFT_ADAPTER points at a dir with adapter_config.json) ─────
BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
SFT_ADAPTER=/scratch/phalle.y/bird_sft_adapter_7b/final_adapter   # verified: dir holds adapter_config.json
DB_DIR=/scratch/phalle.y/bird_train/train/train_databases
OUT=/scratch/phalle.y/results_self_sampling_7b/bird_ss_pairs_7b.json
MAXQ=${MAXQ:-1500}                                        # override: MAXQ=20 sbatch ... for smoke test

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model    "$BASE_MODEL" \
    --adapter       "$SFT_ADAPTER" \
    --output-file   "$OUT" \
    --db-dir        "$DB_DIR" \
    --k             4 \
    --temperature   0.8 \
    --max-questions "$MAXQ" \
    --use-hf
