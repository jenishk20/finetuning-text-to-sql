#!/bin/bash
# ExCoT Step 2.2 — CoT-SFT (7B, GPU)
# ---------------------------------------------------------------------------
# SFT a FRESH LoRA on the BASE Qwen2.5-Coder-7B-Instruct using the execution-
# verified CoT seed (cot_sft_data.json). Teaches the model to REASON step by
# step, then emit SQL — the opposite of bird_sft_adapter_7b (direct SQL).
# Pure GPU training (no CPU execution phase) -> no idle-watchdog risk.
# Expected ~2.5-3h for 2 epochs over ~5.6k CoT examples on H200.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-cot-sft-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cot_sft_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cot_sft_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Same pin as the DPO job — keep trl/transformers consistent across the pipeline.
pip install -q "trl==0.12.2" "transformers==4.46.3" 2>&1 | tail -3

# NOTE: sft_train.py defaults to the 14B base — must pass the 7B explicitly.
PYTHONUNBUFFERED=1 python -m src.bird.sft_train \
    --sft-data    /scratch/phalle.y/cot_sft_data.json \
    --output-dir  /scratch/phalle.y/bird_cot_sft_adapter_7b \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs      2 \
    --cutoff-len  8192 \
    --save-steps  100
