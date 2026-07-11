#!/bin/bash
# STaR / RFT — job 3 of 3: SFT on (teacher CoT seed + self-generated correct CoT) (7B, GPU)
# ---------------------------------------------------------------------------
# Fresh LoRA on BASE Qwen2.5-Coder-7B, same recipe as the CoT-SFT that hit 52.1%,
# but on the STaR-augmented data (teacher seed + the model's own execution-verified
# correct CoTs). Goal: distill what Best-of-N discovers into the GREEDY model.
# Expected ~3-4h for 2 epochs (data is larger than the 5.6k seed).
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-star-sft-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_star_sft_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_star_sft_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

pip install -q "trl==0.12.2" "transformers==4.46.3" 2>&1 | tail -3

# sft_train.py defaults to 14B — pass the 7B explicitly.
PYTHONUNBUFFERED=1 python -m src.bird.sft_train \
    --sft-data    /scratch/phalle.y/results_star_7b/bird_star_sft_data.json \
    --output-dir  /scratch/phalle.y/bird_star_sft_adapter_7b \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs      2 \
    --cutoff-len  8192 \
    --save-steps  100

# Eval greedy: scripts/bird_eval_7b_cot_sft.sh with
#   ADAPTER=/scratch/phalle.y/bird_star_sft_adapter_7b/final_adapter
# Compare to CoT-SFT 52.1% greedy. (Then optionally re-run Best-of-N on top.)
