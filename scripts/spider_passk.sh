#!/bin/bash
# pass@K headroom diagnostic for the Spider SFT adapter. Tells you whether on-policy
# training (GRPO) can beat greedy 77% BEFORE you spend GPU on it.
#   sbatch scripts/spider_passk.sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00          # ~20-40 min for 300 questions x (greedy + K=8)
#SBATCH --job-name=spider-passk
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_passk_%j.out
#SBATCH --error=/scratch/phalle.y/spider_passk_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.spider.passk_diagnostic \
    --adapter    /scratch/phalle.y/spider_sft_adapter_7b/final_adapter \
    --data-dir   /home/phalle.y/Jenish-DPO-GRPO \
    --output-dir /scratch/phalle.y/results_spider_passk \
    --limit 300 --k 8 --temperature 0.8
