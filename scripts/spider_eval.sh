#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # H200 not required — any GPU ≥12GB works
#SBATCH --time=03:00:00          # ~2-3h for full 1034-question eval
#SBATCH --job-name=spider-eval
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_eval_%j.out
#SBATCH --error=/scratch/phalle.y/spider_eval_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Stage 3b: cross-eval the 7B BIRD DPO adapter on the Spider dev set
PYTHONUNBUFFERED=1 python -m src.spider.eval_finetuned \
    --adapter    /scratch/phalle.y/bird_dpo_adapter_7b/final_adapter \
    --data-dir   /home/phalle.y/Jenish-DPO-GRPO/spider_data \
    --output-dir /scratch/phalle.y/results_spider_crosseval_7b
