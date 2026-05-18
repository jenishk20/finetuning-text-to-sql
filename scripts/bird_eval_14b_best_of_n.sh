#!/bin/bash
# Pipeline 1 — single job
# vLLM-batched Best-of-N evaluation on BIRD dev using 14B SFT adapter
# Expected: ~2h on H200, target +3-7pp over greedy SFT (~58-62%)

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-eval-bon
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_bon_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_bon_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Install vLLM if not already (one-time, ~5 min)
pip install -q "vllm==0.6.3" 2>&1 | tail -3

PYTHONUNBUFFERED=1 python -m src.bird.eval_best_of_n \
    --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter     /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --dev-json    /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir      /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir  /scratch/phalle.y/results_14b_best_of_n_eval \
    --k           4 \
    --temperature 0.8
