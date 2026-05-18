#!/bin/bash
# Pipeline 2 — job 1 of 3
# Generate self-sampling DPO pairs from 14B SFT adapter on BIRD train
# vLLM-batched generation, then execute candidates, then build (correct, wrong) pairs
# Expected: ~4-5h on H200, produces ~600-1500 pairs depending on yield

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --job-name=bird-selfsample-gen
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_selfsample_gen_%j.out
#SBATCH --error=/scratch/phalle.y/bird_selfsample_gen_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" datasets 2>&1 | tail -3

# Use the HF BIRD-SQL-data-train dataset (since train.json isn't on cluster)
# Cap at 3000 questions to fit comfortably in time budget
PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model    Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter       /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --output-file   /scratch/phalle.y/results_self_sampling/bird_self_sampling_pairs.json \
    --db-dir        /scratch/phalle.y/bird_train/train/train_databases/train_databases \
    --k             4 \
    --temperature   0.8 \
    --max-questions 3000 \
    --use-hf
