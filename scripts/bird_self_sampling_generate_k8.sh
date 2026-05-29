#!/bin/bash
# Pipeline 2B — alternative self-sampling generation with K=8 and T=1.0
# Goal: extract more diverse candidates → more pairs from the "all correct" bucket
# Runs in parallel with the DPO training on the K=4 pairs (different output dir)

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:00:00
#SBATCH --job-name=bird-selfsample-gen-k8
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_selfsample_gen_k8_%j.out
#SBATCH --error=/scratch/phalle.y/bird_selfsample_gen_k8_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" datasets 2>&1 | tail -3

# K=8 + T=1.0 for higher diversity. Different output dir so it doesn't
# clobber the existing K=4 pairs.
PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model    Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter       /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \
    --output-file   /scratch/phalle.y/results_self_sampling_k8/bird_self_sampling_pairs_k8.json \
    --db-dir        /scratch/phalle.y/bird_train/train/train_databases \
    --k             8 \
    --temperature   1.0 \
    --max-questions 3000 \
    --use-hf
