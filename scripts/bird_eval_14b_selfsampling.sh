#!/bin/bash
# Pipeline 2 — job 3 of 3
# Evaluate the self-sampling DPO adapter on BIRD dev
# Also uses Best-of-N at inference (K=4) to stack with the trained model

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-eval-selfsample
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_selfsample_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_selfsample_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pip install -q "vllm==0.6.3" 2>&1 | tail -3

# Auto-pick latest DPO checkpoint
ADAPTER_DIR=/scratch/phalle.y/bird_dpo_adapter_14b_selfsampling
LATEST_CKPT=$(ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Using self-sampling DPO checkpoint: $LATEST_CKPT"

# K=1 for pure DPO accuracy; bump to K=4 to stack with Best-of-N
PYTHONUNBUFFERED=1 python -m src.bird.eval_best_of_n \
    --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter     "$LATEST_CKPT" \
    --dev-json    /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir      /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir  /scratch/phalle.y/results_14b_selfsampling_eval \
    --k           4 \
    --temperature 0.8
