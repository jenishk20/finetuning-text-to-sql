#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00          # 14B inference takes ~3-4h for 1534 questions
#SBATCH --job-name=bird-eval-14b-dpo
#SBATCH --mem=70GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_14b_dpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_14b_dpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# Auto-pick latest DPO checkpoint (no final_adapter since training hit time limit)
ADAPTER_DIR=/scratch/phalle.y/bird_dpo_adapter_14b_1219pairs
LATEST_CKPT=$(ls -d $ADAPTER_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "Using DPO checkpoint: $LATEST_CKPT"

PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter    "$LATEST_CKPT" \
    --dev-json   /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \
    --db-dir     /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \
    --output-dir /scratch/phalle.y/results_14b_dpo_1219pairs_eval
