#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00          # ~5.5h for 5,500 examples x 3 epochs
#SBATCH --job-name=spider-sft-split
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/spider_sft_split_%j.out
#SBATCH --error=/scratch/phalle.y/spider_sft_split_%j.err

# Re-SFT on the DISJOINT SFT split (~5,500), holding out ~1,500 for GRPO.
# Same recipe as the original SFT (r=32, alpha=64, 4-bit NF4, 3 epochs, lr 2e-4).
# Run scripts/... split first to produce spider_sft_split.json.

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

python -m src.bird.sft_train \
    --sft-data    /scratch/phalle.y/spider_splits/spider_sft_split.json \
    --output-dir  /scratch/phalle.y/spider_sft_split_adapter_7b \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs      3 \
    --lr          2e-4 \
    --cutoff-len  8192 \
    --save-steps  100

# Output: /scratch/phalle.y/spider_sft_split_adapter_7b/final_adapter
# Next: eval it (spider_eval_full.sh) for the new baseline, then GRPO on the holdout.
