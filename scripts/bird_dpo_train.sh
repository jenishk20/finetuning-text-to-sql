#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=bird-dpo
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_dpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_dpo_%j.err

source /scratch/phalle.y/finetuning-text-to-sql/myenv/bin/activate
cd /scratch/phalle.y/finetuning-text-to-sql

python -m src.bird.dpo_train \
    --pairs-file /scratch/phalle.y/results_frontier_pairs/bird_frontier_dpo_data.json \
    --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
    --output-dir /scratch/phalle.y/bird_frontier_dpo_adapter \
    --cutoff-len 8192 \
    --beta 0.05 \
    --epochs 1 \
    --save-steps 50
