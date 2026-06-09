#!/bin/bash
# ExCoT Stage 1 — job 3 of 3 : GREEDY EVAL (7B)
# ---------------------------------------------------------------------------
# Greedy (deterministic) eval of the new self-sampling DPO adapter on BIRD dev.
# Greedy is the apples-to-apples way to measure the DPO gain in the weights
# (Best-of-N is a separate inference lever you can stack later).
# Compare the result_accuracy printed here against your 50.3% baseline
# (results_bird_dpo_7b) and the SFT baseline.
# Expected: ~3-4h for full 1534-question dev (any GPU >=12GB works).
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-eval-ss-7b
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_eval_ss_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_eval_ss_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# NOTE: dpo_train.py saves to <output-dir>/final_adapter — eval must include it.
# Dev set now lives on scratch at bird_dev/dev_20240627/ (dev.json + dev_databases).
PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --adapter    /scratch/phalle.y/bird_dpo_adapter_7b_selfsampling/final_adapter \
    --dev-json   /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
    --db-dir     /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
    --output-dir /scratch/phalle.y/results_self_sampling_7b_eval
