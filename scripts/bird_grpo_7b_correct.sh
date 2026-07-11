#!/bin/bash
# CORRECT BIRD GRPO — fixes the under-dosed first run (600 Q, LR 1e-6, cosine died → 46.9%, no movement).
# Lessons applied: CONSTANT LR 3e-6 (cosine decayed to ~0 and froze the model), low KL beta 0.02
# (verifiable-reward RL likes low KL), 2,000 train Q via --limit (not the full 9,428 → ~63h),
# num_gen 4 (num_gen 8 was ~3.2 min/step on BIRD's long prompts → too slow to finish in 8h).
# NO vLLM (grpo_env — vllm pins break TRL).
#
# Why no holdout split (unlike Spider): BIRD is unsaturated (base 27% → SFT 46% → DPO 50%) and the
# SFT model does NOT memorize BIRD train (first-run reward was ~0.5-0.7, not Spider's 0.98). So GRPO
# on BIRD train has real signal directly — no need to hold data out.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00          # num_gen 8 x 2,000 Q may not finish in 8h — resume (constant LR → clean)
#SBATCH --job-name=bird-grpo-correct
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_correct_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_correct_%j.err

source activate /scratch/phalle.y/grpo_env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# 2,000-question BIRD train subset (bird_grpo_7b_from_sft.sh builds this once; rebuild if missing)
TRAIN_JSON=/scratch/phalle.y/bird_train/train/train.json
if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: $TRAIN_JSON missing — run bird_grpo_7b_from_sft.sh once to build it."
    exit 1
fi

PYTHONUNBUFFERED=1 python -m src.bird.grpo_train \
    --train-json   /scratch/phalle.y/bird_train/train/train.json \
    --db-dir       /scratch/phalle.y/bird_train/train/train_databases \
    --sft-adapter  /scratch/phalle.y/bird_sft_adapter_7b/final_adapter \
    --output-dir   /scratch/phalle.y/bird_grpo_adapter_7b_correct \
    --epochs        1 \
    --limit         2000 \
    --num-gen       4 \
    --batch-size    4 \
    --grad-accum    8 \
    --lr            3e-6 \
    --lr-scheduler  constant \
    --beta          0.02 \
    --max-tokens    512 \
    --save-steps    25

# WATCH THE REWARD: on BIRD it should CLIMB (unsaturated). Loss ≈ 0 is normal (group-relative advantage).
# If OOM (BIRD's ~3072-token prompts x 8 completions is memory-heavy): drop --num-gen and --batch-size to 6 or 4.
# If it hits the 8h wall (constant LR → clean resume):
#   LATEST=$(ls -d /scratch/phalle.y/bird_grpo_adapter_7b_correct/checkpoint-* | sort -t- -k2 -n | tail -1)
#   ...append: --resume "$LATEST"
# Eval after: bird_eval_7b_grpo.sh with ADAPTER_DIR=/scratch/phalle.y/bird_grpo_adapter_7b_correct
#   (compare to SFT 46.9% and DPO 50.3%)
