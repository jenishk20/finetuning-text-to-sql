#!/bin/bash
# ExCoT Stage 1 — job 1 of 3 : ON-POLICY GENERATION ONLY (7B, GPU)
# ---------------------------------------------------------------------------
# vLLM generates K self-samples per BIRD train question from the 7B SFT adapter
# and SAVES THE RAW CANDIDATES, then exits. It does NOT execute SQL here.
# Execution + pair-building is a SEPARATE CPU job (bird_selfsample_buildpairs_7b.sh)
# so the GPU is never left idle (the idle watchdog killed the old single-job run).
# Expected: ~5-10 min on H200 for 1500 questions — GPU busy the whole time.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=bird-ss-gen-7b
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_ss_gen_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_ss_gen_7b_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoids vLLM fork crashes

# Pin transformers to the 4.46 line: newer transformers needs torch>=2.5,
# but vllm 0.6.3 pins torch 2.4 -> import of torch.distributed.tensor.device_mesh fails.
pip install -q "vllm==0.6.3" "transformers==4.46.3" datasets 2>&1 | tail -3

# ── Config (verify SFT_ADAPTER points at a dir with adapter_config.json) ─────
BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
SFT_ADAPTER=/scratch/phalle.y/bird_sft_adapter_7b/final_adapter   # verified: dir holds adapter_config.json
DB_DIR=/scratch/phalle.y/bird_train/train/train_databases
OUT=/scratch/phalle.y/results_self_sampling_7b/bird_ss_pairs_7b.json   # final pairs (written by the CPU job)
CAND=/scratch/phalle.y/results_self_sampling_7b/candidates_7b.json     # raw candidates (this job's output)
MAXQ=${MAXQ:-1500}                                        # override: MAXQ=20 sbatch ... for smoke test

PYTHONUNBUFFERED=1 python -m src.bird.build_self_sampling_pairs \
    --base-model      "$BASE_MODEL" \
    --adapter         "$SFT_ADAPTER" \
    --output-file     "$OUT" \
    --db-dir          "$DB_DIR" \
    --k               4 \
    --temperature     0.8 \
    --max-questions   "$MAXQ" \
    --use-hf \
    --generate-only \
    --save-candidates "$CAND"
