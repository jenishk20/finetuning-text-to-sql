#!/bin/bash
# ExCoT — VALIDATE Step 2.3: did on-policy CoT-DPO (round 1) actually help?
# ---------------------------------------------------------------------------
# Fully self-contained: (1) CoT-evals bird_cot_dpo_r1_7b on the FIRST 500 BIRD
# dev questions, then (2) prints an apples-to-apples comparison vs CoT-SFT on the
# SAME 500 (read from the existing CoT-SFT eval file). Just `sbatch` and read the
# .out — no interactive commands needed.
# Greedy CoT eval, interleaved (one GPU gen per question) -> idle-safe. ~2-2.5h.
# ---------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bird-validate-cotdpo
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_validate_cotdpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_validate_cotdpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

DPO_ADAPTER=/scratch/phalle.y/bird_cot_dpo_r1_7b/final_adapter
DPO_OUT=/scratch/phalle.y/results_cot_dpo_r1_eval
SFT_OUT=/scratch/phalle.y/results_cot_sft_7b_eval   # the existing CoT-SFT eval (1450 results)

# ── 1. CoT-eval the DPO-r1 adapter on the first 500 dev questions ──────────────
PYTHONUNBUFFERED=1 python -m src.bird.eval_finetuned \
    --adapter    "$DPO_ADAPTER" \
    --dev-json   /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
    --db-dir     /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
    --output-dir "$DPO_OUT" \
    --cot \
    --limit 500

# ── 2. Apples-to-apples: CoT-DPO vs CoT-SFT on the SAME first 500 ──────────────
python - "$DPO_OUT" "$SFT_OUT" <<'PY'
import json, glob, sys
dpo_dir, sft_dir = sys.argv[1], sys.argv[2]
dpo_files = sorted(glob.glob(dpo_dir + '/*.json'))
sft_files = sorted(glob.glob(sft_dir + '/*.json'))
if not dpo_files:
    print("No DPO eval file found — eval step may have failed."); raise SystemExit
d = json.load(open(dpo_files[-1]))['results']
n = len(d)
dc = sum(1 for x in d if x['eval'].get('result_match'))
print("\n" + "=" * 60)
print("  STEP 2.3 VALIDATION — CoT-DPO vs CoT-SFT (same first 500)")
print("=" * 60)
if sft_files:
    s = json.load(open(sft_files[-1]))['results'][:n]
    sc = sum(1 for x in s if x['eval'].get('result_match'))
    print(f"  CoT-SFT  first-{n}: {sc/n:.1%}  ({sc}/{n})")
    print(f"  CoT-DPO  first-{n}: {dc/n:.1%}  ({dc}/{n})")
    print(f"  delta (DPO - SFT): {(dc-sc)/n*100:+.1f} pp")
else:
    print(f"  CoT-DPO  first-{n}: {dc/n:.1%}  ({dc}/{n})  (no CoT-SFT file to compare)")
print("=" * 60)
print("  ~0 = DPO flat (pivot to Best-of-N) | >=+2pp = helped | - = hurt")
print("=" * 60)
PY
