#!/bin/bash
# ExCoT — compare CoT-DPO-r1 vs CoT-SFT on the overlapping first-N dev questions.
# Reads EXISTING eval JSONs only (works on the PARTIAL DPO eval too) — no model,
# no GPU, runs in seconds. Just `sbatch` and read the .out.
# ---------------------------------------------------------------------------

#SBATCH --partition=short        # CPU partition (verify with `sinfo -s`)
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=bird-cmp-cotdpo
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_cmp_cotdpo_%j.out
#SBATCH --error=/scratch/phalle.y/bird_cmp_cotdpo_%j.err

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/finetuning-text-to-sql

python - <<'PY'
import json, glob
dpo = sorted(glob.glob('/scratch/phalle.y/results_cot_dpo_r1_eval/*.json'))
sft = sorted(glob.glob('/scratch/phalle.y/results_cot_sft_7b_eval/*.json'))
if not dpo:
    print("No DPO eval file in results_cot_dpo_r1_eval — nothing to compare."); raise SystemExit
d = json.load(open(dpo[-1]))['results']
n = len(d)
dc = sum(1 for x in d if x['eval'].get('result_match'))
de = sum(1 for x in d if x['eval'].get('generated_executed'))
print("=" * 60)
print(f"  CoT-DPO-r1 ({n} questions evaluated{' — PARTIAL' if n < 500 else ''})")
print(f"    result_accuracy:    {dc/n:.1%}  ({dc}/{n})")
print(f"    execution_accuracy: {de/n:.1%}")
if sft:
    s = json.load(open(sft[-1]))['results'][:n]   # same first-N questions
    sc = sum(1 for x in s if x['eval'].get('result_match'))
    se = sum(1 for x in s if x['eval'].get('generated_executed'))
    print(f"  CoT-SFT (same first {n}):")
    print(f"    result_accuracy:    {sc/n:.1%}  ({sc}/{n})")
    print(f"    execution_accuracy: {se/n:.1%}")
    print("-" * 60)
    print(f"  delta result (DPO - SFT): {(dc-sc)/n*100:+.1f} pp")
    print(f"  delta exec   (DPO - SFT): {(de-se)/n*100:+.1f} pp  (very negative => DPO made it verbose/unstable)")
print("=" * 60)
print("  result ~0 = DPO flat (pivot to Best-of-N) | >=+2pp = helped | - = hurt")
PY
