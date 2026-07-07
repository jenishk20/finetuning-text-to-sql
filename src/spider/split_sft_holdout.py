"""
Split the Spider SFT data into a disjoint SFT set + a GRPO held-out set.

WHY: the SFT model memorized all 7,000 train_spider questions (~98% train acc), so
GRPO on that same data had zero reward variance → no learning signal. Holding out a
slice that SFT never sees gives GRPO non-saturated reward (~0.8, real variance) →
actual gradient. This is the fix for the flat-reward GRPO run.

Split is question-level (seed 42): the held-out questions live on the same DBs the
SFT still trains on, but the model won't have memorized those exact (question→SQL)
pairs, so it's uncertain on them.

Produces:
  spider_sft_split.json      — alpaca format {instruction,input,output}, for re-SFT
  spider_grpo_holdout.json   — raw {db_id,question,query}, for src.spider.grpo_dataset

Run on the login node:
  python -m src.spider.split_sft_holdout \
      --sft-data   /home/phalle.y/Jenish-DPO-GRPO/sft_data.json \
      --train-json /home/phalle.y/Jenish-DPO-GRPO/train_spider.json \
      --out-dir    /scratch/phalle.y/spider_splits \
      --holdout 1500
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(s.strip().lower().rstrip(";").split())


def main():
    ap = argparse.ArgumentParser(description="Disjoint SFT / GRPO-holdout split for Spider")
    ap.add_argument("--sft-data", type=Path, required=True, help="alpaca sft_data.json (7,000)")
    ap.add_argument("--train-json", type=Path, required=True, help="raw train_spider.json (7,000)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--holdout", type=int, default=1500, help="# questions held out for GRPO")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sft = json.load(open(args.sft_data))
    raw = json.load(open(args.train_json))
    assert len(sft) == len(raw), f"length mismatch: sft {len(sft)} vs raw {len(raw)}"

    # Confirm the two files are in the same order (sft[i].output == raw[i].query).
    for i in random.Random(0).sample(range(len(sft)), 20):
        assert norm(sft[i]["output"]) == norm(raw[i]["query"]), (
            f"alignment broken at index {i}: sft output != train_spider query — "
            "files are not in the same order; split by index is unsafe.")
    print(f"Alignment verified (20 spot-checks over {len(sft)} examples).")

    idx = list(range(len(sft)))
    random.Random(args.seed).shuffle(idx)
    hold_idx = sorted(idx[:args.holdout])
    sft_idx = sorted(idx[args.holdout:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sft_out = [sft[i] for i in sft_idx]
    hold_out = [{"db_id": raw[i]["db_id"], "question": raw[i]["question"], "query": raw[i]["query"]}
                for i in hold_idx]

    sft_path = args.out_dir / "spider_sft_split.json"
    hold_path = args.out_dir / "spider_grpo_holdout.json"
    json.dump(sft_out, open(sft_path, "w"), indent=2)
    json.dump(hold_out, open(hold_path, "w"), indent=2)

    sft_dbs = {raw[i]["db_id"] for i in sft_idx}
    hold_dbs = {raw[i]["db_id"] for i in hold_idx}
    print(f"\nSFT set:      {len(sft_out):>5} → {sft_path}")
    print(f"GRPO holdout: {len(hold_out):>5} → {hold_path}")
    print(f"DBs — SFT {len(sft_dbs)}, holdout {len(hold_dbs)}, shared {len(sft_dbs & hold_dbs)} "
          f"(question-level split → DBs overlap, expected & fine)")
    print("\nDisjoint by question ✓ — the SFT model will NOT have seen the holdout Q→SQL pairs.")


if __name__ == "__main__":
    main()
