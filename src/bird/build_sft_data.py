"""
Build BIRD SFT training data from xu3kev/BIRD-SQL-data-train HuggingFace dataset.

Outputs JSONL with `instruction` + `output` fields matching the same prompt
format used in DPO training (eval_finetuned.py:build_instruction).

Usage:
    python -m src.bird.build_sft_data \\
        --output-file /scratch/phalle.y/bird_sft_data.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.bird.inference import build_instruction


def main():
    parser = argparse.ArgumentParser(description="Build BIRD SFT training data")
    parser.add_argument("--output-file", type=Path, required=True,
                        help="Where to save the SFT JSON file")
    parser.add_argument("--dataset-id",  type=str,
                        default="xu3kev/BIRD-SQL-data-train",
                        help="HuggingFace dataset ID")
    args = parser.parse_args()

    print(f"Loading {args.dataset_id} from HuggingFace...")
    ds = load_dataset(args.dataset_id, split="train")
    print(f"Loaded {len(ds)} examples")

    examples = []
    skipped  = 0
    for row in ds:
        question = row.get("question", "").strip()
        schema   = row.get("schema",   "").strip()
        evidence = row.get("evidence", "").strip()
        sql      = row.get("SQL",      "").strip()

        if not (question and schema and sql):
            skipped += 1
            continue

        examples.append({
            "instruction": build_instruction(question, schema, evidence),
            "input":       "",
            "output":      sql,
        })

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nSaved {len(examples)} SFT examples to {args.output_file}")
    if skipped:
        print(f"  Skipped {skipped} rows with missing fields")

    # Quick token-length stats
    lens = [len(ex["instruction"] + ex["output"]) // 4 for ex in examples]  # rough token estimate
    avg  = sum(lens) // len(lens)
    print(f"\n  Avg estimated tokens: {avg}")
    print(f"  Max estimated tokens: {max(lens)}")
    print(f"  Over 4096 tokens:     {sum(1 for l in lens if l > 4096)}")
    print(f"  Over 8192 tokens:     {sum(1 for l in lens if l > 8192)}")


if __name__ == "__main__":
    main()
