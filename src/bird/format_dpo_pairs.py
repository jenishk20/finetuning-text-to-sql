"""
Format BIRD preference pairs into LLaMA-Factory DPO training format.

Reads bird_preference_pairs.json (output of build_pairs.py) and writes
the pairwise DPO JSON that LLaMA-Factory expects.

By default only uses `clear_preference` pairs (one model right, one wrong)
which are the strongest training signal. Use --include-judge to also include
judge-resolved pairs from bird_preference_pairs_final.json.

Usage:
    # Clear preference only (recommended — 1,226 pairs)
    python -m src.bird.format_dpo_pairs \\
        --pairs-dir /scratch/phalle.y/results_frontier_pairs \\
        --output-dir /scratch/phalle.y/results_frontier_pairs

    # Include judge-resolved pairs too (4,684 total — run after llm_judge)
    python -m src.bird.format_dpo_pairs \\
        --pairs-dir /scratch/phalle.y/results_frontier_pairs \\
        --output-dir /scratch/phalle.y/results_frontier_pairs \\
        --include-judge

    # Dry run — print stats without saving
        --dry-run
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_sql(s: str) -> str:
    return " ".join(s.strip().lower().rstrip(";").split())


def is_degenerate(chosen: str, rejected: str) -> bool:
    """Skip pairs where chosen and rejected are identical after normalization."""
    return normalize_sql(chosen) == normalize_sql(rejected)


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def format_pairs(
    pairs_dir: Path,
    output_dir: Path,
    include_judge: bool = False,
    dry_run: bool = False,
):
    print("=" * 60)
    print("BIRD Frontier DPO — Format for LLaMA-Factory")
    print("=" * 60)

    # Decide which input file to read
    if include_judge:
        input_file = pairs_dir / "bird_preference_pairs_final.json"
        if not input_file.exists():
            raise FileNotFoundError(
                f"Final pairs file not found: {input_file}\n"
                "Run llm_judge.py first, or omit --include-judge to use clear pairs only."
            )
        print(f"\nInput: {input_file.name} (clear + judge-resolved pairs)")
    else:
        input_file = pairs_dir / "bird_preference_pairs.json"
        print(f"\nInput: {input_file.name} (clear preference pairs only)")

    with open(input_file) as f:
        pairs = json.load(f)
    print(f"Loaded: {len(pairs)} pairs")

    # Filter: only keep clear_preference (and judge_resolved if --include-judge)
    allowed_categories = {"clear_preference"}
    if include_judge:
        allowed_categories.add("judge_resolved")

    dpo_examples: list[dict] = []
    skipped_degenerate = 0
    skipped_null = 0
    skipped_category = 0
    category_counts: Counter = Counter()
    chosen_source_counts: Counter = Counter()
    token_lengths: list[int] = []

    for pair in pairs:
        category = pair.get("category", "")

        if category not in allowed_categories:
            skipped_category += 1
            continue

        chosen_sql   = pair.get("chosen_sql")
        rejected_sql = pair.get("rejected_sql")

        if not chosen_sql or not rejected_sql:
            skipped_null += 1
            continue

        if is_degenerate(chosen_sql, rejected_sql):
            skipped_degenerate += 1
            continue

        instruction = pair["instruction"]

        dpo_examples.append({
            "instruction": instruction,
            "input":       "",
            "chosen":      chosen_sql,
            "rejected":    rejected_sql,
        })

        category_counts[category] += 1
        chosen_source_counts[pair.get("chosen_source", "unknown")] += 1
        token_lengths.append(estimate_tokens(instruction + chosen_sql + rejected_sql))

    # Stats
    avg_tokens = int(sum(token_lengths) / len(token_lengths)) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    over_4096  = sum(1 for t in token_lengths if t > 4096)
    over_2048  = sum(1 for t in token_lengths if t > 2048)

    print(f"\n{'─' * 50}")
    print("FORMATTING SUMMARY")
    print(f"{'─' * 50}")
    print(f"  DPO examples produced:    {len(dpo_examples)}")
    print(f"  Skipped (wrong category): {skipped_category}")
    print(f"  Skipped (null SQL):       {skipped_null}")
    print(f"  Skipped (degenerate):     {skipped_degenerate}")
    print(f"\n  Category breakdown:")
    for cat, n in category_counts.items():
        print(f"    {cat:<22}: {n}")
    print(f"\n  Chosen source breakdown:")
    for src, n in chosen_source_counts.items():
        print(f"    {src:<22}: {n}")
    print(f"\n  Token length stats (rough estimate):")
    print(f"    Avg tokens per example:  {avg_tokens:,}")
    print(f"    Max tokens:              {max_tokens:,}")
    print(f"    Over 2048 tokens:        {over_2048}")
    print(f"    Over 4096 tokens:        {over_4096}")

    # dataset_info.json entry
    dataset_info = {
        "bird_frontier_dpo": {
            "file_name": "bird_frontier_dpo_data.json",
            "ranking":   True,
            "columns": {
                "prompt":    "instruction",
                "query":     "input",
                "chosen":    "chosen",
                "rejected":  "rejected",
            },
        }
    }

    if dry_run:
        print(f"\n[dry-run] Would save {len(dpo_examples)} examples to {output_dir}/bird_frontier_dpo_data.json")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_output_file    = output_dir / "bird_frontier_dpo_data.json"
    dataset_info_file  = output_dir / "dataset_info.json"

    with open(dpo_output_file, "w") as f:
        json.dump(dpo_examples, f, indent=2)

    with open(dataset_info_file, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n  Saved:")
    print(f"    {dpo_output_file}")
    print(f"    {dataset_info_file}")
    print(f"\nDone ✓")
    print(f"\nNext: copy these files to your LLaMA-Factory data/ directory,")
    print(f"then reference 'bird_frontier_dpo' in your training YAML.")


def main():
    parser = argparse.ArgumentParser(
        description="Format BIRD preference pairs for LLaMA-Factory DPO training"
    )
    parser.add_argument(
        "--pairs-dir", type=Path, required=True,
        help="Directory containing bird_preference_pairs.json (output of build_pairs.py)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save output files (default: same as --pairs-dir)"
    )
    parser.add_argument(
        "--include-judge", action="store_true",
        help="Also include judge-resolved pairs (reads bird_preference_pairs_final.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, do not save any files"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.pairs_dir

    format_pairs(
        pairs_dir=args.pairs_dir,
        output_dir=output_dir,
        include_judge=args.include_judge,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
