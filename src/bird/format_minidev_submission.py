"""
Convert eval_finetuned.py output to BIRD mini-dev submission format.

Reads bird_eval_*.json (output of src.bird.eval_finetuned) and writes
a JSON file in BIRD's required submission format:

    {
      "0": "<SQL>\\t----- bird -----\\t<db_id>",
      "1": "<SQL>\\t----- bird -----\\t<db_id>",
      ...
    }

Their evaluator at mini_dev/evaluation/evaluation_utils.py:package_sqls()
iterates this dict in insertion order, splits each value on the literal
'\\t----- bird -----\\t' separator, and pairs the SQL with the gold query
in the same position.

Usage:
    python -m src.bird.format_minidev_submission \\
        --eval-file   /scratch/phalle.y/results_bird_minidev/bird_eval_*.json \\
        --output-file /scratch/phalle.y/results_bird_minidev/predict_minidev_qwen7b_bird_dpo.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_submission(eval_file: Path, output_file: Path):
    with open(eval_file) as f:
        data = json.load(f)

    results = data["results"]

    # Sort by question_id so ordering matches the gold SQL file order
    results = sorted(results, key=lambda r: r["question_id"])

    submission: dict[str, str] = {}
    skipped_empty = 0

    for r in results:
        qid   = str(r["question_id"])
        sql   = (r.get("generated_sql") or "").strip().rstrip(";").strip()
        db_id = r["db_id"]

        if not sql:
            sql = "SELECT 1"  # placeholder so eval doesn't crash on empty
            skipped_empty += 1

        submission[qid] = f"{sql}\t----- bird -----\t{db_id}"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved {len(submission)} predictions to {output_file}")
    if skipped_empty:
        print(f"  Note: {skipped_empty} predictions were empty — replaced with 'SELECT 1' placeholder")
    print(f"\nNext: edit {Path(output_file).name} into mini_dev/evaluation/run_evaluation.sh")
    print(f"      as predicted_sql_path, then run: sh run_evaluation.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Format eval_finetuned output for BIRD mini-dev leaderboard submission"
    )
    parser.add_argument("--eval-file",   type=Path, required=True,
                        help="Path to bird_eval_*.json (output of src.bird.eval_finetuned)")
    parser.add_argument("--output-file", type=Path, required=True,
                        help="Where to save the BIRD-format submission JSON")
    args = parser.parse_args()

    format_submission(args.eval_file, args.output_file)


if __name__ == "__main__":
    main()
