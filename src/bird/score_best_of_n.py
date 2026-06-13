"""
ExCoT — Best-of-N self-consistency SCORING from pre-generated CoT candidates (CPU).

Reads the K-candidate file produced by:
  build_self_sampling_pairs --generate-only --cot --save-candidates <file>
(run on the BIRD DEV set). For each question: extract each candidate's final SQL,
execute it, pick the result set shared by the MOST candidates (execution-based
self-consistency / majority vote), and score that pick against gold.

CPU only -> no GPU, so no idle-watchdog kill. Splitting generation (GPU) from
scoring (CPU) is why this is two jobs.

Usage:
  python -m src.bird.score_best_of_n \
      --candidates-file /scratch/phalle.y/results_bon_7b/dev_cot_candidates.json \
      --output-dir      /scratch/phalle.y/results_bon_7b_eval
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results, compute_metrics
from src.bird.inference import extract_final_sql

EXEC_TIMEOUT = 15


def result_sig(exec_result: dict):
    """Order-independent signature of a result set (matches evaluator.normalize_rows)."""
    if not exec_result["success"]:
        return None
    return frozenset(tuple(str(v) for v in row) for row in exec_result["rows"])


def pick_self_consistency(sqls: list[str], db_path: str):
    """Execute all K, majority-vote by result set, ties broken by lowest rank.
    Returns (chosen_sql, chosen_exec, rank). Matches the 14B Best-of-N method."""
    execs = [execute_sqlite_query(s, db_path, timeout=EXEC_TIMEOUT) for s in sqls]
    sigs = [result_sig(e) for e in execs]
    valid = [s for s in sigs if s is not None]
    if valid:
        top = Counter(valid).most_common(1)[0][0]   # most-shared result set
        for i, s in enumerate(sigs):
            if s == top:
                return sqls[i], execs[i], i
    return sqls[0], execs[0], 0   # all failed -> fall back to first


def main():
    ap = argparse.ArgumentParser(description="Best-of-N self-consistency scoring (CPU)")
    ap.add_argument("--candidates-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cands = json.load(open(args.candidates_file))
    k = len(cands[0]["candidates"]) if cands else 0
    print(f"Loaded {len(cands)} questions with candidates (K={k})")

    results = []
    for i, q in enumerate(cands):
        sqls = [extract_final_sql(c) for c in q["candidates"]]
        best_sql, best_exec, rank = pick_self_consistency(sqls, q["db_path"])
        gold_exec = execute_sqlite_query(q["gold_sql"], q["db_path"], timeout=EXEC_TIMEOUT)
        ev = compare_results(best_exec, gold_exec)
        results.append({
            "question_id": q.get("question_id", i), "db_id": q["db_id"],
            "difficulty": q.get("difficulty", "unknown"),
            "generated_sql": best_sql, "picked_rank": rank, "eval": ev,
        })
        if (i + 1) % 200 == 0:
            acc = sum(1 for r in results if r["eval"]["result_match"]) / len(results)
            print(f"  [{i+1}/{len(cands)}] running result_acc={acc:.1%}", flush=True)

    metrics = compute_metrics(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"bird_bon_eval_k{k}_{ts}.json"
    json.dump({"k": k, "metrics": metrics, "results": results}, open(out, "w"), indent=2, default=str)

    rc = Counter(r["picked_rank"] for r in results)
    print("\n" + "=" * 60)
    print(f"  BEST-OF-N (K={k}) SELF-CONSISTENCY — BIRD dev")
    print("=" * 60)
    print(f"  Questions:          {metrics.get('total_questions', 0)}")
    print(f"  Result accuracy:    {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Execution accuracy: {metrics.get('execution_accuracy', 0):.1%}")
    for diff, s in metrics.get("per_difficulty", {}).items():
        print(f"    {diff:<12}: {s.get('result_accuracy', 0):.1%}  ({s.get('total', 0)})")
    print(f"  Pick-rank dist: " + ", ".join(f"r{j}={rc.get(j, 0)}" for j in range(k)))
    print(f"  Saved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
