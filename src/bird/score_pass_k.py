"""
Pass@k scoring for BIRD CoT candidates (CPU only).

WHAT THIS ANSWERS
-----------------
Your headline 58.5% is Best-of-N *self-consistency*: generate 8 candidates, then
pick the answer whose result set the most candidates agree on. That throws away
every question where the correct SQL was a minority of the 8.

pass@k asks a different question: if we could magically pick the best of k
samples (an oracle that peeks at gold), how often is at least one correct?

    pass@k  >=  self-consistency@k  >=  greedy

The gap between pass@8 and 58.5% is the headroom a better *selector* or an RL
method (e.g. CoT-GRPO) could in principle recover. If that gap is tiny, there is
nothing to harvest and RL is pointless. If it is large, the model already knows
the answer often enough that making it the majority is a real, learnable target.

This script reuses the SAME extraction (`extract_final_sql`), execution
(`execute_sqlite_query`), and comparison (`compare_results`) as
`score_best_of_n.py`, so its self-consistency number is directly comparable to
your 58.5% and its pass@k is on the identical yardstick.

INPUT
-----
The candidates file produced (on the BIRD DEV set) by:
    python -m src.bird.build_self_sampling_pairs \
        --generate-only --cot --save-candidates <file> \
        --k 8 --temperature 0.8 ...
Each record: {question_id, db_id, gold_sql, db_path, difficulty, candidates:[K str]}

USAGE
-----
    python -m src.bird.score_pass_k \
        --candidates-file /scratch/phalle.y/results_bon_7b/dev_cot_candidates.json \
        --output-dir      /scratch/phalle.y/results_cot_passk

CPU only. No GPU -> no idle-watchdog kill. This is deliberately a separate job
from the GPU generation step.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from math import comb
from multiprocessing import Pool
from pathlib import Path

from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results
from src.bird.inference import extract_final_sql

# Match score_best_of_n.py exactly so self-consistency here == your 58.5% run.
EXEC_TIMEOUT = 15
DEFAULT_KS = (1, 2, 4, 8)


# ─────────────────────────────────────────────────────────────────────────────
# pass@k — the unbiased estimator from Chen et al. 2021 (HumanEval)
# ─────────────────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Probability that at least one of k samples, drawn WITHOUT replacement from
    the n generated samples (of which c are correct), is correct.

    Using the closed form avoids the ordering bias of a naive "any of the first
    k are right" and gives a smooth curve across k. Note:
      - pass@1 == c / n            (expected accuracy of a single sample)
      - pass@n == 1.0 if c >= 1     (the pure oracle "did ANY of the n match")
    """
    if c <= 0:
        return 0.0
    if n - c < k:            # not enough wrong samples to fill k slots -> guaranteed hit
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ─────────────────────────────────────────────────────────────────────────────
# self-consistency (majority vote) — identical logic to score_best_of_n.py
# ─────────────────────────────────────────────────────────────────────────────

def result_sig(exec_result: dict):
    """Order-independent signature of a result set (matches evaluator.normalize_rows)."""
    if not exec_result["success"]:
        return None
    return frozenset(tuple(str(v) for v in row) for row in exec_result["rows"])


def majority_correct(execs: list[dict], correct: list[bool]) -> bool:
    """
    Reproduce Best-of-N self-consistency from already-executed candidates:
    group by result set, take the most-shared one (ties -> lowest rank), and
    report whether that chosen candidate is correct. Fallback: first candidate.
    """
    sigs = [result_sig(e) for e in execs]
    valid = [s for s in sigs if s is not None]
    if valid:
        top = Counter(valid).most_common(1)[0][0]
        for i, s in enumerate(sigs):
            if s == top:
                return correct[i]
    return correct[0]


# ─────────────────────────────────────────────────────────────────────────────
# per-question scoring (runs in worker processes)
# ─────────────────────────────────────────────────────────────────────────────

def score_question(item) -> dict:
    """
    Score one question. The gold query runs once, and each DISTINCT candidate
    query runs once: the k samples often repeat the same final SQL, so this cuts
    most of the execution time without changing any result. Connections are
    read-only so a generated DELETE/DROP can't damage the dev database.
    """
    i, qid, difficulty, gold_sql, db_path, candidates = item
    gold_exec = execute_sqlite_query(gold_sql, db_path, timeout=EXEC_TIMEOUT, read_only=True)
    sqls = [extract_final_sql(c) for c in candidates]
    cache: dict[str, dict] = {}
    for s in sqls:
        if s not in cache:
            cache[s] = execute_sqlite_query(s, db_path, timeout=EXEC_TIMEOUT, read_only=True)
    execs = [cache[s] for s in sqls]
    correct = [compare_results(e, gold_exec)["result_match"] for e in execs]
    return {
        "question_id":      qid,
        "difficulty":       difficulty,
        "n":                len(correct),
        "c":                sum(correct),
        "majority_correct": majority_correct(execs, correct),
        "distinct_sql":     len(cache),
    }


def available_cpus() -> int:
    """CPUs this process may use (respects a Slurm/cgroup allocation on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="pass@k + self-consistency scoring (CPU)")
    ap.add_argument("--candidates-file", type=Path, required=True,
                    help="Candidates JSON from build_self_sampling_pairs --save-candidates (DEV set)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS),
                    help="Which k values to report pass@k for (default: 1 2 4 8)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel scoring processes (0 = CPUs available to this job, capped at 16)")
    args = ap.parse_args()

    records = json.load(open(args.candidates_file))
    n_avail = min((len(r.get("candidates", [])) for r in records), default=0)
    print(f"Loaded {len(records)} questions; min candidates/question = {n_avail}")
    ks = [k for k in args.ks if k <= n_avail]
    if not ks:
        raise SystemExit(f"No usable k values: requested {args.ks}, but only {n_avail} candidates/question")

    items, skipped = [], 0
    for i, q in enumerate(records):
        if not q.get("db_path"):
            skipped += 1
            continue
        items.append((i, q.get("question_id", i), q.get("difficulty", "unknown"),
                      q["gold_sql"], q["db_path"], q["candidates"]))

    workers = args.workers or min(16, available_cpus())
    print(f"Scoring {len(items)} questions with {workers} worker process(es) ...", flush=True)
    per_q = []          # one dict per question, same order as the input file
    start = time.time()
    with Pool(workers) as pool:
        for done, row in enumerate(pool.imap(score_question, items, chunksize=1), 1):
            per_q.append(row)
            if done % 100 == 0 or done == len(items):
                elapsed = time.time() - start
                left = elapsed / done * (len(items) - done)
                print(f"  [{done}/{len(items)}] scored  {elapsed/60:.1f} min elapsed, "
                      f"~{left/60:.1f} min left", flush=True)

    total = len(per_q)
    if total == 0:
        raise SystemExit("No scorable questions (all missing db_path?)")

    # Aggregate pass@k (mean of the per-question estimator) and self-consistency.
    passk = {k: round(sum(pass_at_k(q["n"], q["c"], k) for q in per_q) / total, 4) for k in ks}
    self_consistency = round(sum(1 for q in per_q if q["majority_correct"]) / total, 4)
    oracle = passk[max(ks)]  # pass@(largest k): "did ANY of the k match"

    # Per-difficulty pass@(max k) and self-consistency — shows WHERE the headroom is.
    per_diff = {}
    for q in per_q:
        d = q["difficulty"]
        per_diff.setdefault(d, {"total": 0, "sc": 0, "passk_sum": 0.0})
        per_diff[d]["total"] += 1
        per_diff[d]["sc"] += int(q["majority_correct"])
        per_diff[d]["passk_sum"] += pass_at_k(q["n"], q["c"], max(ks))
    for d, s in per_diff.items():
        s["self_consistency"] = round(s["sc"] / s["total"], 4)
        s[f"pass@{max(ks)}"] = round(s["passk_sum"] / s["total"], 4)

    # Group outcomes. GRPO's advantage is (reward - group mean), so only questions
    # whose samples DISAGREE (some right, some wrong) produce any gradient.
    all_right = sum(1 for q in per_q if q["c"] == q["n"])
    all_wrong = sum(1 for q in per_q if q["c"] == 0)
    groups = {
        "mixed":       round((total - all_right - all_wrong) / total, 4),
        "all_correct": round(all_right / total, 4),
        "all_wrong":   round(all_wrong / total, 4),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"bird_passk_{args.candidates_file.stem}_{ts}.json"
    payload = {
        "candidates_file": str(args.candidates_file),
        "total_questions": total,
        "skipped_no_db":   skipped,
        "pass_at_k":       passk,
        "self_consistency@%d" % max(ks): self_consistency,
        "oracle_pass@%d" % max(ks): oracle,
        "headroom": {
            "pass@%d_minus_self_consistency" % max(ks): round(oracle - self_consistency, 4),
            "note": "This gap is the ceiling a better selector or CoT-GRPO could recover.",
        },
        "per_difficulty": per_diff,
        "group_outcomes": groups,
        # One row per question so runs can be compared question by question.
        "per_question": per_q,
    }
    json.dump(payload, open(out, "w"), indent=2, default=str)

    kmax = max(ks)
    print("\n" + "=" * 64)
    print(f"  PASS@K — BIRD dev  ({total} questions, {n_avail} samples each)")
    print("=" * 64)
    for k in ks:
        print(f"    pass@{k:<2}            : {passk[k]:.1%}")
    if n_avail == 1:
        print("    (one sample per question: pass@1 is plain accuracy)")
    else:
        print(f"    self-consistency@{kmax}: {self_consistency:.1%}   (full dev should reproduce ~58.5%)")
        print("-" * 64)
        print(f"    HEADROOM  pass@{kmax} - self-consistency = {oracle - self_consistency:+.1%}")
        print(f"    GROUPS (GRPO signal): mixed {groups['mixed']:.1%} | "
              f"all-correct {groups['all_correct']:.1%} | all-wrong {groups['all_wrong']:.1%}")
    print("-" * 64)
    print("    per difficulty (self-consistency  ->  pass@%d):" % kmax)
    for d, s in sorted(per_diff.items()):
        print(f"      {d:<12}: {s['self_consistency']:.1%}  ->  {s[f'pass@{kmax}']:.1%}   ({s['total']})")
    if skipped:
        print(f"    NOTE: skipped {skipped} questions with no db_path")
    print(f"  Saved: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
