"""
Best-of-N inference for BIRD dev evaluation using vLLM.

Generates K candidate SQLs per question, executes each against the database,
and selects the first one that runs successfully and returns non-empty results.
Falls back to the first generated candidate if all fail.

Output JSON is compatible with the existing analysis tools and matches the
format produced by src.bird.eval_finetuned (same `results` schema).

Usage:
    python -m src.bird.eval_best_of_n \\
        --base-model Qwen/Qwen2.5-Coder-14B-Instruct \\
        --adapter    /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \\
        --dev-json   /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev.json \\
        --db-dir     /home/phalle.y/Jenish-DPO-GRPO/bird_data/dev_databases \\
        --output-dir /scratch/phalle.y/results_14b_best_of_n_eval \\
        --k          4 \\
        --temperature 0.8
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from src.bird.inference import BIRDvLLMEngine
from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers (kept inline to avoid Path coupling with shared/schema_loader)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_path(db_dir: Path, db_id: str) -> str:
    p = Path(db_dir) / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(f"DB not found: {p}")
    return str(p)


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    parts = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Best-of-N selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_candidate(
    candidates: list[str],
    db_path: str,
) -> tuple[str, dict, int]:
    """
    Self-consistency selection: execute all K candidates, group by result set,
    and return the SQL whose result is shared by the most other candidates
    (majority vote). Ties broken by lowest rank.

    This is the standard "execution-based self-consistency" used in SQL-PaLM,
    MAC-SQL, CodeS — far stronger than 'first that runs'.
    """
    from collections import Counter, defaultdict

    # 1. Execute all candidates
    exec_results = [execute_sqlite_query(sql, db_path) for sql in candidates]

    # 2. Hash result rows for grouping (only for successful executions)
    def result_key(res: dict) -> str | None:
        if not res["success"] or res.get("row_count", 0) == 0:
            return None
        rows = res.get("rows", [])
        try:
            return json.dumps(rows, sort_keys=True, default=str)
        except Exception:
            return str(rows)

    keys = [result_key(r) for r in exec_results]

    # 3. Vote — count occurrences of each non-null result key
    votes = Counter(k for k in keys if k is not None)

    if votes:
        winning_key, _ = votes.most_common(1)[0]
        # Pick the lowest-index candidate that produced the winning result
        for idx, k in enumerate(keys):
            if k == winning_key:
                return candidates[idx], exec_results[idx], idx

    # 4. All failed — fall back to first candidate
    return candidates[0], exec_results[0], 0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    base_model: str,
    adapter: Path | None,
    dev_json: Path,
    db_dir: Path,
    output_dir: Path,
    k: int,
    temperature: float,
    limit: int | None = None,
    cutoff_len: int = 8192,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    short_name  = base_model.split("/")[-1].lower().replace("qwen2.5-coder-", "qwen").replace("-instruct", "")
    model_label = f"{short_name}-bird-best-of-{k}" if adapter else f"{short_name}-bird-base-best-of-{k}"

    # ── Load questions ────────────────────────────────────────────────────────
    with open(dev_json) as f:
        dev_questions = json.load(f)
    if limit:
        dev_questions = dev_questions[:limit]
    total = len(dev_questions)

    # ── Build schema cache before loading model (avoid duplicate work) ────────
    print(f"\n[1/3] Loading schemas ...")
    schema_cache: dict[str, tuple[str, str]] = {}
    for q in dev_questions:
        db_id = q["db_id"]
        if db_id not in schema_cache:
            try:
                dp = get_db_path(db_dir, db_id)
                schema_cache[db_id] = (get_schema(dp), dp)
            except FileNotFoundError as e:
                print(f"  WARN: {e}")
    print(f"      Loaded schemas for {len(schema_cache)} databases")

    # Build (question, schema, evidence) tuples in the same order as dev_questions
    items = []
    valid_indices: list[int] = []
    for idx, q in enumerate(dev_questions):
        if q["db_id"] not in schema_cache:
            continue
        schema, _ = schema_cache[q["db_id"]]
        items.append((q["question"], schema, q.get("evidence", "").strip()))
        valid_indices.append(idx)

    print(f"\n[2/3] Loading vLLM engine (base={base_model}, adapter={adapter}) ...")
    engine = BIRDvLLMEngine(
        base_model=base_model,
        adapter_path=str(adapter) if adapter else None,
        max_model_len=cutoff_len,
    )

    # ── Generate K candidates per question (batched by vLLM) ──────────────────
    print(f"\n[3/3] Generating K={k} candidates for {len(items)} questions ...")
    gen_start  = time.time()
    candidates = engine.generate(items, k=k, temperature=temperature)
    gen_elapsed = round((time.time() - gen_start) / 60, 1)
    print(f"      Generation done in {gen_elapsed} min ({len(items)*k} total samples)")

    # ── Execute candidates, pick best, evaluate ───────────────────────────────
    print(f"\n[4/3] Executing candidates + comparing to gold ...")
    results = []
    pick_dist = [0] * k   # how often each rank wins

    for cand_idx, original_idx in enumerate(valid_indices):
        q              = dev_questions[original_idx]
        question_id    = str(q.get("question_id", original_idx))
        db_id          = q["db_id"]
        question       = q["question"]
        evidence       = q.get("evidence", "").strip()
        gold_sql       = q.get("SQL") or q.get("query", "")
        difficulty     = q.get("difficulty", "unknown")
        schema, db_path = schema_cache[db_id]

        cands               = candidates[cand_idx]
        chosen_sql, exec_r, picked_at = pick_best_candidate(cands, db_path)
        pick_dist[picked_at] += 1

        gold_exec  = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(exec_r, gold_exec)

        if (cand_idx + 1) % 50 == 0:
            print(f"  [{cand_idx+1}/{len(valid_indices)}] picked@{picked_at} | {db_id}")

        results.append({
            "question_id":     question_id,
            "db_id":           db_id,
            "question":        question,
            "evidence":        evidence,
            "gold_sql":        gold_sql,
            "generated_sql":   chosen_sql,
            "all_candidates":  cands,
            "picked_index":    picked_at,
            "difficulty":      difficulty,
            "gen_execution":  {"success": exec_r["success"],  "row_count": exec_r["row_count"],  "error": exec_r["error"]},
            "gold_execution": {"success": gold_exec["success"], "row_count": gold_exec["row_count"], "error": gold_exec["error"]},
            "eval": evaluation,
        })

    metrics = compute_metrics(results)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bird_eval_{model_label}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model":           model_label,
                "base_model":      base_model,
                "adapter":         str(adapter) if adapter else None,
                "k":               k,
                "temperature":     temperature,
                "timestamp":       timestamp,
                "benchmark":       "bird_dev",
                "total_questions": total,
                "completed":       len(results),
            },
            "metrics":   metrics,
            "pick_distribution": {
                f"rank_{i}": pick_dist[i] for i in range(k)
            },
            "results":   results,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  BEST-OF-{k} RESULTS — {model_label}")
    print(f"{'=' * 70}")
    print(f"  Total questions:    {metrics.get('total_questions', 0)}")
    print(f"  Execution accuracy: {metrics.get('execution_accuracy', 0):.1%}")
    print(f"  Result accuracy:    {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Row count accuracy: {metrics.get('row_count_accuracy', 0):.1%}")
    print(f"\n  Pick distribution (which rank candidate won):")
    for i in range(k):
        pct = pick_dist[i] / max(1, sum(pick_dist)) * 100
        print(f"    rank {i}: {pick_dist[i]:>4} ({pct:.1f}%)")
    print(f"{'=' * 70}")
    print(f"  Saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Best-of-N BIRD evaluation with vLLM")
    parser.add_argument("--base-model",  type=str,  required=True)
    parser.add_argument("--adapter",     type=Path, default=None)
    parser.add_argument("--dev-json",    type=Path, required=True)
    parser.add_argument("--db-dir",      type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, required=True)
    parser.add_argument("--k",           type=int,   default=4,
                        help="Number of candidates per question (Best-of-K)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (ignored if K=1)")
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Eval only first N questions (smoke test)")
    parser.add_argument("--cutoff-len",  type=int,   default=8192)
    args = parser.parse_args()

    run_evaluation(
        base_model=args.base_model,
        adapter=args.adapter,
        dev_json=args.dev_json,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        k=args.k,
        temperature=args.temperature,
        limit=args.limit,
        cutoff_len=args.cutoff_len,
    )


if __name__ == "__main__":
    main()
