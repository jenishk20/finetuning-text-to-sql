"""
Generate clear_preference DPO pairs from two teachers on the Spider TRAIN set.

For each Spider train question, run teacher A and teacher B, execute both (and the
gold SQL) against the train DB, and keep ONLY clear_preference pairs — exactly one
teacher correct. Both-correct (judge/style) and both-wrong (gold-fallback) are
SKIPPED, because both regressed prior runs (-9.6pp judge, crash on gold-fallback).

This is the clean redo of the Spider DPO data: pairs come from the TRAIN set, so
nothing leaks into the dev eval. Runs LOCALLY — teachers via W&B Inference, SQL
executed against local train SQLite DBs. Concurrent (thread pool) since the calls
are I/O-bound. Saves raw teacher results (resumable + reusable) AND the pairs.

Usage:
    # full train set, both confirmed teachers
    python -m src.spider.build_train_pairs

    # quick smoke test
    python -m src.spider.build_train_pairs --limit 5 --workers 4

    # resume an interrupted run
    python -m src.spider.build_train_pairs --resume results/spider_train_pairs/spider_train_teacher_results.json
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.shared.llm_client import call_llm, resolve_model
from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.schema_loader import get_schema_from_sqlite, get_db_path
from src.shared.evaluator import compare_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Confirmed teachers from the dev bake-off (Qwen 78.8%, V4-Pro 76.2%, cross-family).
TEACHER_A = "qwen3-coder-480b"
TEACHER_B = "deepseek-v4-pro"


def build_instruction(question: str, schema: str) -> str:
    """Must match the SFT/inference template (same as format_training.build_instruction)."""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


def call_with_retry(retries: int, backoff: float, **kwargs) -> dict:
    last = None
    for i in range(retries):
        try:
            return call_llm(**kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))
    raise last


def process_question(idx, q, schema, db_path, provider, teacher_a, teacher_b,
                     temperature, max_tokens, retries, backoff) -> dict:
    question = q["question"]
    gold_sql = q["query"]
    db_id = q["db_id"]

    out_a = call_with_retry(retries, backoff, question=question, schema=schema,
                            provider=provider, model=teacher_a,
                            temperature=temperature, max_tokens=max_tokens)
    out_b = call_with_retry(retries, backoff, question=question, schema=schema,
                            provider=provider, model=teacher_b,
                            temperature=temperature, max_tokens=max_tokens)
    sql_a, sql_b = out_a["generated_sql"], out_b["generated_sql"]

    ex_g = execute_sqlite_query(gold_sql, db_path)
    a_ok = compare_results(execute_sqlite_query(sql_a, db_path), ex_g)["result_match"]
    b_ok = compare_results(execute_sqlite_query(sql_b, db_path), ex_g)["result_match"]

    if a_ok and b_ok:
        category = "both_correct"
    elif a_ok or b_ok:
        category = "clear_preference"
    else:
        category = "both_wrong"

    return {
        "train_index": idx, "db_id": db_id, "question": question,
        "gold_sql": gold_sql, "sql_a": sql_a, "sql_b": sql_b,
        "a_correct": a_ok, "b_correct": b_ok, "category": category,
        "instruction": build_instruction(question, schema),
    }


def derive_pairs(results: list[dict], teacher_a: str, teacher_b: str) -> list[dict]:
    """clear_preference only: chosen = correct teacher's SQL, rejected = wrong one's."""
    pairs = []
    for r in results:
        if r["category"] != "clear_preference":
            continue
        if r["a_correct"]:
            chosen, chosen_src, rejected, rejected_src = r["sql_a"], teacher_a, r["sql_b"], teacher_b
        else:
            chosen, chosen_src, rejected, rejected_src = r["sql_b"], teacher_b, r["sql_a"], teacher_a
        if chosen.strip() == rejected.strip():
            continue  # degenerate — no signal
        pairs.append({
            "instruction": r["instruction"], "input": "",
            "chosen": chosen, "rejected": rejected,
            "category": "clear_preference",
            "chosen_source": chosen_src, "rejected_source": rejected_src,
            "db_id": r["db_id"], "train_index": r["train_index"],
        })
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Build clear_preference Spider train DPO pairs from two teachers")
    ap.add_argument("--teacher-a", default=TEACHER_A)
    ap.add_argument("--teacher-b", default=TEACHER_B)
    ap.add_argument("--provider", default="wandb")
    ap.add_argument("--train-json", type=Path, default=PROJECT_ROOT / "data" / "spider_data" / "train_spider.json")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "spider_data")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "spider_train_pairs")
    ap.add_argument("--limit", type=int, default=None, help="Only first N train questions (smoke test)")
    ap.add_argument("--max-pairs", type=int, default=None, help="Stop once this many clear_preference pairs collected")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent question workers")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--save-every", type=int, default=100, help="Checkpoint raw results every N questions")
    ap.add_argument("--resume", type=Path, default=None, help="Resume from a raw-results JSON")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / "spider_train_teacher_results.json"
    pairs_file = args.output_dir / "spider_train_pairs.json"
    stats_file = args.output_dir / "spider_train_pairs_stats.json"

    with open(args.train_json) as f:
        train = json.load(f)
    if args.limit:
        train = train[:args.limit]

    full_a = resolve_model(args.teacher_a, args.provider)
    full_b = resolve_model(args.teacher_b, args.provider)

    # Resume: reload completed indices
    results: list[dict] = []
    done_idx: set[int] = set()
    resume_path = args.resume or (results_file if results_file.exists() else None)
    if resume_path and Path(resume_path).exists():
        with open(resume_path) as f:
            results = json.load(f).get("results", [])
        done_idx = {r["train_index"] for r in results}
        print(f"Resuming — {len(done_idx)} questions already done")

    todo = [(i, q) for i, q in enumerate(train) if i not in done_idx]

    # Preload schemas single-threaded (thread-safe read afterwards)
    print(f"Preloading schemas for {len({q['db_id'] for _, q in todo})} databases...")
    schema_cache: dict[str, str] = {}
    for _, q in todo:
        db = q["db_id"]
        if db not in schema_cache:
            try:
                schema_cache[db] = get_schema_from_sqlite(get_db_path(str(args.data_dir), db))
            except FileNotFoundError as e:
                print(f"  WARN missing DB {db}: {e}")

    print(f"\n{'='*78}")
    print(f"  SPIDER TRAIN PAIR GENERATION (clear_preference only)")
    print(f"  Teacher A: {full_a}")
    print(f"  Teacher B: {full_b}")
    print(f"  Questions to run: {len(todo)} (of {len(train)}) | workers: {args.workers}")
    print(f"{'='*78}\n")

    lock = threading.Lock()
    counts = {"clear_preference": 0, "both_correct": 0, "both_wrong": 0}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    t0 = time.time()
    stop = threading.Event()

    def worker(item):
        idx, q = item
        if stop.is_set():
            return None
        db = q["db_id"]
        if db not in schema_cache:
            return None
        return process_question(idx, q, schema_cache[db], get_db_path(str(args.data_dir), db),
                                args.provider, args.teacher_a, args.teacher_b,
                                args.temperature, args.max_tokens, args.retries, args.backoff)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, item): item for item in todo}
        for fut in as_completed(futures):
            idx, q = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [{idx}] FAILED: {str(e)[:120]}")
                continue
            if r is None:
                continue
            with lock:
                results.append(r)
                counts[r["category"]] += 1
                completed += 1
                cp = counts["clear_preference"]
                if r["category"] == "clear_preference":
                    tag = "PAIR ✓"
                elif r["category"] == "both_correct":
                    tag = "skip(both-ok)"
                else:
                    tag = "skip(both-wrong)"
                print(f"  [{completed}/{len(todo)}] {r['db_id']:<18} {tag:<16} clear_pref={cp}")

                if completed % args.save_every == 0:
                    _save_results(results_file, results, full_a, full_b, counts, len(train))
                    print(f"    --- checkpoint: {completed} done, {cp} clear_preference pairs ---")

                if args.max_pairs and cp >= args.max_pairs:
                    print(f"\n  Hit --max-pairs={args.max_pairs}; stopping early.")
                    stop.set()
                    break

    # Final save + derive pairs
    _save_results(results_file, results, full_a, full_b, counts, len(train))
    pairs = derive_pairs(results, args.teacher_a, args.teacher_b)
    with open(pairs_file, "w") as f:
        json.dump(pairs, f, indent=2)

    n = len(results) or 1
    a_acc = sum(1 for r in results if r["a_correct"]) / n
    b_acc = sum(1 for r in results if r["b_correct"]) / n
    stats = {
        "teacher_a": full_a, "teacher_b": full_b,
        "questions_processed": len(results),
        "category_counts": counts,
        "clear_preference_pairs": len(pairs),
        "clear_preference_rate": round(counts["clear_preference"] / n, 4),
        "teacher_a_accuracy": round(a_acc, 4),
        "teacher_b_accuracy": round(b_acc, 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*78}")
    print(f"  DONE — {len(pairs)} clear_preference pairs from {len(results)} questions")
    print(f"  clear_preference rate: {stats['clear_preference_rate']:.1%}")
    print(f"  teacher A acc: {a_acc:.1%} | teacher B acc: {b_acc:.1%}")
    print(f"  both_correct (skipped): {counts['both_correct']} | both_wrong (skipped): {counts['both_wrong']}")
    print(f"  pairs → {pairs_file}")
    print(f"{'='*78}")


def _save_results(path, results, full_a, full_b, counts, total):
    with open(path, "w") as f:
        json.dump({
            "metadata": {
                "teacher_a": full_a, "teacher_b": full_b,
                "total_train_questions": total, "completed": len(results),
                "category_counts": counts,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            },
            "results": results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
