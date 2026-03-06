"""
Spider benchmark evaluation pipeline.

Runs Grok against the Spider dev set (1,034 questions across 140+ databases)
using SQLite for execution-based evaluation.

Usage:
    python -m src.spider_pipeline                    # run all 1034 dev questions
    python -m src.spider_pipeline --limit 10         # run first 10 only
    python -m src.spider_pipeline --resume results/spider_run_XXXX.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.grok_client import call_grok
from src.sqlite_executor import execute_sqlite_query
from src.schema_loader import get_schema_from_sqlite, get_db_path
from src.evaluator import compare_results, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DATA_DIR = PROJECT_ROOT / "data" / "spider_data"


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_spider_dev() -> list[dict]:
    with open(SPIDER_DATA_DIR / "dev.json") as f:
        return json.load(f)


def run_spider_pipeline(limit: int | None = None, resume_file: str | None = None):
    config = load_config()
    dev_questions = load_spider_dev()

    if limit:
        dev_questions = dev_questions[:limit]

    # Resume support: skip already-completed questions
    completed_ids = set()
    existing_results = []
    if resume_file and Path(resume_file).exists():
        with open(resume_file) as f:
            saved = json.load(f)
            existing_results = saved.get("results", [])
            completed_ids = {r["spider_index"] for r in existing_results}
        print(f"  Resuming from {resume_file} — {len(completed_ids)} already done")

    print(f"\n{'=' * 80}")
    print(f"  SPIDER BENCHMARK EVALUATION")
    print(f"  Model: {config['model']['name']}")
    print(f"  Questions: {len(dev_questions)} (dev set)")
    print(f"  Already completed: {len(completed_ids)}")
    print(f"{'=' * 80}\n")

    # Cache schemas per db_id to avoid re-reading
    schema_cache: dict[str, str] = {}
    results = list(existing_results)

    total_to_run = len(dev_questions) - len(completed_ids)
    run_count = 0
    total_tokens = 0

    for idx, q in enumerate(dev_questions):
        if idx in completed_ids:
            continue

        run_count += 1
        db_id = q["db_id"]
        question = q["question"]
        gold_sql = q["query"]

        print(f"  [{run_count}/{total_to_run}] Q{idx} | db={db_id}")
        print(f"    Question: {question[:100]}...")

        # Get schema (cached)
        if db_id not in schema_cache:
            try:
                db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)
                schema_cache[db_id] = get_schema_from_sqlite(db_path)
            except FileNotFoundError as e:
                print(f"    SKIP — {e}")
                continue
        schema = schema_cache[db_id]
        db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)

        # Step 1: Call Grok
        try:
            grok_result = call_grok(question, schema, config, dialect="sqlite")
            generated_sql = grok_result["generated_sql"]
            tokens = grok_result["usage"].get("total_tokens", 0)
            total_tokens += tokens
            print(f"    Generated: {generated_sql[:120]}...")
        except Exception as e:
            print(f"    API ERROR: {e}")
            results.append({
                "spider_index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "api_error": str(e),
                "eval": {
                    "generated_executed": False,
                    "gold_executed": False,
                    "result_match": False,
                    "row_count_match": False,
                    "generated_row_count": 0,
                    "gold_row_count": 0,
                    "column_match": False,
                    "details": f"API error: {e}",
                },
            })
            continue

        # Step 2: Execute generated SQL
        gen_exec = execute_sqlite_query(generated_sql, db_path)

        # Step 3: Execute gold SQL
        gold_exec = execute_sqlite_query(gold_sql, db_path)

        # Step 4: Compare
        evaluation = compare_results(gen_exec, gold_exec)
        status = "PASS" if evaluation["result_match"] else "FAIL"
        exec_ok = "OK" if gen_exec["success"] else "ERR"
        print(f"    Exec: {exec_ok} | {status} | {evaluation['details'][:80]}")

        results.append({
            "spider_index": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "usage": grok_result["usage"],
            "gen_execution": {
                "success": gen_exec["success"],
                "row_count": gen_exec["row_count"],
                "execution_time_ms": gen_exec["execution_time_ms"],
                "error": gen_exec["error"],
                "columns": gen_exec["columns"],
            },
            "gold_execution": {
                "success": gold_exec["success"],
                "row_count": gold_exec["row_count"],
                "execution_time_ms": gold_exec["execution_time_ms"],
                "error": gold_exec["error"],
                "columns": gold_exec["columns"],
            },
            "eval": evaluation,
        })

        # Save incrementally every 25 questions
        if run_count % 25 == 0:
            _save_results(config, results, total_tokens, len(dev_questions))
            print(f"    --- Checkpoint saved ({run_count} done) ---")

    # Final save and metrics
    metrics = compute_metrics(results)
    output_file = _save_results(config, results, total_tokens, len(dev_questions), metrics)

    print(f"\n{'=' * 80}")
    print(f"  SPIDER BENCHMARK RESULTS")
    print(f"{'=' * 80}")
    print(f"  Total questions:      {metrics.get('total_questions', 0)}")
    print(f"  Execution accuracy:   {metrics.get('execution_accuracy', 0):.1%}")
    print(f"  Result accuracy:      {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Row count accuracy:   {metrics.get('row_count_accuracy', 0):.1%}")
    print(f"  Total tokens used:    {total_tokens:,}")
    print(f"{'=' * 80}")
    print(f"  Results saved to: {output_file}\n")

    return metrics


def _save_results(config, results, total_tokens, total_questions, metrics=None):
    output_dir = PROJECT_ROOT / config["evaluation"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"spider_run_{timestamp}.json"

    save_data = {
        "metadata": {
            "model": config["model"]["name"],
            "timestamp": timestamp,
            "benchmark": "spider_dev",
            "total_questions": total_questions,
            "completed": len(results),
            "total_tokens": total_tokens,
        },
        "metrics": metrics or {},
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spider benchmark evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a previous results file")
    args = parser.parse_args()

    run_spider_pipeline(limit=args.limit, resume_file=args.resume)
