"""
BIRD benchmark evaluation pipeline.

Runs any LLM against the BIRD dev set (1,534 questions across 11 databases)
using SQLite for execution-based evaluation.

BIRD differs from Spider in two key ways:
  1. Each question has an "evidence" field (external knowledge / domain hints)
     that must be injected into the prompt.
  2. The gold SQL field is "SQL" (not "query").

Usage:
    # List available models
    python -m src.bird.pipeline --list-models

    # Run with specific model via OpenRouter
    python -m src.bird.pipeline --provider openrouter --model deepseek-v3
    python -m src.bird.pipeline --provider openrouter --model gpt-4o --limit 10

    # Run with Grok (xAI)
    python -m src.bird.pipeline --provider xai --model grok-4-1-fast-reasoning

    # Resume a previous run
    python -m src.bird.pipeline --resume results/bird_deepseek-v3_XXXX.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.shared.llm_client import call_llm, list_available_models, resolve_model, PROVIDERS
from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.schema_loader import get_schema_from_sqlite
from src.shared.evaluator import compare_results, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIRD_DATA_DIR = PROJECT_ROOT / "data" / "bird_data"
BIRD_DEV_JSON = BIRD_DATA_DIR / "dev.json"
BIRD_DB_DIR = BIRD_DATA_DIR / "dev_databases"


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_bird_dev() -> list[dict]:
    with open(BIRD_DEV_JSON) as f:
        return json.load(f)


def get_bird_db_path(db_id: str) -> str:
    """Resolve the full path to a BIRD SQLite database file."""
    db_path = BIRD_DB_DIR / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)

    # Fallback: search for any .sqlite in the directory
    db_dir = BIRD_DB_DIR / db_id
    if db_dir.exists():
        for f in db_dir.glob("*.sqlite"):
            return str(f)

    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {BIRD_DB_DIR}")


def make_model_label(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def run_bird_pipeline(
    provider: str,
    model: str,
    limit: int | None = None,
    resume_file: str | None = None,
):
    config = load_config()
    dev_questions = load_bird_dev()
    full_model = resolve_model(model, provider)
    model_label = make_model_label(model)

    if limit:
        dev_questions = dev_questions[:limit]

    # Resume support
    completed_ids = set()
    existing_results = []
    if resume_file and Path(resume_file).exists():
        with open(resume_file) as f:
            saved = json.load(f)
            existing_results = saved.get("results", [])
            completed_ids = {r["question_id"] for r in existing_results}
        print(f"  Resuming from {resume_file} — {len(completed_ids)} already done")

    print(f"\n{'=' * 80}")
    print(f"  BIRD BENCHMARK EVALUATION")
    print(f"  Provider: {provider}")
    print(f"  Model:    {full_model}")
    print(f"  Questions: {len(dev_questions)} (dev set)")
    print(f"  Already completed: {len(completed_ids)}")
    print(f"{'=' * 80}\n")

    schema_cache: dict[str, str] = {}
    results = list(existing_results)

    total_to_run = len(dev_questions) - len(completed_ids)
    run_count = 0
    total_tokens = 0

    for q in dev_questions:
        question_id = q["question_id"]
        if question_id in completed_ids:
            continue

        run_count += 1
        db_id = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "").strip()
        gold_sql = q["SQL"]
        difficulty = q.get("difficulty", "unknown")

        print(f"  [{run_count}/{total_to_run}] Q{question_id} | db={db_id} | {difficulty}")
        print(f"    Question: {question[:100]}...")
        if evidence:
            print(f"    Evidence: {evidence[:80]}...")

        if db_id not in schema_cache:
            try:
                db_path = get_bird_db_path(db_id)
                schema_cache[db_id] = get_schema_from_sqlite(db_path)
            except FileNotFoundError as e:
                print(f"    SKIP — {e}")
                continue
        schema = schema_cache[db_id]
        db_path = get_bird_db_path(db_id)

        try:
            llm_result = call_llm(
                question=question,
                schema=schema,
                provider=provider,
                model=model,
                temperature=config["model"].get("temperature", 0.0),
                max_tokens=config["model"].get("max_tokens", 512),
                evidence=evidence if evidence else None,
            )
            generated_sql = llm_result["generated_sql"]
            tokens = llm_result["usage"].get("total_tokens", 0)
            total_tokens += tokens
            print(f"    Generated: {generated_sql[:120]}...")
        except Exception as e:
            print(f"    API ERROR: {e}")
            results.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "difficulty": difficulty,
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

        gen_exec = execute_sqlite_query(generated_sql, db_path)
        gold_exec = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(gen_exec, gold_exec)

        status = "PASS" if evaluation["result_match"] else "FAIL"
        exec_ok = "OK" if gen_exec["success"] else "ERR"
        print(f"    Exec: {exec_ok} | {status} | {evaluation['details'][:80]}")

        results.append({
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "usage": llm_result["usage"],
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

        if run_count % 25 == 0:
            _save_results(provider, full_model, model_label, results, total_tokens, len(dev_questions))
            print(f"    --- Checkpoint saved ({run_count} done) ---")

    metrics = compute_metrics(results)
    output_file = _save_results(provider, full_model, model_label, results, total_tokens, len(dev_questions), metrics)

    # Breakdown by difficulty
    for diff in ["simple", "moderate", "challenging"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            correct = sum(1 for r in diff_results if r["eval"]["result_match"])
            print(f"  {diff.capitalize():<12}: {correct}/{len(diff_results)} ({correct/len(diff_results):.1%})")

    print(f"\n{'=' * 80}")
    print(f"  BIRD BENCHMARK RESULTS — {full_model}")
    print(f"{'=' * 80}")
    print(f"  Total questions:      {metrics.get('total_questions', 0)}")
    print(f"  Execution accuracy:   {metrics.get('execution_accuracy', 0):.1%}")
    print(f"  Result accuracy:      {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Row count accuracy:   {metrics.get('row_count_accuracy', 0):.1%}")
    print(f"  Total tokens used:    {total_tokens:,}")
    print(f"{'=' * 80}")
    print(f"  Results saved to: {output_file}\n")

    return metrics


def _save_results(provider, full_model, model_label, results, total_tokens, total_questions, metrics=None):
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bird_{model_label}_{timestamp}.json"

    save_data = {
        "metadata": {
            "provider": provider,
            "model": full_model,
            "timestamp": timestamp,
            "benchmark": "bird_dev",
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
    parser = argparse.ArgumentParser(description="Run BIRD benchmark evaluation")
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=list(PROVIDERS.keys()),
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or shortcut (run --list-models to see options)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N questions")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous results file")
    parser.add_argument("--list-models", action="store_true",
                        help="List available model shortcuts and exit")
    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    model = args.model or PROVIDERS[args.provider]["default_model"]

    run_bird_pipeline(
        provider=args.provider,
        model=model,
        limit=args.limit,
        resume_file=args.resume,
    )
