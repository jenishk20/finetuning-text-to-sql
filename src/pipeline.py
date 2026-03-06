"""
Main pipeline orchestrator for the Text-to-SQL evaluation.

Flow:
  1. Load schema and questions from data/
  2. For each question:
     a. Send schema + question to Grok -> get generated SQL
     b. Execute generated SQL against MySQL -> get result or error
     c. Execute gold SQL against MySQL -> get expected result
     d. Compare generated vs gold result
  3. Log all results to results/
  4. Compute and display aggregate metrics
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.grok_client import call_grok
from src.sql_executor import execute_query
from src.evaluator import compare_results, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_schema(path: str) -> str:
    with open(PROJECT_ROOT / path) as f:
        return f.read()


def load_questions(path: str) -> list[dict]:
    with open(PROJECT_ROOT / path) as f:
        return json.load(f)


def print_separator():
    print("=" * 80)


def print_question_header(q: dict):
    print_separator()
    print(f"  Q{q['id']} [{q['difficulty']}]")
    print(f"  {q['question'][:120]}...")
    print_separator()


def run_pipeline():
    """Run the full evaluation pipeline."""
    config = load_config()
    schema = load_schema(config["database"]["schema_file"])
    questions = load_questions(config["database"]["questions_file"])

    print(f"\n{'=' * 80}")
    print(f"  TEXT-TO-SQL BASELINE EVALUATION")
    print(f"  Model: {config['model']['name']}")
    print(f"  Questions: {len(questions)}")
    print(f"  Database: {os.getenv('MYSQL_DATABASE')}")
    print(f"{'=' * 80}\n")

    results = []

    for q in questions:
        print_question_header(q)

        # Step 1: Call Grok
        print("  [1/4] Calling Grok API...")
        try:
            grok_result = call_grok(q["question"], schema, config)
            generated_sql = grok_result["generated_sql"]
            print(f"  Generated SQL ({len(generated_sql)} chars):")
            for line in generated_sql.split("\n"):
                print(f"    {line}")
            if grok_result["usage"]:
                print(f"  Tokens: {grok_result['usage'].get('total_tokens', '?')}")
        except Exception as e:
            print(f"  ERROR calling Grok API: {e}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "gold_sql": q["gold_sql"],
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
                    "details": f"API call failed: {e}",
                },
            })
            continue

        # Step 2: Execute generated SQL against local MySQL
        print("  [2/4] Executing generated SQL on MySQL...")
        gen_exec = execute_query(generated_sql, config["evaluation"]["timeout_seconds"])
        if gen_exec["success"]:
            print(f"  OK — {gen_exec['row_count']} rows in {gen_exec['execution_time_ms']}ms")
        else:
            print(f"  FAILED — {gen_exec['error']}")

        # Step 3: Execute gold SQL against local MySQL
        print("  [3/4] Executing gold SQL on MySQL...")
        gold_exec = execute_query(q["gold_sql"], config["evaluation"]["timeout_seconds"])
        if gold_exec["success"]:
            print(f"  OK — {gold_exec['row_count']} rows in {gold_exec['execution_time_ms']}ms")
        else:
            print(f"  FAILED — {gold_exec['error']}")

        # Step 4: Compare results
        print("  [4/4] Comparing results...")
        evaluation = compare_results(gen_exec, gold_exec)
        status = "PASS" if evaluation["result_match"] else "FAIL"
        print(f"  Result: {status} — {evaluation['details']}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "gold_sql": q["gold_sql"],
            "generated_sql": generated_sql,
            "raw_response": grok_result["raw_response"],
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

    # Compute and display aggregate metrics
    metrics = compute_metrics(results)

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'=' * 80}")
    print(f"  Total questions:      {metrics['total_questions']}")
    print(f"  Execution accuracy:   {metrics['execution_accuracy']:.1%}")
    print(f"  Result accuracy:      {metrics['result_accuracy']:.1%}")
    print(f"  Row count accuracy:   {metrics['row_count_accuracy']:.1%}")
    print(f"\n  Per difficulty:")
    for diff, stats in metrics["per_difficulty"].items():
        print(
            f"    {diff:8s}: exec={stats['execution_accuracy']:.1%}  "
            f"result={stats['result_accuracy']:.1%}  "
            f"(n={stats['total']})"
        )
    print(f"{'=' * 80}\n")

    # Save full results to JSON
    output_dir = PROJECT_ROOT / config["evaluation"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"run_{timestamp}.json"

    save_data = {
        "metadata": {
            "model": config["model"]["name"],
            "timestamp": timestamp,
            "database": os.getenv("MYSQL_DATABASE"),
            "num_questions": len(questions),
        },
        "metrics": metrics,
        "results": results,
    }

    # Rows can contain non-serializable types; skip them in the saved output
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"  Results saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    run_pipeline()
