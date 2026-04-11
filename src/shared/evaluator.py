"""
Evaluator for comparing generated SQL against gold-standard SQL.

Two levels of evaluation:
  1. Execution Accuracy (EX): Did the generated SQL run without errors?
  2. Result Accuracy (RES): Did it produce the same result set as the gold SQL?
"""
from __future__ import annotations


def normalize_rows(rows: list[tuple]) -> set[tuple]:
    """Convert rows to a set of tuples for order-independent comparison."""
    normalized = []
    for row in rows:
        normalized.append(tuple(str(v) for v in row))
    return set(normalized)


def compare_results(generated_result: dict, gold_result: dict) -> dict:
    """
    Compare two query results and return evaluation metrics.

    Returns a dict with:
      - "generated_executed": bool
      - "gold_executed": bool
      - "result_match": bool (same output rows, ignoring order)
      - "row_count_match": bool
      - "generated_row_count": int
      - "gold_row_count": int
      - "column_match": bool
      - "details": str
    """
    gen_ok = generated_result["success"]
    gold_ok = gold_result["success"]

    if not gen_ok:
        return {
            "generated_executed": False,
            "gold_executed": gold_ok,
            "result_match": False,
            "row_count_match": False,
            "generated_row_count": 0,
            "gold_row_count": gold_result["row_count"],
            "column_match": False,
            "details": f"Generated SQL failed: {generated_result['error']}",
        }

    if not gold_ok:
        return {
            "generated_executed": True,
            "gold_executed": False,
            "result_match": False,
            "row_count_match": False,
            "generated_row_count": generated_result["row_count"],
            "gold_row_count": 0,
            "column_match": False,
            "details": f"Gold SQL failed: {gold_result['error']}",
        }

    gen_rows = normalize_rows(generated_result["rows"])
    gold_rows = normalize_rows(gold_result["rows"])

    row_count_match = generated_result["row_count"] == gold_result["row_count"]
    result_match = gen_rows == gold_rows

    gen_cols = [c.lower() for c in generated_result["columns"]]
    gold_cols = [c.lower() for c in gold_result["columns"]]
    column_match = gen_cols == gold_cols

    if result_match:
        details = "Exact match"
    elif row_count_match:
        details = f"Row counts match ({gold_result['row_count']}) but content differs"
    else:
        details = (
            f"Row count mismatch: generated={generated_result['row_count']}, "
            f"gold={gold_result['row_count']}"
        )

    return {
        "generated_executed": True,
        "gold_executed": True,
        "result_match": result_match,
        "row_count_match": row_count_match,
        "generated_row_count": generated_result["row_count"],
        "gold_row_count": gold_result["row_count"],
        "column_match": column_match,
        "details": details,
    }


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics across all questions.

    Returns:
      - "total_questions": int
      - "execution_accuracy": float (0-1)
      - "result_accuracy": float (0-1)
      - "row_count_accuracy": float (0-1)
      - "per_difficulty": dict of metrics broken down by difficulty
    """
    total = len(results)
    if total == 0:
        return {}

    exec_correct = sum(1 for r in results if r["eval"]["generated_executed"])
    result_correct = sum(1 for r in results if r["eval"]["result_match"])
    rowcount_correct = sum(1 for r in results if r["eval"]["row_count_match"])

    per_difficulty = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in per_difficulty:
            per_difficulty[diff] = {"total": 0, "exec_correct": 0, "result_correct": 0}
        per_difficulty[diff]["total"] += 1
        if r["eval"]["generated_executed"]:
            per_difficulty[diff]["exec_correct"] += 1
        if r["eval"]["result_match"]:
            per_difficulty[diff]["result_correct"] += 1

    for diff, stats in per_difficulty.items():
        stats["execution_accuracy"] = round(stats["exec_correct"] / stats["total"], 4)
        stats["result_accuracy"] = round(stats["result_correct"] / stats["total"], 4)

    return {
        "total_questions": total,
        "execution_accuracy": round(exec_correct / total, 4),
        "result_accuracy": round(result_correct / total, 4),
        "row_count_accuracy": round(rowcount_correct / total, 4),
        "per_difficulty": per_difficulty,
    }
