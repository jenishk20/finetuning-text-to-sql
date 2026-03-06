"""
Analyze Spider benchmark results to understand failure patterns.

Classifies queries by SQL complexity, database, and error type,
then produces a summary report useful for guiding fine-tuning strategy.

Usage:
    python -m src.analyze_spider_results results/spider_run_XXXX.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def classify_sql_complexity(sql: str) -> str:
    """
    Classify a SQL query into difficulty buckets based on structural features.
    Mirrors the Spider hardness classification logic.
    """
    sql_upper = sql.upper()

    has_subquery = "SELECT" in sql_upper[sql_upper.find("FROM"):] if "FROM" in sql_upper else False
    has_join = bool(re.search(r'\bJOIN\b', sql_upper))
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_order = "ORDER BY" in sql_upper
    has_union = "UNION" in sql_upper
    has_intersect = "INTERSECT" in sql_upper
    has_except = "EXCEPT" in sql_upper
    has_nested = sql_upper.count("SELECT") > 1

    agg_count = sum(1 for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
                     if fn in sql_upper)

    set_ops = has_union or has_intersect or has_except
    complexity_score = (
        has_join + has_group + has_having + has_order +
        has_nested + set_ops + (agg_count > 1)
    )

    if complexity_score == 0:
        return "easy"
    elif complexity_score <= 1:
        return "medium"
    elif complexity_score <= 3:
        return "hard"
    else:
        return "extra_hard"


def classify_error_type(result: dict) -> str:
    """Classify the type of failure for a wrong result."""
    ev = result["eval"]

    if not ev["generated_executed"]:
        error_msg = result.get("gen_execution", {}).get("error", "") or ""
        if "no such table" in error_msg.lower():
            return "wrong_table_name"
        if "no such column" in error_msg.lower():
            return "wrong_column_name"
        if "syntax" in error_msg.lower() or "near" in error_msg.lower():
            return "syntax_error"
        return "execution_error"

    if ev["result_match"]:
        return "correct"

    gen_rows = result.get("gen_execution", {}).get("row_count", 0)
    gold_rows = result.get("gold_execution", {}).get("row_count", 0)

    if gen_rows == 0 and gold_rows > 0:
        return "empty_result"
    if gen_rows > gold_rows:
        return "too_many_rows"
    if gen_rows < gold_rows:
        return "too_few_rows"

    return "wrong_content"


def classify_sql_features(sql: str) -> list[str]:
    """Extract SQL features present in a query."""
    sql_upper = sql.upper()
    features = []

    if "JOIN" in sql_upper:
        features.append("JOIN")
    if "GROUP BY" in sql_upper:
        features.append("GROUP BY")
    if "HAVING" in sql_upper:
        features.append("HAVING")
    if "ORDER BY" in sql_upper:
        features.append("ORDER BY")
    if sql_upper.count("SELECT") > 1:
        features.append("subquery")
    if "UNION" in sql_upper:
        features.append("UNION")
    if "INTERSECT" in sql_upper:
        features.append("INTERSECT")
    if "EXCEPT" in sql_upper:
        features.append("EXCEPT")
    if "LIKE" in sql_upper:
        features.append("LIKE")
    if "DISTINCT" in sql_upper:
        features.append("DISTINCT")
    if "LIMIT" in sql_upper:
        features.append("LIMIT")

    for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]:
        if fn in sql_upper:
            features.append(fn.rstrip("("))

    if not features:
        features.append("simple_select")

    return features


def analyze(results_file: str):
    with open(results_file) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]

    print(f"\n{'=' * 80}")
    print(f"  SPIDER RESULTS ANALYSIS")
    print(f"  Model: {metadata['model']}")
    print(f"  Total questions: {metadata['total_questions']}")
    print(f"  Total tokens: {metadata['total_tokens']:,}")
    print(f"{'=' * 80}")

    # --- Overall metrics ---
    correct = [r for r in results if r["eval"]["result_match"]]
    failed_exec = [r for r in results if not r["eval"]["generated_executed"]]
    wrong_result = [r for r in results if r["eval"]["generated_executed"] and not r["eval"]["result_match"]]

    print(f"\n  OVERALL BREAKDOWN")
    print(f"  {'Correct results:':<30} {len(correct):>5}  ({len(correct)/len(results):.1%})")
    print(f"  {'Wrong results (ran OK):':<30} {len(wrong_result):>5}  ({len(wrong_result)/len(results):.1%})")
    print(f"  {'Execution failures:':<30} {len(failed_exec):>5}  ({len(failed_exec)/len(results):.1%})")

    # --- By SQL complexity ---
    complexity_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        gold_sql = r.get("gold_sql", "")
        c = classify_sql_complexity(gold_sql)
        complexity_stats[c]["total"] += 1
        if r["eval"]["result_match"]:
            complexity_stats[c]["correct"] += 1

    print(f"\n  ACCURACY BY QUERY COMPLEXITY (based on gold SQL)")
    print(f"  {'Difficulty':<15} {'Total':>7} {'Correct':>9} {'Accuracy':>10}")
    print(f"  {'-' * 45}")
    for diff in ["easy", "medium", "hard", "extra_hard"]:
        if diff in complexity_stats:
            s = complexity_stats[diff]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"  {diff:<15} {s['total']:>7} {s['correct']:>9} {acc:>10.1%}")

    # --- By SQL feature ---
    feature_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        gold_sql = r.get("gold_sql", "")
        features = classify_sql_features(gold_sql)
        for feat in features:
            feature_stats[feat]["total"] += 1
            if r["eval"]["result_match"]:
                feature_stats[feat]["correct"] += 1

    print(f"\n  ACCURACY BY SQL FEATURE (in gold SQL)")
    print(f"  {'Feature':<18} {'Total':>7} {'Correct':>9} {'Accuracy':>10}")
    print(f"  {'-' * 48}")
    sorted_feats = sorted(feature_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for feat, s in sorted_feats:
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {feat:<18} {s['total']:>7} {s['correct']:>9} {acc:>10.1%}")

    # --- Error type distribution ---
    error_types = Counter()
    for r in results:
        etype = classify_error_type(r)
        if etype != "correct":
            error_types[etype] += 1

    print(f"\n  FAILURE TYPE DISTRIBUTION ({sum(error_types.values())} failures)")
    print(f"  {'Error Type':<25} {'Count':>7} {'% of Failures':>15}")
    print(f"  {'-' * 50}")
    for etype, count in error_types.most_common():
        pct = count / sum(error_types.values()) if error_types else 0
        print(f"  {etype:<25} {count:>7} {pct:>15.1%}")

    # --- Hardest databases ---
    db_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        db = r["db_id"]
        db_stats[db]["total"] += 1
        if r["eval"]["result_match"]:
            db_stats[db]["correct"] += 1

    # Sort by accuracy (ascending) — worst databases first
    db_ranked = sorted(db_stats.items(),
                       key=lambda x: (x[1]["correct"] / x[1]["total"], -x[1]["total"]))

    print(f"\n  HARDEST DATABASES (lowest accuracy, min 3 questions)")
    print(f"  {'Database':<35} {'Total':>7} {'Correct':>9} {'Accuracy':>10}")
    print(f"  {'-' * 65}")
    shown = 0
    for db, s in db_ranked:
        if s["total"] < 3:
            continue
        acc = s["correct"] / s["total"]
        print(f"  {db:<35} {s['total']:>7} {s['correct']:>9} {acc:>10.1%}")
        shown += 1
        if shown >= 15:
            break

    # --- Easiest databases (for contrast) ---
    db_ranked_best = sorted(db_stats.items(),
                            key=lambda x: (-x[1]["correct"] / x[1]["total"], -x[1]["total"]))

    print(f"\n  EASIEST DATABASES (highest accuracy, min 3 questions)")
    print(f"  {'Database':<35} {'Total':>7} {'Correct':>9} {'Accuracy':>10}")
    print(f"  {'-' * 65}")
    shown = 0
    for db, s in db_ranked_best:
        if s["total"] < 3:
            continue
        acc = s["correct"] / s["total"]
        if acc < 1.0:
            break
        print(f"  {db:<35} {s['total']:>7} {s['correct']:>9} {acc:>10.1%}")
        shown += 1
        if shown >= 10:
            break

    # --- Sample failures for review ---
    print(f"\n  SAMPLE FAILURES (5 examples)")
    print(f"  {'-' * 75}")
    fail_samples = [r for r in results if not r["eval"]["result_match"]][:5]
    for r in fail_samples:
        etype = classify_error_type(r)
        print(f"\n  Q{r['spider_index']} | db={r['db_id']} | type={etype}")
        print(f"  Question: {r['question'][:100]}")
        print(f"  Gold:      {r['gold_sql'][:100]}")
        print(f"  Generated: {r['generated_sql'][:100]}")
        print(f"  Detail:    {r['eval']['details'][:100]}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analyze_spider_results <results_file.json>")
        sys.exit(1)
    analyze(sys.argv[1])
