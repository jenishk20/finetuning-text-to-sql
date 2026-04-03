"""
Step 1: Build preference pairs from Grok and DeepSeek Spider evaluation results.

WHAT THIS SCRIPT DOES
─────────────────────
We have two models (Grok 4 and DeepSeek V3) that each generated SQL for all
1,034 Spider dev questions. For every question we know whether each model's SQL
produced the correct result (eval.result_match = True/False).

This script matches those two runs question-by-question and creates "preference
pairs" — (chosen_sql, rejected_sql) pairs that DPO/GRPO training uses as its
learning signal. The idea is simple: if we can tell the model "SQL A is better
than SQL B for this question", the model can learn to prefer A-style answers.

FOUR CATEGORIES OF PAIRS
─────────────────────────
1. clear_preference (118 pairs)
   One model got it right, the other got it wrong.
   chosen  = the correct model's SQL
   rejected = the wrong model's SQL
   → These are the cleanest training signal — no judgment call needed.

2. judge_needed (504 pairs)
   Both models produced correct SQL, but they wrote different queries.
   Example: one uses a subquery, the other uses a JOIN. Both are correct,
   but we want to teach style/efficiency preferences. This goes to Step 2
   (LLM judge) to decide which is "better" SQL.
   → NOT included in the output pairs yet — saved separately for Step 2.

3. gold_vs_wrong (223 pairs × 2 = up to 446 pairs)
   Both models got it wrong. But we have the gold SQL from Spider.
   chosen  = gold_sql (we know this is correct)
   rejected = each model's wrong SQL
   → Free training data — no API calls needed. We get two pairs per question
      (gold vs Grok-wrong AND gold vs DeepSeek-wrong).

4. skip (189 pairs)
   Both models got it right AND wrote identical SQL (or near-identical).
   No preference signal — we can't tell which is "better" if they're the same.
   → Discarded.

OUTPUT FILES
────────────
results/preference_pairs.json        — ready-to-use pairs (categories 1 + 3)
results/judge_queue.json             — needs LLM judge (category 2)
results/preference_pairs_stats.json  — summary statistics
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.shared.schema_loader import get_schema_from_sqlite, get_db_path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"
SPIDER_DATA_DIR = PROJECT_ROOT / "data" / "spider_data"

# The final (largest) checkpoint for each model — these contain all 1034 entries
GROK_FILE = RESULTS_DIR / "spider_run_20260305_142700.json"
DEEPSEEK_FILE = RESULTS_DIR / "spider_deepseek-v3_20260318_112505.json"

OUTPUT_PAIRS_FILE = RESULTS_DIR / "preference_pairs.json"
OUTPUT_JUDGE_FILE = RESULTS_DIR / "judge_queue.json"
OUTPUT_STATS_FILE = RESULTS_DIR / "preference_pairs_stats.json"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict[int, dict]:
    """
    Load a spider pipeline result file and return a dict keyed by spider_index.

    The result files have this structure:
      { "metadata": {...}, "metrics": {...}, "results": [ {...}, {...}, ... ] }

    Each entry in "results" has:
      spider_index, db_id, question, gold_sql, generated_sql,
      gen_execution, gold_execution, eval
    """
    with open(path) as f:
        data = json.load(f)

    return {entry["spider_index"]: entry for entry in data["results"]}


def sqls_are_equivalent(sql_a: str, sql_b: str) -> bool:
    """
    Rough check: are two SQL strings functionally the same?

    We normalize whitespace, case, and trailing semicolons before comparing.
    This is intentionally loose — the goal is to detect obviously identical
    queries. When in doubt we call them "different" and send to the judge.
    """
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().rstrip(";").split())

    return normalize(sql_a) == normalize(sql_b)


def build_instruction(question: str, schema: str) -> str:
    """
    Build the instruction string that will be the input to the model during training.

    Format:
      Convert the following natural language question into a valid SQL query.

      Database Schema:
      <DDL>

      Question: <question>

      Return only the SQL query with no explanation.

    This must match exactly the prompt format used during inference, otherwise
    training on these pairs won't transfer to real query generation.
    """
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: Building Preference Pairs")
    print("=" * 60)

    # ── 1. Load both result files ─────────────────────────────────────────────
    print(f"\n[1/4] Loading result files...")
    grok = load_results(GROK_FILE)
    deepseek = load_results(DEEPSEEK_FILE)
    print(f"      Grok entries:     {len(grok)}")
    print(f"      DeepSeek entries: {len(deepseek)}")

    # Verify both cover the same questions
    assert set(grok.keys()) == set(deepseek.keys()), \
        "Grok and DeepSeek don't cover the same spider_index values!"
    indices = sorted(grok.keys())
    print(f"      Matched on {len(indices)} shared questions ✓")

    # ── 2. Load schemas, caching by db_id ────────────────────────────────────
    print(f"\n[2/4] Loading schemas from SQLite databases...")
    schema_cache: dict[str, str] = {}
    schema_errors: list[str] = []

    all_db_ids = {grok[i]["db_id"] for i in indices}
    for db_id in sorted(all_db_ids):
        try:
            db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)
            schema_cache[db_id] = get_schema_from_sqlite(db_path)
        except FileNotFoundError as e:
            schema_errors.append(db_id)
            print(f"      WARNING: {e}")

    print(f"      Loaded schemas for {len(schema_cache)} databases")
    if schema_errors:
        print(f"      MISSING: {schema_errors}")

    # ── 3. Categorize and build pairs ─────────────────────────────────────────
    print(f"\n[3/4] Categorizing pairs...")

    # Accumulators
    preference_pairs: list[dict] = []   # Categories 1 + 3 (ready to use)
    judge_queue: list[dict] = []        # Category 2 (needs LLM judge)

    counts = defaultdict(int)

    for idx in indices:
        g = grok[idx]
        d = deepseek[idx]

        db_id = g["db_id"]
        question = g["question"]
        gold_sql = g["gold_sql"]
        schema = schema_cache.get(db_id, "-- Schema not available")

        grok_correct = g["eval"]["result_match"]
        deepseek_correct = d["eval"]["result_match"]
        grok_sql = g["generated_sql"]
        deepseek_sql = d["generated_sql"]

        # Some entries may be missing gen_execution if an API error occurred
        # during the pipeline run. Treat those as failed/incorrect.
        grok_exec_error = (
            g["gen_execution"].get("error") if "gen_execution" in g else g.get("api_error")
        )
        deepseek_exec_error = (
            d["gen_execution"].get("error") if "gen_execution" in d else d.get("api_error")
        )

        # Common metadata attached to every pair — makes downstream scripts easier
        base = {
            "spider_index": idx,
            "db_id": db_id,
            "question": question,
            "schema": schema,
            "gold_sql": gold_sql,
            "instruction": build_instruction(question, schema),
            "grok_sql": grok_sql,
            "deepseek_sql": deepseek_sql,
            "grok_correct": grok_correct,
            "deepseek_correct": deepseek_correct,
            "grok_exec_error": grok_exec_error,
            "deepseek_exec_error": deepseek_exec_error,
        }

        # ── CATEGORY 1: clear preference ──────────────────────────────────────
        # One model correct, one wrong. The correct model's SQL is "chosen",
        # the wrong model's SQL is "rejected".
        if grok_correct and not deepseek_correct:
            counts["clear_preference"] += 1
            preference_pairs.append({
                **base,
                "category": "clear_preference",
                "chosen_sql": grok_sql,
                "chosen_source": "grok",
                "rejected_sql": deepseek_sql,
                "rejected_source": "deepseek",
            })

        elif deepseek_correct and not grok_correct:
            counts["clear_preference"] += 1
            preference_pairs.append({
                **base,
                "category": "clear_preference",
                "chosen_sql": deepseek_sql,
                "chosen_source": "deepseek",
                "rejected_sql": grok_sql,
                "rejected_source": "grok",
            })

        # ── CATEGORY 2: judge needed ───────────────────────────────────────────
        # Both correct, but wrote different SQL. We need a judge to decide which
        # is "better" (more readable, efficient, idiomatic SQL).
        elif grok_correct and deepseek_correct:
            if sqls_are_equivalent(grok_sql, deepseek_sql):
                # ── CATEGORY 4: skip ──────────────────────────────────────────
                # Both correct, identical SQL. No preference signal to extract.
                counts["skip"] += 1
            else:
                counts["judge_needed"] += 1
                judge_queue.append({
                    **base,
                    "category": "judge_needed",
                    "sql_a": grok_sql,
                    "sql_a_source": "grok",
                    "sql_b": deepseek_sql,
                    "sql_b_source": "deepseek",
                    # These get filled in by Step 2 (LLM judge)
                    "judge_winner": None,
                    "judge_reason": None,
                    "chosen_sql": None,
                    "chosen_source": None,
                    "rejected_sql": None,
                    "rejected_source": None,
                })

        # ── CATEGORY 3: gold vs wrong ──────────────────────────────────────────
        # Both models got it wrong. We use the gold SQL as the chosen answer
        # and each model's wrong SQL as the rejected answer. This gives us 2
        # free training pairs per question (gold vs Grok AND gold vs DeepSeek).
        else:  # both wrong
            counts["gold_vs_wrong"] += 1

            # Pair 1: gold (chosen) vs Grok's wrong SQL (rejected)
            preference_pairs.append({
                **base,
                "category": "gold_vs_wrong",
                "chosen_sql": gold_sql,
                "chosen_source": "gold",
                "rejected_sql": grok_sql,
                "rejected_source": "grok",
                "pair_variant": "gold_vs_grok",
            })

            # Pair 2: gold (chosen) vs DeepSeek's wrong SQL (rejected)
            preference_pairs.append({
                **base,
                "category": "gold_vs_wrong",
                "chosen_sql": gold_sql,
                "chosen_source": "gold",
                "rejected_sql": deepseek_sql,
                "rejected_source": "deepseek",
                "pair_variant": "gold_vs_deepseek",
            })

    # ── 4. Save outputs ────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving output files...")

    with open(OUTPUT_PAIRS_FILE, "w") as f:
        json.dump(preference_pairs, f, indent=2)
    print(f"      preference_pairs.json → {len(preference_pairs)} pairs")

    with open(OUTPUT_JUDGE_FILE, "w") as f:
        json.dump(judge_queue, f, indent=2)
    print(f"      judge_queue.json      → {len(judge_queue)} pairs needing judge")

    # Detailed statistics
    stats = {
        "source_files": {
            "grok": str(GROK_FILE),
            "deepseek": str(DEEPSEEK_FILE),
        },
        "total_questions": len(indices),
        "breakdown": {
            "clear_preference": counts["clear_preference"],
            "judge_needed": counts["judge_needed"],
            "gold_vs_wrong_questions": counts["gold_vs_wrong"],
            "skip_identical": counts["skip"],
        },
        "output_pairs": {
            "ready_to_use": len(preference_pairs),
            "needs_judge": len(judge_queue),
            "after_judge_total_estimate": len(preference_pairs) + len(judge_queue),
        },
        "pair_source_breakdown": {
            "clear_preference_pairs": counts["clear_preference"],
            "gold_vs_wrong_pairs": counts["gold_vs_wrong"] * 2,
        },
        "accuracy": {
            "grok_correct": sum(1 for i in indices if grok[i]["eval"]["result_match"]),
            "deepseek_correct": sum(1 for i in indices if deepseek[i]["eval"]["result_match"]),
            "grok_accuracy_pct": round(
                sum(1 for i in indices if grok[i]["eval"]["result_match"]) / len(indices) * 100, 1
            ),
            "deepseek_accuracy_pct": round(
                sum(1 for i in indices if deepseek[i]["eval"]["result_match"]) / len(indices) * 100, 1
            ),
        },
    }
    with open(OUTPUT_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nPair Categories:")
    print(f"  ✅ Clear preference (1 right, 1 wrong): {counts['clear_preference']:>4} questions → {counts['clear_preference']:>4} pairs")
    print(f"  🔍 Needs LLM judge (both correct):      {counts['judge_needed']:>4} questions → {counts['judge_needed']:>4} pairs")
    print(f"  📚 Gold vs wrong (both wrong):           {counts['gold_vs_wrong']:>4} questions → {counts['gold_vs_wrong'] * 2:>4} pairs")
    print(f"  ⏭️  Skip (both correct, same SQL):        {counts['skip']:>4} questions")
    print(f"  {'─' * 50}")
    print(f"  Total questions:                        {len(indices):>4}")

    print(f"\nOutput Pairs:")
    print(f"  Ready to use NOW:     {len(preference_pairs)} pairs  (Step 1 complete)")
    print(f"  Pending judge:        {len(judge_queue)} pairs  (Step 2 will resolve these)")
    print(f"  Grand total after Step 2: ~{len(preference_pairs) + len(judge_queue)} pairs")

    print(f"\nModel Accuracy:")
    print(f"  Grok 4:     {stats['accuracy']['grok_accuracy_pct']}%  ({stats['accuracy']['grok_correct']}/{len(indices)})")
    print(f"  DeepSeek V3: {stats['accuracy']['deepseek_accuracy_pct']}%  ({stats['accuracy']['deepseek_correct']}/{len(indices)})")

    print(f"\nOutput files:")
    print(f"  {OUTPUT_PAIRS_FILE}")
    print(f"  {OUTPUT_JUDGE_FILE}")
    print(f"  {OUTPUT_STATS_FILE}")
    print(f"\nStep 1 complete ✓  →  Run Step 2 (judge) next: python -m src.llm_judge")


if __name__ == "__main__":
    main()
