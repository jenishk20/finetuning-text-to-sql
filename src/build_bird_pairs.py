"""
Step 1 (BIRD): Build preference pairs from Grok and DeepSeek BIRD evaluation results.

Same 4-category logic as build_preference_pairs.py (Spider), adapted for BIRD:
  - Uses question_id (not spider_index)
  - DB path is data/bird_data/dev_databases/<db_id>/<db_id>.sqlite
  - Instruction includes the evidence field (external knowledge)
  - Output files prefixed with bird_

FOUR CATEGORIES:
  1. clear_preference  — one model right, one wrong → direct pair
  2. judge_needed      — both correct, different SQL → needs LLM judge (Step 2)
  3. gold_vs_wrong     — both wrong → gold SQL as chosen, 2 pairs per question
  4. skip              — both correct, identical SQL → no signal

Usage:
    python -m src.build_bird_pairs
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_loader import get_schema_from_sqlite

RESULTS_DIR = PROJECT_ROOT / "results"
BIRD_DATA_DIR = PROJECT_ROOT / "data" / "bird_data"
BIRD_DB_DIR = BIRD_DATA_DIR / "dev_databases"

GROK_FILE     = RESULTS_DIR / "bird_grok-4-1-fast-reasoning_20260324_155728.json"
DEEPSEEK_FILE = RESULTS_DIR / "bird_deepseek-v3_20260323_230921.json"

OUTPUT_PAIRS_FILE = RESULTS_DIR / "bird_preference_pairs.json"
OUTPUT_JUDGE_FILE = RESULTS_DIR / "bird_judge_queue.json"
OUTPUT_STATS_FILE = RESULTS_DIR / "bird_preference_pairs_stats.json"


def get_bird_db_path(db_id: str) -> str:
    db_path = BIRD_DB_DIR / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)
    db_dir = BIRD_DB_DIR / db_id
    if db_dir.exists():
        for f in db_dir.glob("*.sqlite"):
            return str(f)
    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {BIRD_DB_DIR}")


def load_results(path: Path) -> dict[int, dict]:
    with open(path) as f:
        data = json.load(f)
    return {entry["question_id"]: entry for entry in data["results"]}


def sqls_are_equivalent(sql_a: str, sql_b: str) -> bool:
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().rstrip(";").split())
    return normalize(sql_a) == normalize(sql_b)


def build_instruction(question: str, schema: str, evidence: str = "") -> str:
    """Build the instruction string for BIRD — includes evidence field."""
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


def main():
    print("=" * 60)
    print("Step 1 (BIRD): Building Preference Pairs")
    print("=" * 60)

    # ── 1. Load both result files ─────────────────────────────────────────────
    print(f"\n[1/4] Loading result files...")
    grok     = load_results(GROK_FILE)
    deepseek = load_results(DEEPSEEK_FILE)
    print(f"      Grok entries:     {len(grok)}")
    print(f"      DeepSeek entries: {len(deepseek)}")

    shared_ids = set(grok.keys()) & set(deepseek.keys())
    if len(shared_ids) != len(grok) or len(shared_ids) != len(deepseek):
        print(f"      WARNING: files don't cover identical question_ids. Using {len(shared_ids)} shared.")
    question_ids = sorted(shared_ids)
    print(f"      Matched on {len(question_ids)} shared questions ✓")

    # ── 2. Load schemas ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading schemas from SQLite databases...")
    schema_cache: dict[str, str] = {}
    all_db_ids = {grok[i]["db_id"] for i in question_ids}
    for db_id in sorted(all_db_ids):
        try:
            db_path = get_bird_db_path(db_id)
            schema_cache[db_id] = get_schema_from_sqlite(db_path)
        except FileNotFoundError as e:
            print(f"      WARNING: {e}")
    print(f"      Loaded schemas for {len(schema_cache)} databases")

    # ── 3. Categorize and build pairs ─────────────────────────────────────────
    print(f"\n[3/4] Categorizing pairs...")
    preference_pairs: list[dict] = []
    judge_queue: list[dict] = []
    counts = defaultdict(int)

    for qid in question_ids:
        g = grok[qid]
        d = deepseek[qid]

        db_id    = g["db_id"]
        question = g["question"]
        evidence = g.get("evidence", "")
        gold_sql = g["gold_sql"]
        schema   = schema_cache.get(db_id, "-- Schema not available")
        difficulty = g.get("difficulty", "unknown")

        grok_correct     = g["eval"]["result_match"]
        deepseek_correct = d["eval"]["result_match"]
        grok_sql         = g["generated_sql"]
        deepseek_sql     = d["generated_sql"]

        grok_exec_error = (
            g["gen_execution"].get("error") if "gen_execution" in g else g.get("api_error")
        )
        deepseek_exec_error = (
            d["gen_execution"].get("error") if "gen_execution" in d else d.get("api_error")
        )

        base = {
            "question_id": qid,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "schema": schema,
            "gold_sql": gold_sql,
            "instruction": build_instruction(question, schema, evidence),
            "grok_sql": grok_sql,
            "deepseek_sql": deepseek_sql,
            "grok_correct": grok_correct,
            "deepseek_correct": deepseek_correct,
            "grok_exec_error": grok_exec_error,
            "deepseek_exec_error": deepseek_exec_error,
        }

        # ── CATEGORY 1: clear preference ──────────────────────────────────────
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
        elif grok_correct and deepseek_correct:
            if sqls_are_equivalent(grok_sql, deepseek_sql):
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
                    "judge_winner": None,
                    "judge_reason": None,
                    "chosen_sql": None,
                    "chosen_source": None,
                    "rejected_sql": None,
                    "rejected_source": None,
                })

        # ── CATEGORY 3: gold vs wrong ──────────────────────────────────────────
        else:
            counts["gold_vs_wrong"] += 1
            preference_pairs.append({
                **base,
                "category": "gold_vs_wrong",
                "chosen_sql": gold_sql,
                "chosen_source": "gold",
                "rejected_sql": grok_sql,
                "rejected_source": "grok",
                "pair_variant": "gold_vs_grok",
            })
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
    print(f"      bird_preference_pairs.json → {len(preference_pairs)} pairs")

    with open(OUTPUT_JUDGE_FILE, "w") as f:
        json.dump(judge_queue, f, indent=2)
    print(f"      bird_judge_queue.json      → {len(judge_queue)} pairs needing judge")

    stats = {
        "source_files": {
            "grok": str(GROK_FILE),
            "deepseek": str(DEEPSEEK_FILE),
        },
        "total_questions": len(question_ids),
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
        "accuracy": {
            "grok_correct": sum(1 for i in question_ids if grok[i]["eval"]["result_match"]),
            "deepseek_correct": sum(1 for i in question_ids if deepseek[i]["eval"]["result_match"]),
            "grok_accuracy_pct": round(
                sum(1 for i in question_ids if grok[i]["eval"]["result_match"]) / len(question_ids) * 100, 1
            ),
            "deepseek_accuracy_pct": round(
                sum(1 for i in question_ids if deepseek[i]["eval"]["result_match"]) / len(question_ids) * 100, 1
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
    print(f"  Clear preference (1 right, 1 wrong): {counts['clear_preference']:>4} questions → {counts['clear_preference']:>4} pairs")
    print(f"  Needs LLM judge (both correct):      {counts['judge_needed']:>4} questions → {counts['judge_needed']:>4} pairs")
    print(f"  Gold vs wrong (both wrong):           {counts['gold_vs_wrong']:>4} questions → {counts['gold_vs_wrong'] * 2:>4} pairs")
    print(f"  Skip (both correct, same SQL):        {counts['skip']:>4} questions")
    print(f"  {'─' * 48}")
    print(f"  Total:                               {len(question_ids):>4}")
    print(f"\nOutput Pairs:")
    print(f"  Ready to use NOW:         {len(preference_pairs)} pairs")
    print(f"  Pending judge:            {len(judge_queue)} pairs")
    print(f"  Grand total after judge:  ~{len(preference_pairs) + len(judge_queue)} pairs")
    print(f"\nModel Accuracy:")
    print(f"  Grok:       {stats['accuracy']['grok_accuracy_pct']}%  ({stats['accuracy']['grok_correct']}/{len(question_ids)})")
    print(f"  DeepSeek V3: {stats['accuracy']['deepseek_accuracy_pct']}%  ({stats['accuracy']['deepseek_correct']}/{len(question_ids)})")
    print(f"\nOutput files:")
    print(f"  {OUTPUT_PAIRS_FILE}")
    print(f"  {OUTPUT_JUDGE_FILE}")
    print(f"  {OUTPUT_STATS_FILE}")
    print(f"\nStep 1 complete ✓  → Run Step 2 next: python -m src.bird_llm_judge")


if __name__ == "__main__":
    main()
