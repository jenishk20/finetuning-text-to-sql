"""
Build DPO preference pairs from two frontier model BIRD evaluation runs.

Compares Grok and DeepSeek results question-by-question and produces
preference pairs for DPO training. Designed for the BIRD train set
(9,428 questions) but works for dev set too.

FOUR CATEGORIES:
  1. clear_preference  — one model right, one wrong → direct pair (strongest signal)
  2. judge_needed      — both correct, different SQL → needs LLM judge (Step 2)
  3. gold_vs_wrong     — both wrong → gold SQL chosen (skipped by default, use --include-gold-fallback)
  4. skip              — both correct, identical SQL OR both wrong (default)

Usage:
    # On HPC — train set runs
    python -m src.bird.build_pairs \\
        --grok   /scratch/phalle.y/results_frontier_bird_dpo_grok/bird_train_grok-4-1-fast_XXXX.json \\
        --deepseek /scratch/phalle.y/results_frontier_bird_dpo_deepseek/bird_train_deepseek-v3_XXXX.json \\
        --db-dir /scratch/phalle.y/bird_train/train/train_databases/train_databases \\
        --output-dir /scratch/phalle.y/results_frontier_pairs

    # Include gold fallback pairs (both-wrong questions)
        --include-gold-fallback

    # Dry run — print stats without saving
        --dry-run
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.shared.schema_loader import get_schema_from_sqlite


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_db_path(db_id: str, db_dir: Path) -> str:
    """Find the SQLite file for a given db_id inside db_dir."""
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)
    db_subdir = db_dir / db_id
    if db_subdir.exists():
        for f in db_subdir.glob("*.sqlite"):
            return str(f)
    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {db_dir}")


def load_results(path: Path) -> dict[int, dict]:
    """Load a pipeline result file and index by question_id."""
    with open(path) as f:
        data = json.load(f)
    return {entry["question_id"]: entry for entry in data["results"]}


def sqls_are_equivalent(sql_a: str, sql_b: str) -> bool:
    """Normalize and compare two SQL strings."""
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().rstrip(";").split())
    return normalize(sql_a) == normalize(sql_b)


def build_instruction(question: str, schema: str, evidence: str = "") -> str:
    """
    Build the instruction string for DPO training.
    Must exactly match the format used during SFT training.
    """
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(
    grok_file: Path,
    deepseek_file: Path,
    db_dir: Path,
    output_dir: Path,
    include_gold_fallback: bool = False,
    dry_run: bool = False,
):
    print("=" * 60)
    print("BIRD Frontier DPO — Building Preference Pairs")
    print("=" * 60)

    # ── 1. Load both result files ─────────────────────────────────────────────
    print(f"\n[1/4] Loading result files...")
    grok     = load_results(grok_file)
    deepseek = load_results(deepseek_file)
    print(f"      Grok entries:     {len(grok)}")
    print(f"      DeepSeek entries: {len(deepseek)}")

    shared_ids = set(grok.keys()) & set(deepseek.keys())
    if len(shared_ids) != len(grok) or len(shared_ids) != len(deepseek):
        print(f"      WARNING: files don't cover identical question_ids.")
        print(f"               Grok-only: {len(set(grok.keys()) - shared_ids)}")
        print(f"               DeepSeek-only: {len(set(deepseek.keys()) - shared_ids)}")
    question_ids = sorted(shared_ids)
    print(f"      Matched on {len(question_ids)} shared questions ✓")

    # ── 2. Load schemas ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading schemas from SQLite databases in {db_dir}...")
    schema_cache: dict[str, str] = {}
    all_db_ids = {grok[i]["db_id"] for i in question_ids}
    skipped_dbs = []
    for db_id in sorted(all_db_ids):
        try:
            db_path = resolve_db_path(db_id, db_dir)
            schema_cache[db_id] = get_schema_from_sqlite(db_path)
        except FileNotFoundError as e:
            print(f"      WARNING: {e}")
            skipped_dbs.append(db_id)
    print(f"      Loaded schemas for {len(schema_cache)}/{len(all_db_ids)} databases")
    if skipped_dbs:
        print(f"      Skipped (no DB found): {skipped_dbs}")

    # ── 3. Categorize and build pairs ─────────────────────────────────────────
    print(f"\n[3/4] Categorizing pairs...")
    print(f"      Gold fallback (both-wrong): {'INCLUDED' if include_gold_fallback else 'SKIPPED (use --include-gold-fallback to enable)'}")

    preference_pairs: list[dict] = []
    judge_queue: list[dict] = []
    counts = defaultdict(int)
    skipped_no_schema = 0

    for qid in question_ids:
        g = grok[qid]
        d = deepseek[qid]

        db_id    = g["db_id"]
        question = g["question"]
        evidence = g.get("evidence", "")
        gold_sql = g["gold_sql"]
        difficulty = g.get("difficulty", "unknown")

        if db_id not in schema_cache:
            skipped_no_schema += 1
            continue

        schema      = schema_cache[db_id]
        instruction = build_instruction(question, schema, evidence)

        grok_correct     = g["eval"]["result_match"]
        deepseek_correct = d["eval"]["result_match"]
        grok_sql         = g["generated_sql"]
        deepseek_sql     = d["generated_sql"]

        # Capture execution errors for diagnostics
        grok_exec_error = (
            g["gen_execution"].get("error") if "gen_execution" in g else g.get("api_error")
        )
        deepseek_exec_error = (
            d["gen_execution"].get("error") if "gen_execution" in d else d.get("api_error")
        )

        base = {
            "question_id":        qid,
            "db_id":              db_id,
            "question":           question,
            "evidence":           evidence,
            "difficulty":         difficulty,
            "schema":             schema,
            "gold_sql":           gold_sql,
            "instruction":        instruction,
            "grok_sql":           grok_sql,
            "deepseek_sql":       deepseek_sql,
            "grok_correct":       grok_correct,
            "deepseek_correct":   deepseek_correct,
            "grok_exec_error":    grok_exec_error,
            "deepseek_exec_error": deepseek_exec_error,
        }

        # ── CATEGORY 1: clear preference (one right, one wrong) ───────────────
        if grok_correct and not deepseek_correct:
            counts["clear_preference"] += 1
            preference_pairs.append({
                **base,
                "category":        "clear_preference",
                "chosen_sql":      grok_sql,
                "chosen_source":   "grok",
                "rejected_sql":    deepseek_sql,
                "rejected_source": "deepseek",
            })

        elif deepseek_correct and not grok_correct:
            counts["clear_preference"] += 1
            preference_pairs.append({
                **base,
                "category":        "clear_preference",
                "chosen_sql":      deepseek_sql,
                "chosen_source":   "deepseek",
                "rejected_sql":    grok_sql,
                "rejected_source": "grok",
            })

        # ── CATEGORY 2: both correct ──────────────────────────────────────────
        elif grok_correct and deepseek_correct:
            if sqls_are_equivalent(grok_sql, deepseek_sql):
                counts["skip_identical"] += 1
            else:
                counts["judge_needed"] += 1
                judge_queue.append({
                    **base,
                    "category":      "judge_needed",
                    "sql_a":         grok_sql,
                    "sql_a_source":  "grok",
                    "sql_b":         deepseek_sql,
                    "sql_b_source":  "deepseek",
                    "judge_winner":  None,
                    "judge_reason":  None,
                    "chosen_sql":    None,
                    "chosen_source": None,
                    "rejected_sql":  None,
                    "rejected_source": None,
                })

        # ── CATEGORY 3: both wrong ────────────────────────────────────────────
        else:
            counts["both_wrong"] += 1
            if include_gold_fallback:
                counts["gold_vs_wrong"] += 1
                # Gold vs grok
                preference_pairs.append({
                    **base,
                    "category":        "gold_vs_wrong",
                    "chosen_sql":      gold_sql,
                    "chosen_source":   "gold",
                    "rejected_sql":    grok_sql,
                    "rejected_source": "grok",
                    "pair_variant":    "gold_vs_grok",
                })
                # Gold vs deepseek
                preference_pairs.append({
                    **base,
                    "category":        "gold_vs_wrong",
                    "chosen_sql":      gold_sql,
                    "chosen_source":   "gold",
                    "rejected_sql":    deepseek_sql,
                    "rejected_source": "deepseek",
                    "pair_variant":    "gold_vs_deepseek",
                })
            else:
                counts["skip_both_wrong"] += 1

    # ── 4. Print summary ──────────────────────────────────────────────────────
    total_pairs = len(preference_pairs)
    gold_pairs  = counts["gold_vs_wrong"] * 2 if include_gold_fallback else 0

    grok_correct_total = sum(1 for i in question_ids if grok[i]["eval"]["result_match"])
    ds_correct_total   = sum(1 for i in question_ids if deepseek[i]["eval"]["result_match"])

    print(f"\n{'=' * 60}")
    print("PAIR BREAKDOWN")
    print("=" * 60)
    print(f"  Clear preference (1 right, 1 wrong): {counts['clear_preference']:>5} pairs")
    print(f"  Needs LLM judge  (both correct):     {counts['judge_needed']:>5} pairs")
    if include_gold_fallback:
        print(f"  Gold fallback    (both wrong):       {counts['both_wrong']:>5} questions → {gold_pairs} pairs")
    else:
        print(f"  Both wrong (skipped):                {counts['both_wrong']:>5} questions")
    print(f"  Identical SQL (skipped):             {counts['skip_identical']:>5}")
    if skipped_no_schema:
        print(f"  No schema found (skipped):           {skipped_no_schema:>5}")
    print(f"  {'─' * 45}")
    print(f"  Ready to use now:                    {total_pairs:>5} pairs")
    print(f"  Needs judge:                         {len(judge_queue):>5} pairs")
    print(f"  Total after judge:                   {total_pairs + len(judge_queue):>5} pairs")
    print(f"\nModel Accuracy (on {len(question_ids)} matched questions):")
    print(f"  Grok:     {grok_correct_total}/{len(question_ids)} ({grok_correct_total/len(question_ids):.1%})")
    print(f"  DeepSeek: {ds_correct_total}/{len(question_ids)} ({ds_correct_total/len(question_ids):.1%})")

    if dry_run:
        print(f"\n[dry-run] No files written.")
        return

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving output files to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_file  = output_dir / "bird_preference_pairs.json"
    judge_file  = output_dir / "bird_judge_queue.json"
    stats_file  = output_dir / "bird_preference_pairs_stats.json"

    with open(pairs_file, "w") as f:
        json.dump(preference_pairs, f, indent=2)
    print(f"      {pairs_file.name} → {total_pairs} pairs")

    with open(judge_file, "w") as f:
        json.dump(judge_queue, f, indent=2)
    print(f"      {judge_file.name} → {len(judge_queue)} pairs")

    stats = {
        "source_files": {
            "grok":     str(grok_file),
            "deepseek": str(deepseek_file),
        },
        "db_dir": str(db_dir),
        "include_gold_fallback": include_gold_fallback,
        "total_questions": len(question_ids),
        "breakdown": {
            "clear_preference":        counts["clear_preference"],
            "judge_needed":            counts["judge_needed"],
            "both_wrong_questions":    counts["both_wrong"],
            "gold_fallback_pairs":     gold_pairs,
            "skip_identical":          counts["skip_identical"],
            "skip_no_schema":          skipped_no_schema,
        },
        "output_pairs": {
            "ready_now":        total_pairs,
            "needs_judge":      len(judge_queue),
            "total_after_judge": total_pairs + len(judge_queue),
        },
        "accuracy": {
            "grok_correct":     grok_correct_total,
            "deepseek_correct": ds_correct_total,
            "grok_pct":         round(grok_correct_total / len(question_ids) * 100, 1),
            "deepseek_pct":     round(ds_correct_total / len(question_ids) * 100, 1),
        },
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"      {stats_file.name} → stats")

    print(f"\nStep 1 complete ✓")
    print(f"Next: python -m src.bird.llm_judge --pairs-dir {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from Grok + DeepSeek BIRD evaluation runs"
    )
    parser.add_argument(
        "--grok", type=Path, required=True,
        help="Path to Grok pipeline result JSON"
    )
    parser.add_argument(
        "--deepseek", type=Path, required=True,
        help="Path to DeepSeek pipeline result JSON"
    )
    parser.add_argument(
        "--db-dir", type=Path, required=True,
        help="Path to BIRD databases directory (e.g. .../train_databases/train_databases)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results",
        help="Where to save output files (default: project_root/results)"
    )
    parser.add_argument(
        "--include-gold-fallback", action="store_true",
        help="Include gold SQL pairs for both-wrong questions (risky — see CLAUDE.md)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, do not save any files"
    )
    args = parser.parse_args()

    build_pairs(
        grok_file=args.grok,
        deepseek_file=args.deepseek,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        include_gold_fallback=args.include_gold_fallback,
        dry_run=args.dry_run,
    )
