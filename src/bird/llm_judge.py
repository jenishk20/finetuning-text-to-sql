"""
Step 2 (BIRD): LLM Judge — resolve "both correct, different SQL" pairs.

Same logic as llm_judge.py (Spider), adapted for BIRD:
  - Uses bird_judge_queue.json / bird_preference_pairs.json
  - Uses question_id (not spider_index)
  - Evidence field included in judge prompt for context

Usage:
    python -m src.bird.llm_judge               # full run
    python -m src.bird.llm_judge --dry-run     # test 1 call
    python -m src.bird.llm_judge --limit 10    # test N calls
    python -m src.bird.llm_judge --merge-only  # just merge completed into final
"""

import json
import random
import re
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.shared.llm_client import get_client

RESULTS_DIR = PROJECT_ROOT / "results"

JUDGE_QUEUE_FILE   = RESULTS_DIR / "bird_judge_queue.json"
JUDGED_OUTPUT_FILE = RESULTS_DIR / "bird_judge_queue_completed.json"
PAIRS_FILE         = RESULTS_DIR / "bird_preference_pairs.json"
FINAL_PAIRS_FILE   = RESULTS_DIR / "bird_preference_pairs_final.json"

JUDGE_MODEL      = "google/gemini-2.5-flash"
JUDGE_PROVIDER   = "openrouter"
TEMPERATURE      = 0.0
MAX_TOKENS       = 200
CHECKPOINT_EVERY = 25
SLEEP_BETWEEN_CALLS = 0.2

SYSTEM_PROMPT = """You are an expert SQL code reviewer with deep knowledge of SQL best practices.
You will be shown two SQL queries that are both confirmed CORRECT — they produce the right result.
Your job is to judge which is the BETTER query based on style, readability, and SQL best practices.

Respond with JSON only. No explanations outside the JSON. No markdown fences.
Format: {"winner": "A" or "B", "reason": "one concise sentence"}

If both queries are essentially equivalent in quality, pick the one that better matches the gold SQL style."""


def build_judge_prompt(question: str, schema: str, gold_sql: str, sql_a: str, sql_b: str, evidence: str = "") -> str:
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        f"Gold SQL (reference):\n{gold_sql}\n\n"
        f"Query A:\n{sql_a}\n\n"
        f"Query B:\n{sql_b}\n\n"
        "Evaluate on: readability, SQL best practices, simplicity, alignment with gold SQL style.\n"
        'Respond with JSON only: {"winner": "A" or "B", "reason": "one sentence"}'
    )


def parse_judge_response(raw: str) -> dict | None:
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()

    try:
        result = json.loads(text)
        if "winner" in result and result["winner"] in ("A", "B"):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"winner"\s*:\s*"([AB])"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    winner_match = re.search(r'"winner"\s*:\s*"([AB])"', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
    if winner_match:
        return {
            "winner": winner_match.group(1),
            "reason": reason_match.group(1) if reason_match else "No reason provided",
        }

    return None


def judge_pair(client, entry: dict, rng: random.Random) -> dict:
    swap = rng.random() < 0.5
    if swap:
        display_a, display_b = entry["sql_b"], entry["sql_a"]
        source_a,  source_b  = entry["sql_b_source"], entry["sql_a_source"]
    else:
        display_a, display_b = entry["sql_a"], entry["sql_b"]
        source_a,  source_b  = entry["sql_a_source"], entry["sql_b_source"]

    prompt = build_judge_prompt(
        question  = entry["question"],
        schema    = entry["schema"],
        gold_sql  = entry["gold_sql"],
        sql_a     = display_a,
        sql_b     = display_b,
        evidence  = entry.get("evidence", ""),
    )

    response = client.chat.completions.create(
        model       = JUDGE_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
    )

    raw     = response.choices[0].message.content or ""
    parsed  = parse_judge_response(raw)

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

    result = {**entry, "judge_usage": usage, "judge_raw_response": raw}

    if parsed is None:
        result["judge_winner"]    = None
        result["judge_reason"]    = f"PARSE_ERROR: {raw[:200]}"
        result["chosen_sql"]      = None
        result["chosen_source"]   = None
        result["rejected_sql"]    = None
        result["rejected_source"] = None
        return result

    raw_winner = parsed["winner"]
    actual_winner_source = source_a if raw_winner == "A" else source_b
    winner_sql   = display_a if raw_winner == "A" else display_b
    rejected_sql = display_b if raw_winner == "A" else display_a
    rejected_src = source_b  if raw_winner == "A" else source_a

    result["judge_winner"]    = actual_winner_source
    result["judge_reason"]    = parsed.get("reason", "")
    result["judge_display_a"] = source_a
    result["judge_swapped"]   = swap
    result["chosen_sql"]      = winner_sql
    result["chosen_source"]   = actual_winner_source
    result["rejected_sql"]    = rejected_sql
    result["rejected_source"] = rejected_src

    return result


def merge_into_final(dry_run: bool = False):
    with open(PAIRS_FILE) as f:
        existing_pairs = json.load(f)

    with open(JUDGED_OUTPUT_FILE) as f:
        judged = json.load(f)

    judge_pairs = []
    skipped_errors = 0
    for entry in judged:
        if entry.get("judge_winner") is None:
            skipped_errors += 1
            continue
        judge_pairs.append({
            "question_id":      entry["question_id"],
            "db_id":            entry["db_id"],
            "question":         entry["question"],
            "evidence":         entry.get("evidence", ""),
            "difficulty":       entry.get("difficulty", "unknown"),
            "schema":           entry["schema"],
            "gold_sql":         entry["gold_sql"],
            "instruction":      entry["instruction"],
            "grok_sql":         entry["grok_sql"],
            "deepseek_sql":     entry["deepseek_sql"],
            "grok_correct":     entry["grok_correct"],
            "deepseek_correct": entry["deepseek_correct"],
            "category":         "judge_resolved",
            "chosen_sql":       entry["chosen_sql"],
            "chosen_source":    entry["chosen_source"],
            "rejected_sql":     entry["rejected_sql"],
            "rejected_source":  entry["rejected_source"],
            "judge_winner":     entry["judge_winner"],
            "judge_reason":     entry["judge_reason"],
        })

    final_pairs = existing_pairs + judge_pairs

    print(f"\n{'─' * 50}")
    print("MERGE SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Step 1 pairs (clear + gold_vs_wrong): {len(existing_pairs)}")
    print(f"  Step 2 judged pairs:                  {len(judge_pairs)}")
    print(f"  Parse errors (skipped):               {skipped_errors}")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL preference pairs:               {len(final_pairs)}")

    if judge_pairs:
        grok_wins = sum(1 for p in judge_pairs if p["chosen_source"] == "grok")
        ds_wins   = sum(1 for p in judge_pairs if p["chosen_source"] == "deepseek")
        print(f"\n  Judge winner breakdown ({len(judge_pairs)} judged):")
        print(f"    Grok preferred:     {grok_wins} ({grok_wins/len(judge_pairs)*100:.1f}%)")
        print(f"    DeepSeek preferred: {ds_wins} ({ds_wins/len(judge_pairs)*100:.1f}%)")

    if not dry_run:
        with open(FINAL_PAIRS_FILE, "w") as f:
            json.dump(final_pairs, f, indent=2)
        print(f"\n  Saved: {FINAL_PAIRS_FILE}")
    else:
        print(f"\n  [dry-run] Would save to: {FINAL_PAIRS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Step 2 (BIRD): LLM judge for SQL preference pairs")
    parser.add_argument("--dry-run",    action="store_true", help="Test one call then exit")
    parser.add_argument("--limit",      type=int, default=None, help="Only judge N pairs")
    parser.add_argument("--merge-only", action="store_true", help="Skip judging, just merge")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2 (BIRD): LLM Judge")
    print(f"Model: {JUDGE_MODEL}  |  Provider: {JUDGE_PROVIDER}")
    print("=" * 60)

    if args.merge_only:
        merge_into_final(dry_run=args.dry_run)
        return

    with open(JUDGE_QUEUE_FILE) as f:
        queue = json.load(f)

    # Resume support
    completed: dict[int, dict] = {}
    if JUDGED_OUTPUT_FILE.exists():
        with open(JUDGED_OUTPUT_FILE) as f:
            for entry in json.load(f):
                if entry.get("judge_winner") is not None:
                    completed[entry["question_id"]] = entry
        print(f"\nResuming: {len(completed)} already judged, {len(queue) - len(completed)} remaining")
    else:
        print(f"\nStarting fresh: {len(queue)} pairs to judge")

    todo = [e for e in queue if e["question_id"] not in completed]
    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("All pairs already judged. Running merge...")
        merge_into_final()
        return

    print(f"Pairs to judge this run: {len(todo)}")
    if args.dry_run:
        print("[dry-run mode: will judge 1 pair then exit]\n")
        todo = todo[:1]

    client = get_client(JUDGE_PROVIDER)
    rng    = random.Random(42)

    all_results  = list(completed.values())
    errors       = 0
    total_tokens = 0

    print(f"\n{'qid':>6}  {'db_id':<30}  {'winner':<12}  {'tokens':>8}  {'reason'}")
    print("─" * 90)

    for i, entry in enumerate(todo):
        try:
            result = judge_pair(client, entry, rng)
        except Exception as exc:
            print(f"{entry['question_id']:>6}  {'ERROR':<30}  {str(exc)[:50]}")
            errors += 1
            time.sleep(2)
            continue

        all_results.append(result)

        winner_display = result.get("judge_winner") or "PARSE_ERR"
        tokens         = result.get("judge_usage", {}).get("total_tokens", 0)
        total_tokens  += tokens
        reason_preview = (result.get("judge_reason") or "")[:55]

        print(f"{entry['question_id']:>6}  {entry['db_id']:<30}  {winner_display:<12}  {tokens:>8}  {reason_preview}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(JUDGED_OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [checkpoint — {i+1}/{len(todo)} done, {total_tokens:,} tokens so far]")

        if args.dry_run:
            print("\n[dry-run] First call successful.")
            print(f"  Winner: {result.get('judge_winner')}")
            print(f"  Reason: {result.get('judge_reason')}")
            return

        time.sleep(SLEEP_BETWEEN_CALLS)

    with open(JUDGED_OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    judged_this_run = len(todo) - errors
    parse_errors    = sum(1 for r in all_results if r.get("judge_winner") is None)
    total_judged    = len([r for r in all_results if r.get("judge_winner") is not None])
    grok_wins       = sum(1 for r in all_results if r.get("judge_winner") == "grok")
    ds_wins         = sum(1 for r in all_results if r.get("judge_winner") == "deepseek")

    print(f"\n{'=' * 60}")
    print("RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Judged this run:   {judged_this_run}")
    print(f"  API errors:        {errors}")
    print(f"  Parse errors:      {parse_errors}")
    print(f"  Total judged:      {total_judged} / {len(queue)}")
    print(f"  Total tokens used: {total_tokens:,}")
    est_cost = (total_tokens / 1_000_000) * 0.15
    print(f"  Estimated cost:    ~${est_cost:.4f}")

    if total_judged > 0:
        print(f"\n  Winner breakdown:")
        print(f"    Grok preferred:     {grok_wins} ({grok_wins/total_judged*100:.1f}%)")
        print(f"    DeepSeek preferred: {ds_wins}  ({ds_wins/total_judged*100:.1f}%)")

    if total_judged == len(queue):
        print(f"\n  All {len(queue)} pairs judged ✓ — merging into final preference pairs...")
        merge_into_final()
    else:
        remaining = len(queue) - total_judged
        print(f"\n  {remaining} pairs still pending. Re-run to continue.")
        print(f"  When complete, run: python -m src.bird_llm_judge --merge-only")


if __name__ == "__main__":
    main()
