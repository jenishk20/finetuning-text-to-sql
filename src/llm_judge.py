"""
Step 2: LLM Judge — resolve "both correct, different SQL" pairs.

WHAT THIS SCRIPT DOES
─────────────────────
From Step 1 we have 478 pairs where both Grok and DeepSeek produced correct
SQL but wrote it differently. We can't automatically say which is "better" —
we need a judge.

We use Gemini Flash 2.0 (via OpenRouter) as the judge because:
  - It's a third party: no conflict of interest (not Grok, not DeepSeek)
  - It understands SQL best practices well
  - It reliably returns structured JSON
  - It costs ~$0.036 for all 478 calls

HOW THE JUDGE WORKS
────────────────────
For each pair the judge sees:
  - The database schema (DDL)
  - The natural language question
  - The gold SQL (correct reference answer)
  - Query A and Query B (randomly assigned to control position bias)

It returns: { "winner": "A" or "B", "reason": "..." }

POSITION BIAS CONTROL
─────────────────────
LLMs slightly prefer whichever answer comes first ("A"). To cancel this out,
we randomly decide whether Grok's SQL goes in position A or B for each entry.
When we reverse the assignment, we flip the winner label before saving.
This prevents systematic bias toward one model's style.

CHECKPOINTING
─────────────
Results are saved every CHECKPOINT_EVERY calls. If the script is interrupted,
re-run it and it will skip already-judged entries automatically.

FINAL MERGE
───────────
After all pairs are judged, this script merges them into preference_pairs.json
so you have one complete file of (instruction, chosen_sql, rejected_sql) pairs
ready for DPO training.
"""

import json
import random
import re
import sys
import time
import argparse
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_client import get_client, resolve_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"

JUDGE_QUEUE_FILE    = RESULTS_DIR / "judge_queue.json"
JUDGED_OUTPUT_FILE  = RESULTS_DIR / "judge_queue_completed.json"
PAIRS_FILE          = RESULTS_DIR / "preference_pairs.json"
FINAL_PAIRS_FILE    = RESULTS_DIR / "preference_pairs_final.json"

JUDGE_MODEL    = "google/gemini-2.5-flash"   # third party, no conflict of interest
JUDGE_PROVIDER = "openrouter"
TEMPERATURE    = 0.0    # deterministic — same pair should always get same answer
MAX_TOKENS     = 200    # winner (A or B) + one sentence reason
CHECKPOINT_EVERY = 25   # save progress every N calls
SLEEP_BETWEEN_CALLS = 0.2   # seconds — avoid hammering the API


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL code reviewer with deep knowledge of SQL best practices.
You will be shown two SQL queries that are both confirmed CORRECT — they produce the right result.
Your job is to judge which is the BETTER query based on style, readability, and SQL best practices.

Respond with JSON only. No explanations outside the JSON. No markdown fences.
Format: {"winner": "A" or "B", "reason": "one concise sentence"}

If both queries are essentially equivalent in quality, pick the one that better matches the gold SQL style."""


def build_judge_prompt(question: str, schema: str, gold_sql: str, sql_a: str, sql_b: str) -> str:
    return (
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"Gold SQL (reference):\n{gold_sql}\n\n"
        f"Query A:\n{sql_a}\n\n"
        f"Query B:\n{sql_b}\n\n"
        "Evaluate on: readability, SQL best practices, simplicity, alignment with gold SQL style.\n"
        'Respond with JSON only: {"winner": "A" or "B", "reason": "one sentence"}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — robust fallback for imperfect model output
# ─────────────────────────────────────────────────────────────────────────────

def parse_judge_response(raw: str) -> dict | None:
    """
    Parse the judge's response into {"winner": "A"|"B", "reason": str}.

    Tries three strategies in order:
    1. Direct json.loads (model returned clean JSON)
    2. Regex extraction (model wrapped JSON in markdown or added extra text)
    3. Simple keyword search (fallback — look for "winner" and "A"/"B")

    Returns None if all strategies fail.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if "winner" in result and result["winner"] in ("A", "B"):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first {...} block
    match = re.search(r'\{[^{}]*"winner"\s*:\s*"([AB])"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: keyword fallback
    winner_match = re.search(r'"winner"\s*:\s*"([AB])"', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
    if winner_match:
        return {
            "winner": winner_match.group(1),
            "reason": reason_match.group(1) if reason_match else "No reason provided",
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# JUDGING
# ─────────────────────────────────────────────────────────────────────────────

def judge_pair(client, entry: dict, rng: random.Random) -> dict:
    """
    Call the LLM judge on one (sql_a, sql_b) pair.

    Handles position-bias control: randomly swaps which model goes in slot A vs B.
    Returns the entry dict with judge fields filled in.
    """
    # Randomly decide which model occupies position A
    swap = rng.random() < 0.5
    if swap:
        display_a, display_b = entry["sql_b"], entry["sql_a"]
        source_a, source_b   = entry["sql_b_source"], entry["sql_a_source"]
    else:
        display_a, display_b = entry["sql_a"], entry["sql_b"]
        source_a, source_b   = entry["sql_a_source"], entry["sql_b_source"]

    prompt = build_judge_prompt(
        question  = entry["question"],
        schema    = entry["schema"],
        gold_sql  = entry["gold_sql"],
        sql_a     = display_a,
        sql_b     = display_b,
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

    raw = response.choices[0].message.content or ""
    parsed = parse_judge_response(raw)

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

    result = {**entry, "judge_usage": usage, "judge_raw_response": raw}

    if parsed is None:
        # Parsing failed entirely — mark as error, don't include in final pairs
        result["judge_winner"]   = None
        result["judge_reason"]   = f"PARSE_ERROR: {raw[:200]}"
        result["chosen_sql"]     = None
        result["chosen_source"]  = None
        result["rejected_sql"]   = None
        result["rejected_source"] = None
        return result

    # Un-swap the winner back to the original source names
    raw_winner = parsed["winner"]   # "A" or "B" in display order
    if swap:
        actual_winner_source = source_a if raw_winner == "A" else source_b
        # source_a is sql_b_source after swap, source_b is sql_a_source
        winner_sql   = display_a if raw_winner == "A" else display_b
        rejected_sql = display_b if raw_winner == "A" else display_a
        rejected_src = source_b  if raw_winner == "A" else source_a
    else:
        actual_winner_source = source_a if raw_winner == "A" else source_b
        winner_sql   = display_a if raw_winner == "A" else display_b
        rejected_sql = display_b if raw_winner == "A" else display_a
        rejected_src = source_b  if raw_winner == "A" else source_a

    result["judge_winner"]    = actual_winner_source   # "grok" or "deepseek"
    result["judge_reason"]    = parsed.get("reason", "")
    result["judge_display_a"] = source_a               # for auditing: what model was shown as A
    result["judge_swapped"]   = swap
    result["chosen_sql"]      = winner_sql
    result["chosen_source"]   = actual_winner_source
    result["rejected_sql"]    = rejected_sql
    result["rejected_source"] = rejected_src

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MERGE — combine judged pairs into the main preference_pairs.json
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_final(dry_run: bool = False):
    """
    After all judging is done, merge the completed judge entries into
    preference_pairs.json and write preference_pairs_final.json.

    Only includes judge entries where parsing succeeded (judge_winner is not None).
    """
    with open(PAIRS_FILE) as f:
        existing_pairs = json.load(f)

    with open(JUDGED_OUTPUT_FILE) as f:
        judged = json.load(f)

    # Build clean judge pairs (drop parse errors and audit-only fields)
    judge_pairs = []
    skipped_errors = 0
    for entry in judged:
        if entry.get("judge_winner") is None:
            skipped_errors += 1
            continue
        # Keep only the fields needed for training + debugging
        judge_pairs.append({
            "spider_index":   entry["spider_index"],
            "db_id":          entry["db_id"],
            "question":       entry["question"],
            "schema":         entry["schema"],
            "gold_sql":       entry["gold_sql"],
            "instruction":    entry["instruction"],
            "grok_sql":       entry["grok_sql"],
            "deepseek_sql":   entry["deepseek_sql"],
            "grok_correct":   entry["grok_correct"],
            "deepseek_correct": entry["deepseek_correct"],
            "category":       "judge_resolved",
            "chosen_sql":     entry["chosen_sql"],
            "chosen_source":  entry["chosen_source"],
            "rejected_sql":   entry["rejected_sql"],
            "rejected_source": entry["rejected_source"],
            "judge_winner":   entry["judge_winner"],
            "judge_reason":   entry["judge_reason"],
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

    # Winner breakdown
    grok_wins = sum(1 for p in judge_pairs if p["chosen_source"] == "grok")
    ds_wins   = sum(1 for p in judge_pairs if p["chosen_source"] == "deepseek")
    print(f"\n  Judge winner breakdown (among {len(judge_pairs)} judged):")
    print(f"    Grok preferred:     {grok_wins} ({grok_wins/len(judge_pairs)*100:.1f}%)")
    print(f"    DeepSeek preferred: {ds_wins} ({ds_wins/len(judge_pairs)*100:.1f}%)")

    if not dry_run:
        with open(FINAL_PAIRS_FILE, "w") as f:
            json.dump(final_pairs, f, indent=2)
        print(f"\n  Saved: {FINAL_PAIRS_FILE}")
    else:
        print(f"\n  [dry-run] Would save to: {FINAL_PAIRS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2: LLM judge for SQL preference pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test one call then exit — don't save results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge this many pairs (for testing)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip judging, just merge existing completed file into final pairs")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: LLM Judge")
    print(f"Model: {JUDGE_MODEL}  |  Provider: {JUDGE_PROVIDER}")
    print("=" * 60)

    # ── Merge-only mode ────────────────────────────────────────────────────────
    if args.merge_only:
        merge_into_final(dry_run=args.dry_run)
        return

    # ── Load judge queue ───────────────────────────────────────────────────────
    with open(JUDGE_QUEUE_FILE) as f:
        queue = json.load(f)

    # Load any already-completed results (for resuming)
    completed: dict[int, dict] = {}
    if JUDGED_OUTPUT_FILE.exists():
        with open(JUDGED_OUTPUT_FILE) as f:
            for entry in json.load(f):
                if entry.get("judge_winner") is not None:
                    completed[entry["spider_index"]] = entry
        print(f"\nResuming: {len(completed)} already judged, {len(queue) - len(completed)} remaining")
    else:
        print(f"\nStarting fresh: {len(queue)} pairs to judge")

    # Filter to only unjudged entries
    todo = [e for e in queue if e["spider_index"] not in completed]
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

    # ── Set up client ──────────────────────────────────────────────────────────
    client = get_client(JUDGE_PROVIDER)
    rng    = random.Random(42)   # fixed seed so reruns produce same A/B assignments

    # ── Judging loop ────────────────────────────────────────────────────────────
    all_results = list(completed.values())
    errors = 0
    total_tokens = 0

    print(f"\n{'idx':>6}  {'db_id':<30}  {'winner':<12}  {'tokens':>8}  {'reason'}")
    print("─" * 90)

    for i, entry in enumerate(todo):
        try:
            result = judge_pair(client, entry, rng)
        except Exception as exc:
            print(f"{entry['spider_index']:>6}  {'ERROR':<30}  {str(exc)[:50]}")
            errors += 1
            # Don't add to results — will be retried on next run
            time.sleep(2)   # back off on errors
            continue

        all_results.append(result)

        winner_display = result.get("judge_winner") or "PARSE_ERR"
        tokens = result.get("judge_usage", {}).get("total_tokens", 0)
        total_tokens += tokens
        reason_preview = (result.get("judge_reason") or "")[:55]

        print(f"{entry['spider_index']:>6}  {entry['db_id']:<30}  {winner_display:<12}  {tokens:>8}  {reason_preview}")

        # Checkpoint every N entries
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(JUDGED_OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [checkpoint saved — {i+1}/{len(todo)} done, {total_tokens:,} tokens so far]")

        if args.dry_run:
            print("\n[dry-run] First call successful. Full response:")
            print(f"  Raw:    {result.get('judge_raw_response', '')[:300]}")
            print(f"  Winner: {result.get('judge_winner')}")
            print(f"  Reason: {result.get('judge_reason')}")
            print(f"  Chosen SQL:   {result.get('chosen_sql', '')[:100]}")
            print(f"  Rejected SQL: {result.get('rejected_sql', '')[:100]}")
            return

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Final save
    with open(JUDGED_OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print run summary ──────────────────────────────────────────────────────
    judged_this_run = len(todo) - errors
    parse_errors    = sum(1 for r in all_results if r.get("judge_winner") is None)
    grok_wins       = sum(1 for r in all_results if r.get("judge_winner") == "grok")
    ds_wins         = sum(1 for r in all_results if r.get("judge_winner") == "deepseek")
    total_judged    = len([r for r in all_results if r.get("judge_winner") is not None])

    print(f"\n{'=' * 60}")
    print("RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Judged this run:   {judged_this_run}")
    print(f"  API errors:        {errors}")
    print(f"  Parse errors:      {parse_errors}")
    print(f"  Total judged:      {total_judged} / {len(queue)}")
    print(f"  Total tokens used: {total_tokens:,}")
    est_cost = (total_tokens / 1_000_000) * 0.25   # blended in/out rate
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
        print(f"  When complete, run: python -m src.llm_judge --merge-only")


if __name__ == "__main__":
    main()
