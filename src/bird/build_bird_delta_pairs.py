"""
Build delta learning preference pairs from Qwen 3B and 1.5B BIRD train results.

DELTA LEARNING HYPOTHESIS (Geng et al., COLM 2025 — arXiv:2507.06187):
    Preference pairs where both chosen and rejected are weak (small models)
    can still improve a stronger student via DPO, because the *relative
    quality delta* drives learning — even when absolute quality is low.
    SFT on the same weak data hurts; DPO on the pairs helps.

PAIR CATEGORIES:
    1. clean_delta      — 3B correct, 1.5B wrong → strongest signal
                          chosen = qwen3b_sql, rejected = qwen1b_sql

    2. both_correct     — both correct, different SQL → size as quality proxy
                          chosen = qwen3b_sql, rejected = qwen1b_sql
                          (same result, but 3B is "more capable" model)

    3. gold_fallback    — both wrong → gold SQL as chosen
                          chosen = gold_sql, rejected = qwen1b_sql   (pair A)
                          chosen = gold_sql, rejected = qwen3b_sql   (pair B)
                          (maximum delta; consistent with paper spirit)

    4. skip_inverted    — 1.5B correct, 3B wrong → inverted quality signal,
                          cannot trust size as proxy here → discard

    5. skip_identical   — both correct, identical SQL → no delta → discard

Usage:
    python src/bird/build_bird_delta_pairs.py \\
        --qwen3b  ~/results/bird_local_Qwen_Qwen2-5-Coder-3B-Instruct_XXXX.json \\
        --qwen1b  ~/results/bird_local_Qwen_Qwen2-5-Coder-1-5B-Instruct_XXXX.json \\
        --output  ~/results/bird_delta_pairs.json

The output file is fed directly into LLaMA-Factory format_training (or a
format_training script you write for BIRD delta pairs).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict[int, dict]:
    """Load a local_pipeline output file, keyed by question_id."""
    with open(path) as f:
        data = json.load(f)
    return {entry["question_id"]: entry for entry in data["results"]}


def sqls_are_equivalent(sql_a: str, sql_b: str) -> bool:
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().rstrip(";").split())
    return normalize(sql_a) == normalize(sql_b)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_delta_pairs(
    qwen3b_file: Path,
    qwen1b_file: Path,
    output_file: Path,
):
    print("=" * 60)
    print("Delta Learning: Build BIRD Preference Pairs")
    print("=" * 60)

    # ── Load results ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading result files...")
    qwen3b = load_results(qwen3b_file)
    qwen1b = load_results(qwen1b_file)
    print(f"      Qwen 3B entries:  {len(qwen3b)}")
    print(f"      Qwen 1.5B entries: {len(qwen1b)}")

    shared_ids = set(qwen3b.keys()) & set(qwen1b.keys())
    if len(shared_ids) != len(qwen3b):
        print(f"      WARNING: files don't have identical question_ids. Using {len(shared_ids)} shared.")
    question_ids = sorted(shared_ids)
    print(f"      Matched on {len(question_ids)} shared questions")

    # ── Categorize ────────────────────────────────────────────────────────────
    print(f"\n[2/3] Building preference pairs...")

    pairs: list[dict] = []
    counts = defaultdict(int)

    for qid in question_ids:
        r3b  = qwen3b[qid]
        r1b  = qwen1b[qid]

        db_id      = r3b["db_id"]
        question   = r3b["question"]
        evidence   = r3b.get("evidence", "")
        gold_sql   = r3b["gold_sql"]
        difficulty = r3b.get("difficulty", "unknown")
        instruction = r3b.get("instruction", "")

        qwen3b_correct = r3b["eval"]["result_match"]
        qwen1b_correct = r1b["eval"]["result_match"]
        qwen3b_sql     = r3b["generated_sql"]
        qwen1b_sql     = r1b["generated_sql"]

        # Skip if either model had a generation error (empty SQL)
        if not qwen3b_sql.strip() or not qwen1b_sql.strip():
            counts["skip_error"] += 1
            continue

        base = {
            "question_id": qid,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "instruction": instruction,
            "qwen3b_sql": qwen3b_sql,
            "qwen1b_sql": qwen1b_sql,
            "qwen3b_correct": qwen3b_correct,
            "qwen1b_correct": qwen1b_correct,
        }

        # ── CATEGORY 1: clean delta — strongest signal ────────────────────────
        if qwen3b_correct and not qwen1b_correct:
            counts["clean_delta"] += 1
            pairs.append({
                **base,
                "category": "clean_delta",
                "chosen_sql": qwen3b_sql,
                "chosen_source": "qwen3b",
                "rejected_sql": qwen1b_sql,
                "rejected_source": "qwen1b",
            })

        # ── CATEGORY 4: inverted — 1.5B beats 3B, skip ───────────────────────
        elif qwen1b_correct and not qwen3b_correct:
            counts["skip_inverted"] += 1
            # Cannot trust size as quality proxy — discard

        # ── CATEGORY 2: both correct ──────────────────────────────────────────
        elif qwen3b_correct and qwen1b_correct:
            if sqls_are_equivalent(qwen3b_sql, qwen1b_sql):
                counts["skip_identical"] += 1
                # No delta — discard
            else:
                # Different SQL, both correct. Use size as quality proxy:
                # 3B is "more capable" so prefer its output.
                counts["both_correct"] += 1
                pairs.append({
                    **base,
                    "category": "both_correct",
                    "chosen_sql": qwen3b_sql,
                    "chosen_source": "qwen3b",
                    "rejected_sql": qwen1b_sql,
                    "rejected_source": "qwen1b",
                })

        # ── CATEGORY 3: both wrong — gold SQL fallback ────────────────────────
        else:  # not qwen3b_correct and not qwen1b_correct
            counts["gold_fallback"] += 1

            # Pair A: gold vs 1.5B (larger delta — 1.5B is weaker)
            pairs.append({
                **base,
                "category": "gold_fallback",
                "chosen_sql": gold_sql,
                "chosen_source": "gold",
                "rejected_sql": qwen1b_sql,
                "rejected_source": "qwen1b",
                "pair_variant": "gold_vs_qwen1b",
            })

            # Pair B: gold vs 3B (smaller delta — 3B is stronger but still wrong)
            pairs.append({
                **base,
                "category": "gold_fallback",
                "chosen_sql": gold_sql,
                "chosen_source": "gold",
                "rejected_sql": qwen3b_sql,
                "rejected_source": "qwen3b",
                "pair_variant": "gold_vs_qwen3b",
            })

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving {len(pairs)} pairs → {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(pairs, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_q = len(question_ids)
    qwen3b_acc = sum(1 for qid in question_ids if qwen3b[qid]["eval"]["result_match"]) / total_q
    qwen1b_acc = sum(1 for qid in question_ids if qwen1b[qid]["eval"]["result_match"]) / total_q

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel accuracy on BIRD train set:")
    print(f"  Qwen 3B:   {qwen3b_acc:.1%}  ({int(qwen3b_acc * total_q)}/{total_q})")
    print(f"  Qwen 1.5B: {qwen1b_acc:.1%}  ({int(qwen1b_acc * total_q)}/{total_q})")

    print(f"\nPair categories:")
    print(f"  clean_delta   (3B right, 1.5B wrong):  {counts['clean_delta']:>5} → {counts['clean_delta']:>5} pairs")
    print(f"  both_correct  (both right, diff SQL):   {counts['both_correct']:>5} → {counts['both_correct']:>5} pairs")
    print(f"  gold_fallback (both wrong):             {counts['gold_fallback']:>5} → {counts['gold_fallback'] * 2:>5} pairs")
    print(f"  skip_inverted (1.5B beat 3B):           {counts['skip_inverted']:>5}")
    print(f"  skip_identical (both right, same SQL):  {counts['skip_identical']:>5}")
    print(f"  skip_error    (empty SQL):              {counts['skip_error']:>5}")
    print(f"  {'─' * 44}")
    print(f"  Total pairs produced:                   {len(pairs):>5}")

    print(f"\nOutput: {output_file}")
    print(f"\nNext step: format pairs for LLaMA-Factory DPO training.")
    print(f"  DPO must be applied on top of the BIRD SFT checkpoint")
    print(f"  (~/checkpoints/bird_sft/), NOT the already-DPO'd model.\n")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HOME = Path.home()

    parser = argparse.ArgumentParser(
        description="Build delta learning preference pairs from Qwen 3B and 1.5B results"
    )
    parser.add_argument(
        "--qwen3b", type=Path, required=True,
        help="Path to local_pipeline output for Qwen2.5-Coder-3B-Instruct"
    )
    parser.add_argument(
        "--qwen1b", type=Path, required=True,
        help="Path to local_pipeline output for Qwen2.5-Coder-1.5B-Instruct"
    )
    parser.add_argument(
        "--output", type=Path,
        default=HOME / "results" / "bird_delta_pairs.json",
        help="Output file path (default: ~/results/bird_delta_pairs.json)"
    )
    args = parser.parse_args()

    build_delta_pairs(
        qwen3b_file=args.qwen3b,
        qwen1b_file=args.qwen1b,
        output_file=args.output,
    )
