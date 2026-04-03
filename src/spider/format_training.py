"""
Step 3: Format preference pairs into LLaMA-Factory training format.

WHY TWO DATASETS?
─────────────────
Fine-tuning a model for text-to-SQL usually requires two stages:

  Stage 1 — SFT (Supervised Fine-Tuning)
    The base Qwen model has never specifically learned to generate SQL from a
    schema + question prompt. We first train it on (prompt → correct SQL) pairs
    so it learns the basic task before we teach preferences. Without this stage,
    DPO can fail because the model's outputs are too random to learn from.
    Source: Spider train set — 7,000 gold question/SQL pairs.

  Stage 2 — DPO (Direct Preference Optimization)
    Now we refine the model using our preference pairs: for the same prompt,
    prefer chosen_sql over rejected_sql. The model learns SQL style, best
    practices, and which patterns are better.
    Source: preference_pairs_final.json — 1,042 pairs we built in Steps 1 & 2.

WHAT "FORMATTING" MEANS
────────────────────────
LLaMA-Factory reads data from JSON files. It needs fields with specific names
in a specific structure. This script is a pure data transformation — it reads
our files and writes them out in the exact shape LLaMA-Factory expects.

No model calls, no training — just JSON reshaping.

LLaMA-FACTORY FORMATS
──────────────────────
SFT (alpaca format):
  {
    "instruction": "Convert the following question to SQL...\nSchema: ...\nQuestion: ...",
    "input": "",          ← always empty in our case (everything is in instruction)
    "output": "SELECT COUNT(*) FROM singer;"
  }

DPO (pairwise format):
  {
    "instruction": "Convert the following question to SQL...\nSchema: ...\nQuestion: ...",
    "input": "",
    "chosen": "SELECT COUNT(*) FROM singer;",
    "rejected": "SELECT COUNT(*) FROM \"singer\";"
  }

dataset_info.json tells LLaMA-Factory which fields map to which roles and
whether to treat the dataset as ranking (DPO) or standard (SFT).

OUTPUT
──────
data/training/
  sft_data.json         — 7,000 SFT pairs from Spider train set
  dpo_data.json         — 1,042 DPO pairs from our preference pipeline
  dataset_info.json     — LLaMA-Factory dataset registry
  formatting_stats.json — counts, lengths, quality checks
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.shared.schema_loader import get_schema_from_sqlite, get_db_path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR       = PROJECT_ROOT / "results"
SPIDER_DATA_DIR   = PROJECT_ROOT / "data" / "spider_data"
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"

FINAL_PAIRS_FILE  = RESULTS_DIR / "preference_pairs_final.json"
SPIDER_TRAIN_FILE = SPIDER_DATA_DIR / "train_spider.json"

SFT_OUTPUT_FILE   = TRAINING_DATA_DIR / "sft_data.json"
DPO_OUTPUT_FILE   = TRAINING_DATA_DIR / "dpo_data.json"
DATASET_INFO_FILE = TRAINING_DATA_DIR / "dataset_info.json"
STATS_FILE        = TRAINING_DATA_DIR / "formatting_stats.json"


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTION TEMPLATE
# This MUST match what you use at inference time. If you change this template
# during training, the model will expect a different format when you evaluate.
# ─────────────────────────────────────────────────────────────────────────────
def build_instruction(question: str, schema: str) -> str:
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def is_degenerate(chosen: str, rejected: str) -> bool:
    """
    A pair is degenerate if chosen and rejected are identical after normalization.
    These provide no learning signal — skip them.
    """
    def normalize(s: str) -> str:
        return " ".join(s.strip().lower().rstrip(";").split())
    return normalize(chosen) == normalize(rejected)


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Build SFT data from Spider train set
# ─────────────────────────────────────────────────────────────────────────────
def build_sft_data() -> tuple[list[dict], dict]:
    """
    Load the Spider train set (7,000 examples) and format each one as:
      {instruction: "...", input: "", output: "gold_sql"}

    The instruction includes the full schema DDL loaded from the SQLite file.
    Schemas are cached by db_id — most of the 7,000 examples share databases.
    """
    print("\n[SFT] Loading Spider train set...")
    with open(SPIDER_TRAIN_FILE) as f:
        train_data = json.load(f)
    print(f"      {len(train_data)} examples loaded")

    # Cache schemas so we only load each SQLite file once
    schema_cache: dict[str, str] = {}
    schema_errors: list[str] = []

    # Unique db_ids needed
    needed_dbs = {ex["db_id"] for ex in train_data}
    print(f"      Loading schemas for {len(needed_dbs)} unique databases...")

    for db_id in sorted(needed_dbs):
        try:
            db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)
            schema_cache[db_id] = get_schema_from_sqlite(db_path)
        except FileNotFoundError:
            schema_errors.append(db_id)

    if schema_errors:
        print(f"      WARNING: Missing schemas for {len(schema_errors)} databases: {schema_errors[:5]}")

    # Build formatted examples
    sft_examples: list[dict] = []
    skipped_missing_schema = 0
    token_lengths: list[int] = []

    for ex in train_data:
        db_id    = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]       # Spider uses "query" as the field name

        if db_id not in schema_cache:
            skipped_missing_schema += 1
            continue

        schema      = schema_cache[db_id]
        instruction = build_instruction(question, schema)

        sft_examples.append({
            "instruction": instruction,
            "input":       "",          # LLaMA-Factory expects this field even if empty
            "output":      gold_sql,
        })

        token_lengths.append(estimate_tokens(instruction + gold_sql))

    stats = {
        "total_spider_train":         len(train_data),
        "sft_examples_produced":      len(sft_examples),
        "skipped_missing_schema":     skipped_missing_schema,
        "avg_tokens_per_example":     int(sum(token_lengths) / len(token_lengths)) if token_lengths else 0,
        "max_tokens":                 max(token_lengths) if token_lengths else 0,
        "examples_over_2048_tokens":  sum(1 for t in token_lengths if t > 2048),
        "examples_over_4096_tokens":  sum(1 for t in token_lengths if t > 4096),
    }

    return sft_examples, stats


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Build DPO data from preference pairs
# ─────────────────────────────────────────────────────────────────────────────
def build_dpo_data() -> tuple[list[dict], dict]:
    """
    Load preference_pairs_final.json and format each pair as:
      {instruction: "...", input: "", chosen: "...", rejected: "..."}

    Applies quality filters:
      - Skip degenerate pairs (chosen == rejected after normalization)
      - Track category breakdown for stats
    """
    print("\n[DPO] Loading preference pairs...")
    with open(FINAL_PAIRS_FILE) as f:
        pairs = json.load(f)
    print(f"      {len(pairs)} pairs loaded")

    dpo_examples: list[dict] = []
    skipped_degenerate  = 0
    skipped_null        = 0
    category_counts     = Counter()
    chosen_source_counts = Counter()
    token_lengths: list[int] = []

    # Track which spider_index values we've seen — avoid duplicate prompts
    # (a single question could appear as both gold_vs_grok AND gold_vs_deepseek,
    #  meaning the same instruction appears twice with different rejected SQL.
    #  Both are valid training signal, so we keep both.)
    for pair in pairs:
        chosen_sql   = pair.get("chosen_sql")
        rejected_sql = pair.get("rejected_sql")

        # Skip if either SQL is missing (parse error from judge)
        if not chosen_sql or not rejected_sql:
            skipped_null += 1
            continue

        # Skip degenerate pairs
        if is_degenerate(chosen_sql, rejected_sql):
            skipped_degenerate += 1
            continue

        instruction = pair["instruction"]

        dpo_examples.append({
            "instruction": instruction,
            "input":       "",
            "chosen":      chosen_sql,
            "rejected":    rejected_sql,
        })

        category_counts[pair["category"]] += 1
        chosen_source_counts[pair["chosen_source"]] += 1
        token_lengths.append(estimate_tokens(instruction + chosen_sql + rejected_sql))

    stats = {
        "total_pairs_loaded":         len(pairs),
        "dpo_examples_produced":      len(dpo_examples),
        "skipped_null_sql":           skipped_null,
        "skipped_degenerate":         skipped_degenerate,
        "category_breakdown":         dict(category_counts),
        "chosen_source_breakdown":    dict(chosen_source_counts),
        "avg_tokens_per_example":     int(sum(token_lengths) / len(token_lengths)) if token_lengths else 0,
        "max_tokens":                 max(token_lengths) if token_lengths else 0,
        "examples_over_4096_tokens":  sum(1 for t in token_lengths if t > 4096),
    }

    return dpo_examples, stats


# ─────────────────────────────────────────────────────────────────────────────
# DATASET_INFO.JSON
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset_info() -> dict:
    """
    LLaMA-Factory requires a dataset_info.json that registers your datasets.

    Each entry tells LLaMA-Factory:
      - Which file to load
      - What each field means (prompt, response, chosen, rejected)
      - Whether it's a ranking dataset (DPO) or standard (SFT)

    When you run LLaMA-Factory training, you reference these dataset names
    in your training YAML config.
    """
    return {
        # ── SFT dataset ───────────────────────────────────────────────────────
        # Used in Stage 1: trains the model to generate SQL from schema + question
        "spider_sft": {
            "file_name": "sft_data.json",
            "columns": {
                "prompt":    "instruction",   # the schema+question input
                "query":     "input",         # always empty string in our data
                "response":  "output",        # the gold SQL to learn
            },
        },

        # ── DPO dataset ───────────────────────────────────────────────────────
        # Used in Stage 2: trains the model to prefer chosen over rejected SQL
        "spider_dpo": {
            "file_name":  "dpo_data.json",
            "ranking":    True,              # tells LLaMA-Factory this is pairwise data
            "columns": {
                "prompt":    "instruction",   # same schema+question input
                "query":     "input",         # always empty string
                "chosen":    "chosen",        # the better SQL
                "rejected":  "rejected",      # the worse SQL
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 3: Format Training Data for LLaMA-Factory")
    print("=" * 60)

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build SFT data ────────────────────────────────────────────────────────
    sft_examples, sft_stats = build_sft_data()
    with open(SFT_OUTPUT_FILE, "w") as f:
        json.dump(sft_examples, f, indent=2)
    print(f"      ✓ Wrote {len(sft_examples)} SFT examples → {SFT_OUTPUT_FILE.name}")

    # ── Build DPO data ────────────────────────────────────────────────────────
    dpo_examples, dpo_stats = build_dpo_data()
    with open(DPO_OUTPUT_FILE, "w") as f:
        json.dump(dpo_examples, f, indent=2)
    print(f"      ✓ Wrote {len(dpo_examples)} DPO examples → {DPO_OUTPUT_FILE.name}")

    # ── Write dataset_info.json ───────────────────────────────────────────────
    dataset_info = build_dataset_info()
    with open(DATASET_INFO_FILE, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"      ✓ Wrote dataset registry → {DATASET_INFO_FILE.name}")

    # ── Write stats ───────────────────────────────────────────────────────────
    stats = {"sft": sft_stats, "dpo": dpo_stats}
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  SFT Stage (train the task):")
    print(f"    Examples:          {sft_stats['sft_examples_produced']:,}")
    print(f"    Avg tokens/ex:     {sft_stats['avg_tokens_per_example']:,}")
    print(f"    Max tokens:        {sft_stats['max_tokens']:,}")
    print(f"    Over 4096 tokens:  {sft_stats['examples_over_4096_tokens']}")

    print(f"\n  DPO Stage (refine preferences):")
    print(f"    Examples:          {dpo_stats['dpo_examples_produced']:,}")
    print(f"    Skipped degen.:    {dpo_stats['skipped_degenerate']}")
    print(f"    Avg tokens/ex:     {dpo_stats['avg_tokens_per_example']:,}")

    print(f"\n  DPO category breakdown:")
    for cat, n in dpo_stats["category_breakdown"].items():
        print(f"    {cat:<22}: {n}")

    print(f"\n  DPO chosen source breakdown:")
    for src, n in dpo_stats["chosen_source_breakdown"].items():
        print(f"    {src:<22}: {n}")

    print(f"\n  Output files:")
    for f in [SFT_OUTPUT_FILE, DPO_OUTPUT_FILE, DATASET_INFO_FILE, STATS_FILE]:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<30} {size_kb:>6} KB")

    print(f"\n  Next step: copy data/training/ to your GPU machine and")
    print(f"  point LLaMA-Factory's dataset_dir at it in your YAML config.")
    print(f"\nStep 3 complete ✓")


if __name__ == "__main__":
    main()
