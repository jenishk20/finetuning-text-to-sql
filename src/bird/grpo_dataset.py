"""
Format BIRD train set as a HuggingFace Dataset for GRPO training.

Each example contains:
  - prompt: the formatted instruction (schema + question + evidence)
  - gold_sql: ground truth SQL for reward computation
  - db_path: absolute path to the SQLite database

Run to verify the dataset loads correctly:
    python -m src.bird.grpo_dataset \
        --train-json /scratch/phalle.y/bird_train/train/train.json \
        --db-dir /scratch/phalle.y/bird_train/train/train_databases/train_databases
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from datasets import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE — must match grpo_train.py and eval scripts exactly
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def build_instruction(question: str, schema: str, evidence: str = "") -> str:
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


def build_chat_prompt(question: str, schema: str, evidence: str, tokenizer) -> str:
    """Build a fully formatted chat prompt string using the tokenizer's template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_instruction(question, schema, evidence)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    )
    tables = cursor.fetchall()
    conn.close()
    return "\n\n".join(sql for _, sql in tables if sql)


def resolve_db_path(db_id: str, db_dir: Path) -> str | None:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)
    db_subdir = db_dir / db_id
    if db_subdir.exists():
        for f in db_subdir.glob("*.sqlite"):
            return str(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_dataset(
    train_json: Path,
    db_dir: Path,
    tokenizer=None,
) -> Dataset:
    """
    Load BIRD train.json and return a HuggingFace Dataset ready for GRPOTrainer.

    Each row:
      prompt    — formatted chat string (if tokenizer provided) OR raw instruction
      gold_sql  — ground truth SQL for reward function
      db_path   — absolute path to the SQLite database
      db_id     — database name
      question  — original question text
      evidence  — domain knowledge hint
    """
    with open(train_json) as f:
        questions = json.load(f)

    schema_cache: dict[str, str] = {}
    rows = []
    skipped = 0

    for i, q in enumerate(questions):
        db_id    = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "").strip()
        gold_sql = q.get("SQL") or q.get("query", "")

        if not gold_sql:
            skipped += 1
            continue

        db_path = resolve_db_path(db_id, db_dir)
        if db_path is None:
            skipped += 1
            continue

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = get_schema(db_path)
            except Exception:
                skipped += 1
                continue

        schema = schema_cache[db_id]

        if tokenizer is not None:
            prompt = build_chat_prompt(question, schema, evidence, tokenizer)
        else:
            prompt = build_instruction(question, schema, evidence)

        rows.append({
            "prompt":   prompt,
            "gold_sql": gold_sql,
            "db_path":  db_path,
            "db_id":    db_id,
            "question": question,
            "evidence": evidence,
        })

    print(f"Dataset: {len(rows)} examples loaded, {skipped} skipped")
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GRPO dataset loading")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--db-dir",     type=Path, required=True)
    args = parser.parse_args()

    ds = build_grpo_dataset(args.train_json, args.db_dir)
    print(f"\nDataset size: {len(ds)}")
    print(f"\nSample row keys: {list(ds[0].keys())}")
    print(f"\nSample prompt (first 300 chars):\n{ds[0]['prompt'][:300]}")
    print(f"\nSample gold_sql: {ds[0]['gold_sql']}")
    print(f"\nSample db_path:  {ds[0]['db_path']}")
