"""
GRPO dataset for the Spider TRAIN set.

Each row: {prompt, gold_sql, db_path} — the prompt is built with the SAME template
as src/spider/eval_finetuned.py, so GRPO optimizes exactly what we score at eval
(no train/eval prompt mismatch). gold_sql + db_path feed the execution reward.

    python -m src.spider.grpo_dataset \
        --train-json /home/phalle.y/Jenish-DPO-GRPO/train_spider.json \
        --data-dir   /home/phalle.y/Jenish-DPO-GRPO --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from src.shared.schema_loader import get_schema_from_sqlite, get_db_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Identical to src/spider/eval_finetuned.py so GRPO trains on the eval prompt.
SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def build_chat_prompt(schema: str, question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### Database Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL Query:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_grpo_dataset(train_json: Path, data_dir: Path, tokenizer=None, limit: int | None = None) -> Dataset:
    with open(train_json) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    schema_cache: dict[str, str] = {}
    rows, skipped = [], 0
    for q in questions:
        db_id = q["db_id"]
        question = q["question"]
        gold_sql = q.get("query") or q.get("SQL", "")   # Spider uses "query"
        if not gold_sql:
            skipped += 1
            continue
        try:
            db_path = get_db_path(str(data_dir), db_id)
            if db_id not in schema_cache:
                schema_cache[db_id] = get_schema_from_sqlite(db_path)
        except FileNotFoundError:
            skipped += 1
            continue
        schema = schema_cache[db_id]
        prompt = build_chat_prompt(schema, question, tokenizer) if tokenizer else f"{schema}\n\n{question}"
        rows.append({"prompt": prompt, "gold_sql": gold_sql, "db_path": db_path,
                     "db_id": db_id, "question": question})

    print(f"Spider GRPO dataset: {len(rows)} examples loaded, {skipped} skipped")
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify the Spider GRPO dataset loads")
    ap.add_argument("--train-json", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ds = build_grpo_dataset(a.train_json, a.data_dir, limit=a.limit)
    print(f"\nsize: {len(ds)} | keys: {list(ds[0].keys())}")
    print(f"gold_sql: {ds[0]['gold_sql'][:80]!r}")
    print(f"db_path:  {ds[0]['db_path']}")
