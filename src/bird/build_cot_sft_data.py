"""
ExCoT Step 2.1 — Generate execution-verified CoT seed data for CoT-SFT.

A strong teacher (Qwen3-Coder-480B on W&B Inference) produces chain-of-thought
+ SQL for each BIRD question; we execute the final SQL and KEEP ONLY the traces
whose result matches the gold result. Output is alpaca format ready for
src.bird.sft_train, where `output` is the full reasoning + final ```sql block.

Execution-filtering is what makes a ~52%-on-BIRD teacher safe to learn from:
we discard its wrong CoTs, so the SFT seed is 100% correct reasoning.

CPU + API only — NO GPU. Runs locally (dev smoke) or on the HPC login node
(real train run). Needs a WANDB_API_KEY in .env and the databases to execute.

Smoke (local, ~cents — uses BIRD dev DBs you have locally):
  python -m src.bird.build_cot_sft_data \
      --questions-json data/bird_data/dev.json \
      --db-dir         data/bird_data/dev_databases \
      --output-file    /tmp/cot_seed_smoke.json \
      --max-questions  5

Real (HPC login node, BIRD train set):
  python -m src.bird.build_cot_sft_data \
      --questions-json /scratch/phalle.y/bird_train/train/train.json \
      --db-dir         /scratch/phalle.y/bird_train/train/train_databases \
      --output-file    /scratch/phalle.y/cot_sft_data.json \
      --concurrency    8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

# Read-only reuse of the existing pipeline (prompt + executor + metric).
from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results
from src.bird.inference import build_cot_instruction, extract_final_sql, COT_SYSTEM_PROMPT

load_dotenv()

DEFAULT_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
EXEC_TIMEOUT = 15


# ─────────────────────────────────────────────────────────────────────────────
# W&B INFERENCE CLIENT (self-contained — does not touch shared/llm_client.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_client(base_url: str, project: str | None) -> OpenAI:
    key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_INFERENCE_API_KEY")
    if not key:
        sys.exit("Missing API key. Add WANDB_API_KEY=... to your .env "
                 "(get one at https://wandb.ai/authorize).")
    kwargs: dict = {"api_key": key, "base_url": base_url}
    if project:
        kwargs["default_headers"] = {"OpenAI-Project": project}
    return OpenAI(**kwargs)


def generate_cot(client, model, schema, question, evidence, max_tokens, temperature, retries=3):
    """Return (full_cot_text, extracted_final_sql, usage_dict)."""
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": build_cot_instruction(question, schema, evidence)},
    ]
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            return text, extract_final_sql(THINK_RE.sub("", text)), {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise  # config error — never succeeds on retry
            last_err = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API call failed after {retries} retries: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_db_path(db_dir: Path, db_id: str) -> str | None:
    p = Path(db_dir) / db_id / f"{db_id}.sqlite"
    return str(p) if p.exists() else None


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    schema = "\n\n".join(r[0] for r in cur.fetchall() if r[0])
    conn.close()
    return schema


def load_questions(questions_json: Path, db_dir: Path) -> list[dict]:
    """Works for BIRD train.json or dev.json (both have db_id/question/evidence/SQL)."""
    with open(questions_json) as f:
        data = json.load(f)
    out = []
    for i, q in enumerate(data):
        db_id = q["db_id"]
        db_path = get_db_path(db_dir, db_id)
        if not db_path:
            continue
        out.append({
            "question_id": q.get("question_id", i),
            "db_id": db_id,
            "question": q["question"],
            "evidence": q.get("evidence", ""),
            "gold_sql": q.get("SQL") or q.get("query", ""),
            "db_path": db_path,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ExCoT 2.1 — execution-verified CoT seed via W&B teacher")
    ap.add_argument("--questions-json", type=Path, required=True, help="BIRD train.json or dev.json")
    ap.add_argument("--db-dir", type=Path, required=True, help="dir of <db_id>/<db_id>.sqlite")
    ap.add_argument("--output-file", type=Path, required=True, help="alpaca CoT seed JSON (for sft_train)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="W&B teacher model ID")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--project", default=None, help="W&B team/project for usage attribution (optional)")
    ap.add_argument("--max-questions", type=int, default=None, help="cap for smoke test / budget")
    ap.add_argument("--max-tokens", type=int, default=2048, help="room for reasoning + SQL")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-execution", action="store_true",
                    help="keep ALL traces unfiltered (inspect CoT format without DBs)")
    args = ap.parse_args()

    questions = load_questions(args.questions_json, args.db_dir)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} questions with valid databases")
    if not questions:
        sys.exit("No questions with resolvable DBs — check --db-dir layout (<db_id>/<db_id>.sqlite).")

    client = make_client(args.base_url, args.project)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Schema cache (one read per db_id)
    schema_cache: dict[str, str] = {}

    def work(item):
        idx, q = item
        schema = schema_cache.get(q["db_id"]) or get_schema(q["db_path"])
        schema_cache[q["db_id"]] = schema
        try:
            cot_text, gen_sql, usage = generate_cot(
                client, args.model, schema, q["question"], q["evidence"],
                args.max_tokens, args.temperature,
            )
        except Exception as e:  # noqa: BLE001
            return {"keep": False, "reason": f"api_error: {e}", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        if args.skip_execution:
            correct = True
        else:
            gen_exec = execute_sqlite_query(gen_sql, q["db_path"], timeout=EXEC_TIMEOUT)
            gold_exec = execute_sqlite_query(q["gold_sql"], q["db_path"], timeout=EXEC_TIMEOUT)
            correct = compare_results(gen_exec, gold_exec)["result_match"]

        example = None
        if correct:
            example = {
                "instruction": build_cot_instruction(q["question"], schema, q["evidence"]),
                "input": "",
                "output": cot_text,           # full reasoning + final ```sql block
            }
        return {"keep": correct, "example": example, "usage": usage}

    kept, attempted = [], 0
    tok_in = tok_out = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for r in pool.map(work, list(enumerate(questions))):
            attempted += 1
            tok_in += r["usage"]["prompt_tokens"]
            tok_out += r["usage"]["completion_tokens"]
            if r.get("keep") and r.get("example"):
                kept.append(r["example"])
            if attempted % 50 == 0:
                print(f"  [{attempted}/{len(questions)}] kept={len(kept)} "
                      f"({len(kept)/attempted:.0%}) | toks {tok_in:,}/{tok_out:,} "
                      f"| {(time.time()-start)/60:.1f}min", flush=True)
                json.dump(kept, open(args.output_file, "w"), indent=2)  # intermediate save

    json.dump(kept, open(args.output_file, "w"), indent=2)

    print(f"\n{'='*70}")
    print(f"  COT SEED DONE")
    print(f"{'='*70}")
    print(f"  Attempted:   {attempted}")
    print(f"  Kept (correct CoT): {len(kept)}  ({len(kept)/max(1,attempted):.1%} yield)")
    print(f"  Tokens:      {tok_in:,} in / {tok_out:,} out")
    print(f"  Saved:       {args.output_file}")
    print(f"{'='*70}")
    if kept:
        print("\n--- first kept example (output preview, 500 chars) ---")
        print(kept[0]["output"][:500])


if __name__ == "__main__":
    main()
